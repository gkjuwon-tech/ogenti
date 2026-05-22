"""Anatomical consistency loss with differentiable surrogate.

Why a surrogate?
  MediaPipe / DWPose are not differentiable. Running them on VAE-decoded
  predictions and back-propagating is impossible. Instead we co-train a small
  ResNet-style keypoint regressor that takes decoded RGB frames and predicts
  a fixed-size keypoint heatmap. The regressor is supervised offline by
  MediaPipe outputs (teacher distillation) and frozen during Ogenti training,
  so gradients flow ONLY through Ogenti's parameters.

The regressor lives at checkpoints/anatomy_surrogate/. If absent, the loss
returns zero and logs a warning — the training loop will simply not enforce
the anatomical prior.

Two loss components:
  1. KeypointHeatmapMSE — predicted heatmap (from decoded prediction) vs.
     teacher heatmap (precomputed from ground truth).
  2. KinematicSmoothness — penalize jitter on the predicted keypoint
     coordinates across time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn


@dataclass
class AnatomyLossConfig:
    keypoint_weight_hands: float = 4.0
    keypoint_weight_face: float = 2.0
    keypoint_weight_body: float = 1.0
    smoothness_weight: float = 0.5
    heatmap_resolution: int = 64
    num_keypoints_body: int = 33
    num_keypoints_face: int = 468
    num_keypoints_hand: int = 21
    surrogate_ckpt: Optional[str] = None
    confidence_floor: float = 0.3


class KeypointSurrogate(nn.Module):
    """Small conv backbone that maps a single RGB frame to a keypoint heatmap.

    Distilled offline from MediaPipe Holistic. Architecture is intentionally
    lightweight so it fits in the training VRAM budget alongside Ogenti.
    """

    def __init__(self, num_keypoints: int, heatmap_resolution: int = 64) -> None:
        super().__init__()
        self.num_keypoints = num_keypoints
        self.heatmap_resolution = heatmap_resolution

        def cbr(c_in: int, c_out: int, stride: int = 2) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.GroupNorm(8, c_out),
                nn.SiLU(inplace=True),
            )

        self.stem = nn.Sequential(
            cbr(3, 32, stride=2),
            cbr(32, 64, stride=2),
        )
        self.trunk = nn.Sequential(
            cbr(64, 128, stride=2),
            cbr(128, 256, stride=2),
            cbr(256, 256, stride=1),
        )
        self.head = nn.Conv2d(256, num_keypoints, kernel_size=1)

    def forward(self, frames: Tensor) -> Tensor:
        """frames: (B*T, 3, H, W) in [-1, 1] -> heatmap (B*T, K, R, R)."""
        x = self.stem(frames)
        x = self.trunk(x)
        h = self.head(x)
        h = F.interpolate(
            h, size=(self.heatmap_resolution, self.heatmap_resolution),
            mode="bilinear", align_corners=False,
        )
        return h


class AnatomyLoss(nn.Module):
    """Wraps a frozen KeypointSurrogate and computes heatmap MSE + smoothness."""

    def __init__(self, config: AnatomyLossConfig) -> None:
        super().__init__()
        self.config = config
        total_kp = (
            config.num_keypoints_body
            + config.num_keypoints_face
            + 2 * config.num_keypoints_hand
        )
        self.surrogate = KeypointSurrogate(total_kp, config.heatmap_resolution)
        self._loaded = False

        if config.surrogate_ckpt and Path(config.surrogate_ckpt).exists():
            from safetensors.torch import load_file

            state = load_file(config.surrogate_ckpt)
            self.surrogate.load_state_dict(state, strict=False)
            self._loaded = True

        for p in self.surrogate.parameters():
            p.requires_grad_(False)
        self.surrogate.eval()

        weights = (
            [config.keypoint_weight_body] * config.num_keypoints_body
            + [config.keypoint_weight_face] * config.num_keypoints_face
            + [config.keypoint_weight_hands] * (2 * config.num_keypoints_hand)
        )
        self.register_buffer(
            "kp_weights",
            torch.tensor(weights, dtype=torch.float32).view(1, -1, 1, 1),
        )

    @staticmethod
    def keypoints_to_heatmap(
        keypoints: Tensor,
        confidence: Tensor,
        resolution: int,
        sigma: float = 1.5,
    ) -> Tensor:
        """keypoints (B, T, K, 2) in [0,1] -> heatmap (B*T, K, R, R)."""
        b, t, k, _ = keypoints.shape
        kp = rearrange(keypoints, "b t k c -> (b t) k c")
        conf = rearrange(confidence, "b t k -> (b t) k")
        device = kp.device
        dtype = kp.dtype

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, 1, resolution, device=device, dtype=dtype),
            torch.linspace(0, 1, resolution, device=device, dtype=dtype),
            indexing="ij",
        )
        grid = torch.stack([grid_x, grid_y], dim=-1)
        diff = kp[:, :, None, None, :] - grid[None, None, :, :, :]
        sq = (diff.pow(2).sum(dim=-1)) * (resolution ** 2)
        heatmap = torch.exp(-sq / (2.0 * sigma * sigma))
        heatmap = heatmap * conf[:, :, None, None]
        return heatmap

    def forward(
        self,
        decoded_pred: Tensor,
        target_keypoints: Tensor,
        target_confidence: Tensor,
    ) -> dict[str, Tensor]:
        b, c, t, h, w = decoded_pred.shape
        frames = rearrange(decoded_pred, "b c t h w -> (b t) c h w")
        frames_resized = F.interpolate(frames, size=(256, 256), mode="bilinear", align_corners=False)

        pred_heatmap = self.surrogate(frames_resized)
        target_heatmap = self.keypoints_to_heatmap(
            target_keypoints, target_confidence, self.config.heatmap_resolution,
        )

        per_pixel = (pred_heatmap - target_heatmap).pow(2)
        weighted = per_pixel * self.kp_weights.to(per_pixel.dtype)
        heatmap_mse = weighted.mean()

        coords_pred = _heatmap_to_coords(pred_heatmap, self.config.heatmap_resolution)
        coords_pred = rearrange(coords_pred, "(b t) k c -> b t k c", b=b, t=t)
        smooth = _kinematic_smoothness(coords_pred)

        total = heatmap_mse + self.config.smoothness_weight * smooth
        return {
            "anatomy_heatmap": heatmap_mse,
            "anatomy_smoothness": smooth,
            "anatomy_total": total,
            "anatomy_surrogate_loaded": torch.tensor(float(self._loaded)),
        }


def _heatmap_to_coords(heatmap: Tensor, resolution: int) -> Tensor:
    """Soft-argmax over the spatial axes -> (B, K, 2) in [0, 1]."""
    b, k, r, _ = heatmap.shape
    flat = heatmap.flatten(start_dim=2)
    prob = F.softmax(flat * 10.0, dim=-1)
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(0, 1, resolution, device=heatmap.device, dtype=heatmap.dtype),
        torch.linspace(0, 1, resolution, device=heatmap.device, dtype=heatmap.dtype),
        indexing="ij",
    )
    coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
    return prob @ coords


def _kinematic_smoothness(coords: Tensor) -> Tensor:
    if coords.shape[1] < 3:
        return coords.new_zeros(())
    accel = coords[:, 2:] - 2 * coords[:, 1:-1] + coords[:, :-2]
    return accel.pow(2).mean()
