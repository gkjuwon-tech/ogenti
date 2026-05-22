"""Velocity-conditioned motion blur post-process head.

Real cameras integrate light over a shutter interval. AI generates each
frame as a sharp instantaneous snapshot, which is itself an AI-tell.

We implement motion blur as a per-pixel line-integral of the predicted
optical flow, with shutter duration as a controllable parameter.

This is a pure post-process — it operates on the VAE-decoded video and
the predicted/estimated flow field. Differentiable.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn


@dataclass
class MotionBlurConfig:
    num_samples: int = 7
    shutter_init: float = 0.0
    learnable_shutter: bool = True


class MotionBlurHead(nn.Module):
    """Approximate motion blur by sampling along the optical-flow line per pixel.

    shutter_amount controls how much of the inter-frame motion is integrated.
    Zero shutter => identity (no blur). 1.0 => full integration across the
    next-frame motion vector.
    """

    def __init__(self, config: MotionBlurConfig) -> None:
        super().__init__()
        self.config = config
        if config.learnable_shutter:
            self.shutter = nn.Parameter(torch.tensor(config.shutter_init))
        else:
            self.register_buffer("shutter", torch.tensor(config.shutter_init), persistent=False)

    def forward(self, video: Tensor, flow: Tensor) -> Tensor:
        """
        Args:
            video: (B, 3, T, H, W) in [-1, 1]
            flow:  (B, 2, T, H, W) — per-pixel xy displacement to next frame
        """
        b, c, t, h, w = video.shape
        n = self.config.num_samples
        shutter = torch.tanh(self.shutter) if self.config.learnable_shutter else self.shutter

        if torch.abs(shutter) < 1e-4:
            return video

        x = rearrange(video, "b c t h w -> (b t) c h w")
        flow_t = rearrange(flow, "b c t h w -> (b t) c h w")

        ys = torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype)
        xs = torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        base_grid = torch.stack([gx, gy], dim=-1).unsqueeze(0).expand(x.shape[0], -1, -1, -1)

        flow_norm = flow_t.clone()
        flow_norm[:, 0] = flow_t[:, 0] * 2.0 / max(w - 1, 1)
        flow_norm[:, 1] = flow_t[:, 1] * 2.0 / max(h - 1, 1)
        flow_grid = rearrange(flow_norm, "n c h w -> n h w c")

        accumulated = torch.zeros_like(x)
        offsets = torch.linspace(0.0, float(shutter), n, device=x.device, dtype=x.dtype)
        for off in offsets:
            sample_grid = base_grid + off * flow_grid
            sampled = F.grid_sample(x, sample_grid, mode="bilinear", padding_mode="border", align_corners=True)
            accumulated = accumulated + sampled
        accumulated = accumulated / n

        out = rearrange(accumulated, "(b t) c h w -> b c t h w", b=b, t=t)
        return out


def estimate_flow_from_frames(video: Tensor) -> Tensor:
    """Cheap flow proxy: pixel-wise next-frame difference.

    Used when we don't have a learned flow estimator inline. Returns a
    (B, 2, T, H, W) tensor where the first 2 channels are dummy xy
    displacements set proportional to luminance gradient × frame diff —
    a stand-in for proper optical flow.
    """
    b, c, t, h, w = video.shape
    if t < 2:
        return torch.zeros(b, 2, t, h, w, device=video.device, dtype=video.dtype)

    lum = video.mean(dim=1, keepdim=True)
    dt = torch.zeros_like(lum)
    dt[:, :, :-1] = lum[:, :, 1:] - lum[:, :, :-1]

    dy = torch.zeros_like(lum)
    dx = torch.zeros_like(lum)
    dy[:, :, :, 1:-1, :] = (lum[:, :, :, 2:, :] - lum[:, :, :, :-2, :]) * 0.5
    dx[:, :, :, :, 1:-1] = (lum[:, :, :, :, 2:] - lum[:, :, :, :, :-2]) * 0.5

    eps = 1e-3
    denom = (dx * dx + dy * dy + eps).sqrt()
    fx = -dt * dx / denom
    fy = -dt * dy / denom
    flow = torch.cat([fx, fy], dim=1)
    return flow.clamp(-0.5, 0.5)
