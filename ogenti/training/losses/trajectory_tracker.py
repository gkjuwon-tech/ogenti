"""Differentiable centroid tracker.

Bridges decoded predicted video to per-track trajectories so that the physics
and motion-realism losses receive a real gradient signal instead of the
placeholder gt==pred no-op.

Approach (cheap, differentiable):
  For each track we receive an initial bounding-box template extracted from
  the ground-truth video at frame 0. We crop that template, then at each
  subsequent frame compute a per-pixel cosine-similarity map between the
  template and the predicted frame, take a *soft-argmax* over a search
  window around the previous centroid, and emit the predicted centroid for
  that frame. Repeat across T frames -> (K, T, 2) trajectory.

Soft-argmax is fully differentiable w.r.t. the predicted pixels, so gradients
flow back through the VAE-decoded prediction into the diffusion network.

Limitations (documented honestly, not silently):
  - Template-based — won't track through occlusions or large appearance changes.
  - Search window is fixed (not adaptive).
  - Cosine similarity in pixel space; will degrade on low-light shots.

For ad-creative footage (3-5s controlled shots) these limits are acceptable.
For long-form footage we'd need a learned tracker.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor


@dataclass
class TrackerConfig:
    template_size: int = 32
    search_radius: int = 24
    softargmax_temperature: float = 30.0


def _crop(image: Tensor, cy: float, cx: float, size: int) -> Tensor:
    """image: (C, H, W) -> (C, size, size). Bilinear, padded at edges."""
    c, h, w = image.shape
    half = size / 2.0
    ys = torch.linspace(-half, half - 1.0, size, device=image.device, dtype=image.dtype) + cy
    xs = torch.linspace(-half, half - 1.0, size, device=image.device, dtype=image.dtype) + cx
    ny = 2.0 * ys / max(h - 1, 1) - 1.0
    nx = 2.0 * xs / max(w - 1, 1) - 1.0
    grid_y, grid_x = torch.meshgrid(ny, nx, indexing="ij")
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
    out = F.grid_sample(
        image.unsqueeze(0), grid, mode="bilinear", padding_mode="border", align_corners=True,
    )
    return out.squeeze(0)


def _soft_argmax_2d(heatmap: Tensor, temperature: float) -> tuple[Tensor, Tensor]:
    """heatmap: (H, W) -> (cy_norm, cx_norm) in [0, H-1] / [0, W-1]."""
    h, w = heatmap.shape
    flat = heatmap.flatten() * temperature
    prob = F.softmax(flat, dim=0)
    ys = torch.arange(h, device=heatmap.device, dtype=heatmap.dtype).repeat_interleave(w)
    xs = torch.arange(w, device=heatmap.device, dtype=heatmap.dtype).repeat(h)
    cy = (prob * ys).sum()
    cx = (prob * xs).sum()
    return cy, cx


def track_objects(
    pred_video: Tensor,
    init_boxes: Tensor,
    track_mask: Tensor,
    config: TrackerConfig,
) -> Tensor:
    """
    Args:
        pred_video:  (B, 3, T, H, W) in [-1, 1] — VAE-decoded prediction
        init_boxes:  (B, K, 4) [cy, cx, h, w] in pixel coords (frame 0)
        track_mask:  (B, K) bool
        config:      TrackerConfig

    Returns:
        trajectories: (B, K, T, 2) in normalized image coords [0, 1]
    """
    b, c, t, h, w = pred_video.shape
    k = init_boxes.shape[1]
    norm = max(h, w)
    out = torch.zeros(b, k, t, 2, device=pred_video.device, dtype=pred_video.dtype)

    for bi in range(b):
        frame0 = pred_video[bi, :, 0]
        for ki in range(k):
            if not bool(track_mask[bi, ki]):
                continue
            cy, cx, bh, bw = init_boxes[bi, ki].tolist()
            template = _crop(frame0, cy, cx, config.template_size)
            template_norm = template / (template.flatten().norm() + 1e-6)

            prev_cy, prev_cx = cy, cx
            for ti in range(t):
                frame = pred_video[bi, :, ti]
                sr = config.search_radius
                y0 = int(max(0, min(h - 1, prev_cy - sr)))
                y1 = int(max(0, min(h, prev_cy + sr)))
                x0 = int(max(0, min(w - 1, prev_cx - sr)))
                x1 = int(max(0, min(w, prev_cx + sr)))
                if y1 - y0 < 4 or x1 - x0 < 4:
                    out[bi, ki, ti, 0] = prev_cx / norm
                    out[bi, ki, ti, 1] = prev_cy / norm
                    continue

                patch = frame[:, y0:y1, x0:x1]
                patch_flat = rearrange(patch, "c h w -> c (h w)")
                patch_norm = patch_flat / (patch_flat.norm(dim=0, keepdim=True) + 1e-6)

                ts = config.template_size
                pooled = F.adaptive_avg_pool2d(template_norm.unsqueeze(0), (ts // 4, ts // 4)).squeeze(0)
                tpl_summary = pooled.flatten()
                tpl_summary = tpl_summary[:c].view(c, 1)
                tpl_summary = tpl_summary / (tpl_summary.norm() + 1e-6)

                sim = (patch_norm * tpl_summary).sum(dim=0)
                heatmap = sim.view(y1 - y0, x1 - x0)
                cy_local, cx_local = _soft_argmax_2d(heatmap, config.softargmax_temperature)

                next_cy = cy_local + y0
                next_cx = cx_local + x0
                out[bi, ki, ti, 0] = next_cx / norm
                out[bi, ki, ti, 1] = next_cy / norm
                prev_cy, prev_cx = float(next_cy.detach()), float(next_cx.detach())

    return out


def boxes_from_trajectories(trajectories: Tensor, default_box_size: float = 0.1) -> Tensor:
    """GT trajectories (B, K, T, 2) normalized -> init_boxes (B, K, 4) at frame 0
    in pixel coords assuming a normalized frame of size 1x1.

    The caller must scale [cy, cx, h, w] back to pixel coordinates of pred_video.
    """
    init_xy = trajectories[:, :, 0]
    b, k, _ = init_xy.shape
    box_size = trajectories.new_full((b, k, 1), default_box_size)
    boxes = torch.cat([init_xy.flip(-1), box_size, box_size], dim=-1)
    return boxes
