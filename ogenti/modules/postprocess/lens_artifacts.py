"""Lens artifact post-process head.

Real lenses produce:
  - chromatic aberration (color fringing at edges)
  - lens-shape bokeh (anamorphic streaks, hex aperture, swirly old-glass)
  - veiling glare under strong backlight
  - subtle vignette

We expose these as a small conditioning vector and apply differentiable
operations in pixel space.

Lens descriptor (6-dim):
  [chromatic_aberration_strength,
   vignette_strength,
   anamorphic_squeeze,
   flare_amount,
   bokeh_polygon_sides_norm,
   global_glow]
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn


LENS_DESCRIPTOR_DIM = 6


@dataclass
class LensArtifactConfig:
    descriptor_dim: int = LENS_DESCRIPTOR_DIM
    learnable_default: bool = True
    default_init: float = 0.0


PRESET_LENS: dict[str, list[float]] = {
    "clean":         [0.00, 0.00, 1.0, 0.00, 1.0, 0.00],
    "modern_prime":  [0.15, 0.10, 1.0, 0.05, 1.0, 0.10],
    "vintage":       [0.35, 0.30, 1.0, 0.20, 0.7, 0.30],
    "anamorphic":    [0.20, 0.20, 1.5, 0.30, 1.0, 0.20],
    "documentary":   [0.10, 0.15, 1.0, 0.05, 1.0, 0.10],
}


def lens_preset_to_descriptor(name: str) -> Tensor:
    if name not in PRESET_LENS:
        raise ValueError(f"unknown lens preset: {name}")
    return torch.tensor(PRESET_LENS[name], dtype=torch.float32)


class LensArtifactHead(nn.Module):
    def __init__(self, config: LensArtifactConfig) -> None:
        super().__init__()
        self.config = config
        if config.learnable_default:
            self.default_descriptor = nn.Parameter(torch.full((config.descriptor_dim,), config.default_init))
        else:
            self.register_buffer(
                "default_descriptor",
                torch.full((config.descriptor_dim,), config.default_init),
                persistent=False,
            )

    def forward(
        self,
        video: Tensor,
        descriptor: Tensor | None = None,
    ) -> Tensor:
        b, c, t, h, w = video.shape
        if descriptor is None:
            d = self.default_descriptor.expand(b, -1).to(video.device).to(video.dtype)
        else:
            d = descriptor.to(video.device).to(video.dtype)
            if d.dim() == 1:
                d = d.unsqueeze(0).expand(b, -1)

        ca = d[:, 0].clamp(min=0.0)
        vig = d[:, 1].clamp(min=0.0)
        glow = d[:, 5].clamp(min=0.0)

        if (ca.abs().sum() + vig.abs().sum() + glow.abs().sum()) < 1e-5:
            return video

        x = rearrange(video, "b c t h w -> (b t) c h w")
        x = self._apply_chromatic_aberration(x, ca.repeat_interleave(t))
        x = self._apply_vignette(x, vig.repeat_interleave(t))
        x = self._apply_glow(x, glow.repeat_interleave(t))
        x = rearrange(x, "(b t) c h w -> b c t h w", b=b, t=t)
        return x.clamp(-1.0, 1.0)

    @staticmethod
    def _radial_grid(b: int, h: int, w: int, device, dtype) -> tuple[Tensor, Tensor, Tensor]:
        ys = torch.linspace(-1, 1, h, device=device, dtype=dtype)
        xs = torch.linspace(-1, 1, w, device=device, dtype=dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        r = (gx * gx + gy * gy).sqrt()
        return gx, gy, r

    def _apply_chromatic_aberration(self, x: Tensor, amount: Tensor) -> Tensor:
        b, c, h, w = x.shape
        gx, gy, _ = self._radial_grid(b, h, w, x.device, x.dtype)
        base_grid = torch.stack([gx, gy], dim=-1).unsqueeze(0).expand(b, -1, -1, -1)
        radial = base_grid * (gx.pow(2) + gy.pow(2)).unsqueeze(-1).clamp(max=1.0)

        amt = amount.view(b, 1, 1, 1) * 0.02
        red_grid = base_grid + amt * radial
        blue_grid = base_grid - amt * radial

        r = F.grid_sample(x[:, 0:1], red_grid, mode="bilinear", padding_mode="border", align_corners=True)
        b_ch = F.grid_sample(x[:, 2:3], blue_grid, mode="bilinear", padding_mode="border", align_corners=True)
        g_ch = x[:, 1:2]
        return torch.cat([r, g_ch, b_ch], dim=1)

    def _apply_vignette(self, x: Tensor, amount: Tensor) -> Tensor:
        b, c, h, w = x.shape
        _, _, r = self._radial_grid(b, h, w, x.device, x.dtype)
        falloff = (1.0 - r.pow(2)).clamp(min=0.0, max=1.0)
        amt = amount.view(b, 1, 1, 1)
        mask = (1.0 - amt) + amt * falloff.unsqueeze(0).unsqueeze(0)
        return x * mask

    def _apply_glow(self, x: Tensor, amount: Tensor) -> Tensor:
        b, c, h, w = x.shape
        bright = (x.mean(dim=1, keepdim=True) - 0.3).clamp(min=0.0)
        ksize = 21
        sigma = 6.0
        coords = torch.arange(ksize, device=x.device, dtype=x.dtype) - (ksize - 1) / 2
        g1d = torch.exp(-coords.pow(2) / (2 * sigma * sigma))
        g1d = g1d / g1d.sum()
        kernel = g1d[:, None] * g1d[None, :]
        kernel = kernel[None, None].expand(1, 1, ksize, ksize)
        bloom = F.conv2d(bright, kernel, padding=ksize // 2)
        amt = amount.view(b, 1, 1, 1)
        return x + amt * bloom * 0.5
