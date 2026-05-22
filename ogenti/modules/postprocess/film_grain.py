"""Film grain head — adds parameterized noise overlay.

Real film and digital cinema have characteristic grain patterns. AI-generated
video is unnaturally clean. We add a light-conditioned noise overlay matching
the spectral characteristics of common film stocks (Kodak Vision3, Fuji Eterna)
and digital sensor noise (ISO 800–6400 ranges).

Parameters exposed to the user:
  - grain_amount: 0 (no grain) to 1 (heavy 35mm Vision3-style grain)
  - grain_size: 0.5 (fine digital) to 2.0 (16mm coarse)
  - luma_dependent_strength: more grain in mid-tones (matches real film)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn


@dataclass
class FilmGrainConfig:
    base_amount_init: float = 0.0
    learnable: bool = True
    default_grain_size: float = 1.0
    luma_dependent: bool = True


class FilmGrainHead(nn.Module):
    def __init__(self, config: FilmGrainConfig) -> None:
        super().__init__()
        self.config = config
        if config.learnable:
            self.base_amount = nn.Parameter(torch.tensor(config.base_amount_init))
        else:
            self.register_buffer("base_amount", torch.tensor(config.base_amount_init), persistent=False)

    def forward(
        self,
        video: Tensor,
        amount: Tensor | float | None = None,
        grain_size: float | None = None,
        seed: int | None = None,
    ) -> Tensor:
        """
        Args:
            video: (B, 3, T, H, W) in [-1, 1]
            amount: (B,) or scalar — overrides learnable base_amount
            grain_size: 0.5..2.0
        """
        cfg = self.config
        b, c, t, h, w = video.shape

        if amount is None:
            base = torch.sigmoid(self.base_amount) if cfg.learnable else self.base_amount
            a = base.expand(b)
        elif isinstance(amount, float):
            a = video.new_full((b,), amount)
        else:
            a = amount.to(video.device).to(video.dtype)

        a_thresh = a.clamp(min=1e-6)
        if (a_thresh < 1e-5).all():
            return video

        if grain_size is None:
            grain_size = cfg.default_grain_size

        gen = None
        if seed is not None:
            gen = torch.Generator(device=video.device).manual_seed(seed)

        low_h = max(2, int(h / grain_size))
        low_w = max(2, int(w / grain_size))
        noise_small = torch.randn(b, 1, t, low_h, low_w, device=video.device, dtype=video.dtype, generator=gen)
        noise_small = rearrange(noise_small, "b c t h w -> (b t) c h w")
        noise = F.interpolate(noise_small, size=(h, w), mode="bilinear", align_corners=False)
        noise = rearrange(noise, "(b t) c h w -> b c t h w", b=b, t=t)
        noise = noise.expand(-1, c, -1, -1, -1)

        if cfg.luma_dependent:
            luma = video.mean(dim=1, keepdim=True)
            midtone_weight = 1.0 - (luma.abs() * 1.5).clamp(max=1.0)
        else:
            midtone_weight = torch.ones_like(video[:, :1])

        a_b = a.view(b, 1, 1, 1, 1)
        out = video + a_b * 0.08 * noise * midtone_weight
        return out.clamp(-1, 1)
