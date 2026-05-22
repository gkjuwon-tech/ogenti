"""3D patch embedding for video latents (T x H x W -> token sequence)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from einops import rearrange
from torch import Tensor, nn


@dataclass
class PatchEmbed3DConfig:
    in_channels: int = 16
    embed_dim: int = 3072
    patch_size: tuple[int, int, int] = (1, 2, 2)


class PatchEmbed3D(nn.Module):
    def __init__(self, config: PatchEmbed3DConfig) -> None:
        super().__init__()
        self.config = config
        pt, ph, pw = config.patch_size
        self.proj = nn.Conv3d(
            config.in_channels,
            config.embed_dim,
            kernel_size=(pt, ph, pw),
            stride=(pt, ph, pw),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, tuple[int, int, int]]:
        """x: (B, C, T, H, W) -> tokens (B, N, D), grid (t, h, w)."""
        x = self.proj(x)
        b, d, t, h, w = x.shape
        tokens = rearrange(x, "b d t h w -> b (t h w) d")
        return tokens, (t, h, w)


class PatchUnembed3D(nn.Module):
    def __init__(self, config: PatchEmbed3DConfig, out_channels: int) -> None:
        super().__init__()
        self.config = config
        pt, ph, pw = config.patch_size
        self.proj = nn.Linear(config.embed_dim, out_channels * pt * ph * pw, bias=True)
        self.out_channels = out_channels

    def forward(self, tokens: Tensor, grid: tuple[int, int, int]) -> Tensor:
        t, h, w = grid
        pt, ph, pw = self.config.patch_size
        x = self.proj(tokens)
        x = rearrange(
            x,
            "b (t h w) (c pt ph pw) -> b c (t pt) (h ph) (w pw)",
            t=t, h=h, w=w, pt=pt, ph=ph, pw=pw, c=self.out_channels,
        )
        return x
