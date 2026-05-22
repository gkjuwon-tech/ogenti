"""Skin Detail Residual Head (SDRH) — kills the "tanghulu skin" tell.

Operates in pixel space AFTER VAE decode. Predicts a high-frequency residual
that is added to decoded video, gated by a skin mask. Zero-init output so
retrofit invariant holds.

Architecture: small U-Net-ish conv stack with frequency-aware modulation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn


@dataclass
class SDRHConfig:
    in_channels: int = 3
    hidden: int = 64
    num_blocks: int = 4
    out_channels: int = 3
    residual_scale_init: float = 0.0
    cond_dim: int = 64


class FreqAwareConvBlock(nn.Module):
    """Conv block with a parallel high-pass branch."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        kernel = torch.tensor(
            [[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]],
            dtype=torch.float32,
        ) / 8.0
        kernel = kernel.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        self.register_buffer("hp_kernel", kernel, persistent=False)
        self.channels = channels

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm(x)
        h = F.silu(self.conv1(h))
        hp = F.conv2d(h, self.hp_kernel.to(h.dtype), padding=1, groups=self.channels)
        h = F.silu(self.conv2(h + 0.5 * hp))
        return x + h


class SkinDetailResidualHead(nn.Module):
    """Pixel-space HF residual generator, gated by a skin mask."""

    def __init__(self, config: SDRHConfig) -> None:
        super().__init__()
        self.config = config

        self.stem = nn.Conv2d(config.in_channels + 1, config.hidden, 3, padding=1)
        self.blocks = nn.ModuleList(
            [FreqAwareConvBlock(config.hidden) for _ in range(config.num_blocks)]
        )
        self.out_conv = nn.Conv2d(config.hidden, config.out_channels, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

        self.residual_scale = nn.Parameter(torch.full((1,), config.residual_scale_init))

    def forward(self, decoded_video: Tensor, skin_mask: Tensor) -> Tensor:
        """
        Args:
            decoded_video: (B, 3, T, H, W) in [-1, 1]
            skin_mask:     (B, 1, T, H, W) in [0, 1]
        Returns:
            refined_video: (B, 3, T, H, W), same range
        """
        b, c, t, h, w = decoded_video.shape
        x = rearrange(decoded_video, "b c t h w -> (b t) c h w")
        m = rearrange(skin_mask, "b c t h w -> (b t) c h w")

        if m.shape[-2:] != (h, w):
            m = F.interpolate(m, size=(h, w), mode="bilinear", align_corners=False)

        h_in = torch.cat([x, m], dim=1)
        z = self.stem(h_in)
        for blk in self.blocks:
            z = blk(z)
        residual = self.out_conv(z)

        residual = residual * m
        residual = residual * torch.tanh(self.residual_scale)

        out = x + residual
        out = rearrange(out, "(b t) c h w -> b c t h w", b=b, t=t)
        return out.clamp(-1.0, 1.0)
