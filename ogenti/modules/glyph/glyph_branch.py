"""Dedicated glyph branch — high-resolution token stream for text rendering.

Strategy (decided in RFC-0001 + D5 keyframe+propagation):
  1. Off-the-shelf OCR-region predictor (e.g. easyocr) extracts text-region
     bboxes from the *target* video during training, or from a generated
     keyframe at inference. Boxes are dilated and tracked across frames via
     optical flow (handled in preprocess; here we just receive token tensors).
  2. For each detected glyph region we crop a high-res patch and embed it
     with a separate patch embedder operating at 2x main backbone resolution.
  3. The resulting tokens form a small stream (typically 64-512 tokens per
     shot) that fuses into the main DiT via GatedCrossAttention in late
     blocks only.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor, nn


@dataclass
class GlyphBranchConfig:
    in_channels: int = 16
    embed_dim: int = 1024
    patch_size: tuple[int, int] = (8, 8)
    num_blocks: int = 4
    num_heads: int = 16
    head_dim: int = 64
    out_dim: int = 3072
    max_regions: int = 16
    tokens_per_region: int = 32


class GlyphPatchEmbed(nn.Module):
    def __init__(self, config: GlyphBranchConfig) -> None:
        super().__init__()
        ph, pw = config.patch_size
        self.proj = nn.Conv2d(
            config.in_channels,
            config.embed_dim,
            kernel_size=(ph, pw),
            stride=(ph, pw),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        return rearrange(x, "b c h w -> b (h w) c")


class GlyphTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, head_dim: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        from ogenti.modules.attention.multihead import FeedForward, MultiHeadAttention

        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim=dim, num_heads=num_heads, head_dim=head_dim, qk_norm=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, mlp_ratio=mlp_ratio)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class GlyphBranch(nn.Module):
    """High-resolution sub-transformer for text-region tokens."""

    def __init__(self, config: GlyphBranchConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_embed = GlyphPatchEmbed(config)
        self.region_pos_embed = nn.Parameter(
            torch.zeros(config.max_regions, config.tokens_per_region, config.embed_dim)
        )
        nn.init.trunc_normal_(self.region_pos_embed, std=0.02)

        self.blocks = nn.ModuleList(
            [
                GlyphTransformerBlock(
                    dim=config.embed_dim,
                    num_heads=config.num_heads,
                    head_dim=config.head_dim,
                )
                for _ in range(config.num_blocks)
            ]
        )
        self.norm_out = nn.LayerNorm(config.embed_dim)
        self.proj_out = nn.Linear(config.embed_dim, config.out_dim, bias=False)

    def forward(self, region_crops: Tensor, region_mask: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            region_crops: (B, R, C, H, W) — R high-res text-region crops per sample.
            region_mask:  (B, R) bool — True where a region slot is occupied.

        Returns:
            tokens: (B, R*T, out_dim) — glyph stream tokens
            mask:   (B, R*T) bool    — expanded mask (False -> ignore in cross-attn)
        """
        b, r, c, h, w = region_crops.shape
        x = rearrange(region_crops, "b r c h w -> (b r) c h w")
        x = self.patch_embed(x)
        n = x.shape[1]

        if n != self.config.tokens_per_region:
            x = self._resize_tokens(x, self.config.tokens_per_region)
            n = self.config.tokens_per_region

        pos = repeat(self.region_pos_embed[:r, :n, :], "r n d -> (b r) n d", b=b)
        x = x + pos

        for block in self.blocks:
            x = block(x)

        x = self.norm_out(x)
        x = self.proj_out(x)
        x = rearrange(x, "(b r) n d -> b (r n) d", b=b, r=r)

        token_mask = repeat(region_mask, "b r -> b (r n)", n=n)
        return x, token_mask

    @staticmethod
    def _resize_tokens(x: Tensor, target: int) -> Tensor:
        b, n, d = x.shape
        x = x.transpose(1, 2)
        x = F.interpolate(x, size=target, mode="linear", align_corners=False)
        return x.transpose(1, 2)
