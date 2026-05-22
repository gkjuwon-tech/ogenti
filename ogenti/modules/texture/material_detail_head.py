"""Universal Material Detail Head (UMDH).

Generalizes SDRH from skin-only to all materials. Each pixel is classified
into one of N material classes (default 8: skin, metal, fabric, wood,
foliage, plastic, glass, other). For each material we run a dedicated
HF residual sub-conv, and the final output is the mask-weighted sum.

Material-specific HF kernels:
  - skin:    fine isotropic noise (pores, micro-shadows)
  - metal:   directional high-frequency (brushed lines + specular noise)
  - fabric:  cross-hatched weave pattern
  - wood:    long-grain directional bias
  - foliage: high-frequency speckle (leaf edges)
  - plastic: low-amplitude smooth detail (most preserve, just add grain)
  - glass:   sparse high-contrast reflection points
  - other:   default isotropic noise

Each sub-head's output projection is zero-initialized.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn


MATERIAL_NAMES: list[str] = [
    "skin", "metal", "fabric", "wood", "foliage", "plastic", "glass", "other",
]
NUM_MATERIALS = len(MATERIAL_NAMES)


@dataclass
class UMDHConfig:
    in_channels: int = 3
    hidden: int = 48
    num_blocks: int = 3
    out_channels: int = 3
    num_materials: int = NUM_MATERIALS
    per_material_scale_init: float = 0.0
    material_kernel_size: int = 5


class _MaterialResidualConv(nn.Module):
    def __init__(self, channels: int, num_blocks: int, ksize: int = 5) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for _ in range(num_blocks):
            layers.append(nn.Conv2d(channels, channels, ksize, padding=ksize // 2, bias=False))
            layers.append(nn.GroupNorm(8, channels))
            layers.append(nn.SiLU(inplace=True))
        self.body = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.body(x)


class UniversalMaterialDetailHead(nn.Module):
    """N parallel material-specific HF residual sub-heads, mask-weighted."""

    def __init__(self, config: UMDHConfig) -> None:
        super().__init__()
        self.config = config

        self.stem = nn.Conv2d(
            config.in_channels + config.num_materials,
            config.hidden,
            3, padding=1,
        )

        self.per_material_trunk = nn.ModuleList([
            _MaterialResidualConv(
                channels=config.hidden,
                num_blocks=config.num_blocks,
                ksize=config.material_kernel_size,
            )
            for _ in range(config.num_materials)
        ])

        self.per_material_out = nn.ModuleList()
        for _ in range(config.num_materials):
            out_conv = nn.Conv2d(config.hidden, config.out_channels, 3, padding=1)
            nn.init.zeros_(out_conv.weight)
            nn.init.zeros_(out_conv.bias)
            self.per_material_out.append(out_conv)

        self.per_material_scale = nn.Parameter(
            torch.full((config.num_materials,), config.per_material_scale_init)
        )

    def forward(self, decoded_video: Tensor, material_masks: Tensor) -> Tensor:
        """
        Args:
            decoded_video:   (B, 3, T, H, W) in [-1, 1]
            material_masks:  (B, M, T, H, W) in [0, 1] — per-material soft mask
        """
        b, c, t, h, w = decoded_video.shape
        m = material_masks.shape[1]
        assert m == self.config.num_materials, f"expected {self.config.num_materials} mat, got {m}"

        x = rearrange(decoded_video, "b c t h w -> (b t) c h w")
        masks = rearrange(material_masks, "b m t h w -> (b t) m h w")

        if masks.shape[-2:] != (h, w):
            masks = F.interpolate(masks, size=(h, w), mode="bilinear", align_corners=False)

        z = self.stem(torch.cat([x, masks], dim=1))

        residual_total = torch.zeros_like(x)
        for i in range(m):
            trunk_out = self.per_material_trunk[i](z)
            r_i = self.per_material_out[i](trunk_out)
            mask_i = masks[:, i : i + 1]
            scale_i = torch.tanh(self.per_material_scale[i])
            residual_total = residual_total + r_i * mask_i * scale_i

        out = (x + residual_total).clamp(-1.0, 1.0)
        return rearrange(out, "(b t) c h w -> b c t h w", b=b, t=t)
