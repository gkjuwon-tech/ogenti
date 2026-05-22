"""Rotary positional embeddings for 3D (T, H, W) spatio-temporal tokens.

Axial RoPE: separate frequency bands for time / height / width axes, concatenated
in the head dimension. This matches the layout used by Wan2.2 and HunyuanVideo
(easier weight import).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from einops import rearrange
from torch import Tensor, nn


@dataclass
class RopeConfig:
    head_dim: int = 128
    t_split: int = 32
    h_split: int = 48
    w_split: int = 48
    theta: float = 10000.0

    def __post_init__(self) -> None:
        if self.t_split + self.h_split + self.w_split != self.head_dim:
            raise ValueError(
                f"axial splits {self.t_split}+{self.h_split}+{self.w_split} "
                f"must equal head_dim={self.head_dim}"
            )


def _axis_freqs(num_positions: int, dim: int, theta: float, device, dtype) -> Tensor:
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    pos = torch.arange(num_positions, device=device, dtype=torch.float32)
    freqs = torch.einsum("n,d->nd", pos, inv_freq)
    return torch.polar(torch.ones_like(freqs), freqs).to(dtype=torch.complex64)


class AxialRoPE3D(nn.Module):
    def __init__(self, config: RopeConfig) -> None:
        super().__init__()
        self.config = config

    def freqs(self, t: int, h: int, w: int, device, dtype=torch.float32) -> Tensor:
        cfg = self.config
        ft = _axis_freqs(t, cfg.t_split, cfg.theta, device, dtype)
        fh = _axis_freqs(h, cfg.h_split, cfg.theta, device, dtype)
        fw = _axis_freqs(w, cfg.w_split, cfg.theta, device, dtype)

        ft = ft[:, None, None, :].expand(t, h, w, -1)
        fh = fh[None, :, None, :].expand(t, h, w, -1)
        fw = fw[None, None, :, :].expand(t, h, w, -1)

        freqs = torch.cat([ft, fh, fw], dim=-1)
        return rearrange(freqs, "t h w d -> (t h w) d")

    @staticmethod
    def apply(x: Tensor, freqs: Tensor) -> Tensor:
        """x: (B, H, N, D) real -> rotated real of same shape."""
        b, h, n, d = x.shape
        x_pairs = x.float().reshape(b, h, n, d // 2, 2)
        x_complex = torch.view_as_complex(x_pairs.contiguous())
        rotated = x_complex * freqs[None, None, :, :]
        out = torch.view_as_real(rotated).reshape(b, h, n, d)
        return out.to(x.dtype)
