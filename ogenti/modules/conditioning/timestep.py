"""Timestep embedding + AdaLN-Zero modulation matching DiT/Wan2.2 layout."""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn


class SinusoidalTimestepEmbedding(nn.Module):
    def __init__(self, dim: int, max_period: int = 10000) -> None:
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: Tensor) -> Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(0, half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float()[:, None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


class TimestepConditioningHead(nn.Module):
    """Embeds scalar timestep into a per-block conditioning vector."""

    def __init__(self, dim: int, hidden_dim: Optional[int] = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.sinusoidal = SinusoidalTimestepEmbedding(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t: Tensor) -> Tensor:
        return self.mlp(self.sinusoidal(t))


class AdaLNZero(nn.Module):
    """AdaLN-Zero modulation: produces (scale_msa, shift_msa, gate_msa, scale_ffn, shift_ffn, gate_ffn).

    Output projection is zero-initialized so blocks behave as identity at step 0.
    """

    NUM_PARAMS = 6

    def __init__(self, cond_dim: int, target_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(cond_dim, self.NUM_PARAMS * target_dim, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        self.target_dim = target_dim

    def forward(self, cond: Tensor) -> tuple[Tensor, ...]:
        params = self.linear(nn.functional.silu(cond))
        return params.chunk(self.NUM_PARAMS, dim=-1)


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
