"""Gated cross-attention — used for glyph stream fusion into the main backbone.

Gating starts at zero so the fusion is a no-op until trained. This preserves
the retrofit invariant: a Wan2.2 model wrapped with glyph fusion produces
identical output until the gate is unlocked during training stage 2.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from ogenti.modules.attention.multihead import MultiHeadAttention


class GatedCrossAttention(nn.Module):
    """Wraps MHA with a learnable scalar gate (tanh-bounded, zero-init)."""

    def __init__(
        self,
        dim: int,
        kv_dim: int,
        num_heads: int,
        head_dim: int,
        gate_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(kv_dim)
        self.attn = MultiHeadAttention(
            dim=dim,
            num_heads=num_heads,
            head_dim=head_dim,
            kv_dim=kv_dim,
            qk_norm=True,
            zero_init_out=True,
        )
        self.gate = nn.Parameter(torch.full((1,), gate_init))

    def forward(
        self,
        x: Tensor,
        context: Tensor,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        delta = self.attn(self.norm_q(x), self.norm_kv(context), attn_mask=attn_mask)
        return x + torch.tanh(self.gate) * delta
