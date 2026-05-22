"""Multi-head attention used inside OgentiBlock.

Wraps torch.nn.functional.scaled_dot_product_attention so we automatically pick
flash / mem-efficient / math backend depending on hardware. RoPE is applied
externally so that cross-attention (where q and k come from different streams)
can opt out cleanly.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn


class MultiHeadAttention(nn.Module):
    """Standard MHA with optional QK normalization and external RoPE injection."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: Optional[int] = None,
        qk_norm: bool = True,
        kv_dim: Optional[int] = None,
        out_dim: Optional[int] = None,
        zero_init_out: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim is not None else dim // num_heads
        self.inner_dim = self.num_heads * self.head_dim
        self.kv_dim = kv_dim if kv_dim is not None else dim
        self.out_dim = out_dim if out_dim is not None else dim
        self.dropout = dropout

        self.q_proj = nn.Linear(dim, self.inner_dim, bias=True)
        self.k_proj = nn.Linear(self.kv_dim, self.inner_dim, bias=True)
        self.v_proj = nn.Linear(self.kv_dim, self.inner_dim, bias=True)
        self.out_proj = nn.Linear(self.inner_dim, self.out_dim, bias=True)

        self.q_norm = nn.RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(self.head_dim) if qk_norm else nn.Identity()

        if zero_init_out:
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        q_in: Tensor,
        kv_in: Optional[Tensor] = None,
        *,
        rope_apply_fn=None,
        rope_freqs_q: Optional[Tensor] = None,
        rope_freqs_k: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        kv_in = q_in if kv_in is None else kv_in

        q = rearrange(self.q_proj(q_in), "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(self.k_proj(kv_in), "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(self.v_proj(kv_in), "b n (h d) -> b h n d", h=self.num_heads)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if rope_apply_fn is not None:
            if rope_freqs_q is not None:
                q = rope_apply_fn(q, rope_freqs_q)
            if rope_freqs_k is not None:
                k = rope_apply_fn(k, rope_freqs_k)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.out_proj(out)


class FeedForward(nn.Module):
    """SwiGLU FFN — matches Wan2.2 / Llama-style FFN for clean weight import."""

    def __init__(self, dim: int, mlp_ratio: float = 4.0, multiple_of: int = 256) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio * 2 / 3)
        hidden = ((hidden + multiple_of - 1) // multiple_of) * multiple_of
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
