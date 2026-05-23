"""Multi-head attention used inside OgentiBlock.

Wraps torch.nn.functional.scaled_dot_product_attention so we automatically pick
flash / mem-efficient / math backend depending on hardware. RoPE is applied
externally so that cross-attention (where q and k come from different streams)
can opt out cleanly.

Supports two QK-norm modes:
  - `per_head`: RMSNorm with shape `(head_dim,)`, applied AFTER head split.
                Matches Ogenti's original retrofit target (TI2V-5B in old configs)
                and our entity / glyph cross-attention helpers.
  - `per_channel`: RMSNorm with shape `(inner_dim,)`, applied BEFORE head split.
                Matches Wan2.2-{TI2V-5B, T2V-A14B, I2V-A14B} `self_attn.norm_q` /
                `norm_k` parametrization (a single (dim,) RMSNorm scale shared
                across all heads).
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
        qk_norm_mode: str = "per_head",
        kv_dim: Optional[int] = None,
        out_dim: Optional[int] = None,
        zero_init_out: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if qk_norm_mode not in ("per_head", "per_channel"):
            raise ValueError(f"qk_norm_mode must be 'per_head' or 'per_channel', got {qk_norm_mode!r}")
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim is not None else dim // num_heads
        self.inner_dim = self.num_heads * self.head_dim
        self.kv_dim = kv_dim if kv_dim is not None else dim
        self.out_dim = out_dim if out_dim is not None else dim
        self.dropout = dropout
        self.qk_norm_mode = qk_norm_mode

        self.q_proj = nn.Linear(dim, self.inner_dim, bias=True)
        self.k_proj = nn.Linear(self.kv_dim, self.inner_dim, bias=True)
        self.v_proj = nn.Linear(self.kv_dim, self.inner_dim, bias=True)
        self.out_proj = nn.Linear(self.inner_dim, self.out_dim, bias=True)

        if qk_norm:
            norm_dim = self.inner_dim if qk_norm_mode == "per_channel" else self.head_dim
            self.q_norm: nn.Module = nn.RMSNorm(norm_dim)
            self.k_norm: nn.Module = nn.RMSNorm(norm_dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

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

        q_flat = self.q_proj(q_in)
        k_flat = self.k_proj(kv_in)
        v_flat = self.v_proj(kv_in)

        if self.qk_norm_mode == "per_channel":
            q_flat = self.q_norm(q_flat)
            k_flat = self.k_norm(k_flat)

        q = rearrange(q_flat, "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(k_flat, "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(v_flat, "b n (h d) -> b h n d", h=self.num_heads)

        if self.qk_norm_mode == "per_head":
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
    """SwiGLU FFN — legacy, used by Ogenti's entity_refine helper modules.

    Wan2.2's per-block FFN is *not* SwiGLU; see `WanFeedForward` below.
    """

    def __init__(self, dim: int, mlp_ratio: float = 4.0, multiple_of: int = 256) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio * 2 / 3)
        hidden = ((hidden + multiple_of - 1) // multiple_of) * multiple_of
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class WanFeedForward(nn.Sequential):
    """Wan2.2-style block FFN: ``Linear(dim, ffn_dim, bias=True) -> GELU(tanh) -> Linear(ffn_dim, dim, bias=True)``.

    Subclassing `nn.Sequential` keeps the state dict keys flat
    (``0.weight``, ``0.bias``, ``2.weight``, ``2.bias``) so that
    ``self.ffn = WanFeedForward(...)`` produces ``blocks.N.ffn.0.weight`` etc.
    matching Wan2.2 exactly.
    """

    def __init__(self, dim: int, ffn_dim: int) -> None:
        super().__init__(
            nn.Linear(dim, ffn_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim, bias=True),
        )
