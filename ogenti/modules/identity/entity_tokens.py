"""Entity token bank — persistent identity anchors across diffusion timesteps.

Inspired by Slot Attention (Locatello et al.) and Perceiver IO's latent array,
adapted for diffusion:

  - K learned slot prototypes (K = num_slots, default 32)
  - Slots are bound to the *batch sample*, not the timestep — they are
    re-used across denoising steps to maintain identity continuity.
  - At step 0 of training (retrofit), slot influence is gated to zero so the
    network behaves as a vanilla DiT.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from einops import repeat
from torch import Tensor, nn


@dataclass
class EntityBankConfig:
    num_slots: int = 32
    dim: int = 3072
    init_scale: float = 0.02


class EntityTokenBank(nn.Module):
    """Holds K learned slot prototypes, expanded per-batch on forward."""

    def __init__(self, config: EntityBankConfig) -> None:
        super().__init__()
        self.config = config
        self.prototypes = nn.Parameter(
            torch.randn(config.num_slots, config.dim) * config.init_scale
        )

    def expand(self, batch_size: int) -> Tensor:
        return repeat(self.prototypes, "k d -> b k d", b=batch_size).contiguous()

    @property
    def num_slots(self) -> int:
        return self.config.num_slots


class EntityRefineLayer(nn.Module):
    """One round of slot refinement: slots cross-attend to patch tokens, then FFN.

    Used inside OgentiBlock Pass A. Gate is zero-init so refinement is a no-op
    at step 0 (retrofit invariant).
    """

    def __init__(self, dim: int, num_heads: int, head_dim: int, mlp_ratio: float = 2.0) -> None:
        super().__init__()
        from ogenti.modules.attention.multihead import FeedForward, MultiHeadAttention

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = MultiHeadAttention(
            dim=dim,
            num_heads=num_heads,
            head_dim=head_dim,
            qk_norm=True,
            zero_init_out=True,
        )

        self.norm_ff = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, mlp_ratio=mlp_ratio)
        self.ff_gate = nn.Parameter(torch.zeros(1))

    def forward(self, slots: Tensor, patches: Tensor) -> Tensor:
        slots = slots + self.cross_attn(self.norm_q(slots), self.norm_kv(patches))
        slots = slots + self.ff_gate * self.ffn(self.norm_ff(slots))
        return slots
