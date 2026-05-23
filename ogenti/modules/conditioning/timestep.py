"""Timestep / text / AdaLN conditioning matching Wan2.2 layout.

Wan2.2 (both TI2V-5B and T2V/I2V-A14B) places conditioning as:

    time_embedding   = nn.Sequential(Linear(freq_dim, dim), SiLU, Linear(dim, dim))
    text_embedding   = nn.Sequential(Linear(text_dim, dim), GELU, Linear(dim, dim))
    time_projection  = nn.Sequential(SiLU, Linear(dim, 6*dim))         # global per-step e
    block.modulation = nn.Parameter((1, 6, dim))                       # per-block offset
    head.modulation  = nn.Parameter((1, 2, dim))                       # output AdaLN

so the per-block AdaLN params are `(block.modulation + e.reshape(B,6,dim)).chunk(6, dim=1)`.

We keep the legacy `TimestepConditioningHead` / `AdaLNZero` classes around so older
configs / tests / curriculum stages that targeted the 5B / TI2V layout still load,
but the A14B path uses the new `WanTimeEmbedding` + `WanTextEmbedding` +
`WanTimeProjection` + `WanBlockModulation` quartet to be byte-for-byte
state-dict compatible with the upstream Wan2.2 snapshots.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn


class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal positional encoding of scalar timesteps."""

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


class WanTimeEmbedding(nn.Sequential):
    """Wan2.2 time_embedding: Linear -> SiLU -> Linear over a *pre-sinusoidal*
    input of shape ``(B, freq_dim)``. Apply a `SinusoidalTimestepEmbedding`
    externally before feeding it in.

    Subclassing `nn.Sequential` keeps the state dict keys flat
    (``0.weight``, ``0.bias``, ``2.weight``, ``2.bias``) so that
    ``self.time_embedding = WanTimeEmbedding(...)`` produces keys
    ``time_embedding.0.weight`` etc. exactly matching Wan2.2.
    """

    def __init__(self, freq_dim: int, dim: int) -> None:
        super().__init__(
            nn.Linear(freq_dim, dim, bias=True),
            nn.SiLU(),
            nn.Linear(dim, dim, bias=True),
        )


class WanTextEmbedding(nn.Sequential):
    """Wan2.2 text_embedding: Linear -> GELU -> Linear over T5 embeddings.

    State dict keys (relative to parent attribute name `text_embedding`):
        text_embedding.0.weight/bias (Linear(text_dim, dim))
        text_embedding.2.weight/bias (Linear(dim, dim))
    """

    def __init__(self, text_dim: int, dim: int) -> None:
        super().__init__(
            nn.Linear(text_dim, dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim, dim, bias=True),
        )


class WanTimeProjection(nn.Sequential):
    """Wan2.2 time_projection: SiLU -> Linear(dim, 6*dim).

    State dict keys (relative to parent attribute name `time_projection`):
        time_projection.1.weight/bias  (Linear; index 0 is SiLU activation, no params).

    Output is `(B, 6*dim)` which the caller reshapes to `(B, 6, dim)` before adding
    per-block modulation offsets.
    """

    def __init__(self, dim: int) -> None:
        super().__init__(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),
        )


class WanOutputHead(nn.Module):
    """Wan2.2 output head.

    State-dict layout (relative to the parent attribute name, conventionally `head`):
        head.modulation        - nn.Parameter((1, 2, dim))     learned (shift, scale) offset
        head.head.weight/bias  - nn.Linear(dim, out_dim)        final projection to pixel-patch space

    The (norm-free) `nn.LayerNorm` is registered as `self.norm` for forward use but,
    being created with `elementwise_affine=False`, contributes no state-dict entries.

    Forward conditioning: the time embedding `t_emb: (B, dim)` is broadcast and
    added to `self.modulation` to produce `(shift, scale)` for the AdaLN-zero
    output. This matches Wan2.2 where the head modulation is conditioned on the
    raw time embedding (NOT the 6*dim `time_projection` output).
    """

    def __init__(self, dim: int, out_dim: int) -> None:
        super().__init__()
        self.modulation = nn.Parameter(torch.empty(1, 2, dim))
        nn.init.normal_(self.modulation, std=1.0 / math.sqrt(dim))
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim, bias=True)

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        # x: (B, N, dim), t_emb: (B, dim)
        e = self.modulation + t_emb.unsqueeze(1)              # (B, 2, dim)
        shift, scale = e.chunk(2, dim=1)                      # each (B, 1, dim)
        return self.head(self.norm(x) * (1 + scale) + shift)


# ─── legacy modules (TI2V-5B / pre-A14B configs) ──────────────────────────────


class TimestepConditioningHead(nn.Module):
    """Legacy single-MLP timestep head (used by ogenti_5b.yaml).

    Replaced by `WanTimeEmbedding` for the A14B path which separates the
    sinusoidal projection from the MLP and uses a 256-dim sinusoidal input.
    """

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
    """Legacy AdaLN-Zero (Linear from cond -> 6*dim, zero-init).

    Replaced by `WanBlockModulation` for the A14B path which uses a per-block
    learned `(1, 6, dim)` parameter added to a globally-shared
    `time_projection(cond)` output.
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
    """Apply DiT-style AdaLN modulation.

    Accepts shift/scale either as `(B, dim)` (legacy AdaLNZero output) or
    `(B, 1, dim)` (Wan-style block modulation). The latter is already broadcast
    along the token dim, the former needs an explicit unsqueeze.
    """
    if shift.dim() == 2:
        shift = shift.unsqueeze(1)
        scale = scale.unsqueeze(1)
    return x * (1 + scale) + shift
