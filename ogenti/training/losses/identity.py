"""Identity persistence loss for entity tokens.

Two components:
  1. Entity-patch alignment: each entity slot should be assigned to a coherent
     spatio-temporal cluster of patches. We use a soft-assignment regularizer
     pushing the slot-patch attention map to be temporally consistent.

  2. Cross-frame CLIP similarity of entity-token-modulated crops vs. their
     counterparts in adjacent frames.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class IdentityLossConfig:
    temporal_consistency_weight: float = 1.0
    diversity_weight: float = 0.1


def temporal_attention_consistency(
    attn_maps: Tensor,
) -> Tensor:
    """attn_maps: (B, T, K, HW). Penalize change across t per slot."""
    if attn_maps.shape[1] < 2:
        return attn_maps.new_zeros(())
    diff = attn_maps[:, 1:] - attn_maps[:, :-1]
    return diff.pow(2).mean()


def slot_diversity_penalty(slots: Tensor) -> Tensor:
    """Encourage slots to occupy distinct subspaces."""
    s = F.normalize(slots, dim=-1)
    sim = torch.einsum("bkd,bjd->bkj", s, s)
    k = sim.shape[-1]
    eye = torch.eye(k, device=sim.device, dtype=sim.dtype).unsqueeze(0)
    off = (sim - eye).pow(2)
    return off.mean()


def compute_identity_loss(
    slots: Tensor,
    attn_maps: Tensor | None,
    config: IdentityLossConfig,
) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    total = slots.new_zeros(())
    if attn_maps is not None:
        tc = temporal_attention_consistency(attn_maps)
        out["id_temporal"] = tc
        total = total + config.temporal_consistency_weight * tc
    div = slot_diversity_penalty(slots)
    out["id_diversity"] = div
    total = total + config.diversity_weight * div
    out["id_total"] = total
    return out
