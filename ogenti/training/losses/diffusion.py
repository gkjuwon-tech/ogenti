"""Flow-matching velocity prediction loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def flow_matching_loss(
    predicted_velocity: Tensor,
    target_velocity: Tensor,
    weights: Tensor | None = None,
) -> Tensor:
    per_sample = (predicted_velocity - target_velocity).pow(2).mean(dim=tuple(range(1, predicted_velocity.ndim)))
    if weights is not None:
        per_sample = per_sample * weights
    return per_sample.mean()
