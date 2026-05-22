"""Rectified flow / flow matching scheduler (Wan2.2-compatible).

Forward: x_t = (1-t) * x_0 + t * noise
Target velocity: v = noise - x_0

Sigmoid timestep sampling (logit-normal) per Wan2.2 / SD3 recipe.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class FlowMatchingConfig:
    shift: float = 7.0
    logit_mean: float = 0.0
    logit_std: float = 1.0
    num_train_timesteps: int = 1000


class FlowMatchingScheduler:
    def __init__(self, config: FlowMatchingConfig) -> None:
        self.config = config

    def sample_timesteps(self, batch_size: int, device, dtype=torch.float32) -> Tensor:
        u = torch.normal(
            mean=self.config.logit_mean,
            std=self.config.logit_std,
            size=(batch_size,),
            device=device,
            dtype=dtype,
        )
        t = torch.sigmoid(u)
        s = self.config.shift
        t = (s * t) / (1.0 + (s - 1.0) * t)
        return t

    def add_noise(self, x0: Tensor, noise: Tensor, t: Tensor) -> Tensor:
        t_b = t.view(-1, *([1] * (x0.ndim - 1)))
        return (1.0 - t_b) * x0 + t_b * noise

    def velocity_target(self, x0: Tensor, noise: Tensor) -> Tensor:
        return noise - x0

    def to_model_timesteps(self, t: Tensor) -> Tensor:
        return t * self.config.num_train_timesteps
