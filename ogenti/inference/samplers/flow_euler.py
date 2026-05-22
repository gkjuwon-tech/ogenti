"""Euler sampler for rectified flow (Wan2.2-compatible)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor


@dataclass
class FlowEulerConfig:
    num_inference_steps: int = 50
    shift: float = 7.0
    guidance_scale: float = 5.0


class FlowEulerSampler:
    def __init__(self, config: FlowEulerConfig) -> None:
        self.config = config

    def timesteps(self, device) -> Tensor:
        n = self.config.num_inference_steps
        t = torch.linspace(1.0, 0.0, n + 1, device=device)
        s = self.config.shift
        t = (s * t) / (1.0 + (s - 1.0) * t)
        return t

    @torch.no_grad()
    def sample(
        self,
        velocity_fn: Callable[[Tensor, Tensor], Tensor],
        latents_shape: tuple,
        device,
        dtype=torch.bfloat16,
        velocity_fn_uncond: Callable[[Tensor, Tensor], Tensor] | None = None,
    ) -> Tensor:
        x = torch.randn(latents_shape, device=device, dtype=dtype)
        ts = self.timesteps(device)
        for i in range(len(ts) - 1):
            t = ts[i].expand(x.shape[0])
            dt = ts[i + 1] - ts[i]

            v_cond = velocity_fn(x, t * 1000.0)
            if velocity_fn_uncond is not None and self.config.guidance_scale != 1.0:
                v_uncond = velocity_fn_uncond(x, t * 1000.0)
                v = v_uncond + self.config.guidance_scale * (v_cond - v_uncond)
            else:
                v = v_cond
            x = x + dt * v
        return x
