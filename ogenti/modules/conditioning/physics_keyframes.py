"""PhysicsKeyframeEmbed — inject simulator-derived trajectories as conditioning.

Each scene object contributes K_tokens temporally-downsampled descriptors
(position 3 + quaternion 4 + velocity 3 = 10 dims). Up to N_max objects.
Tokens are embedded with a small MLP, mixed across objects, and added to
the main conditioning stream.

The output projection is zero-initialized → retrofit invariant: if no scene
is provided, contribution is zero, model behaves exactly as the previous
checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from einops import rearrange
from torch import Tensor, nn


PHYSICS_DESCRIPTOR_DIM = 10
DEFAULT_MAX_OBJECTS = 8
DEFAULT_TOKENS_PER_OBJECT = 16


@dataclass
class PhysicsKeyframeEmbedConfig:
    max_objects: int = DEFAULT_MAX_OBJECTS
    tokens_per_object: int = DEFAULT_TOKENS_PER_OBJECT
    descriptor_per_token: int = PHYSICS_DESCRIPTOR_DIM
    embed_dim: int = 256
    out_dim: int = 3072
    zero_init: bool = True


class PhysicsKeyframeEmbed(nn.Module):
    """Trajectory tokens → single conditioning vector contribution."""

    def __init__(self, config: PhysicsKeyframeEmbedConfig) -> None:
        super().__init__()
        self.config = config

        self.token_mlp = nn.Sequential(
            nn.Linear(config.descriptor_per_token, config.embed_dim),
            nn.SiLU(),
            nn.Linear(config.embed_dim, config.embed_dim),
        )

        self.object_pos_embed = nn.Parameter(
            torch.randn(config.max_objects, 1, config.embed_dim) * 0.02
        )
        self.temporal_pos_embed = nn.Parameter(
            torch.randn(1, config.tokens_per_object, config.embed_dim) * 0.02
        )

        self.mixer = nn.Sequential(
            nn.LayerNorm(config.embed_dim),
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.SiLU(),
            nn.Linear(config.embed_dim, config.out_dim),
        )

        if config.zero_init:
            nn.init.zeros_(self.mixer[-1].weight)
            nn.init.zeros_(self.mixer[-1].bias)

    @staticmethod
    def downsample_trajectory(
        descriptor: Tensor,
        tokens_per_object: int,
    ) -> Tensor:
        """descriptor: (B, K, T_full, D) -> (B, K, tokens_per_object, D)."""
        b, k, t, d = descriptor.shape
        if t == tokens_per_object:
            return descriptor
        x = rearrange(descriptor, "b k t d -> (b k) d t")
        x = nn.functional.interpolate(
            x, size=tokens_per_object, mode="linear", align_corners=False
        )
        return rearrange(x, "(b k) d t -> b k t d", b=b, k=k)

    def forward(
        self,
        descriptor: Optional[Tensor],
        object_mask: Optional[Tensor],
        batch_size: int,
        device,
        dtype,
    ) -> Tensor:
        if descriptor is None:
            return torch.zeros(batch_size, self.config.out_dim, device=device, dtype=dtype)

        b, k, t, d = descriptor.shape
        assert d == self.config.descriptor_per_token, (
            f"expected descriptor dim {self.config.descriptor_per_token}, got {d}"
        )

        x = self.downsample_trajectory(descriptor.to(dtype), self.config.tokens_per_object)
        x = self.token_mlp(x)
        x = x + self.object_pos_embed[:k].unsqueeze(0).to(dtype)
        x = x + self.temporal_pos_embed.unsqueeze(0).to(dtype)

        if object_mask is not None:
            m = object_mask.to(dtype).view(b, k, 1, 1)
            x = x * m
            pooled_object = x.sum(dim=2)
            denom = m.squeeze(-1).sum(dim=1).clamp(min=1.0)
            pooled = pooled_object.sum(dim=1) / denom
        else:
            pooled = x.mean(dim=(1, 2))

        return self.mixer(pooled)
