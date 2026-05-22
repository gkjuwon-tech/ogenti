"""Subject motion field conditioning.

Camera motion (camera_motion.py) handles whole-frame motion. Subject motion
handles the *internal* motion of objects/characters within the frame.

Strategy: extract a per-track motion descriptor for the most prominent
foreground subjects (up to K_max tracks). Each track produces:

  [vel_mean_x, vel_mean_y, vel_std_x, vel_std_y,
   accel_max, hesitation_score, decel_ratio, articulation_energy]

These descriptors are aggregated into a per-track token, embedded by an MLP,
and added to the conditioning stream alongside the camera motion token.

At inference, the user can either supply a preset ("calm walking",
"explosive sprint", "hesitant browsing") OR omit entirely (zero-conditioning).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from einops import rearrange
from torch import Tensor, nn


SUBJECT_MOTION_DIM = 8
MAX_SUBJECT_TRACKS = 8


@dataclass
class SubjectMotionEmbedConfig:
    descriptor_dim: int = SUBJECT_MOTION_DIM
    max_tracks: int = MAX_SUBJECT_TRACKS
    embed_dim: int = 256
    out_dim: int = 3072
    zero_init: bool = True


class SubjectMotionEmbed(nn.Module):
    """Embed up to K subject motion descriptors → single conditioning vector."""

    def __init__(self, config: SubjectMotionEmbedConfig) -> None:
        super().__init__()
        self.config = config

        self.per_track_mlp = nn.Sequential(
            nn.Linear(config.descriptor_dim, config.embed_dim),
            nn.SiLU(),
            nn.Linear(config.embed_dim, config.embed_dim),
        )

        self.track_mixer = nn.Sequential(
            nn.LayerNorm(config.embed_dim),
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.SiLU(),
            nn.Linear(config.embed_dim, config.out_dim),
        )

        if config.zero_init:
            nn.init.zeros_(self.track_mixer[-1].weight)
            nn.init.zeros_(self.track_mixer[-1].bias)

    def forward(
        self,
        descriptors: Tensor | None,
        track_mask: Tensor | None,
        batch_size: int,
        device,
        dtype,
    ) -> Tensor:
        if descriptors is None:
            return torch.zeros(batch_size, self.config.out_dim, device=device, dtype=dtype)

        k = descriptors.shape[1]
        x = self.per_track_mlp(descriptors.to(dtype))

        if track_mask is not None:
            m = track_mask.to(dtype).unsqueeze(-1)
            x = x * m
            denom = m.sum(dim=1).clamp(min=1.0)
            pooled = x.sum(dim=1) / denom
        else:
            pooled = x.mean(dim=1)

        return self.track_mixer(pooled)


SUBJECT_MOTION_PRESETS: dict[str, list[float]] = {
    "calm":        [0.005, 0.000, 0.002, 0.001, 0.05, 0.20, 0.30, 0.10],
    "walking":     [0.015, 0.003, 0.008, 0.005, 0.20, 0.30, 0.50, 0.30],
    "running":     [0.060, 0.020, 0.030, 0.015, 0.80, 0.10, 0.40, 0.80],
    "hesitant":    [0.008, 0.002, 0.020, 0.010, 0.15, 0.70, 0.20, 0.20],
    "explosive":   [0.040, 0.030, 0.080, 0.050, 1.20, 0.05, 0.85, 0.95],
    "browsing":    [0.005, 0.005, 0.010, 0.010, 0.10, 0.60, 0.50, 0.40],
    "dance":       [0.030, 0.030, 0.040, 0.040, 0.60, 0.15, 0.45, 0.85],
}


def subject_preset_to_descriptor(name: str, repeat_for_tracks: int = 1) -> Tensor:
    if name not in SUBJECT_MOTION_PRESETS:
        raise ValueError(f"unknown subject motion preset: {name}")
    base = torch.tensor(SUBJECT_MOTION_PRESETS[name], dtype=torch.float32)
    return base.unsqueeze(0).expand(repeat_for_tracks, -1).clone()
