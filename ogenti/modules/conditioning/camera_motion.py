"""Camera motion conditioning — converts shake/jitter from emergent into controlled.

Descriptor vector (8-dim):
  [0] translation_jitter_x   — std of frame-to-frame camera tx
  [1] translation_jitter_y   — std of frame-to-frame camera ty
  [2] rotation_jitter        — std of frame-to-frame camera rotation
  [3] zoom_drift             — mean optical zoom velocity
  [4] motion_entropy         — spectral entropy of motion power spectrum
  [5] handheld_logit         — learned tripod-vs-handheld classifier logit
  [6] global_motion_mag      — overall camera motion magnitude
  [7] high_freq_motion_energy — power above 4Hz (handheld signature)

At inference the user supplies this directly or our prompt parser infers it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


CAMERA_MOTION_DIM = 8


@dataclass
class CameraMotionEmbedConfig:
    descriptor_dim: int = CAMERA_MOTION_DIM
    embed_dim: int = 256
    out_dim: int = 3072
    zero_init: bool = True


class CameraMotionEmbed(nn.Module):
    """Embeds the 8-dim motion descriptor into the model conditioning stream."""

    def __init__(self, config: CameraMotionEmbedConfig) -> None:
        super().__init__()
        self.config = config
        self.mlp = nn.Sequential(
            nn.Linear(config.descriptor_dim, config.embed_dim),
            nn.SiLU(),
            nn.Linear(config.embed_dim, config.out_dim),
        )
        if config.zero_init:
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, descriptor: Tensor | None, batch_size: int, device, dtype) -> Tensor:
        if descriptor is None:
            descriptor = torch.zeros(batch_size, self.config.descriptor_dim, device=device, dtype=dtype)
        return self.mlp(descriptor.to(dtype))


PRESET_DESCRIPTORS: dict[str, list[float]] = {
    "tripod":      [0.001, 0.001, 0.0005, 0.0,  0.10, -3.0, 0.005, 0.001],
    "slider":      [0.020, 0.005, 0.0010, 0.0,  0.15, -2.0, 0.050, 0.005],
    "gimbal":      [0.015, 0.010, 0.0020, 0.05, 0.30, -0.5, 0.080, 0.010],
    "handheld":    [0.040, 0.035, 0.0080, 0.05, 0.55,  2.0, 0.150, 0.060],
    "documentary": [0.060, 0.050, 0.0100, 0.10, 0.65,  3.0, 0.180, 0.090],
    "drone":       [0.030, 0.025, 0.0040, 0.20, 0.40,  0.5, 0.220, 0.030],
}


def preset_to_descriptor(name: str) -> Tensor:
    if name not in PRESET_DESCRIPTORS:
        raise ValueError(f"unknown camera motion preset: {name}. Known: {list(PRESET_DESCRIPTORS)}")
    return torch.tensor(PRESET_DESCRIPTORS[name], dtype=torch.float32)
