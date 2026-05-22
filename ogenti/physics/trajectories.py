"""Trajectory output structs shared by all physics backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class ObjectTrajectory:
    name: str
    positions: np.ndarray
    orientations: np.ndarray
    velocities: np.ndarray
    angular_velocities: np.ndarray
    visibility: np.ndarray
    contact_events: list[tuple[int, str]] = field(default_factory=list)
    semantic_label: Optional[str] = None

    def __post_init__(self) -> None:
        t = self.positions.shape[0]
        assert self.orientations.shape == (t, 4), f"orientations shape {self.orientations.shape}"
        assert self.velocities.shape == (t, 3)
        assert self.visibility.shape == (t,)

    @property
    def num_frames(self) -> int:
        return self.positions.shape[0]

    def to_npz_dict(self, prefix: str) -> dict[str, np.ndarray]:
        return {
            f"{prefix}_positions": self.positions.astype(np.float32),
            f"{prefix}_orientations": self.orientations.astype(np.float32),
            f"{prefix}_velocities": self.velocities.astype(np.float32),
            f"{prefix}_angular_velocities": self.angular_velocities.astype(np.float32),
            f"{prefix}_visibility": self.visibility.astype(np.float32),
        }


@dataclass
class KeyframeTrajectories:
    fps: int
    duration_s: float
    objects: list[ObjectTrajectory] = field(default_factory=list)
    backend: str = "unknown"

    @property
    def num_frames(self) -> int:
        return int(self.duration_s * self.fps)

    def by_name(self, name: str) -> Optional[ObjectTrajectory]:
        for o in self.objects:
            if o.name == name:
                return o
        return None

    def stack_positions(self, max_objects: int = 8) -> np.ndarray:
        """Returns (K, T, 3) array padded to max_objects. Missing rows are zero."""
        t = self.num_frames
        out = np.zeros((max_objects, t, 3), dtype=np.float32)
        for i, obj in enumerate(self.objects[:max_objects]):
            n = min(t, obj.num_frames)
            out[i, :n] = obj.positions[:n]
        return out

    def stack_descriptor(self, max_objects: int = 8) -> tuple[np.ndarray, np.ndarray]:
        """Returns ((K, T, 10), (K,) mask). Descriptor = [pos(3), quat(4), vel(3)]."""
        t = self.num_frames
        desc = np.zeros((max_objects, t, 10), dtype=np.float32)
        mask = np.zeros(max_objects, dtype=bool)
        for i, obj in enumerate(self.objects[:max_objects]):
            n = min(t, obj.num_frames)
            desc[i, :n, 0:3] = obj.positions[:n]
            desc[i, :n, 3:7] = obj.orientations[:n]
            desc[i, :n, 7:10] = obj.velocities[:n]
            mask[i] = True
        return desc, mask
