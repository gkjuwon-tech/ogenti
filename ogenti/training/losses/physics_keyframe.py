"""Hard keyframe loss — MSE between tracker-derived predicted trajectory
and simulator ground-truth trajectory.

This complements (and during inference replaces) the soft `physics_loss`
from training/losses/physics.py. When a SceneSpec was simulated for a
training shot, we use this hard loss to force the generated video to
respect the simulator's trajectory exactly.

We compare:
  - 2D screen-space position trajectories (most important for visual realism)
  - velocity profiles (acceleration envelope matching)
  - optional contact event timing (binary event coincidence)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class PhysicsKeyframeLossConfig:
    position_weight: float = 1.0
    velocity_weight: float = 0.5
    contact_weight: float = 0.3
    eps: float = 1.0e-6


def _velocity(traj: Tensor) -> Tensor:
    return traj[:, :, 1:] - traj[:, :, :-1]


def keyframe_position_loss(
    pred_traj: Tensor,
    sim_traj: Tensor,
    mask: Optional[Tensor] = None,
) -> Tensor:
    err = (pred_traj - sim_traj).pow(2).sum(dim=-1).mean(dim=-1)
    if mask is not None:
        err = err * mask.to(err.dtype)
        denom = mask.to(err.dtype).sum().clamp(min=1.0)
        return err.sum() / denom
    return err.mean()


def keyframe_velocity_loss(
    pred_traj: Tensor,
    sim_traj: Tensor,
    mask: Optional[Tensor] = None,
) -> Tensor:
    p_v = _velocity(pred_traj)
    s_v = _velocity(sim_traj)
    err = (p_v - s_v).pow(2).sum(dim=-1).mean(dim=-1)
    if mask is not None:
        err = err * mask.to(err.dtype)
        denom = mask.to(err.dtype).sum().clamp(min=1.0)
        return err.sum() / denom
    return err.mean()


def keyframe_contact_event_loss(
    pred_contact_logits: Tensor,
    sim_contact_binary: Tensor,
) -> Tensor:
    return F.binary_cross_entropy_with_logits(
        pred_contact_logits, sim_contact_binary.float()
    )


def compute_physics_keyframe_loss(
    pred_traj: Tensor,
    sim_traj: Tensor,
    config: PhysicsKeyframeLossConfig,
    object_mask: Optional[Tensor] = None,
    pred_contact_logits: Optional[Tensor] = None,
    sim_contact_binary: Optional[Tensor] = None,
) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    out["kf_position"] = keyframe_position_loss(pred_traj, sim_traj, object_mask)
    out["kf_velocity"] = keyframe_velocity_loss(pred_traj, sim_traj, object_mask)

    total = (
        config.position_weight * out["kf_position"]
        + config.velocity_weight * out["kf_velocity"]
    )

    if pred_contact_logits is not None and sim_contact_binary is not None:
        out["kf_contact"] = keyframe_contact_event_loss(pred_contact_logits, sim_contact_binary)
        total = total + config.contact_weight * out["kf_contact"]

    out["physics_keyframe_total"] = total
    return out
