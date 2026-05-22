"""Physics priors: gravity, inertia, momentum conservation.

These are auxiliary losses computed on tracked-point trajectories. We don't
require true 3D — we operate in normalized image-space coordinates and
calibrate gravity magnitude per shot from the GT data itself, so the
prior generalizes across scenes regardless of camera distance.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class PhysicsConfig:
    gravity_weight: float = 1.0
    inertia_weight: float = 0.5
    momentum_weight: float = 0.3
    gravity_freefall_threshold: float = 0.005
    direction_reversal_window: int = 3


def _velocity(traj: Tensor) -> Tensor:
    return traj[:, :, 1:] - traj[:, :, :-1]


def _accel(traj: Tensor) -> Tensor:
    v = _velocity(traj)
    return v[:, :, 1:] - v[:, :, :-1]


def calibrate_gravity_per_shot(gt_traj: Tensor) -> Tensor:
    """Estimate per-shot gravity from GT — mean vertical acceleration during
    high-velocity downward motion intervals.

    gt_traj: (B, K, T, 2)
    returns: (B,) gravity magnitude in normalized coords / frame²
    """
    v = _velocity(gt_traj)
    a = _accel(gt_traj)
    vy = v[:, :, 1:, 1]
    ay = a[:, :, :, 1]
    falling = vy > 0.001
    if falling.any():
        masked = ay.masked_fill(~falling, float("nan"))
        per_sample = torch.nanmean(masked.reshape(masked.shape[0], -1), dim=-1)
        per_sample = torch.nan_to_num(per_sample, nan=0.0)
        return per_sample.clamp(min=0.0)
    return torch.zeros(gt_traj.shape[0], device=gt_traj.device, dtype=gt_traj.dtype)


def gravity_consistency_loss(
    pred_traj: Tensor,
    gt_traj: Tensor,
    threshold: float,
) -> Tensor:
    """Penalize vertical-accel mismatch during free-fall intervals only."""
    g_per_shot = calibrate_gravity_per_shot(gt_traj)
    p_v = _velocity(pred_traj)
    p_a = _accel(pred_traj)
    p_vy = p_v[:, :, 1:, 1]
    p_ay = p_a[:, :, :, 1]

    free_fall_mask = (p_vy > threshold).float()
    target = g_per_shot[:, None, None].expand_as(p_ay)
    err = (p_ay - target).pow(2) * free_fall_mask
    denom = free_fall_mask.sum().clamp(min=1.0)
    return err.sum() / denom


def inertia_loss(
    pred_traj: Tensor,
    window: int,
) -> Tensor:
    """Penalize direction reversals without intermediate deceleration."""
    v = _velocity(pred_traj)
    if v.shape[2] < 2 * window + 1:
        return v.new_zeros(())
    v_mag = v.pow(2).sum(dim=-1).sqrt()
    a = _accel(pred_traj)
    dot = (v[:, :, :-1] * v[:, :, 1:]).sum(dim=-1)
    cos = dot / (v_mag[:, :, :-1] * v_mag[:, :, 1:] + 1e-6)
    reversal = (cos < -0.5).float()
    a_mag = a.pow(2).sum(dim=-1).sqrt()
    if a_mag.shape[2] != reversal.shape[2]:
        a_mag = a_mag[:, :, : reversal.shape[2]]
    return (reversal * a_mag).mean()


def momentum_conservation_loss(
    pred_traj: Tensor,
    gt_traj: Tensor,
) -> Tensor:
    p_v = _velocity(pred_traj).pow(2).sum(dim=-1).sqrt().mean(dim=-1)
    g_v = _velocity(gt_traj).pow(2).sum(dim=-1).sqrt().mean(dim=-1)
    return (p_v - g_v).pow(2).mean()


def compute_physics_loss(
    pred_traj: Tensor,
    gt_traj: Tensor,
    config: PhysicsConfig,
) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    out["gravity"] = gravity_consistency_loss(pred_traj, gt_traj, config.gravity_freefall_threshold)
    out["inertia"] = inertia_loss(pred_traj, config.direction_reversal_window)
    out["momentum"] = momentum_conservation_loss(pred_traj, gt_traj)
    out["physics_total"] = (
        config.gravity_weight * out["gravity"]
        + config.inertia_weight * out["inertia"]
        + config.momentum_weight * out["momentum"]
    )
    return out
