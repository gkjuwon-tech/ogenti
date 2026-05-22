"""Subject/object motion realism loss.

Components:
  1. Acceleration envelope matching — penalize mismatch between the *shape*
     of the per-track acceleration profile (windup → impact → recovery)
     instead of just its magnitude. We compare normalized acceleration
     curves track-by-track.
  2. Hesitation preservation — real subjects pause; we measure
     "near-zero-velocity" windows in GT and penalize loss of them in
     the prediction.
  3. Velocity spectrum match — power spectrum of per-track velocity
     time-series should match GT.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class MotionRealismConfig:
    accel_envelope_weight: float = 1.0
    hesitation_weight: float = 0.5
    velocity_spectrum_weight: float = 0.5
    hesitation_threshold: float = 0.01
    eps: float = 1.0e-6


def _track_velocity(track_xy: Tensor) -> Tensor:
    return track_xy[:, :, 1:] - track_xy[:, :, :-1]


def _track_acceleration(track_xy: Tensor) -> Tensor:
    v = _track_velocity(track_xy)
    return v[:, :, 1:] - v[:, :, :-1]


def _normalize_envelope(x: Tensor) -> Tensor:
    norm = x.float().pow(2).sum(dim=-1, keepdim=True).sqrt().clamp(min=1e-6)
    return x / norm


def acceleration_envelope_loss(
    pred_tracks: Tensor,
    target_tracks: Tensor,
    track_mask: Tensor | None = None,
) -> Tensor:
    p_acc = _track_acceleration(pred_tracks)
    g_acc = _track_acceleration(target_tracks)
    p_env = _normalize_envelope(p_acc)
    g_env = _normalize_envelope(g_acc)
    err = (p_env - g_env).pow(2).mean(dim=-1)
    if track_mask is not None:
        err = err * track_mask.unsqueeze(-1)
        denom = track_mask.sum().clamp(min=1.0)
        return err.sum() / denom
    return err.mean()


def hesitation_loss(
    pred_tracks: Tensor,
    target_tracks: Tensor,
    threshold: float,
) -> Tensor:
    p_v = _track_velocity(pred_tracks).pow(2).sum(dim=-1).sqrt()
    g_v = _track_velocity(target_tracks).pow(2).sum(dim=-1).sqrt()
    p_pause = (p_v < threshold).float().mean(dim=-1)
    g_pause = (g_v < threshold).float().mean(dim=-1)
    return (p_pause - g_pause).pow(2).mean()


def velocity_spectrum_loss(
    pred_tracks: Tensor,
    target_tracks: Tensor,
    eps: float,
) -> Tensor:
    p_v = _track_velocity(pred_tracks)
    g_v = _track_velocity(target_tracks)
    p_mag = p_v.pow(2).sum(dim=-1).sqrt()
    g_mag = g_v.pow(2).sum(dim=-1).sqrt()
    if p_mag.shape[-1] < 4:
        return p_mag.new_zeros(())
    p_ps = torch.fft.rfft(p_mag.float(), dim=-1).abs().pow(2).clamp(min=eps)
    g_ps = torch.fft.rfft(g_mag.float(), dim=-1).abs().pow(2).clamp(min=eps)
    return (p_ps.log() - g_ps.log()).pow(2).mean()


def compute_motion_realism_loss(
    pred_tracks: Tensor,
    target_tracks: Tensor,
    config: MotionRealismConfig,
    track_mask: Tensor | None = None,
) -> dict[str, Tensor]:
    """
    Args:
        pred_tracks:   (B, K, T, 2) — predicted per-track xy positions
        target_tracks: (B, K, T, 2)
        track_mask:    (B, K) bool
    """
    out: dict[str, Tensor] = {}
    out["accel_envelope"] = acceleration_envelope_loss(pred_tracks, target_tracks, track_mask)
    out["hesitation"] = hesitation_loss(pred_tracks, target_tracks, config.hesitation_threshold)
    out["velocity_spectrum"] = velocity_spectrum_loss(pred_tracks, target_tracks, config.eps)
    out["motion_realism_total"] = (
        config.accel_envelope_weight * out["accel_envelope"]
        + config.hesitation_weight * out["hesitation"]
        + config.velocity_spectrum_weight * out["velocity_spectrum"]
    )
    return out
