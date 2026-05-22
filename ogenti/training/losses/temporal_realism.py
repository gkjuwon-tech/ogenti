"""Temporal realism loss — match the motion power spectrum of ground truth.

Flow-matching averages over noise → produces temporally smooth video.
This loss explicitly penalizes mismatch between the temporal power spectra
of predicted and target motion, so the model is forced to reproduce the
correct amount of jitter, micro-shake, and acceleration spikes.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor


@dataclass
class TemporalRealismConfig:
    spectrum_weight: float = 1.0
    accel_weight: float = 0.5
    hf_cutoff_hz_ratio: float = 0.25
    eps: float = 1.0e-6


def _temporal_highpass(video: Tensor, cutoff_ratio: float) -> Tensor:
    """video: (B, C, T, H, W) -> high-pass along T."""
    b, c, t, h, w = video.shape
    x = rearrange(video, "b c t h w -> b c h w t").float()
    fft = torch.fft.rfft(x, dim=-1)
    freqs = torch.fft.rfftfreq(t, d=1.0, device=video.device)
    cutoff = cutoff_ratio * freqs.max()
    mask = (freqs >= cutoff).to(fft.dtype)
    fft = fft * mask
    out = torch.fft.irfft(fft, n=t, dim=-1).to(video.dtype)
    return rearrange(out, "b c h w t -> b c t h w")


def _temporal_power_spectrum(video: Tensor) -> Tensor:
    b, c, t, h, w = video.shape
    x = rearrange(video, "b c t h w -> b c h w t").float()
    fft = torch.fft.rfft(x, dim=-1)
    return fft.abs().pow(2).mean(dim=(2, 3))


def temporal_realism_loss(
    pred_video: Tensor,
    target_video: Tensor,
    config: TemporalRealismConfig,
) -> dict[str, Tensor]:
    pred_ps = _temporal_power_spectrum(pred_video)
    targ_ps = _temporal_power_spectrum(target_video)
    spectrum_loss = (
        (pred_ps.clamp(min=config.eps).log() - targ_ps.clamp(min=config.eps).log())
        .pow(2)
        .mean()
    )

    if pred_video.shape[2] >= 3:
        pred_accel = pred_video[:, :, 2:] - 2 * pred_video[:, :, 1:-1] + pred_video[:, :, :-2]
        targ_accel = target_video[:, :, 2:] - 2 * target_video[:, :, 1:-1] + target_video[:, :, :-2]
        pred_accel_mag = pred_accel.float().pow(2).mean(dim=(3, 4)).sqrt()
        targ_accel_mag = targ_accel.float().pow(2).mean(dim=(3, 4)).sqrt()
        accel_loss = (pred_accel_mag - targ_accel_mag).pow(2).mean()
    else:
        accel_loss = pred_video.new_zeros(())

    total = config.spectrum_weight * spectrum_loss + config.accel_weight * accel_loss
    return {
        "temporal_spectrum": spectrum_loss,
        "temporal_accel": accel_loss,
        "temporal_total": total,
    }
