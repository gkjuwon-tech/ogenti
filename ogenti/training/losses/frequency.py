"""Frequency-domain losses for skin texture preservation.

Standard MSE has no frequency selectivity — gradient descent will happily
sacrifice high-frequency detail (skin pores, fabric weave, hair strands)
to minimize average error. This loss restores frequency parity by comparing
FFT power spectra in skin regions.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor


@dataclass
class FrequencyLossConfig:
    hf_cutoff_ratio: float = 0.125
    skin_weight: float = 4.0
    non_skin_weight: float = 1.0
    eps: float = 1.0e-6


def _spatial_highpass_fft(image: Tensor, cutoff_ratio: float) -> Tensor:
    """Zero out frequencies below `cutoff_ratio * nyquist`. Returns spatial residual."""
    b, c, h, w = image.shape
    fft = torch.fft.fft2(image.float())
    fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))

    cy, cx = h // 2, w // 2
    r_cut = int(cutoff_ratio * min(h, w) / 2)

    yy, xx = torch.meshgrid(
        torch.arange(h, device=image.device),
        torch.arange(w, device=image.device),
        indexing="ij",
    )
    dist = ((yy - cy).float().pow(2) + (xx - cx).float().pow(2)).sqrt()
    hp_mask = (dist >= r_cut).to(fft_shifted.dtype)
    fft_shifted = fft_shifted * hp_mask

    fft_unshifted = torch.fft.ifftshift(fft_shifted, dim=(-2, -1))
    spatial = torch.fft.ifft2(fft_unshifted).real
    return spatial.to(image.dtype)


def skin_aware_frequency_loss(
    pred_video: Tensor,
    target_video: Tensor,
    skin_mask: Tensor,
    config: FrequencyLossConfig,
) -> Tensor:
    """
    Args:
        pred_video:   (B, 3, T, H, W) in [-1, 1]
        target_video: (B, 3, T, H, W)
        skin_mask:    (B, 1, T, H, W) in [0, 1]
    Returns:
        scalar loss.
    """
    b, c, t, h, w = pred_video.shape
    p = rearrange(pred_video, "b c t h w -> (b t) c h w")
    g = rearrange(target_video, "b c t h w -> (b t) c h w")
    m = rearrange(skin_mask, "b c t h w -> (b t) c h w")

    if m.shape[-2:] != (h, w):
        m = F.interpolate(m, size=(h, w), mode="bilinear", align_corners=False)

    p_hp = _spatial_highpass_fft(p, config.hf_cutoff_ratio)
    g_hp = _spatial_highpass_fft(g, config.hf_cutoff_ratio)

    err = (p_hp - g_hp).pow(2)
    weight = config.skin_weight * m + config.non_skin_weight * (1.0 - m)
    weighted_err = err * weight

    denom = weight.mean().clamp(min=config.eps)
    return weighted_err.mean() / denom


def power_spectrum_match_loss(
    pred_video: Tensor,
    target_video: Tensor,
    skin_mask: Tensor | None = None,
) -> Tensor:
    """Match the 2D power spectrum (radially averaged) between pred and target."""
    p = rearrange(pred_video, "b c t h w -> (b t) c h w")
    g = rearrange(target_video, "b c t h w -> (b t) c h w")

    if skin_mask is not None:
        m = rearrange(skin_mask, "b c t h w -> (b t) c h w")
        if m.shape[-2:] != p.shape[-2:]:
            m = F.interpolate(m, size=p.shape[-2:], mode="bilinear", align_corners=False)
        p = p * m
        g = g * m

    p_ps = torch.fft.fft2(p.float()).abs().pow(2)
    g_ps = torch.fft.fft2(g.float()).abs().pow(2)
    log_diff = (p_ps.clamp(min=1e-8).log() - g_ps.clamp(min=1e-8).log()).pow(2)
    return log_diff.mean()
