"""Micro-event conditioning: blinks, breath, idle micro-movements.

Real humans:
  - blink every 4-6 seconds on average, with substantial jitter (LogNormal
    inter-blink interval, μ≈4.2s, σ≈1.5s in log space)
  - breathe at rest ~0.2-0.4 Hz (12-24 BPM)
  - exhibit constant idle micro-movements (sway, weight shifts, scratch)

Generative models tend to drop these events (or homogenize them) because
the loss landscape rewards smooth high-likelihood pixel values, not
specifically-timed micro-events.

We inject these as explicit conditioning channels:
  - blink_signal:  (T,) binary {0,1} per frame
  - breath_signal: (T,) continuous [0, 1] sinusoidal+noise
  - idle_signal:   (T, 3) small random walk for sway

At training time, signals are extracted from GT video where possible
(blinks via eye-aspect-ratio on MediaPipe face landmarks, breath via
chest landmark vertical motion). At inference, we synthesize them with
realistic distributions.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class MicroEventConfig:
    blink_lognormal_mu: float = 1.435
    blink_lognormal_sigma: float = 0.35
    blink_duration_frames: int = 4
    breath_freq_hz: float = 0.25
    breath_amplitude: float = 1.0
    breath_noise_std: float = 0.08
    idle_walk_std: float = 0.005
    embed_dim: int = 128
    cond_dim: int = 3072
    zero_init: bool = True


def synthesize_blink_signal(
    num_frames: int,
    fps: float,
    config: MicroEventConfig,
    seed: int | None = None,
) -> Tensor:
    gen = None
    if seed is not None:
        gen = torch.Generator().manual_seed(seed)

    signal = torch.zeros(num_frames)
    t = 0
    while t < num_frames:
        interval_s = float(torch.distributions.LogNormal(
            torch.tensor(config.blink_lognormal_mu),
            torch.tensor(config.blink_lognormal_sigma),
        ).sample())
        gap = max(1, int(interval_s * fps))
        t += gap
        if t < num_frames:
            end = min(num_frames, t + config.blink_duration_frames)
            signal[t:end] = 1.0
    return signal


def synthesize_breath_signal(
    num_frames: int,
    fps: float,
    config: MicroEventConfig,
    seed: int | None = None,
) -> Tensor:
    gen = None
    if seed is not None:
        gen = torch.Generator().manual_seed(seed)

    t = torch.arange(num_frames, dtype=torch.float32) / fps
    base = torch.sin(2 * 3.14159265 * config.breath_freq_hz * t)
    noise = torch.randn(num_frames, generator=gen) * config.breath_noise_std
    return (config.breath_amplitude * 0.5 * (base + 1.0) + noise).clamp(0.0, 1.0)


def synthesize_idle_walk(
    num_frames: int,
    config: MicroEventConfig,
    seed: int | None = None,
) -> Tensor:
    gen = None
    if seed is not None:
        gen = torch.Generator().manual_seed(seed)
    steps = torch.randn(num_frames, 3, generator=gen) * config.idle_walk_std
    return torch.cumsum(steps, dim=0)


def synthesize_all(
    num_frames: int,
    fps: float = 24.0,
    config: MicroEventConfig | None = None,
    seed: int | None = None,
) -> dict[str, Tensor]:
    cfg = config or MicroEventConfig()
    return {
        "blink": synthesize_blink_signal(num_frames, fps, cfg, seed),
        "breath": synthesize_breath_signal(num_frames, fps, cfg, seed),
        "idle": synthesize_idle_walk(num_frames, cfg, seed),
    }


class MicroEventEmbed(nn.Module):
    """Embed per-frame micro-event signals into a per-shot conditioning vector.

    Signals are summarized via mean+std over the temporal axis, then projected.
    """

    def __init__(self, config: MicroEventConfig) -> None:
        super().__init__()
        self.config = config
        per_frame_dim = 1 + 1 + 3
        summary_dim = per_frame_dim * 2

        self.mlp = nn.Sequential(
            nn.Linear(summary_dim, config.embed_dim),
            nn.SiLU(),
            nn.Linear(config.embed_dim, config.cond_dim),
        )
        if config.zero_init:
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self,
        blink: Tensor | None,
        breath: Tensor | None,
        idle: Tensor | None,
        batch_size: int,
        device,
        dtype,
    ) -> Tensor:
        if blink is None and breath is None and idle is None:
            return torch.zeros(batch_size, self.config.cond_dim, device=device, dtype=dtype)

        parts: list[Tensor] = []
        if blink is not None:
            b = blink.to(device).to(dtype)
            if b.dim() == 1:
                b = b.unsqueeze(0).expand(batch_size, -1)
            parts.append(b.unsqueeze(-1))
        else:
            parts.append(torch.zeros(batch_size, 1, 1, device=device, dtype=dtype))

        if breath is not None:
            br = breath.to(device).to(dtype)
            if br.dim() == 1:
                br = br.unsqueeze(0).expand(batch_size, -1)
            parts.append(br.unsqueeze(-1))
        else:
            parts.append(torch.zeros(batch_size, parts[0].shape[1], 1, device=device, dtype=dtype))

        if idle is not None:
            i = idle.to(device).to(dtype)
            if i.dim() == 2:
                i = i.unsqueeze(0).expand(batch_size, -1, -1)
            parts.append(i)
        else:
            parts.append(torch.zeros(batch_size, parts[0].shape[1], 3, device=device, dtype=dtype))

        target_t = max(p.shape[1] for p in parts)
        aligned: list[Tensor] = []
        for p in parts:
            if p.shape[1] != target_t:
                p = p.expand(-1, target_t, -1) if p.shape[1] == 1 else p[:, :target_t]
            aligned.append(p)
        cat = torch.cat(aligned, dim=-1)
        summary = torch.cat([cat.mean(dim=1), cat.std(dim=1)], dim=-1)
        return self.mlp(summary)
