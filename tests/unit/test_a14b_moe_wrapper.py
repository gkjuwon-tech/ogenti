"""Unit tests for the A14B dual-expert inference wrapper.

We don't load real 14B experts in CI (and the surrounding OgentiTransformer
forward path is exercised separately by the retrofit-invariant integration
tests). These tests cover the *routing* logic in isolation: which expert
gets called for which timestep, under which routing mode.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import torch
from torch import nn

from ogenti.inference.moe_wrapper import A14BMoEConfig, A14BMoEWrapper


class _SpyExpert(nn.Module):
    """Stand-in for OgentiTransformer that just records that it was called."""

    def __init__(self, tag: str) -> None:
        super().__init__()
        self.tag = tag
        self.calls: list[tuple] = []
        # Mimic the patch_embed.proj.weight chain used by _maybe_swap_device.
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Linear(1, 1)
        self.config = MagicMock(name=f"cfg_{tag}")

    def forward(self, latents, timesteps, text_embeddings, **kwargs):
        self.calls.append((self.tag, float(timesteps.float().mean())))
        return torch.full_like(latents, fill_value=(1.0 if self.tag == "hi" else -1.0))


def _make_wrapper(**moe_kwargs) -> tuple[A14BMoEWrapper, _SpyExpert, _SpyExpert]:
    hi = _SpyExpert("hi")
    lo = _SpyExpert("lo")
    wrapper = A14BMoEWrapper(hi, lo, A14BMoEConfig(**moe_kwargs))
    return wrapper, hi, lo


def test_threshold_routing_picks_high_noise_for_large_t():
    wrapper, hi, lo = _make_wrapper(boundary_t=0.5)
    latents = torch.zeros(1, 4, 2, 4, 4)
    text = torch.zeros(1, 8, 16)

    out = wrapper(latents, torch.tensor([800.0]), text)
    assert len(hi.calls) == 1
    assert len(lo.calls) == 0
    assert out.unique().item() == 1.0


def test_threshold_routing_picks_low_noise_for_small_t():
    wrapper, hi, lo = _make_wrapper(boundary_t=0.5)
    latents = torch.zeros(1, 4, 2, 4, 4)
    text = torch.zeros(1, 8, 16)

    out = wrapper(latents, torch.tensor([100.0]), text)
    assert len(hi.calls) == 0
    assert len(lo.calls) == 1
    assert out.unique().item() == -1.0


def test_threshold_routing_picks_low_noise_exactly_at_boundary_minus_one():
    # boundary_t = 0.5 -> exactly t=499/1000 should be low_noise
    wrapper, hi, lo = _make_wrapper(boundary_t=0.5)
    latents = torch.zeros(1, 4, 2, 4, 4)
    text = torch.zeros(1, 8, 16)

    wrapper(latents, torch.tensor([499.0]), text)
    assert len(hi.calls) == 0 and len(lo.calls) == 1


def test_threshold_routing_picks_high_noise_at_boundary():
    # t=500/1000 == 0.5 == boundary -> high_noise (>=)
    wrapper, hi, lo = _make_wrapper(boundary_t=0.5)
    latents = torch.zeros(1, 4, 2, 4, 4)
    text = torch.zeros(1, 8, 16)

    wrapper(latents, torch.tensor([500.0]), text)
    assert len(hi.calls) == 1 and len(lo.calls) == 0


def test_high_only_routing_ignores_timestep():
    wrapper, hi, lo = _make_wrapper(routing="high_only")
    latents = torch.zeros(1, 4, 2, 4, 4)
    text = torch.zeros(1, 8, 16)

    wrapper(latents, torch.tensor([10.0]), text)
    wrapper(latents, torch.tensor([900.0]), text)
    assert len(hi.calls) == 2 and len(lo.calls) == 0


def test_low_only_routing_ignores_timestep():
    wrapper, hi, lo = _make_wrapper(routing="low_only")
    latents = torch.zeros(1, 4, 2, 4, 4)
    text = torch.zeros(1, 8, 16)

    wrapper(latents, torch.tensor([10.0]), text)
    wrapper(latents, torch.tensor([900.0]), text)
    assert len(hi.calls) == 0 and len(lo.calls) == 2


def test_shared_config_returns_low_noise_config():
    wrapper, hi, lo = _make_wrapper()
    assert wrapper.shared_config is lo.config


def test_custom_boundary_threshold():
    # boundary_t = 0.9 -> t=850 (==0.85) below threshold -> low_noise
    wrapper, hi, lo = _make_wrapper(boundary_t=0.9)
    latents = torch.zeros(1, 4, 2, 4, 4)
    text = torch.zeros(1, 8, 16)

    wrapper(latents, torch.tensor([850.0]), text)
    assert len(hi.calls) == 0 and len(lo.calls) == 1

    wrapper(latents, torch.tensor([950.0]), text)
    assert len(hi.calls) == 1 and len(lo.calls) == 1
