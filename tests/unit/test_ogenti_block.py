"""Unit tests for OgentiBlock — focus on retrofit invariant and shape contracts.

After the RFC-0006 architecture fix, OgentiBlock takes a per-step `e6: (B, 6, dim)`
from the top-level `time_projection` (NOT a `cond: (B, dim)` like the legacy
TI2V-5B path). The block's own `modulation: (1, 6, dim)` parameter is added to
`e6` and then chunked into the six AdaLN scalars.
"""

from __future__ import annotations

import pytest
import torch

from ogenti.models.blocks.ogenti_block import OgentiBlock, OgentiBlockConfig
from ogenti.modules.identity.entity_tokens import EntityBankConfig, EntityTokenBank


@pytest.fixture
def small_config() -> OgentiBlockConfig:
    return OgentiBlockConfig(
        dim=128,
        num_heads=4,
        head_dim=32,
        mlp_ratio=2.0,
        enable_entity_attention=True,
        entity_attn_heads=4,
        entity_attn_head_dim=32,
        enable_glyph_fusion=True,
        glyph_kv_dim=128,
        glyph_heads=4,
        glyph_head_dim=32,
        enable_cross_attn=False,
    )


def _e6(b: int, d: int) -> torch.Tensor:
    return torch.randn(b, 6, d)


def test_forward_shape(small_config):
    block = OgentiBlock(small_config)
    b, n, d = 2, 64, 128
    patches = torch.randn(b, n, d)
    entities = torch.randn(b, 8, d)
    glyph = torch.randn(b, 16, d)

    out, ent_out = block(patches, _e6(b, d), entity_tokens=entities, glyph_stream=glyph)
    assert out.shape == patches.shape
    assert ent_out is not None
    assert ent_out.shape == entities.shape


def test_retrofit_invariant_without_entity_or_glyph(small_config):
    """Same e6 / same inputs => deterministic identical output."""
    block = OgentiBlock(small_config)
    block.eval()
    b, n, d = 2, 32, 128
    patches = torch.randn(b, n, d)
    e6 = _e6(b, d)

    with torch.no_grad():
        out_a, _ = block(patches, e6, entity_tokens=None, glyph_stream=None)
        out_b, _ = block(patches, e6, entity_tokens=None, glyph_stream=None)

    assert torch.allclose(out_a, out_b)


def test_zero_gate_means_no_glyph_effect(small_config):
    block = OgentiBlock(small_config)
    block.eval()
    b, n, d = 2, 32, 128
    patches = torch.randn(b, n, d)
    e6 = _e6(b, d)
    glyph = torch.randn(b, 16, d)

    with torch.no_grad():
        out_with_glyph, _ = block(patches, e6, glyph_stream=glyph)
        out_no_glyph, _ = block(patches, e6, glyph_stream=None)

    assert torch.allclose(out_with_glyph, out_no_glyph, atol=1e-5)


def test_entity_bank_expansion():
    bank = EntityTokenBank(EntityBankConfig(num_slots=32, dim=128))
    expanded = bank.expand(batch_size=4)
    assert expanded.shape == (4, 32, 128)
    assert torch.allclose(expanded[0], expanded[1])
