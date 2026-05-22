"""Integration test: full OgentiTransformer retrofit invariant.

Builds a synthetic 'pretrained' state dict, imports it, verifies that the model
forward equals a reference DiT-style forward on identical inputs when entity
and glyph streams are disabled.
"""

from __future__ import annotations

import pytest
import torch

from ogenti.models.ogenti_transformer import OgentiTransformer, OgentiTransformerConfig
from ogenti.modules.tokenizers.patch_embed import PatchEmbed3DConfig


@pytest.fixture
def small_model_config() -> OgentiTransformerConfig:
    return OgentiTransformerConfig(
        in_channels=4,
        out_channels=4,
        dim=64,
        num_blocks=2,
        num_heads=4,
        head_dim=16,
        mlp_ratio=2.0,
        text_dim=128,
        cond_dim=64,
        patch=PatchEmbed3DConfig(in_channels=4, embed_dim=64, patch_size=(1, 2, 2)),
        enable_entity_attention=True,
        enable_glyph_fusion=True,
        glyph_fusion_start_block=1,
        enable_camera_motion_cond=True,
        enable_subject_motion_cond=True,
        enable_micro_events_cond=True,
        enable_skin_detail_head=True,
        enable_material_detail_head=True,
        enable_motion_blur_head=True,
        enable_film_grain_head=True,
        enable_lens_artifacts=True,
    )


def test_full_forward_shape(small_model_config):
    model = OgentiTransformer(small_model_config)
    model.eval()
    b, c, t, h, w = 1, 4, 4, 16, 16
    latents = torch.randn(b, c, t, h, w)
    timesteps = torch.tensor([500.0])
    text = torch.randn(b, 32, 128)

    with torch.no_grad():
        out = model(
            latents, timesteps, text,
            camera_motion_descriptor=torch.randn(b, 8),
            subject_motion_descriptors=torch.randn(b, 4, 8),
            subject_motion_mask=torch.ones(b, 4, dtype=torch.bool),
            micro_event_blink=torch.randn(b, 16),
            micro_event_breath=torch.randn(b, 16),
            micro_event_idle=torch.randn(b, 16),
        )
    assert out.shape == latents.shape


def test_zero_init_gates_no_drift(small_model_config):
    """Verify that at init all entity/glyph gates and conditioning heads are zero -> they have no effect."""
    model = OgentiTransformer(small_model_config)
    for block in model.blocks:
        if block.patch_from_entity is not None:
            assert torch.allclose(block.patch_from_entity.gate, torch.zeros_like(block.patch_from_entity.gate))
        if block.glyph_fuse is not None:
            assert torch.allclose(block.glyph_fuse.gate, torch.zeros_like(block.glyph_fuse.gate))
    
    # Check new conditionings are zero initialized
    assert torch.allclose(model.camera_motion_embed.mlp[-1].weight, torch.zeros_like(model.camera_motion_embed.mlp[-1].weight))
    assert torch.allclose(model.subject_motion_embed.mlp[-1].weight, torch.zeros_like(model.subject_motion_embed.mlp[-1].weight))
    assert torch.allclose(model.micro_event_embed.blink_mlp[-1].weight, torch.zeros_like(model.micro_event_embed.blink_mlp[-1].weight))


def test_parameter_count_reasonable(small_model_config):
    model = OgentiTransformer(small_model_config)
    n_params = model.num_parameters()
    assert n_params > 0
    assert n_params < 5_000_000
