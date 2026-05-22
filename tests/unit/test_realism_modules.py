"""Unit tests for skin detail head, frequency loss, camera motion, temporal realism."""

from __future__ import annotations

import pytest
import torch

from ogenti.modules.conditioning.camera_motion import (
    CAMERA_MOTION_DIM,
    CameraMotionEmbed,
    CameraMotionEmbedConfig,
    preset_to_descriptor,
)
from ogenti.modules.texture.skin_detail_head import SDRHConfig, SkinDetailResidualHead
from ogenti.training.losses.frequency import (
    FrequencyLossConfig,
    skin_aware_frequency_loss,
)
from ogenti.training.losses.temporal_realism import (
    TemporalRealismConfig,
    temporal_realism_loss,
)
from ogenti.modules.texture.material_detail_head import UMDHConfig, UniversalMaterialDetailHead
from ogenti.modules.postprocess.motion_blur import MotionBlurConfig, MotionBlurHead
from ogenti.modules.postprocess.film_grain import FilmGrainConfig, FilmGrainHead
from ogenti.modules.postprocess.lens_artifacts import LensArtifactConfig, LensArtifactHead
from ogenti.modules.conditioning.subject_motion import SubjectMotionEmbedConfig, SubjectMotionEmbed
from ogenti.modules.conditioning.micro_events import MicroEventConfig, MicroEventEmbed
from ogenti.training.losses.physics import PhysicsConfig, compute_physics_loss
from ogenti.training.losses.motion_realism import MotionRealismConfig, compute_motion_realism_loss


def test_sdrh_zero_init_is_identity():
    head = SkinDetailResidualHead(SDRHConfig())
    head.eval()
    video = torch.randn(1, 3, 4, 32, 32) * 0.5
    mask = torch.rand(1, 1, 4, 32, 32)
    with torch.no_grad():
        out = head(video, mask)
    assert torch.allclose(out, video.clamp(-1, 1), atol=1e-6)


def test_sdrh_mask_gating():
    head = SkinDetailResidualHead(SDRHConfig())
    with torch.no_grad():
        head.residual_scale.fill_(1.0)
        head.out_conv.weight.normal_(std=0.1)
        head.out_conv.bias.normal_(std=0.1)
    video = torch.zeros(1, 3, 2, 16, 16)
    mask_zero = torch.zeros(1, 1, 2, 16, 16)
    out = head(video, mask_zero)
    assert torch.allclose(out, video, atol=1e-6), "residual must be zero where mask=0"


def test_frequency_loss_shapes():
    pred = torch.randn(1, 3, 4, 32, 32)
    target = torch.randn(1, 3, 4, 32, 32)
    mask = torch.rand(1, 1, 4, 32, 32)
    loss = skin_aware_frequency_loss(pred, target, mask, FrequencyLossConfig())
    assert loss.dim() == 0
    assert loss.item() >= 0


def test_camera_motion_preset_dim():
    for name in ["tripod", "handheld", "documentary", "drone"]:
        d = preset_to_descriptor(name)
        assert d.shape == (CAMERA_MOTION_DIM,)


def test_camera_motion_embed_zero_init():
    cfg = CameraMotionEmbedConfig(out_dim=64, zero_init=True)
    embed = CameraMotionEmbed(cfg)
    desc = preset_to_descriptor("handheld").unsqueeze(0)
    out = embed(desc, batch_size=1, device="cpu", dtype=torch.float32)
    assert torch.allclose(out, torch.zeros_like(out))


def test_temporal_realism_loss_decreases_with_match():
    target = torch.randn(2, 3, 8, 16, 16)
    cfg = TemporalRealismConfig()
    out_same = temporal_realism_loss(target.clone(), target, cfg)
    out_diff = temporal_realism_loss(torch.randn_like(target), target, cfg)
    assert out_same["temporal_total"].item() <= out_diff["temporal_total"].item()


def test_umdh_zero_init_is_identity():
    head = UniversalMaterialDetailHead(UMDHConfig())
    head.eval()
    video = torch.randn(1, 3, 4, 32, 32) * 0.5
    mask = torch.rand(1, 8, 4, 32, 32)
    with torch.no_grad():
        out = head(video, mask)
    assert torch.allclose(out, video.clamp(-1, 1), atol=1e-6)

def test_motion_blur_zero_shutter_identity():
    head = MotionBlurHead(MotionBlurConfig())
    head.eval()
    video = torch.randn(1, 3, 4, 32, 32)
    flow = torch.zeros(1, 4, 32, 32, 2)
    with torch.no_grad():
        out = head(video, flow)
    assert torch.allclose(out, video, atol=1e-6)

def test_film_grain_zero_amount_identity():
    head = FilmGrainHead(FilmGrainConfig())
    head.eval()
    video = torch.randn(1, 3, 4, 32, 32)
    with torch.no_grad():
        out = head(video, amount=0.0)
    assert torch.allclose(out, video, atol=1e-6)

def test_lens_artifacts_zero_init_identity():
    head = LensArtifactHead(LensArtifactConfig(zero_init=True))
    head.eval()
    video = torch.randn(1, 3, 4, 32, 32)
    with torch.no_grad():
        out = head(video)
    assert torch.allclose(out, video, atol=1e-6)

def test_subject_motion_embed_zero_init():
    cfg = SubjectMotionEmbedConfig(out_dim=64, zero_init=True)
    embed = SubjectMotionEmbed(cfg)
    desc = torch.randn(1, 4, 8)
    mask = torch.ones(1, 4, dtype=torch.bool)
    out = embed(desc, mask, batch_size=1, device="cpu", dtype=torch.float32)
    assert torch.allclose(out, torch.zeros_like(out))

def test_micro_event_embed_zero_init():
    cfg = MicroEventConfig(cond_dim=64, zero_init=True)
    embed = MicroEventEmbed(cfg)
    blink = torch.randn(1, 16)
    out = embed(blink, None, None, batch_size=1, device="cpu", dtype=torch.float32)
    assert torch.allclose(out, torch.zeros_like(out))

def test_physics_loss_shapes():
    traj = torch.randn(1, 4, 10, 2)
    out = compute_physics_loss(traj, traj, PhysicsConfig())
    assert "physics_total" in out
    assert out["physics_total"].item() >= 0

def test_motion_realism_loss_shapes():
    traj = torch.randn(1, 4, 10, 2)
    mask = torch.ones(1, 4, dtype=torch.bool)
    out = compute_motion_realism_loss(traj, traj, MotionRealismConfig(), mask)
    assert "motion_realism_total" in out
    assert out["motion_realism_total"].item() >= 0
