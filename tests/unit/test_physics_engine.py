"""Unit tests for the physics engine integration (RFC-0005)."""

from __future__ import annotations

import pytest
import torch
import numpy as np

from ogenti.modules.conditioning.physics_keyframes import (
    PHYSICS_DESCRIPTOR_DIM,
    PhysicsKeyframeEmbed,
    PhysicsKeyframeEmbedConfig,
)
from ogenti.physics.scene import (
    ForceType,
    SceneObject,
    SceneSpec,
    ShapeType,
    make_falling_object_scene,
    make_rolling_object_scene,
)
from ogenti.physics.scene_parser import parse_prompt_heuristic
from ogenti.physics.trajectories import KeyframeTrajectories, ObjectTrajectory
from ogenti.training.losses.physics_keyframe import (
    PhysicsKeyframeLossConfig,
    compute_physics_keyframe_loss,
)


def test_scene_spec_round_trip_json():
    spec = make_falling_object_scene()
    s = spec.to_json()
    spec2 = SceneSpec.from_json(s)
    assert spec2.duration_s == spec.duration_s
    assert len(spec2.objects) == len(spec.objects)
    assert spec2.objects[0].shape == spec.objects[0].shape


def test_scene_object_static_zeroes_mass():
    obj = SceneObject(name="floor", shape=ShapeType.PLANE, dimensions=(10, 10), mass=5.0, is_static=True)
    assert obj.mass == 0.0


def test_scene_add_gravity_idempotent():
    spec = SceneSpec(duration_s=1.0, objects=[], forces=[])
    spec.add_gravity_if_missing()
    spec.add_gravity_if_missing()
    grav_count = sum(1 for f in spec.forces if f.type == ForceType.GRAVITY)
    assert grav_count == 1


def test_template_falling_matches():
    spec = parse_prompt_heuristic("A red bottle falls off the wooden table.")
    assert spec is not None
    assert spec.num_frames > 0
    assert any(f.type == ForceType.GRAVITY for f in spec.forces)


def test_template_rolling_matches():
    spec = parse_prompt_heuristic("The ball rolls to the right across the floor.")
    assert spec is not None


def test_template_no_match_returns_none():
    assert parse_prompt_heuristic("static product shot, studio lighting") is None


def test_physics_keyframe_embed_zero_init_returns_zero():
    cfg = PhysicsKeyframeEmbedConfig(out_dim=64, zero_init=True)
    embed = PhysicsKeyframeEmbed(cfg)

    desc = torch.randn(1, 4, 24, PHYSICS_DESCRIPTOR_DIM)
    mask = torch.ones(1, 4, dtype=torch.bool)
    out = embed(desc, mask, batch_size=1, device="cpu", dtype=torch.float32)
    assert torch.allclose(out, torch.zeros_like(out))


def test_physics_keyframe_embed_none_returns_zero():
    cfg = PhysicsKeyframeEmbedConfig(out_dim=64, zero_init=False)
    embed = PhysicsKeyframeEmbed(cfg)
    out = embed(None, None, batch_size=2, device="cpu", dtype=torch.float32)
    assert out.shape == (2, 64)
    assert torch.allclose(out, torch.zeros_like(out))


def test_physics_keyframe_embed_temporal_downsample():
    cfg = PhysicsKeyframeEmbedConfig(out_dim=64, tokens_per_object=8, zero_init=False)
    embed = PhysicsKeyframeEmbed(cfg)
    desc = torch.randn(1, 3, 81, PHYSICS_DESCRIPTOR_DIM)
    mask = torch.ones(1, 3, dtype=torch.bool)
    out = embed(desc, mask, batch_size=1, device="cpu", dtype=torch.float32)
    assert out.shape == (1, 64)
    assert torch.isfinite(out).all()


def test_physics_keyframe_loss_zero_on_perfect_match():
    pred = torch.randn(1, 3, 16, 2)
    out = compute_physics_keyframe_loss(pred, pred.clone(), PhysicsKeyframeLossConfig())
    assert out["physics_keyframe_total"].item() < 1e-5


def test_physics_keyframe_loss_nonneg():
    pred = torch.randn(2, 4, 16, 2)
    tgt = torch.randn(2, 4, 16, 2)
    mask = torch.ones(2, 4, dtype=torch.bool)
    out = compute_physics_keyframe_loss(pred, tgt, PhysicsKeyframeLossConfig(), object_mask=mask)
    for v in out.values():
        assert v.item() >= 0


def test_trajectories_stack_descriptor_shape():
    t = 24
    obj = ObjectTrajectory(
        name="ball",
        positions=np.zeros((t, 3), dtype=np.float32),
        orientations=np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (t, 1)),
        velocities=np.zeros((t, 3), dtype=np.float32),
        angular_velocities=np.zeros((t, 3), dtype=np.float32),
        visibility=np.ones(t, dtype=np.float32),
    )
    kt = KeyframeTrajectories(fps=24, duration_s=1.0, objects=[obj], backend="pybullet")
    desc, mask = kt.stack_descriptor(max_objects=4)
    assert desc.shape == (4, t, 10)
    assert mask.tolist() == [True, False, False, False]


@pytest.mark.skipif(
    pytest.importorskip("pybullet", reason="pybullet not installed") is None,
    reason="pybullet not installed",
)
def test_pybullet_smoke():
    from ogenti.physics.simulator import simulate

    spec = make_rolling_object_scene(duration_s=0.5, fps=12)
    traj = simulate(spec, backend="pybullet")
    assert traj.num_frames > 0
    ball = traj.by_name("ball")
    assert ball is not None
    assert ball.positions.shape[0] == traj.num_frames
    assert ball.positions[-1, 0] > ball.positions[0, 0]  # moved right
