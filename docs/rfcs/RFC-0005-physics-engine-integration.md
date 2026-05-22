# RFC-0005 — Physics Engine Integration: Keyframe Rails from Real Simulation

**Status:** Draft
**Date:** 2026-05-22
**Depends on:** RFC-0001, RFC-0004
**Tags:** physics, conditioning, hard-constraints, ad-grade

---

## 1. Motivation

RFC-0004 introduced **soft** physics losses (gravity consistency, inertia,
momentum). These help, but they fight the diffusion model's natural tendency
to produce visually-plausible but physically-wrong motion. A 0.2-weighted
penalty against a model that has billions of parameters and a strong
smoothness prior loses on average.

The diagnosis: **physics is not a learning problem, it is a constraint
problem.** A diffusion model trained to predict pixels should not be asked
to learn gravity from data when `g=9.8 m/s²` is a known constant.

The solution: **delegate the physics, condition on the result.**

We integrate a real physics simulator (PyBullet / Genesis / MuJoCo) into the
Ogenti pipeline. At training and inference time the simulator produces
trajectories that the diffusion model is **required to follow** via
keyframe conditioning. The diffusion model then handles what it is actually
good at — appearance, lighting, materials — while the simulator handles
what it is good at — motion under physical laws.

## 2. Hypothesis

Two strong claims:

1. **Hard keyframe conditioning beats soft physics loss.** A diffusion model
   conditioned on simulator-derived trajectories will produce dramatically
   more physically-plausible motion than the same model with only a soft
   penalty, at the cost of zero generation quality (since appearance is
   still fully learned).

2. **A frozen physics simulator generalizes better than a learned physics
   prior.** PyBullet does not need a training set. It does not overfit.
   It does not hallucinate vertical drift. It produces the right answer
   for any rigid body, every time.

## 3. Architecture

```
                ┌──────────────────────────────────────────────────────┐
                │  Inference                                            │
                └──────────────────────────────────────────────────────┘
                                       │
            "Coca-Cola bottle falls off a table"
                                       │
                  ┌────────────────────▼────────────────────┐
                  │  Scene Parser                            │
                  │  (LLM call + heuristic fallback)         │
                  │  → SceneSpec {                           │
                  │      objects: [bottle, table],           │
                  │      forces: [gravity],                  │
                  │      events: [bottle-falls @t=0.5s]      │
                  │    }                                     │
                  └────────────────────┬────────────────────┘
                                       │
                  ┌────────────────────▼────────────────────┐
                  │  Physics Backend                         │
                  │  (PyBullet / Genesis / MuJoCo)           │
                  │  → KeyframeTrajectories {                │
                  │      bottle: [(x,y,z,quat) × T frames],  │
                  │      table: [(x,y,z,quat) × T frames]    │
                  │    }                                     │
                  └────────────────────┬────────────────────┘
                                       │
                  ┌────────────────────▼────────────────────┐
                  │  PhysicsKeyframeEmbed                    │
                  │  → conditioning tokens (B, K, T, D)       │
                  └────────────────────┬────────────────────┘
                                       │
                  ┌────────────────────▼────────────────────┐
                  │  OgentiTransformer                       │
                  │  (appearance, lighting, materials)       │
                  │  conditioning += keyframe tokens         │
                  └────────────────────┬────────────────────┘
                                       │
                                  output video
```

## 4. Components

### 4.1 SceneSpec (declarative)

A `SceneSpec` is a serializable description of a physics scene:

```python
@dataclass
class SceneObject:
    name: str
    shape: Literal["box", "sphere", "cylinder", "mesh", "soft", "fluid"]
    dimensions: tuple[float, ...]
    mass: float
    restitution: float = 0.5
    friction: float = 0.5
    initial_position: tuple[float, float, float] = (0, 0, 0)
    initial_velocity: tuple[float, float, float] = (0, 0, 0)
    initial_orientation_quat: tuple[float, float, float, float] = (0, 0, 0, 1)
    is_static: bool = False
    mesh_path: Optional[str] = None

@dataclass
class SceneForce:
    type: Literal["gravity", "impulse", "wind"]
    target: Optional[str] = None  # object name, or None for global
    vector: tuple[float, float, float] = (0, -9.81, 0)
    timing: tuple[float, float] = (0.0, float("inf"))  # active interval

@dataclass
class SceneSpec:
    duration_s: float
    fps: int = 24
    objects: list[SceneObject]
    forces: list[SceneForce]
    gravity: tuple[float, float, float] = (0, -9.81, 0)
```

### 4.2 Simulator Backend Interface

```python
class PhysicsBackend(Protocol):
    def simulate(self, scene: SceneSpec) -> KeyframeTrajectories: ...
```

Concrete backends:
- **PyBullet** (`PyBulletBackend`) — rigid body, default. Wide compatibility.
- **Genesis** (`GenesisBackend`) — GPU-accelerated, supports soft body + fluid.
- **MuJoCo** (`MuJoCoBackend`) — best for articulated rigs (humanoids).
- **Brax** (`BraxBackend`) — JAX-based, differentiable (for end-to-end gradient
  experiments down the line).

All four implement the same `simulate(SceneSpec) -> KeyframeTrajectories`
contract.

### 4.3 KeyframeTrajectories (output)

```python
@dataclass
class ObjectTrajectory:
    name: str
    positions: np.ndarray   # (T, 3)
    orientations: np.ndarray  # (T, 4) quaternion xyzw
    velocities: np.ndarray  # (T, 3)
    contact_events: list[tuple[int, str]]  # (frame_idx, contact_with)

@dataclass
class KeyframeTrajectories:
    fps: int
    duration_s: float
    objects: list[ObjectTrajectory]
```

### 4.4 Scene Parser (Prompt → SceneSpec)

Two-tier:

**Tier 1 — Heuristic templates.** Pattern-match the prompt against a
library of common ad-scenarios:
- "X falls off Y" → gravity + collision
- "X rolls across Y" → initial velocity + friction
- "X splashes Y" → fluid + rigid coupling (Genesis-only)
- "X swings on Y" → constraint (hinge)
- "Cloth drapes over X" → soft body + collision

**Tier 2 — LLM call.** For prompts that don't match a template, we call an
LLM with a system prompt instructing it to output a `SceneSpec` JSON. The
LLM call is cached aggressively and a default scene is used on failure.

### 4.5 PhysicsKeyframeEmbed

Converts `KeyframeTrajectories` → conditioning tokens for the
`OgentiTransformer`. Each object contributes K_traj tokens (default 16,
temporally downsampled from T_video frames). Tokens encode position +
orientation + velocity. Concatenated across objects (max 8 objects).

```python
class PhysicsKeyframeEmbedConfig:
    max_objects: int = 8
    tokens_per_object: int = 16
    descriptor_per_token: int = 10  # (x,y,z, qx,qy,qz,qw, vx,vy,vz)
    embed_dim: int = 256
    out_dim: int = 3072
    zero_init: bool = True
```

Injected into the existing conditioning stream alongside camera_motion /
subject_motion. Zero-init → retrofit invariant preserved.

### 4.6 Hard Keyframe Loss

```
hard_kf_loss = MSE( differentiable_track(pred_video, init_boxes),
                    simulator_trajectories )
```

This **replaces** the soft physics loss when a SceneSpec is available, but
both can coexist (soft loss handles edge-case shots where we don't have a
scene).

## 5. Data Strategy

### 5.1 Training-time Pre-sim

For each ad video in our training corpus we:
1. Extract object bounding boxes per frame (via offline detector).
2. Lift 2D → 3D using monocular depth (DepthAnythingV2) + scene heuristic.
3. Reconstruct an approximate SceneSpec from the GT video.
4. Run the simulator on that SceneSpec → "ideal" trajectories.
5. Compare GT trajectories vs ideal → record physics-realism score.
6. For high-score shots: condition the model on the simulator trajectory
   during training (the model learns to render given physics).
7. For low-score shots: down-weight (they teach bad physics).

### 5.2 Inference-time Sim

At inference the user supplies (or we infer from prompt) a SceneSpec, run
the simulator, condition the model. The model never has to "discover"
physics — it's given.

## 6. Backend Selection Policy

| Backend  | When to use                          | Cost    | License        |
| -------- | ------------------------------------ | ------- | -------------- |
| PyBullet | default, rigid-body shots            | low CPU | zlib           |
| Genesis  | fluid/cloth/soft body, hero shots    | GPU     | Apache-2.0     |
| MuJoCo   | articulated characters, dancing      | low CPU | Apache-2.0     |
| Brax     | gradient research, ablations         | GPU     | Apache-2.0     |

Selection is automatic based on `SceneSpec.objects[].shape`: any
`soft` / `fluid` object → Genesis. Any humanoid → MuJoCo. Otherwise →
PyBullet.

## 7. Retrofit Invariant

`PhysicsKeyframeEmbed.out_proj` zero-initialized. When no SceneSpec is
provided, conditioning contribution is zero, model behaves exactly as
previous checkpoint. Hard loss weight starts at 0 and ramps in Phase 5.

## 8. Success Metrics

- **Physics blind test:** cinematographer-blind survey. Identify-as-AI
  rate on physics-heavy shots (objects falling, splashing, swinging)
  target <25% (baseline ~80%).
- **Trajectory MSE:** predicted trajectory MSE against held-out simulator
  ground truth in normalized image coords ≤0.02.
- **Constraint compliance:** detected contact events in generated video
  match simulator events ±2 frames in ≥80% of shots.

## 9. Out of Scope

- True end-to-end differentiable physics (would require Brax/Warp-only
  pipeline — research direction, not v1).
- Active camera control via physics ("camera mounted on falling object") —
  doable in v2 by treating camera as an object in the scene.
- Audio synthesis from collision events — separate RFC.

## 10. Why Not Just Run Unreal/Unity?

People ask. The answer:

1. **Licensing.** UE5 Epic Games License is restrictive on derivative
   distributions; Unity Pro per-seat is expensive. PyBullet (zlib) and
   Genesis (Apache-2.0) are unencumbered.
2. **No render dependency.** We do not need the visual engine — only the
   physics solver. PyBullet IS Bullet, which IS what most game engines
   use under the hood for rigid bodies. Genesis ships its own GPU solvers
   competitive with the modern UE Chaos engine.
3. **Headless training.** Our cluster has no display server. PyBullet runs
   headless trivially.
4. **Programmatic SceneSpec.** Game engines want scenes via their editor;
   we want them as JSON for batch processing.

The fidelity that matters for ad creative — gravity, momentum, contact,
restitution — is **identical** between PyBullet/Genesis and the underlying
solvers in UE/Unity. The visual difference comes from the renderer, and
the renderer is Ogenti.

---

*This RFC lands the next code bundle. See ogenti/physics/ for module layout.*
