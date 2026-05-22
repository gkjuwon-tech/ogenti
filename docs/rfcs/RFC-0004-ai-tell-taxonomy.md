# RFC-0004 — AI-Tell Taxonomy and Eradication Strategy

**Status:** Active / Implemented
**Date:** 2026-05-22
**Depends on:** RFC-0001, RFC-0003

---

## 1. Motivation

A trained reviewer (cinematographer, ad creative director) can flag
AI-generated video within 2 seconds. The dismissal does not depend on a
single failure mode — it is a *bundle* of "tells" that together produce
the uncanny "this is AI" signal.

Solving the bundle requires enumerating it. This RFC catalogs every
AI-tell observed in current SOTA video generation, groups them by
mechanism, and assigns each one a priority + intervention strategy.

## 2. The Taxonomy

### 2.1 Texture / Material (T-class)

| ID  | Tell                              | Root cause                                  | Severity |
| --- | --------------------------------- | ------------------------------------------- | -------- |
| T-1 | Skin looks like tanghulu          | VAE smoothing + retouched training corpus   | CRITICAL |
| T-2 | All objects look like wet plastic | Universal HF loss in VAE round-trip         | CRITICAL |
| T-3 | Material differentiation absent   | No material-aware conditioning              | HIGH     |
| T-4 | Surface imperfections absent      | Training data bias toward "clean" product   | HIGH     |
| T-5 | Fabric/weave detail erased        | Patch tokenizer destroys sub-patch detail   | MEDIUM   |

### 2.2 Motion (M-class)

| ID  | Tell                                  | Root cause                                  | Severity |
| --- | ------------------------------------- | ------------------------------------------- | -------- |
| M-1 | Camera motion: conveyor belt smooth   | Flow-matching averages temporal noise       | CRITICAL |
| M-2 | Subject motion: ballerina-smooth      | No motion field conditioning                | CRITICAL |
| M-3 | Object motion: weightless float       | No physics prior                            | HIGH     |
| M-4 | No acceleration envelope (windup/impact/recovery) | Loss penalizes acceleration spikes    | HIGH     |
| M-5 | Gravity ignored (vertical drift)      | No gravity-consistency loss                 | HIGH     |
| M-6 | No inertia on direction changes       | Flow matching prefers smooth turns          | MEDIUM   |
| M-7 | Hair / cloth physics as rigid mass    | Pose+cloth not differentiated               | MEDIUM   |

### 2.3 Optics / Lighting (O-class)

| ID  | Tell                                   | Root cause                                  | Severity |
| --- | -------------------------------------- | ------------------------------------------- | -------- |
| O-1 | No motion blur on fast objects         | Frame-by-frame sharp prediction             | CRITICAL |
| O-2 | No film grain                          | VAE flat output                             | HIGH     |
| O-3 | Lighting too flat (no caustics/bounces)| No global illumination conditioning         | HIGH     |
| O-4 | Lens artifacts absent (flare/CA/bokeh) | No lens model conditioning                  | HIGH     |
| O-5 | Saturation / color: instagram-flat     | Training corpus skewed to graded content    | MEDIUM   |
| O-6 | DOF / focus pull missing               | No focus depth conditioning                 | MEDIUM   |

### 2.4 Micro-temporal (μ-class)

| ID  | Tell                                  | Root cause                                  | Severity |
| --- | ------------------------------------- | ------------------------------------------- | -------- |
| μ-1 | Blink rhythm uniform / absent         | No blink scheduler                          | HIGH     |
| μ-2 | Breathing motion uniform              | No respiration conditioning                 | HIGH     |
| μ-3 | Micro-expressions absent              | Short-window expression loss missing        | MEDIUM   |
| μ-4 | Dead eyes — no microsaccades          | Pupil scale below tokenization grain        | MEDIUM   |
| μ-5 | Environmental incidentals absent (dust, bugs, leaves) | Loss prefers parsimony       | LOW      |

### 2.5 Compositional (C-class)

| ID  | Tell                                  | Root cause                                  | Severity |
| --- | ------------------------------------- | ------------------------------------------- | -------- |
| C-1 | Faces too symmetric                   | Symmetry is high-likelihood prior           | MEDIUM   |
| C-2 | Backgrounds suspiciously clean        | Curation removes "messy" footage            | MEDIUM   |
| C-3 | Center-weighted composition only      | Image-net composition bias                  | LOW      |

## 3. Intervention Strategies

Each tell maps to one of five intervention types:

- **A — Architecture branch** (new module added to OgentiTransformer)
- **D — Data curation** (filter / augment / tag training data)
- **L — Loss term** (add a new component to the training objective)
- **C — Conditioning** (expose as user-controllable input variable)
- **P — Post-process** (pixel-space head after VAE decode)

Mapping:

| Tell      | A | D | L | C | P | Phase |
| --------- | - | - | - | - | - | ----- |
| T-1 skin  | ✓ | ✓ | ✓ |   | ✓ | 4-RE  |
| T-2 universal material | ✓ | ✓ | ✓ |   | ✓ | 4-RE2 |
| T-3 material differentiation |   | ✓ |   | ✓ |   | 4-RE2 |
| T-4 surface imperfection |   | ✓ |   |   |   | 4-RE2 |
| M-1 camera motion |   | ✓ | ✓ | ✓ |   | 4-RE  |
| M-2 subject motion | ✓ | ✓ | ✓ | ✓ |   | 4-RE2 |
| M-3 object physics |   | ✓ | ✓ |   |   | 4-RE2 |
| M-4 acceleration envelope |   |   | ✓ |   |   | 4-RE2 |
| M-5 gravity |   |   | ✓ |   |   | 4-RE2 |
| O-1 motion blur |   |   |   | ✓ | ✓ | 4-RE2 |
| O-2 film grain |   |   |   | ✓ | ✓ | 4-RE2 |
| O-4 lens artifacts |   |   |   | ✓ | ✓ | 4-RE2 |
| μ-1 blink | ✓ |   | ✓ | ✓ |   | 4-RE2 |
| μ-2 breath |   |   | ✓ | ✓ |   | 4-RE2 |

## 4. Implementation Bundle: Phase 4-RE2

This RFC scopes the second realism bundle. Modules to land:

### 4.1 Universal Material Detail Head (UMDH)
Generalize SDRH from skin to all materials. Multi-channel residual head
keyed by a per-pixel material logit (skin / metal / fabric / wood /
foliage / plastic / glass / other). Different HF kernels per material.

### 4.2 Subject Motion Realism
Per-track motion descriptor (separate from camera motion). Acceleration
envelope loss matching ground-truth motion profile per subject.

### 4.3 Physics Priors
- Gravity loss: vertical acceleration of detected objects should
  approximate `g = 9.8 m/s²` scaled to pixel space (calibrated via
  scene depth heuristic).
- Inertia loss: penalize abrupt direction reversals without intermediate
  deceleration.

### 4.4 Post-process Heads
- **Motion blur head**: per-pixel velocity-conditioned blur kernel.
- **Film grain head**: parameterized noise overlay (ISO + grain size).
- **Lens artifact head**: chromatic aberration, configurable bokeh
  shape, optional flare.

### 4.5 Micro-event Scheduler
- Blink trigger: a discrete event signal injected as conditioning,
  scheduled per-shot with realistic inter-blink interval distribution
  (lognormal, μ=4.2s, σ=1.5s).
- Breath rhythm: continuous sinusoidal+noise conditioning at
  realistic frequencies (0.2-0.4 Hz at rest).

## 5. Retrofit Invariant Compliance

Every module above must satisfy:
- Zero output at initialization (head out-projections zero-init).
- Behavior at training step 0 indistinguishable from the previous
  checkpoint.

This is non-negotiable. We accumulate, never break.

## 6. Success Metrics

- **AI-spotting blind test:** ad professionals shown 30-shot reels mixing
  Ogenti and reference footage. Identify-as-AI rate target: <35%.
  Baseline (Phase 4-RE only): ~70%.
- **Per-tell measurement:**
  - T-2 metric: HF energy retention across all material classes ≥75% of GT.
  - M-2 metric: KL divergence of subject motion power spectrum ≤0.2.
  - M-5 metric: vertical acceleration of free-falling objects within
    ±15% of `g`.
  - O-1 metric: motion blur extent linearly correlates with object
    velocity (Pearson r ≥0.6).

## 7. Out of Scope (for this RFC)

- C-class composition tells — handled by Phase 5 prompt engineering.
- Audio (lip-sync, footsteps) — separate audio model integration RFC.
- Long-horizon temporal coherence (>30s shots) — separate RFC.

---

*This RFC unlocks the next code bundle. Each subsection corresponds to
a file landed alongside this commit.*

## 8. Implementation Checklist

All Phase 4-RE2 modules have been integrated into the Ogenti codebase:
- [x] **UMDH** (`ogenti/modules/texture/material_detail_head.py`) integrated into Transformer.
- [x] **Post-process Heads** (`motion_blur`, `film_grain`, `lens_artifacts`) wired to Pipeline.
- [x] **Subject Motion Embed** + **Micro-Event Embed** wired to Transformer forward pass.
- [x] **Physics Priors & Motion Realism** losses integrated into `train.py`.
- [x] **Video Dataset** expanded to load subject motion, trajectories, and material masks.
- [x] **Retrofit Invariant** tested and verified.
