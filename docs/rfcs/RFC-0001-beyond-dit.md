# RFC-0001 — Beyond DiT: The Ogenti Architecture Manifesto

**Status:** Draft
**Date:** 2026-05-22
**Target:** Photoreal, ad-grade video generation (commercial-spot fidelity)

---

## 1. Problem Statement

Current SOTA open-source video generators (HunyuanVideo, Wan2.2, CogVideoX, Mochi, LTX) all
share a Diffusion Transformer (DiT) backbone. In production ad-creative workloads we observe
three recurring failure modes that DiT does **not** structurally address:

1. **Object identity collapse.** A building morphs into a car mid-shot. A person's jacket
   changes color across frames. The model has no explicit notion of *which token cluster
   denotes which entity over time* — identity is an emergent property of full self-attention,
   and emergence is not a contract.
2. **Anatomical violation.** Fingers bending backwards, six-fingered hands, knees inverting.
   DiT's isotropic attention treats hand pixels and sky pixels with the same prior. Anatomy
   is a *strong* prior the model is forced to re-learn implicitly per sample.
3. **Glyph degradation.** Text on a product label becomes "COCC-CLOA". Patch tokenization
   destroys the high-frequency stroke structure that OCR-grade rendering requires. DiT has
   no dedicated pathway for typographic fidelity.

For ad clients these are not "artifacts" — they are deal-killers. A Pepsi logo that reads
"PEPSL" is not a deliverable.

## 2. Why Not Just Fine-Tune?

Fine-tuning adjusts *weights* within a fixed *forward graph*. The failures above are
properties of the graph, not the weights:

- No tensor in DiT represents "this is entity #3 across frames 0..N."
- No subgraph is responsible for rendering glyphs at glyph-native resolution.
- No structural bias enforces kinematic plausibility.

You cannot fine-tune a missing inductive bias into existence. We must perform **structural
retrofit**: keep most of the pretrained representational capacity (the heavy lift no startup
can afford to redo) but surgically replace key blocks and add new pathways.

## 3. The Ogenti Hypothesis

Ogenti is a hybrid architecture built on three structural commitments:

### 3.1 Identity-Anchored Hierarchical Attention (IAHA)
Replace each DiT block with an `OgentiBlock` that operates in two passes:
- **Pass A — Entity attention.** A small set of learned *entity tokens* (≈16–64 per video)
  cross-attend to spatio-temporal patches. These tokens are *persistent across timesteps*
  and serve as identity anchors. Inspired by Slot Attention + Perceiver IO, but conditioned
  on the diffusion timestep.
- **Pass B — Patch attention.** Standard spatio-temporal attention over patches, but each
  patch additionally attends to entity tokens via cross-attention. Entity tokens act as a
  low-rank "memory" that prevents identity drift.

### 3.2 Dedicated Glyph Branch (DGB)
A parallel high-resolution branch operating on a separate token stream derived from
text-region crops (detected via an auxiliary lightweight OCR-region predictor). The glyph
branch operates at 2× spatial resolution of the main backbone and is fused back via gated
cross-attention only in late blocks. Loss includes OCR-supervised auxiliary head.

### 3.3 Anatomical Consistency Prior (ACP)
Not a new branch — a *loss* and *conditioning signal*. Off-the-shelf pose/hand keypoint
estimators (e.g., MediaPipe, DWPose) extract structure from training videos. We condition
on these signals via ControlNet-style adapters and add a keypoint-reconstruction loss on
generated frames. This is the cheapest of the three commitments and the most leveraged.

## 4. What We Reuse vs. Replace

| Component                  | Strategy                                                |
| -------------------------- | ------------------------------------------------------- |
| VAE (3D causal)            | **Reuse** from base model (frozen)                      |
| Text encoder (T5 / CLIP)   | **Reuse** (frozen)                                      |
| DiT blocks                 | **Replace** with `OgentiBlock` (weight init from DiT)   |
| Patch embedding            | **Reuse** initially, retrain late                       |
| Positional encoding (RoPE) | **Extend** with entity-token positions                  |
| Sampler / scheduler        | **Reuse** flow-matching or EDM from base                |
| Final projection           | **Reuse**                                               |

Key insight: `OgentiBlock` is designed so that with entity tokens disabled and glyph branch
off, it reduces *exactly* to a DiT block. This means we can warm-start from pretrained
weights and progressively unlock new structure during training — a **retrofit curriculum**.

## 5. Non-Goals

- We are not training from scratch. Anyone claiming this in 2026 outside of frontier labs
  is lighting compute on fire.
- We are not chasing 60-second generations. Ad spots are 6–30s. We optimize the regime
  clients pay for.
- We are not building yet another generic T2V demo. The product is **ad-creative
  fidelity**, measured on text rendering, brand consistency, and hand/face quality.

## 6. Open Questions

- **Q1.** Wan2.2 vs HunyuanVideo vs CogVideoX as base? (See RFC-0002.)
- **Q2.** How many entity tokens scale with shot complexity? Fixed or adaptive?
- **Q3.** Glyph branch — single-stage (full video) or keyframe-only with propagation?
- **Q4.** Anatomical loss weighting — does it harm aesthetic quality? (Likely yes early,
  needs annealing schedule.)

## 7. Success Metrics

- **Text rendering:** OCR accuracy on generated product labels vs. ground-truth prompt
  text. Target: >85% character accuracy at 720p.
- **Identity persistence:** CLIP-similarity of entity crops across frames in same shot.
  Target: >0.92 mean.
- **Hand quality:** Hand-keypoint detection success rate × keypoint MSE. Target: SOTA
  among open-source.
- **Client-blind A/B:** ad-agency creatives prefer Ogenti output over base model in
  ≥70% of paired comparisons.

---

*Author: ogenti team. Comments welcome. This RFC is load-bearing — changes here cascade.*
