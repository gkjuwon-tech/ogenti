# RFC-0003 — Killing the AI-Look: Skin Texture and Motion Realism

**Status:** Draft
**Date:** 2026-05-22
**Depends on:** RFC-0001
**Tags:** photorealism, ad-grade, training-data-strategy

---

## 1. Problem Statement

Even after solving identity persistence, glyph rendering, and anatomy (RFC-0001),
ad-creative reviewers can still spot AI-generated video within 2 seconds. Two
remaining tells dominate the dismissal rate:

### 1.1 The "Tanghulu Skin" Problem

Generated human skin looks like glazed candy fruit (탕후루). Specifically:

- Uniform sheen across the whole face — no localized matte/sebum variation
- Pores erased, micro-wrinkles smoothed, blemishes airbrushed away
- Specular highlights too clean, no subsurface scattering variance
- All faces converge toward an "Instagram beauty filter" prior

**Root causes:**
1. **VAE compression** destroys 8x8 patch-scale skin detail before the
   diffusion model ever sees it. Encode → decode round-trip already smooths
   skin even on real footage.
2. **Training data bias** — most large video corpora are over-represented
   with beauty/influencer/cosmetic-ad content, all of which is already
   skin-retouched. The model learns retouched as "normal."
3. **MSE-style flow losses** average over noise samples → produce smooth,
   high-likelihood pixel values → high-frequency detail is statistically
   suppressed.

### 1.2 The "Conveyor Belt Motion" Problem

AI-generated motion is too smooth. Specifically:

- Camera moves as if mounted on a perfect gimbal — no micro-jitter
- Subject motion lacks acceleration spikes (real humans jerk, hesitate)
- Frame-to-frame transitions are uniformly smoothed by the model
- The whole shot has the "stock B-roll" smoothness even when the prompt
  says "handheld documentary"

**Root causes:**
1. **Flow matching velocity targets are temporally low-pass-filtered** by
   the loss landscape — the model converges toward the temporal mean.
2. **Training data is curated toward gimbal/stabilized footage** because
   shaky footage is often discarded as "low quality."
3. **No conditioning signal for camera-motion type** — the model has no
   variable to encode "this should be handheld, this should be on a slider."
   It picks the population-mean smoothness.

## 2. Hypothesis

Both problems require **dual-side intervention**: data-side (what we feed
the model) AND architecture-side (what the model is structurally allowed
to express). Fixing one side alone is insufficient — the model either
loses the signal in the bottleneck (architecture-only) or has no capacity
to use it (data-only).

## 3. Solution Design — Skin Texture

### 3.1 Architecture: Skin Detail Residual Head (SDRH)

A small parallel head operating in the late blocks of OgentiTransformer
that predicts a **high-frequency residual** to be added to the decoded
output specifically inside skin regions.

```
        main DiT stream → [late block N]
                                ↓
                     [low-freq decoded video]
                                ↓
        skin_mask → [SDRH residual head]
                                ↓
                     [+ HF residual on skin]
                                ↓
                       [final pixels]
```

Key design choices:
- SDRH operates in **pixel space** (after VAE decode), not latent space —
  high-freq skin detail is destroyed by the VAE bottleneck, so we restore
  it post-decode.
- SDRH is **mask-gated** — its output is zeroed outside skin regions,
  preventing it from degrading non-skin content.
- SDRH is **zero-init at residual output** — retrofit invariant preserved
  (model behavior unchanged at step 0).

### 3.2 Architecture: Frequency-Domain Loss

Standard MSE loss has no frequency selectivity. We add a loss that
explicitly penalizes loss of high-frequency content inside skin regions:

```
freq_loss = || HF(pred ⊙ skin_mask) - HF(target ⊙ skin_mask) ||²
```

where `HF` is a high-pass FFT filter (cutoff at 1/8 Nyquist). This forces
the model to preserve pore-scale detail in skin regions.

### 3.3 Data: Skin Quality Curation

We score every training shot by **skin detail preservation** at encode-decode
time:

```
skin_score(shot) = mean over skin pixels of (HF energy after VAE round-trip
                                              / HF energy before)
```

Shots scoring below threshold (heavily retouched content) are **down-weighted**,
not removed. We need diverse skin presentations, but don't want the model to
overweight the airbrushed prior.

We also tag training data with explicit metadata:
- `skin_retouched: bool` (manual or auto-classifier)
- `lighting: ["natural", "studio", "harsh", ...]`
- `skin_type: ["matte", "oily", "mixed"]`

Conditioning the model on these gives the user explicit control at inference
time ("matte skin under harsh natural light") and prevents mode collapse.

### 3.4 Data: Don't Discard Imperfection

Most pretraining corpora filter out "low-quality" footage. We **actively
include**:
- Documentary footage with real skin
- Behind-the-scenes BTS clips
- Older film stock (35mm grain)
- Properly graded but not retouched commercial work

This is curation labor, not algorithmic.

## 4. Solution Design — Motion Realism

### 4.1 Architecture: Camera Motion Token (CMT)

A small token (or set of tokens) injected into the conditioning stream that
explicitly encodes camera-motion type. The token is derived from:

```
camera_motion_descriptor = [
    translation_jitter_xy,    # std of frame-to-frame camera translation
    rotation_jitter,          # std of frame-to-frame camera rotation
    zoom_drift,               # mean optical zoom velocity
    motion_entropy,           # spectral entropy of motion power spectrum
    handheld_vs_tripod_logit, # learned binary classifier output
]
```

At training time, this descriptor is extracted from the ground-truth video
and conditioned on. At inference, the user supplies it directly OR our
prompt parser infers it from natural language ("handheld documentary
style" → high jitter).

This converts camera motion from an **emergent, averaged-out property**
into a **controlled variable**.

### 4.2 Architecture: Temporal High-Pass Realism Loss

Standard flow-matching loss averages over noise → produces smooth motion.
We add an explicit penalty on **mismatch in temporal high-frequency
content**:

```
def temporal_realism_loss(pred_video, target_video):
    pred_hp  = temporal_highpass(pred_video)
    target_hp = temporal_highpass(target_video)
    return || power_spectrum(pred_hp) - power_spectrum(target_hp) ||²
```

This forces the model to match the **motion spectral signature** of the
ground truth, not just per-frame pixel values.

### 4.3 Data: Camera Motion Descriptor Extraction

For every training video, we extract the camera_motion_descriptor offline
using:
1. Dense optical flow (Farneback) → global motion field
2. RANSAC affine fit per frame pair → camera translation/rotation per frame
3. Spectral analysis of the per-frame motion → entropy, peak frequencies

Stored as a 8-dim float vector per shot in the manifest.

### 4.4 Data: Keep the Shake

Aggressive curation rule: **never auto-filter footage on basis of shake/jitter
alone.** Stable footage is over-represented in scraped corpora. We
deliberately oversample handheld, gimbal-with-walk, drone-with-wind, and
documentary footage to balance the distribution.

We also add **synthetic shake augmentation** as a data augmentation:
take stable footage, add controlled handheld-style motion via small affine
warps. This expands the diversity of camera-motion descriptors the model
sees.

## 5. Integration Plan

Both interventions plug into Ogenti the same way — additive, retrofit-safe:

| Component             | Stream            | Init        | Activation                |
| --------------------- | ----------------- | ----------- | ------------------------- |
| Skin Detail Head      | post-VAE pixel    | zero output | unlocked in training Phase 4  |
| Frequency loss        | training loss     | weight=0    | ramped in Phase 4         |
| Camera Motion Token   | conditioning      | zero embed  | unlocked in Phase 4       |
| Temporal realism loss | training loss     | weight=0    | ramped in Phase 4         |

All four can be turned off, in which case Ogenti behaves identically to
the RFC-0001 architecture. Retrofit invariant preserved end-to-end.

## 6. Success Metrics

- **Skin tanghulu test:** dermatologist-blind survey. <30% identify-as-AI
  rate on close-up facial shots. (Baseline: ~95%.)
- **Motion stock-feel test:** cinematographer-blind survey. <40% identify
  shots as gimbal-stabilized when prompt requested handheld.
- **Frequency MSE:** high-frequency band MSE on skin regions ≥80% of
  ground-truth GT-to-GT-roundtrip floor.
- **Motion spectrum KL:** KL divergence between predicted and target
  motion power spectra ≤0.15.

## 7. Open Questions

- **Q1.** Should SDRH be a separate small model or another OgentiBlock with
  HF-specialized conditioning? Tradeoff: separation gives clean training
  curriculum; integration shares representations.
- **Q2.** Do we need a discriminator (GAN-style) for skin to escape the
  MSE-smoothing trap, or is the frequency loss alone sufficient?
- **Q3.** Camera motion descriptor extraction is expensive (offline flow
  per shot). Can we precompute it lazily on first training touch?
- **Q4.** Should we expose camera motion control as a continuous slider or
  as discrete presets (tripod/slider/handheld/drone)? Continuous is more
  expressive; discrete is more usable.

---

*This RFC is implementation-ready. Tracking issues land alongside this commit.*
