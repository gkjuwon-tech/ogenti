# RFC-0002 — Base Backbone Selection for Retrofit

**Status:** Draft → seeking decision
**Date:** 2026-05-22
**Depends on:** RFC-0001

---

## Context

We are not training from scratch. We need a pretrained open-source video DiT whose blocks
we will surgically replace with `OgentiBlock`. The choice is load-bearing — it dictates:
- license terms (commercial use for ad clients)
- weight import schema (the surgery layer)
- VAE / text encoder reuse
- max resolution and shot length out-of-the-box
- community tooling depth

## Candidates

### A. HunyuanVideo (Tencent)
- **Params:** 13B DiT, 3D causal VAE
- **License:** Tencent Hunyuan Community License — **restricts commercial use in
  EU/UK/KR; >100M MAU prohibited.** For an ad-creative SaaS this is a yellow flag
  but not blocking for non-EU clients.
- **Architecture:** dual-stream then single-stream DiT (à la Flux). Cleanly separable
  blocks → surgery-friendly.
- **Strengths:** highest open-source quality as of late 2025; T5 + CLIP dual text
  encoder; strong motion.
- **Weaknesses:** large (>26GB fp16); slow inference; license geofencing.

### B. Wan2.2 (Alibaba)
- **Params:** 14B (T2V-A14B) or 5B (TI2V-5B) variants
- **License:** Apache-2.0 — **fully commercial, no geo restrictions.** Green flag.
- **Architecture:** MoE-style dual-expert DiT in A14B; single-expert in 5B. The MoE
  routing complicates surgery (we'd need to retrofit both experts symmetrically).
- **Strengths:** best license; 5B variant is retrofit-tractable on 8×H100; strong
  prompt adherence; native 720p.
- **Weaknesses:** MoE adds complexity if we target A14B; community tooling thinner
  than Hunyuan.

### C. CogVideoX-5B (Zhipu / THUDM)
- **Params:** 5B DiT, 3D VAE
- **License:** CogVideoX License — **commercial use allowed with attribution.**
  Green-ish flag (lighter than Hunyuan, heavier than Apache).
- **Architecture:** classic single-stream DiT with expert adaptive LayerNorm. Cleanest
  reference DiT — easiest to retrofit.
- **Strengths:** smallest viable; well-documented; mature diffusers integration;
  surgery is straightforward.
- **Weaknesses:** quality below Hunyuan/Wan2.2 on motion and detail; 480p native
  (upscale needed); text rendering notably weak (which we want to fix anyway).

### D. Mochi-1 (Genmo) — *considered, rejected*
Apache-2.0 but quality below the above three on ad-relevant criteria.

### E. LTX-Video (Lightricks) — *considered, rejected*
Fast but lower fidelity ceiling.

## Decision Matrix

| Criterion (weight)         | HunyuanVideo | Wan2.2-5B | Wan2.2-A14B | CogVideoX-5B |
| -------------------------- | :----------: | :-------: | :---------: | :----------: |
| License for ads (×3)       |      1       |     3     |      3      |      2       |
| Retrofit tractability (×3) |      2       |     3     |      1      |      3       |
| Out-of-box quality (×2)    |      3       |     2     |      3      |      1       |
| Compute fit, single-node (×2) |   1       |     3     |      1      |      3       |
| Community tooling (×1)     |      3       |     2     |      2      |      3       |
| **Weighted total**         |    **17**    |  **26**   |   **16**    |    **22**    |

## Recommendation

**Wan2.2-TI2V-5B** as the primary retrofit target.

Rationale: Apache-2.0 resolves all commercial concerns for ad clients (the
deal-killer dimension), the 5B parameter count fits a single 8×H100 training node
with room for our auxiliary heads, and the single-expert variant avoids MoE
surgery complexity. We sacrifice some peak quality vs HunyuanVideo, but the
retrofit gains (identity persistence, glyph fidelity, anatomy) should more than
recover the gap on our target use case.

**Fallback:** if Wan2.2-5B's pretrained quality proves insufficient after a 4-week
retrofit pilot, we promote CogVideoX-5B to primary — it has the cleanest DiT
structure for our surgery layer and similar compute footprint.

**Explicitly not chosen:** HunyuanVideo. License geofencing makes it a non-starter
for a global ad-tech product, regardless of quality.

## Next Steps

1. Download Wan2.2-TI2V-5B weights to `third_party/wan2.2/`.
2. Build weight-import map (`ogenti/retrofit/surgery/wan22_import.py`).
3. Verify reproducibility: load Wan2.2 in our wrapper, generate a sample, compare
   against reference output bit-equivalently before any surgery.
4. **Then** begin block replacement.

---

*Decision requested from team lead before proceeding to scripts/retrofit/.*
