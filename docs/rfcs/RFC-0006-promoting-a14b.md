# RFC-0006 — Promoting Wan2.2-A14B over TI2V-5B as Primary Backbone

**Status:** Accepted
**Date:** 2026-05-22
**Supersedes (in part):** RFC-0002 recommendation
**Depends on:** RFC-0001, RFC-0002

---

## Context

RFC-0002 selected **Wan2.2-TI2V-5B** as the primary retrofit target on a
decision matrix that weighted "retrofit tractability" and "compute fit,
single-node" heavily (×3 each) and "out-of-box quality" lightly (×2). The
core argument was: Apache-2.0 + 5B fits on 8×H100 + single-expert avoids MoE
surgery complexity → easiest path to a first retrofit pilot.

That pilot has now been run.

## Empirical findings (operator-reported, 2026-05-22)

Independent qualitative test on the raw Wan2.2-TI2V-5B model (Hugging Face
inference, before any retrofit) on ad-creative prompts:

| Test dimension                | Wan2.2-TI2V-5B (raw) observation                                                                      |
| ----------------------------- | ----------------------------------------------------------------------------------------------------- |
| Max coherent clip length      | ≤ 3 seconds before subject identity drifts or melts                                                   |
| Shape / form fidelity         | Subjects become unrecognizable — "shapes all collapse"                                                |
| Prompt adherence (Korean)     | Largely ignored — the model produces generic motion rather than executing the requested scene         |
| Prompt adherence (English)    | Partial; complex multi-clause prompts (e.g. "X on Y with lighting Z, camera move W") not honored      |
| Verdict for ad creative       | **Unusable as a base.** No amount of retrofit-stage LoRA / fine-tuning recovers a base model this weak |

These findings **invalidate** the "out-of-box quality is good enough" premise
that RFC-0002 used to justify the 5B pick. Retrofit cannot resurrect a base
model that doesn't understand the prompt — our entity / glyph / anatomy heads
add structural commitments on top of a coherent base, but if the base output
isn't a recognizable subject in the first place, identity attention has
nothing to anchor and the glyph branch has no glyphs to fix.

## Constraint that changed

RFC-0002's compute weight (×3 for single-node fit) was driven by the
assumption that we'd be running on consumer / mid-tier hardware. The
operator has now confirmed that **A100 80GB-class hardware is available**
("5090 말고 더 좋은거 a100 이나 그런거 빌려올테니까"), making the 14B-vs-5B
compute delta non-binding.

## Re-scored decision matrix

Re-running RFC-0002's matrix with two changes:
1. **Out-of-box quality** weight bumped ×2 → ×4 (it's now the deal-killer
   dimension, displacing license which already shook out for both Wan2.2 variants).
2. **Compute fit, single-node** weight dropped ×2 → ×1 (A100 80GB available;
   we're no longer single-GPU-bound).

| Criterion (new weight)        | HunyuanVideo | Wan2.2-5B | Wan2.2-A14B | CogVideoX-5B |
| ----------------------------- | :----------: | :-------: | :---------: | :----------: |
| License for ads (×3)          |      1       |     3     |      3      |      2       |
| Retrofit tractability (×3)    |      2       |     3     |      1      |      3       |
| Out-of-box quality (×4)       |      3       |     2     |      3      |      1       |
| Compute fit, single-node (×1) |      1       |     3     |      1      |      3       |
| Community tooling (×1)        |      3       |     2     |      2      |      3       |
| **Weighted total**            |    **23**    |  **27**   |   **27**    |    **23**    |

A14B and 5B now **tie at 27**. The empirical 5B failure breaks the tie:
under-spec on the highest-weighted dimension (out-of-box quality) is fatal,
since retrofit cannot recover a base that doesn't understand prompts.

## Decision

**Promote Wan2.2-T2V-A14B to primary retrofit target.** Wan2.2-I2V-A14B is
the I2V sibling and shares the same architecture / surgery path. Wan2.2-TI2V-5B
is demoted to **legacy / fallback** status for 32GB-class hardware where A14B
won't fit at all.

Specifically:

1. **Primary**: `Wan-AI/Wan2.2-T2V-A14B` — text-to-video, MoE dual-expert.
2. **Variant**: `Wan-AI/Wan2.2-I2V-A14B` — image-to-video, MoE dual-expert.
   Same surgery, same configs, just different model id at backbone load time.
3. **Legacy**: `Wan-AI/Wan2.2-TI2V-5B` — kept reachable via
   `--model-config ogenti/configs/model/ogenti_5b.yaml` but no longer the
   default in CLI / scripts / configs.

## What changes in the codebase

| Surface                                            | Before                          | After                                                |
| -------------------------------------------------- | ------------------------------- | ---------------------------------------------------- |
| `Wan22BackboneConfig.model_id` default             | `Wan-AI/Wan2.2-TI2V-5B`         | `Wan-AI/Wan2.2-T2V-A14B`                             |
| Default `--model-config` (CLI + scripts)           | `ogenti_5b.yaml`                | `ogenti_a14b.yaml`                                   |
| Default retrofit output dir                        | `checkpoints/ogenti_5b_retrofit_init` | `checkpoints/ogenti_a14b_retrofit_init`         |
| Primary smoke runbook                              | `docs/RUNPOD-5090-GUIDE.md`     | `docs/A100-GUIDE.md`                                 |
| Importer keymap variants                           | official / diffusers / ti2v     | + new **a14b** variant + MoE expert-subdir resolver  |
| Inference                                          | single OgentiTransformer        | + optional `A14BMoEWrapper` for dual-expert routing  |

The retrofit invariant tests (`tests/integration/test_retrofit_invariant.py`)
use a tiny synthetic config and are unaffected — the architecture is generic.

## MoE handling

Wan2.2-A14B ships two transformer weight sets (`high_noise_model/` and
`low_noise_model/`) routed by timestep at inference. The importer now:

1. Detects the A14B MoE directory layout and routes into the requested
   expert subdir before loading shards (see `_resolve_expert_dir`).
2. Adds a new `WAN22_KEYMAP_A14B_BLOCK` / `WAN22_KEYMAP_A14B_TOP` keymap
   variant covering A14B's intra-block + top-level key naming.
3. Accepts a new `expert: "low_noise" | "high_noise"` argument throughout
   the API (CLI, standalone script, `dit_to_ogenti` adapter).

Default expert is `low_noise` — it carries the final-step quality bar that
matters most for ad creative.

For full MoE inference, retrofit both experts independently into separate
checkpoints, then wrap them with `ogenti.inference.moe_wrapper.A14BMoEWrapper`,
which presents the standard `OgentiTransformer.forward` signature so existing
samplers don't need to know about MoE.

## Risks and how they're handled

1. **Activation memory** — A14B per-step activations are ~3× the 5B path. We
   mitigate with `gradient_checkpointing: true` in `smoke_a100.yaml` and
   `retrofit_stage1_a14b.yaml`. On 80GB cards this fits at 480×832×81 frames.
2. **Both experts in VRAM** — full MoE inference holds 2 × 14B params
   simultaneously. The `A14BMoEWrapper` has `offload_inactive_expert=True`
   which pins the inactive expert to host RAM and swaps on the boundary
   crossing, halving peak GPU usage.
3. **5B regression in CI** — existing tests use a tiny synthetic
   OgentiTransformer config; they don't care about the upstream model size.
   The 5B retrofit code path is still importable and the keymap variants
   for it (`ti2v`) remain in the importer.

## What is **not** in scope of this RFC

- **Proper MoE training** — the `A14BMoEWrapper` does inference-time routing
  only. Training both experts jointly (so the Ogenti add-on modules learn
  to specialize across noise regions) is deferred to a follow-up. For the
  stage 1 retrofit curriculum, we train the Ogenti add-ons on top of the
  frozen `low_noise` expert only; this is sufficient for the identity /
  glyph / anatomy losses, all of which care most about fine-detail
  (low-noise) timesteps.
- **Switch back to TI2V-5B in extreme low-memory regimes** — supported via
  `--model-config ogenti/configs/model/ogenti_5b.yaml` but not part of the
  default flow. The 5090 runbook stays in tree for that case.

## Next steps

1. Update default configs and CLI to point at A14B. ✅ (this PR)
2. Add `ogenti_a14b.yaml` + `smoke_a100.yaml` + `retrofit_stage1_a14b.yaml`. ✅
3. Add A14B keymap variant + MoE expert resolver to `wan22_import.py`. ✅
4. Add `A14BMoEWrapper` for dual-expert inference. ✅
5. Write `docs/A100-GUIDE.md` for the new primary smoke path. ✅
6. Validate keymap against a real downloaded A14B snapshot (operator task).
7. Run the A14B smoke and confirm "before/after" coherence at 5+ seconds.

---

*Approved by operator (박은실 / @gkjuwon-tech) on 2026-05-22 based on the
empirical Wan2.2-TI2V-5B HF test failure described above.*
