# Ogenti

Ad-grade video generation via **structural retrofit** of pretrained DiT backbones.

We don't train from scratch. We don't fine-tune. We take an open-source video DiT
(**Wan2.2-T2V-A14B**, the 14B MoE big-sibling — see [RFC-0006](docs/rfcs/RFC-0006-promoting-a14b.md)
for why 5B was demoted), surgically replace its blocks with `OgentiBlock`, and
add three new structural commitments that DiT cannot express:

1. **Identity-Anchored Hierarchical Attention** — persistent entity tokens stop
   buildings from morphing into cars mid-shot.
2. **Dedicated Glyph Branch** — a high-resolution sub-transformer for text
   regions, so "COCA-COLA" stops rendering as "COCC-CLOA".
3. **Anatomical Consistency Prior** — keypoint-supervised loss + conditioning
   so fingers stop bending backwards.

See [RFC-0001](docs/rfcs/RFC-0001-beyond-dit.md) for the manifesto,
[RFC-0002](docs/rfcs/RFC-0002-base-backbone-selection.md) for the original
backbone selection, and [RFC-0006](docs/rfcs/RFC-0006-promoting-a14b.md) for the
empirical post-mortem that promoted A14B over 5B as the primary target.

## Status

Pre-alpha. The retrofit invariant is enforced: at step 0, an OgentiTransformer
loaded with Wan2.2 weights produces output bit-equivalent to vanilla Wan2.2.
All Ogenti-specific submodules start dormant (zero-init gates) and unlock
progressively during the retrofit training curriculum.

## ⚡ Smoke test on a single A100 80GB (~$5, ~4 hours)

Want a "before/after" video TODAY on one cloud GPU?
👉 Follow [`docs/A100-GUIDE.md`](docs/A100-GUIDE.md).

End-to-end pipeline (A14B / A100):
```bash
# 1. Build a 1,500-clip royalty-free ad-grade dataset
python -m scripts.data.build_dataset --root data/ --pexels-max 1000 --pixabay-max 400

# 2. Import Wan2.2-T2V-A14B weights into Ogenti (low_noise expert by default)
python -m scripts.retrofit.import_wan22 /workspace/wan2.2-t2v-a14b/

# 3. Smoke-train on a single A100 (frozen-backbone, Ogenti add-ons only)
python -m ogenti.cli train ogenti/configs/training/smoke_a100.yaml \
    --model-config ogenti/configs/model/ogenti_a14b.yaml \
    --init-from checkpoints/ogenti_a14b_retrofit_init

# 4. Compare before vs after
python -m ogenti.cli generate "cola bottle falls on marble" \
    --ckpt outputs/runs/smoke_a100/final \
    --camera-motion handheld --physics-backend pybullet
```

> **Legacy 5B path** (kept for nostalgia / 32GB cards): see
> [`docs/RUNPOD-5090-GUIDE.md`](docs/RUNPOD-5090-GUIDE.md). Do not use this
> path for ad-quality output — empirically Wan2.2-TI2V-5B caps at ~3 s clips
> with mangled form and broken prompt adherence (see RFC-0006 for evidence).

## Quick start

```bash
pip install -e .

# 1. Import Wan2.2-T2V-A14B weights into an Ogenti checkpoint
#    (point at the snapshot root containing high_noise_model/ + low_noise_model/)
python -m ogenti.cli retrofit /path/to/wan2.2-t2v-a14b/ \
    --out checkpoints/ogenti_a14b_retrofit_init

# 2a. Validate keymap against actual downloaded Wan2.2 weights BEFORE training
python -m scripts.retrofit.validate_keymap /path/to/wan2.2-t2v-a14b/

# 2b. Train stage 1 (identity unlock) — single A100 80GB
python -m ogenti.cli train ogenti/configs/training/retrofit_stage1_a14b.yaml \
    --init-from checkpoints/ogenti_a14b_retrofit_init

# 2c. Or multi-GPU via accelerate + deepspeed zero-3 (A14B needs zero-3 for
#     activation + optimizer-state sharding)
accelerate launch --config_file ogenti/configs/training/accelerate_zero3.yaml \
    -m ogenti.cli train ogenti/configs/training/retrofit_stage1_a14b.yaml \
    --init-from checkpoints/ogenti_a14b_retrofit_init

# 3. Generate
python -m ogenti.cli generate "A pristine Coca-Cola bottle on a marble countertop, soft golden light, 4k photoreal" \
    --ckpt outputs/runs/retrofit_stage1_a14b/final \
    --out outputs/samples/coke.mp4

# 3b. (Optional) Full dual-expert MoE inference. Retrofit both experts:
python -m ogenti.cli retrofit /path/to/wan2.2-t2v-a14b/ --expert high_noise \
    --out checkpoints/ogenti_a14b_high
python -m ogenti.cli retrofit /path/to/wan2.2-t2v-a14b/ --expert low_noise  \
    --out checkpoints/ogenti_a14b_low
#  Then load both via ``ogenti.inference.moe_wrapper.load_a14b_moe_from_retrofit_dirs``.
```

## Layout

```
ogenti/
  models/           # OgentiTransformer + OgentiBlock + backbone wrappers
  modules/          # attention, identity slots, glyph branch, tokenizers, conditioning
  retrofit/         # Wan2.2 -> Ogenti weight surgery (incl. A14B MoE dual-expert)
  training/         # losses (diffusion, identity, anatomy, OCR) + schedulers
  data/             # video dataset + offline preprocessors
  inference/        # samplers + end-to-end pipeline + A14B MoE wrapper
  configs/          # model + training YAMLs (a14b is the default; 5b kept as fallback)
scripts/            # train / retrofit / eval entry points
tests/              # unit + integration (retrofit invariant + A14B variant tests)
docs/rfcs/          # architectural decision records
```

## License

Apache-2.0. Wan2.2 base weights are Apache-2.0 (Alibaba). See `third_party/` for
upstream attributions.
