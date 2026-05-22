# Ogenti

Ad-grade video generation via **structural retrofit** of pretrained DiT backbones.

We don't train from scratch. We don't fine-tune. We take an open-source video DiT
(Wan2.2-TI2V-5B), surgically replace its blocks with `OgentiBlock`, and add three
new structural commitments that DiT cannot express:

1. **Identity-Anchored Hierarchical Attention** — persistent entity tokens stop
   buildings from morphing into cars mid-shot.
2. **Dedicated Glyph Branch** — a high-resolution sub-transformer for text
   regions, so "COCA-COLA" stops rendering as "COCC-CLOA".
3. **Anatomical Consistency Prior** — keypoint-supervised loss + conditioning
   so fingers stop bending backwards.

See [RFC-0001](docs/rfcs/RFC-0001-beyond-dit.md) for the manifesto and
[RFC-0002](docs/rfcs/RFC-0002-base-backbone-selection.md) for the backbone
selection rationale.

## Status

Pre-alpha. The retrofit invariant is enforced: at step 0, an OgentiTransformer
loaded with Wan2.2 weights produces output bit-equivalent to vanilla Wan2.2.
All Ogenti-specific submodules start dormant (zero-init gates) and unlock
progressively during the retrofit training curriculum.

## ⚡ Smoke test on a single RTX 5090 (RunPod, ~$2, ~4 hours)

Want a "before/after" video TODAY on one consumer GPU?
👉 Follow [`docs/RUNPOD-5090-GUIDE.md`](docs/RUNPOD-5090-GUIDE.md).

End-to-end pipeline:
```bash
# 1. Build a 1,500-clip royalty-free ad-grade dataset
python -m scripts.data.build_dataset --root data/ --pexels-max 1000 --pixabay-max 400

# 2. Import Wan2.2 weights into Ogenti
python -m scripts.retrofit.import_wan22 /workspace/wan2.2-ti2v-5b/

# 3. Smoke-train on a single 5090
python -m ogenti.cli train ogenti/configs/training/smoke_5090.yaml \
    --model-config ogenti/configs/model/ogenti_5b_5090.yaml \
    --init-from checkpoints/ogenti_5b_retrofit_init

# 4. Compare before vs after
python -m ogenti.cli generate "cola bottle falls on marble" \
    --ckpt outputs/runs/smoke_5090/final \
    --camera-motion handheld --physics-backend pybullet
```

## Quick start

```bash
pip install -e .

# 1. Import Wan2.2 weights into an Ogenti checkpoint
python -m ogenti.cli retrofit /path/to/wan2.2-ti2v-5b/ \
    --out checkpoints/ogenti_5b_retrofit_init

# 2a. Validate keymap against actual downloaded Wan2.2 weights BEFORE training
python -m scripts.retrofit.validate_keymap /path/to/wan2.2-ti2v-5b/

# 2b. Train stage 1 (identity unlock) — single GPU
python -m ogenti.cli train ogenti/configs/training/retrofit_stage1.yaml \
    --init-from checkpoints/ogenti_5b_retrofit_init

# 2c. Or multi-GPU via accelerate + deepspeed zero-2
accelerate launch --config_file ogenti/configs/training/accelerate_zero2.yaml \
    -m ogenti.cli train ogenti/configs/training/retrofit_stage1.yaml \
    --init-from checkpoints/ogenti_5b_retrofit_init

# 3. Generate
python -m ogenti.cli generate "A pristine Coca-Cola bottle on a marble countertop, soft golden light, 4k photoreal" \
    --ckpt outputs/runs/retrofit_stage1/final \
    --out outputs/samples/coke.mp4
```

## Layout

```
ogenti/
  models/           # OgentiTransformer + OgentiBlock + backbone wrappers
  modules/          # attention, identity slots, glyph branch, tokenizers, conditioning
  retrofit/         # Wan2.2 -> Ogenti weight surgery
  training/         # losses (diffusion, identity, anatomy, OCR) + schedulers
  data/             # video dataset + offline preprocessors
  inference/        # samplers + end-to-end pipeline
  configs/          # model + training YAMLs
scripts/            # train / retrofit / eval entry points
tests/              # unit + integration (retrofit invariant verification)
docs/rfcs/          # architectural decision records
```

## License

Apache-2.0. Wan2.2 base weights are Apache-2.0 (Alibaba). See `third_party/` for
upstream attributions.
