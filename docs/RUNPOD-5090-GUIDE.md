# Ogenti — RunPod RTX 5090 Smoke-Test Runbook

> Goal: from a fresh RunPod pod to "before vs after" videos on screen in **≈4 hours of wall clock** and **≈$5 of compute**.

This runbook assumes:
- Single RTX 5090 (32 GB VRAM) on RunPod.
- Ubuntu 22.04 / 24.04 with CUDA 12.4+ available.
- Persistent volume mounted at `/workspace`.
- You already have free API keys for:
  - **Pexels** — https://www.pexels.com/api/
  - **Pixabay** — https://pixabay.com/api/docs/

---

## 0. Spin up the pod

RunPod → Deploy → Community Cloud → "RTX 5090" → "PyTorch 2.4 / CUDA 12.4" template.
Persistent volume: 200 GB (we'll download ~30 GB of clips + model weights).

When the pod boots, open the web terminal.

```bash
cd /workspace
git clone <your-fork>/ogenti.git
cd ogenti
```

(Or `scp` the repo up from your local clone — the repo path here is `C:\Users\wonma\ogenti`.)

## 1. Install dependencies

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -e .[dev,physics]      # core + pybullet
pip install ffmpeg-python
sudo apt-get install -y ffmpeg
```

Optional but recommended:
```bash
pip install -e .[clip-materials]   # CLIP material classifier
pip install -e .[llm-scene]        # Anthropic/OpenAI for scene parser
pip install flash-attn --no-build-isolation   # ~10 min compile
```

## 2. Set keys

```bash
export PEXELS_API_KEY="..."
export PIXABAY_API_KEY="..."
# optional:
export ANTHROPIC_API_KEY="..."
export WANDB_API_KEY="..."
export WANDB_DISABLED=true   # if you don't want wandb
```

## 3. Download Wan2.2-TI2V-5B backbone (≈11 GB)

```bash
huggingface-cli login   # optional, only if your token is gated
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir /workspace/wan2.2-ti2v-5b
```

## 4. Validate the keymap before spending compute

```bash
python -m scripts.retrofit.validate_keymap /workspace/wan2.2-ti2v-5b/ \
    --model-config ogenti/configs/model/ogenti_5b_5090.yaml
```

Expected outcome: `report` JSON shows `mapped_ok` > 100, `shape_mismatch == 0`.
If shape mismatch > 0, the model dims in `ogenti_5b_5090.yaml` don't match the
checkpoint — adjust `dim`, `num_blocks`, `head_dim` to match Wan2.2 spec.

## 5. Retrofit-import weights into an Ogenti checkpoint

```bash
python -m scripts.retrofit.import_wan22 /workspace/wan2.2-ti2v-5b/ \
    --model-config ogenti/configs/model/ogenti_5b_5090.yaml \
    --output-dir checkpoints/ogenti_5b_retrofit_init
```

You now have an initial Ogenti checkpoint that — by retrofit invariant — produces output
**bit-equivalent** to vanilla Wan2.2.

## 6. Generate the "BEFORE" reference video

```bash
mkdir -p outputs/samples/before
python -m ogenti.cli generate \
    "Slow motion shot of a cola bottle falling onto a marble counter" \
    --ckpt checkpoints/ogenti_5b_retrofit_init \
    --out outputs/samples/before/cola_fall.mp4 \
    --no-micro-events --physics-backend off \
    --steps 30 --seed 42
```

This is your **baseline** (= what vanilla Wan2.2 would do).

## 7. Build the dataset

```bash
python -m scripts.data.build_dataset \
    --root data/ \
    --pexels-max 1000 \
    --pixabay-max 400 \
    --skip-archive=false --archive-max 50 \
    --skip-wikimedia=true \
    --caption-backend qwen2vl \
    --caption-device cuda
```

Expected time on 5090: 30-50 minutes (most of it is downloads + Qwen2-VL captioning).

The final manifest lands at `data/manifests/ads_train.jsonl` with auto-captions,
brand-text OCR, keypoints, skin masks, glyph regions, camera motion, subject
motion, micro-events, and material masks all precomputed per shot.

## 8. (Optional) Pre-sim physics scenes

```bash
python -m scripts.data.presim_manifest data/manifests/ads_train.jsonl \
    --root data/ --backend pybullet --duration 5.0 --fps 16
```

Writes `_physics.npz` files alongside each video and produces
`ads_train.presim.jsonl` with `physics_scene_npz` paths.

## 9. Smoke test — 200 step forward+backward sanity

```bash
TOTAL_STEPS=200 python -m ogenti.cli train \
    ogenti/configs/training/smoke_5090.yaml \
    --model-config ogenti/configs/model/ogenti_5b_5090.yaml \
    --init-from checkpoints/ogenti_5b_retrofit_init
```

Watch the loss curve. It should:
- Start near a finite non-zero value (not NaN, not Inf)
- Trend down monotonically over ~50 steps
- VRAM usage should peak around 26-30 GB

If VRAM blows up: lower `target_height` and `target_width` in the smoke config.
If loss is NaN: lower LR to `5e-6` and re-run.

## 10. Full smoke training — ~3000 steps

```bash
python -m ogenti.cli train \
    ogenti/configs/training/smoke_5090.yaml \
    --model-config ogenti/configs/model/ogenti_5b_5090.yaml \
    --init-from checkpoints/ogenti_5b_retrofit_init
```

Expected wall clock: ~2.5 hours on a 5090.

Tail the run dir:
```bash
tail -f outputs/runs/smoke_5090/log
```

## 11. Generate "AFTER" video — same prompt + seed

```bash
mkdir -p outputs/samples/after
python -m ogenti.cli generate \
    "Slow motion shot of a cola bottle falling onto a marble counter" \
    --ckpt outputs/runs/smoke_5090/final \
    --out outputs/samples/after/cola_fall.mp4 \
    --camera-motion handheld \
    --physics-backend pybullet \
    --steps 30 --seed 42
```

## 12. Side-by-side comparison

```bash
ffmpeg -i outputs/samples/before/cola_fall.mp4 \
       -i outputs/samples/after/cola_fall.mp4 \
       -filter_complex hstack \
       outputs/comparison_before_after.mp4
```

Drag it down to your laptop and judge with your own eyes.

---

## Troubleshooting

**OOM during VAE decode at inference**
- Reduce `num_frames` in the smoke config or use `--steps 20`.

**`pybullet` not found / segfault**
- Some headless containers lack OpenGL. Use `--physics-backend off` and the
  soft physics loss still works for now.

**Captioner Qwen2-VL OOM**
- Switch to `--caption-backend blip2` (BLIP-2 fits in <8 GB).

**Pexels rate limit hit**
- Wait 1 hour (200/hour limit) or lower `--pexels-max`. Pixabay has a different
  bucket — it'll keep going.

**Loss still NaN at lr=5e-6**
- The `bfloat16` path on 5090 should be stable. If it isn't, switch
  `precision.dtype: float32` (training slower, but safe).

**"Where's my video?"**
- All checkpoints + samples live under `outputs/runs/smoke_5090/`. The
  final-step model is `outputs/runs/smoke_5090/final/model.safetensors`.

---

## Cost estimate (RunPod RTX 5090, Jan 2026)

| Stage              | Wall clock | $/hr | Cost      |
| ------------------ | ---------- | ---- | --------- |
| Pod boot + setup   | 0.2 h      | 0.60 | $0.12     |
| Dataset fetch      | 0.7 h      | 0.60 | $0.42     |
| Smoke train (3k st)| 2.5 h      | 0.60 | $1.50     |
| Sample generation  | 0.2 h      | 0.60 | $0.12     |
| **Total**          | **~3.6 h** |      | **~$2.2** |

Add ~$0.50 if you keep the volume around for a second run.

---

## What "success" looks like

The smoke test is a **proof-of-life**, not a proof-of-fidelity. You'll see:
- ✅ Loss curve trends down
- ✅ Generated videos still coherent (not garbage)
- ✅ Subtle but visible differences vs the baseline (entity-anchored consistency,
  more stable cameras)
- ❓ Big tells like tanghulu skin and glyph mangling: probably **still present**
  because we haven't enabled SDRH/UMDH/OCR loss in this smoke. Those land in
  later stages (stage4_realism, stage5_physics) which require longer training
  and more compute.

The goal here: prove the **plumbing works end to end**, single GPU, in an
afternoon. Once it does, scale up.
