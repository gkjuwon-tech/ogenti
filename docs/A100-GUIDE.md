# Ogenti — A100 80GB Smoke-Test Runbook (Wan2.2-T2V-A14B)

> Goal: from a fresh A100 80GB pod (RunPod / Lambda / Together / vast.ai)
> to "before vs after" videos on screen in **≈4 hours of wall clock** and
> **≈$5 of compute**, using the primary Wan2.2-**T2V-A14B** retrofit target
> from [RFC-0006](rfcs/RFC-0006-promoting-a14b.md).

This runbook supersedes [RUNPOD-5090-GUIDE.md](RUNPOD-5090-GUIDE.md) — the
5090 / 5B path is kept only as a fallback for 32GB-card environments. Do
**not** use the 5B path for ad-quality output (3-second cap, mangled form,
broken prompt adherence — see RFC-0006).

This runbook assumes:
- Single **A100 80GB** (PCIe or SXM) on any major cloud GPU provider.
- Ubuntu 22.04 / 24.04 with CUDA 12.4+ available.
- Persistent volume mounted at `/workspace`.
- You already have free API keys for:
  - **Pexels** — https://www.pexels.com/api/
  - **Pixabay** — https://pixabay.com/api/docs/
- ~80 GB free storage for the A14B snapshot (high + low noise experts).

---

## 0. Spin up the pod

Pick the cheapest A100 80GB you can find. As of 2026-Q2 the market rate is
~$1.20–$1.80 per hour spot. Templates:
- RunPod → "PyTorch 2.4 / CUDA 12.4 / A100 80GB SXM"
- Lambda → "Ada / Hopper one-click PyTorch"
- Vast.ai → filter `gpu_name="A100 SXM4 80GB"`, sort by `$/hr`

Persistent volume: **400 GB** (we'll download ~80 GB of model weights plus
~30 GB of clips, leaving headroom for activations + sample outputs).

When the pod boots, open the web terminal.

```bash
cd /workspace
git clone <your-fork>/ogenti.git
cd ogenti
```

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
pip install flash-attn --no-build-isolation   # ~10 min compile, big speedup
pip install xformers                          # bf16 memory-efficient attention
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

## 3. Download Wan2.2-T2V-A14B backbone (≈58 GB)

```bash
huggingface-cli login   # optional, only if your token is gated
huggingface-cli download Wan-AI/Wan2.2-T2V-A14B \
    --local-dir /workspace/wan2.2-t2v-a14b
```

The downloaded layout looks like:
```
/workspace/wan2.2-t2v-a14b/
  high_noise_model/         # ~28 GB safetensors shards (expert for noisy steps)
  low_noise_model/          # ~28 GB safetensors shards (expert for clean steps)
  vae/                      # AutoencoderKLWan
  text_encoder/             # T5-XXL bf16
  tokenizer/
  scheduler/
  model_index.json
```

The Ogenti importer auto-detects the MoE layout and picks the **low_noise
expert by default** (it carries the final-step quality bar that matters most
for ad creative). You can override with `--expert high_noise`.

> Want I2V (image-to-video) instead? Swap the model id for
> `Wan-AI/Wan2.2-I2V-A14B`. Same MoE layout, same retrofit pipeline.

## 4. Validate the keymap before spending compute

```bash
python -m scripts.retrofit.validate_keymap /workspace/wan2.2-t2v-a14b/ \
    --model-config ogenti/configs/model/ogenti_a14b.yaml
```

Expected outcome: `report` JSON shows `mapped_ok` > 500 (40 blocks × ~14
mapped tensors per block, plus top-level), `shape_mismatch == 0`.
If shape mismatch > 0, the model dims in `ogenti_a14b.yaml` don't match
the checkpoint — adjust `dim` (5120) / `num_blocks` (40) / `head_dim` (128)
to match upstream Wan2.2-A14B spec.

To validate the *other* expert:
```bash
python -m scripts.retrofit.validate_keymap /workspace/wan2.2-t2v-a14b/ \
    --expert high_noise \
    --model-config ogenti/configs/model/ogenti_a14b.yaml
```

## 5. Retrofit-import weights into an Ogenti checkpoint

```bash
python -m scripts.retrofit.import_wan22 /workspace/wan2.2-t2v-a14b/ \
    --model-config ogenti/configs/model/ogenti_a14b.yaml \
    --output-dir checkpoints/ogenti_a14b_retrofit_init
```

You now have an initial Ogenti checkpoint that — by retrofit invariant — is
**bit-equivalent** to the vanilla Wan2.2-T2V-A14B low_noise expert at step 0.

For full dual-expert inference later, also retrofit the high_noise expert:
```bash
python -m scripts.retrofit.import_wan22 /workspace/wan2.2-t2v-a14b/ \
    --expert high_noise \
    --model-config ogenti/configs/model/ogenti_a14b.yaml \
    --output-dir checkpoints/ogenti_a14b_retrofit_init_high
```

## 6. Generate the "BEFORE" reference video

```bash
mkdir -p outputs/samples/before
python -m ogenti.cli generate \
    "Slow motion shot of a cola bottle falling onto a marble counter" \
    --ckpt checkpoints/ogenti_a14b_retrofit_init \
    --out outputs/samples/before/cola_fall.mp4 \
    --no-micro-events --physics-backend off \
    --steps 30 --seed 42
```

This is your **baseline** (= what vanilla Wan2.2-A14B low_noise expert
produces). Save it; you'll diff against the post-training version in step 11.

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

Expected time on A100: 20-40 minutes (most of it is downloads + Qwen2-VL
captioning, which is much faster on A100 than 5090).

The final manifest lands at `data/manifests/ads_train.jsonl` with auto-captions,
brand-text OCR, keypoints, skin masks, glyph regions, camera motion, subject
motion, micro-events, and material masks all precomputed per shot.

## 8. (Optional) Pre-sim physics scenes

```bash
python -m scripts.data.presim_manifest data/manifests/ads_train.jsonl \
    --root data/ --backend pybullet --duration 5.0 --fps 24
```

## 9. Smoke test — 200 step forward+backward sanity

```bash
TOTAL_STEPS=200 python -m ogenti.cli train \
    ogenti/configs/training/smoke_a100.yaml \
    --model-config ogenti/configs/model/ogenti_a14b.yaml \
    --init-from checkpoints/ogenti_a14b_retrofit_init
```

Watch the loss curve. It should:
- Start near a finite non-zero value (not NaN, not Inf)
- Trend down monotonically over ~50 steps
- VRAM usage should peak around 60–70 GB (A14B is ~3× the 5B activation cost)

If VRAM blows up:
- enable `gradient_checkpointing` (already on in `smoke_a100.yaml`),
- lower `target_height`/`target_width` to 384×704,
- or drop `target_frames` to 65.

If loss is NaN: lower LR to `5e-6` and re-run.

## 10. Full smoke training — ~3000 steps

```bash
python -m ogenti.cli train \
    ogenti/configs/training/smoke_a100.yaml \
    --model-config ogenti/configs/model/ogenti_a14b.yaml \
    --init-from checkpoints/ogenti_a14b_retrofit_init
```

Expected wall clock: ~3 hours on a single A100 80GB (vs ~2.5 hours for the
old 5B path on a 5090 — the 14B backbone is bigger per step but the A100 is
~2× the throughput, so net is similar).

## 11. Generate "AFTER" video — same prompt + seed

```bash
mkdir -p outputs/samples/after
python -m ogenti.cli generate \
    "Slow motion shot of a cola bottle falling onto a marble counter" \
    --ckpt outputs/runs/smoke_a100/final \
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

## (Optional) Step 13. Full dual-expert MoE inference

For final-quality ad creative, run inference with BOTH retrofit experts
loaded simultaneously (the high_noise expert handles the noisy early steps,
the low_noise expert handles the clean late steps):

```python
from ogenti.inference.moe_wrapper import (
    A14BMoEConfig, load_a14b_moe_from_retrofit_dirs,
)
from ogenti.inference.pipelines.ogenti_pipeline import OgentiPipeline, OgentiPipelineConfig

moe = load_a14b_moe_from_retrofit_dirs(
    high_noise_ckpt="checkpoints/ogenti_a14b_retrofit_init_high",
    low_noise_ckpt="outputs/runs/smoke_a100/final",
    moe_config=A14BMoEConfig(boundary_t=0.5, offload_inactive_expert=True),
)
# `moe` is a drop-in replacement for an OgentiTransformer in OgentiPipeline.
```

Holding both 14B experts on a single A100 80GB requires
`offload_inactive_expert=True` (pins the inactive expert to host RAM, swaps
on the boundary crossing).

---

## Troubleshooting

**OOM at activations**
- Confirm `gradient_checkpointing: true` in `smoke_a100.yaml`.
- Drop `target_frames` to 65 or `target_height/width` to 384×704.

**OOM at VAE decode at inference**
- Reduce `num_frames` in the pipeline or use `--steps 20`.

**`pybullet` not found / segfault**
- Some headless containers lack OpenGL. Use `--physics-backend off` and the
  soft physics loss still works for now.

**Captioner Qwen2-VL OOM**
- Switch to `--caption-backend blip2` (BLIP-2 fits in <8 GB).

**Loss still NaN at lr=5e-6**
- The `bfloat16` path on A100 should be stable. If it isn't, switch
  `precision.dtype: float32` (training slower, but safe).

**"Where's my video?"**
- All checkpoints + samples live under `outputs/runs/smoke_a100/`. The
  final-step model is `outputs/runs/smoke_a100/final/model.safetensors`.

---

## Cost estimate (Cloud A100 80GB, 2026-Q2 spot)

| Stage                | Wall clock | $/hr | Cost       |
| -------------------- | ---------- | ---- | ---------- |
| Pod boot + setup     | 0.2 h      | 1.40 | $0.28      |
| Model download (58G) | 0.3 h      | 1.40 | $0.42      |
| Dataset fetch        | 0.5 h      | 1.40 | $0.70      |
| Smoke train (3k st)  | 3.0 h      | 1.40 | $4.20      |
| Sample generation    | 0.3 h      | 1.40 | $0.42      |
| **Total**            | **~4.3 h** |      | **~$6.0**  |

Add ~$1.00 if you keep the volume around for follow-up runs.

---

## What "success" looks like

The smoke test is a **proof-of-life**, not a proof-of-fidelity. With the
A14B backbone you should see significantly better baseline quality than
the legacy 5B path:
- 5+ second clips that hold shape end-to-end (no melting)
- Glyph rendering on labels that's nearly correct (still not perfect — that
  needs the dedicated glyph branch + OCR loss in later stages)
- Prompt adherence on multi-clause prompts (the "smart enough to understand
  what you wrote" property that 5B was missing)
- Loss curve trends down monotonically over ~3000 steps
- Subtle but visible differences vs the baseline (entity-anchored consistency,
  more stable cameras)

The goal here: prove the **A14B retrofit plumbing works end to end** on a
single GPU in an afternoon. Once it does, scale to multi-A100 for the full
retrofit_stage1_a14b run.
