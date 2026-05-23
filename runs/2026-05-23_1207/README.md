# Pod handoff snapshot — 2026-05-23

Snapshot of the RunPod A100 work directory at `/workspace/ogenti/` immediately
before the user terminated the pod. Everything is preserved **except** the
two large binary blobs that are trivially re-downloadable:

- `checkpoints/ogenti_a14b_retrofit_init/model.safetensors` (~82 GB) — the
  Ogenti retrofit-init state dict. Re-derive with
  `ogenti retrofit /workspace/wan2.2-t2v-a14b --expert low_noise --out checkpoints/ogenti_a14b_retrofit_init`
  on a fresh pod after the Wan2.2 download completes. The `meta.json` next to
  it is preserved here so the retrofit metadata (shape map, missing keys,
  zero-init register) is committed.
- `data/skin_masks/` (~16 GB of `.npy` per-frame skin segmentation masks) —
  re-derivable from `data/videos/` via `scripts/data/build_dataset.py
  --resume --run-skin --no-run-glyphs --no-run-subject-motion
  --no-run-micro-events`.

The Wan2.2-T2V-A14B HF snapshot at `/workspace/wan2.2-t2v-a14b/` (~118 GB) is
also not in this commit; it is the unmodified upstream weights and is fetched
from `huggingface-cli download Wan-AI/Wan2.2-T2V-A14B --local-dir ...`.

## Directory layout

```
.runtime/                       master queue + monitoring guide + per-step logs
  run_all.sh                    9-stage bash driver (HF wait → ... → comparison)
  build_presim.py               pod-local PyBullet pre-sim (fixes broken upstream script)
  HANDOFF.md                    monitoring guide written for the next AI / human
  secrets.env                   *** REDACTED *** (Pexels/Pixabay/HF env vars)
  phase.txt                     (not preserved; was reset by pod restart)
  run_all.log                   (not preserved; reset by pod restart)
  run_all.stdout                outer nohup wrapper log
  01_validate_keymap.log        keymap validation stdout
  01_keymap_report.json         pre-restart keymap shape/missing summary
  01b_keymap_report.json        post-restart re-run of the same
  02_retrofit_import.log        ogenti retrofit (Wan2.2 → Ogenti) import log
  02b_build_presim.log          PyBullet pre-sim per-clip output
  03_before_video.log           BEFORE video generation (CRASHED — see below)
  04_curriculum.log             curriculum start (CRASHED at same import)
data/
  manifests/
    ads_train.jsonl             112 ad clips (Pexels + Pixabay)
    ads_train.presim.jsonl      same + physics_keyframes refs
  videos/                       112 source MP4 clips (480×832, 24 fps, 4 s)
  camera_motion/                Farneback flow + RANSAC → 8-dim camera motion .npy
  keypoints/                    MediaPipe pose .npy
  physics_keyframes/            PyBullet pre-sim .npz (descriptor + mask)
  subject_motion/               (mostly empty — was disabled at build time)
outputs/runs/full_curriculum_a14b/
  _phases/phase1_entity_init.yaml   materialized phase 1 config (driver wrote this)
checkpoints/ogenti_a14b_retrofit_init/
  meta.json                     retrofit shape map, zero-init register, vanilla-bit-equivalence proof
```

## How far the queue got

The master queue script ran the following stages successfully:

1. `preflight` — environment sanity OK
2. `waiting_hf_download` — Wan2.2-T2V-A14B fully downloaded (~118 GB, 6+6 shards + VAE + T5)
3. `waiting_dataset` — 112 clips built via `scripts.data.build_dataset` (target was 100; the duplicate-process artifact in session 1 inflated the count)
4. `build_presim` — `02b_build_presim.log` shows PyBullet successfully ran on every clip; physics_keyframes `.npz` files generated
5. `validate_keymap` — both pre- and post-restart runs produced clean shape reports (`01_keymap_report.json`, `01b_keymap_report.json`)
6. `retrofit_import` — Ogenti retrofit checkpoint written (~82 GB safetensors + meta.json)

Then it **crashed** in two consecutive stages with the same root cause:

7. `before_video` — `RuntimeError: Failed to import diffusers.models.autoencoders.autoencoder_kl_wan because of: infer_schema(func): Parameter q has unsupported type torch.Tensor`
8. `curriculum` — phase 1 distributed init succeeded, model loaded (131.1M trainable / 25.4 B total), then the same `AutoencoderKLWan` import fired during `Wan22Backbone.load()`.

### Root cause

`diffusers.models.autoencoders.autoencoder_kl_wan` registers a custom flash-attn-style op with PyTorch's schema inferencer, and the installed combo of `diffusers` + `torch 2.4.1+cu124` + flash-attn / vendored attention kernel does not agree on the op signature. The schema check at op-registration time raises before the autoencoder class is even reachable. This is an environment issue, not a code issue. Possible fixes for the next pod:

- pin `diffusers` to a version that ships an op signature compatible with torch 2.4 (e.g. test `pip install 'diffusers==0.31.0'` or `==0.30.0`), or
- upgrade torch to 2.5+ (which most recent diffusers Wan kernels target), or
- use the older `AutoencoderKLWanInference` path or write a small shim that loads the VAE bypassing the new op-registry entry.

The retrofit checkpoint itself is valid — the crash happened **after** loading and before any forward pass — so re-running on a fixed environment should let phase 1 start immediately.

## Re-running on a new pod

1. SSH in, clone the repo, fetch the Wan2.2-T2V-A14B HF snapshot.
2. Install deps. Then **before launching the master queue**:
   ```
   pip install --force-reinstall 'diffusers==0.31.0'   # or whichever pin works for your torch
   python -c "from diffusers import AutoencoderKLWan; print('import ok')"
   ```
3. Restore `secrets.env` with the real Pexels/Pixabay keys (rotate first if they leaked).
4. `bash /workspace/ogenti/.runtime/run_all.sh` (or copy `.runtime/run_all.sh` from this snapshot if you cloned the repo at this commit). The dataset will be rebuilt from scratch unless you also restore `runs/2026-05-23_1207/data/` to `/workspace/ogenti/data/`.

## Cost / time summary

- ~12 h pod wall-clock.
- 4 sessions of agent activity (RFC analysis → data pipeline → curriculum design → master queue + handoff).
- 0 PRs created from this session — user explicit instruction "pr 더이상 하지마".
