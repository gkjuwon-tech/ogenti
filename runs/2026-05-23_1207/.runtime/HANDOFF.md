# Ogenti Pod Handoff (for the next AI / human picking this up)

You are taking over a running Ogenti retrofit training job on a RunPod A100. The
job is fully queued and self-driving — your role is to **monitor**, **report**,
and **only intervene if something fails or wanders off the rails**. This doc is
the single source of truth; nothing else on the pod has more context.

---

## 1. What we are doing (in one paragraph)

The "ogenti" project is a structural retrofit of `Wan2.2-T2V-A14B` (Alibaba's 14B
parameter mixture-of-experts text-to-video DiT) for **advertising-quality** video
generation. The goal is to fix the failure modes that make raw Wan2.2 unusable for
ads: identity drift across shots, mangled text/logos, broken anatomy, plastic
skin, fake camera motion, missing micro-events, and physics violations. We do
this by inserting Ogenti add-on modules (zero-init-gated, so step 0 ≡ vanilla
Wan2.2) and running a 5-phase curriculum that progressively unlocks parameter
subsets and loss terms. The full retrofit ladder lives in
`ogenti/configs/training/full_curriculum_a14b.yaml`. The driver is
`scripts/train/run_curriculum.py`. The end goal of this session is one BEFORE
video (vanilla Wan2.2) and one AFTER video (post-curriculum Ogenti) on the same
prompt + seed, plus a side-by-side comparison.

---

## 2. Access

### SSH
The user (uwonma) has registered their personal `~/.ssh/id_ed25519` public key
on this pod. From the user's box:

```bash
ssh root@154.54.102.38 -p 11236 -i ~/.ssh/id_ed25519
```

If the user gives you a different SSH key file path, use that. The pod also
accepts the Devin session key (`devin-ogenti-runpod`) for backwards-compatibility
but that is owned by the Devin session that built this and won't be needed.

### Pod facts

| Item | Value |
| --- | --- |
| Public IP / port | 154.54.102.38 : 11236 |
| GPU | NVIDIA A100-SXM4-80GB (compute_cap 8.0) |
| RAM | 250 GB |
| Container disk | 200 GB overlay (fast, local) |
| Volume / workspace | `/workspace` (mfs, very large, **slower for many small writes** but excellent sequential bandwidth) |
| OS | Ubuntu 22.04 |
| Python | `/usr/bin/python3` → 3.11.10 (system); **do not use any venv on /workspace, it is slow** |
| torch | 2.4.1+cu124 (pre-installed in the runpod-torch-v240 template) |
| Ogenti repo | `/workspace/ogenti` (master, with PRs #2 + #3 merged in) |

### Secrets (Pexels / Pixabay / HF cache)

Persisted at `/workspace/ogenti/.runtime/secrets.env`. Source with:

```bash
set -a; source /workspace/ogenti/.runtime/secrets.env; set +a
```

Do **NOT** print these in any chat to the user or post them anywhere public —
they are the user's personal API keys. If they leak, recommend the user rotate
both keys.

---

## 3. The master queue — `run_all.sh`

Everything from "Wan2.2 download done" through to "final comparison video" runs
inside one bash script:

```
/workspace/ogenti/.runtime/run_all.sh
```

It is launched under `nohup` and writes a single phase indicator to:

```
/workspace/ogenti/.runtime/phase.txt
```

Reading this file is the **fastest way** to know what step the job is on. Phases
in order:

| `phase.txt` value | Means |
| --- | --- |
| `preflight` | Sanity checks (python, ffmpeg, GPU). |
| `waiting_hf_download` | Polling for the `huggingface-cli` PID downloading Wan2.2-T2V-A14B. |
| `waiting_dataset` | Polling for the `scripts.data.build_dataset` PID. |
| `build_presim` | Runs `.runtime/build_presim.py` — PyBullet pre-simulates physics trajectories from each clip's caption, writes `data/manifests/ads_train.presim.jsonl` + `data/physics_keyframes/<id>.npz`. Phase 5 prerequisite. |
| `validate_keymap` | `scripts.retrofit.validate_keymap` — dry-run sanity vs the real Wan2.2 weights. |
| `retrofit_import` | `ogenti retrofit` — builds the zero-init Ogenti checkpoint by importing Wan2.2 low_noise expert weights. |
| `before_video` | `ogenti generate` from the retrofit-init checkpoint → `outputs/samples/before.mp4`. |
| `curriculum` | `ogenti curriculum full_curriculum_a14b.yaml` runs phases 1..5 back-to-back. **This is the long one (≈10–11h).** |
| `after_video` | `ogenti generate` from the deepest completed phase checkpoint → `outputs/samples/after.mp4`. |
| `comparison` | `ffmpeg hstack` BEFORE / AFTER → `outputs/samples/comparison.mp4`. |
| `DONE` | Everything finished. |
| `FAILED: <stage>` | A required stage hit a fatal error; see the corresponding log. |

Per-step logs all land under `/workspace/ogenti/.runtime/`:

```
run_all.log                  ← high-level timeline
01_validate_keymap.log       ← keymap dry-run stdout/stderr
01_keymap_report.json        ← shape/missing summary
02_retrofit_import.log
02b_build_presim.log         ← PyBullet pre-sim output
03_before_video.log
04_curriculum.log            ← phase 1..5 training output (long)
05_after_video.log
06_comparison.log
```

---

## 4. Live monitoring cheatsheet

Run any of these from your own machine via SSH. They are all read-only.

### What step is the job on right now?

```bash
ssh root@154.54.102.38 -p 11236 'cat /workspace/ogenti/.runtime/phase.txt; tail -20 /workspace/ogenti/.runtime/run_all.log'
```

### Is the master script still alive?

```bash
ssh root@154.54.102.38 -p 11236 'pgrep -af run_all.sh; pgrep -af curriculum; pgrep -af train.py'
```

If none of those return, either `phase.txt == DONE` or something crashed.
Check `phase.txt` and the latest log to disambiguate.

### Dataset progress (during `waiting_dataset`)

```bash
ssh root@154.54.102.38 -p 11236 '
  wc -l /workspace/ogenti/data/manifests/ads_train.jsonl
  tail -3 /workspace/dataset_build.log
'
```

### Wan2.2 download progress (during `waiting_hf_download`)

```bash
ssh root@154.54.102.38 -p 11236 '
  du -sh /workspace/wan2.2-t2v-a14b
  tail -3 /workspace/hf_download.log
'
```

### Curriculum / training progress (the long one)

```bash
ssh root@154.54.102.38 -p 11236 '
  tail -50 /workspace/ogenti/.runtime/04_curriculum.log
  ls -1 /workspace/ogenti/outputs/runs/full_curriculum_a14b/ 2>/dev/null
'
```

A finished phase leaves a directory like
`outputs/runs/full_curriculum_a14b/phase1_entity_init/final/` containing the
checkpoint. Subsequent phases auto-pick that up.

The curriculum's own resumption state file is
`outputs/runs/full_curriculum_a14b/_state.json` (lists completed phases).

### GPU utilization

```bash
ssh root@154.54.102.38 -p 11236 'nvidia-smi'
```

During the curriculum the A100 should hover at 80–100% GPU util and 60–75 GB
GPU memory. If it stays at 0 % for >5 minutes during phases 1–5 something is
wrong — most likely the dataloader (check `04_curriculum.log` for CUDA OOM,
NaN loss, missing tensor keys, etc).

---

## 5. The 5-phase retrofit curriculum (so you understand what you're watching)

| Phase | RFC | Steps | LR | Unfreezes (delta) | Losses (active) |
| --- | --- | --- | --- | --- | --- |
| `phase1_entity_init` | RFC-0001 IAHA + DGB + CMT warm-start | 1000 | 1e-5 | `entity_bank`, `blocks.*.entity_refine`, `blocks.*.patch_from_entity`, `blocks.*.glyph_fuse`, `glyph_branch`, `camera_motion_embed` | diffusion + identity |
| `phase2_subject_motion_anatomy` | RFC-0001 §3.3 ACP + RFC-0003 §4.1 | 800 | 8e-6 | + `subject_motion_embed` | + anatomy |
| `phase3_ocr_hardening` | RFC-0001 §3.2 DGB tightening | 700 | 6e-6 | (no new params unlocked — fine-tune gates) | + ocr (stub; see warnings below) |
| `phase4_realism` | RFC-0003 SDRH + RFC-0004 4-RE2 | **1500** | 5e-6 | + `micro_event_embed`, `skin_detail_head`, `material_detail_head`, `motion_blur_head`, `film_grain_head`, `lens_artifact_head` | + frequency_skin, temporal_realism, motion_realism (**AI-tell killer phase**) |
| `phase5_physics` | RFC-0005 hard keyframe rails | 800 | 3e-6 | + `physics_keyframe_embed` | + physics, physics_keyframe (hard) — swaps in `ads_train.presim.jsonl` |

**Total: 4 800 steps.** At ~8 s/step on an A100-80GB this is ~10.7 h. With grad
accumulation 4, optimizer state spills, and checkpointing the realistic wall
clock is 10–12 h for the full ladder.

---

## 6. Known caveats (don't panic if you see these)

These are intentional simplifications taken for time-budget reasons and are
**not bugs**:

1. **OCR loss is a 0.0 stub.** `scripts/train/train.py` returns `loss/ocr = 0.0`
   when `loss_weights.ocr > 0` because the glyph branch isn't wired to char
   logits yet. Phase 3 still trains — it just doesn't have OCR gradient.
   Glyph branch keeps training via the diffusion loss.

2. **Heavy preprocessing was skipped during dataset build:**
   `--no-run-glyphs --no-run-subject-motion --no-run-micro-events`. The
   dataset loader has graceful zero-tensor fallback, and zero-init gates on
   the Ogenti modules mean training is bit-stable. You should still see
   meaningful gradient on `entity_bank`, `camera_motion_embed`, `glyph_branch`
   (via diffusion), `skin_detail_head` (via frequency_skin loss), and the
   anatomy loss path (via keypoints).

3. **Phase 5 physics pre-sim is built automatically** by the `build_presim`
   stage between `waiting_dataset` and `validate_keymap`. The upstream
   `scripts/data/presim_manifest.py` references function names that don't
   exist in `ogenti.physics` (`parse_scene_from_prompt`, `simulate_scene`) and
   silently produces zero physics files — so we **do not call it**. Instead
   we use `/workspace/ogenti/.runtime/build_presim.py`, a pod-local rewrite
   that calls the real APIs (`parse_prompt` + `simulate`). Output:
   `data/manifests/ads_train.presim.jsonl` + `data/physics_keyframes/<id>.npz`
   per clip. If a clip has no template match in the heuristic scene parser
   the default factory (`make_falling_object_scene`) is used, so every clip
   gets coverage. Inspect `02b_build_presim.log` for per-batch progress.

4. **Dataset is small (target 100 clips).** Retrofit on a 14B model with
   frozen backbone and zero-init gates is **statistically efficient on tiny
   data** — we're not pre-training, we're nudging an already-trained backbone
   along ~ a dozen low-dim conditioning paths. 100 clips × 4800 steps × bs1×ga4
   = ~6 epochs per clip. Sufficient for a smoke-to-first-checkpoint pass.

5. **ffmpeg is symlinked from `imageio_ffmpeg`** at `/usr/local/bin/ffmpeg`
   (apt's `ffmpeg` package wasn't available in the runpod-torch image). It is
   a static build of ffmpeg 7.0.2 and works for both build-time encoding and
   the final hstack comparison.

---

## 7. Troubleshooting decision tree

| Symptom | Action |
| --- | --- |
| `phase.txt` stuck on `waiting_dataset` for >2h | `pgrep -af build_dataset` and read `/workspace/dataset_build.log`. If process died and manifest has ≥30 entries, you can edit `run_all.sh` to skip ahead, or `kill` the master and re-run starting from `validate_keymap`. If process died with no manifest, check Pexels/Pixabay API rate limits. |
| `phase.txt: FAILED: build_presim` not seen but `02b_build_presim.log` shows 0 .npz files written | The presim script has a soft fallback — if PyBullet or scene_parser fails on a clip it just skips it and the dataloader returns zero descriptors (graceful). 0 files written means PyBullet may not be installed; check `python3 -c 'import pybullet'` and reinstall if needed. The curriculum will still complete; phase 5 will simply not get hard-keyframe gradient. |
| `phase.txt: FAILED: validate_keymap` | Open `01_keymap_report.json` — anything in `"shape_mismatch"` means the model config doesn't match the real Wan2.2 weights. Compare `ogenti/configs/model/ogenti_a14b.yaml` to the actual dims in `/workspace/wan2.2-t2v-a14b/low_noise_model/config.json`. |
| `phase.txt: FAILED: retrofit_import` | Read `02_retrofit_import.log`. Usually a key drift (KeyError) or shape mismatch. Refer back to keymap report. |
| CUDA OOM during curriculum | Open `04_curriculum.log`. The default `dataloader.batch_size: 1, grad_accum: 4` is already conservative. Likely culprits: too high `frames` (lower from 81→49 in curriculum YAML), or activation checkpointing not enabled (look for `gradient_checkpointing: false`; flip to `true`). |
| Loss NaN | Almost always a corrupt clip in the manifest. Read the last few `04_curriculum.log` lines for `clip_id`. Remove that line from `data/manifests/ads_train.jsonl` and let the curriculum auto-resume from the last checkpoint. |
| `pod stop`'d unexpectedly | All checkpoints are on the `/workspace` volume → survive pod restart. On new pod, just `bash /workspace/ogenti/.runtime/run_all.sh` again — it will skip completed phases via `_state.json`. |

---

## 8. Final deliverables (what to send the user when `phase.txt == DONE`)

1. `/workspace/ogenti/outputs/samples/before.mp4`
2. `/workspace/ogenti/outputs/samples/after.mp4`
3. `/workspace/ogenti/outputs/samples/comparison.mp4`
4. `/workspace/ogenti/.runtime/run_all.log` (one-page timeline)
5. (Optional) Plot of loss curves — easiest path: tensorboard inside
   `outputs/runs/full_curriculum_a14b/<phase>/tb/`, but for a quick screenshot
   just grep `04_curriculum.log` for `loss/diffusion=` and pipe into a
   matplotlib script.

**Get those three MP4s to the user any way you can** (scp to user's box, upload
to a cloud bucket, attach in chat). The user's GitHub is `gkjuwon-tech`. The
repo is `gkjuwon-tech/ogenti`. The user does **not** want any further PRs from
this session (their explicit instruction); attach files in chat, do not push.

---

## 9. Style notes from the user (uwonma)

- Speaks Korean and English freely, prefers terse / direct answers with no fluff.
- Has a sense of humor — "개웃기게 답해" = "be funny" — match that tone in
  Korean replies but stay technically rigorous.
- Hates wasted tokens. If you can answer in 1 line, do. If you need 30 lines,
  make every line earn its place.
- Avoid PRs unless the user explicitly asks. Local edits / pod-only scripts
  are fine.

---

Good luck. The hard part is already done — just don't let it die in its sleep.
