#!/bin/bash
# Ogenti master queue: HF wait -> dataset wait -> validate -> import -> BEFORE
# -> full curriculum (phase 1..5) -> AFTER -> side-by-side comparison.
# Resumable; uses content-based waits (file presence + .incomplete absence) so
# it does NOT get tricked by stale pgrep PIDs.

set -uo pipefail

ROOT=/workspace/ogenti
RUNTIME=$ROOT/.runtime
LOG=$RUNTIME/run_all.log
PHASE_FILE=$RUNTIME/phase.txt
PROMPT="Cinematic luxury watch product ad: polished steel case on dark marble surface, soft window light from left, slow push-in dolly, shallow depth of field, photoreal 4K commercial cinematography"
WAN=/workspace/wan2.2-t2v-a14b

cd "$ROOT"
mkdir -p outputs/samples outputs/logs outputs/runs checkpoints

if [ -f "$RUNTIME/secrets.env" ]; then
    set -a; . "$RUNTIME/secrets.env"; set +a
fi

log()       { echo "[$(date +%FT%T%z)] $*" | tee -a "$LOG"; }
set_phase() { echo "$1" > "$PHASE_FILE"; log "==> PHASE: $1"; }
mark_fail() { log "FAIL: $1"; echo "FAILED: $1" > "$PHASE_FILE"; }

# Content-based HF readiness: all 6 low_noise shards + all 6 high_noise shards
# + VAE + at least one text_encoder weight, and no .incomplete files anywhere.
hf_ready() {
    local lo_n hi_n inc_n vae te_n
    lo_n=$(ls "$WAN"/low_noise_model/diffusion_pytorch_model-0000?-of-00006.safetensors 2>/dev/null | wc -l)
    hi_n=$(ls "$WAN"/high_noise_model/diffusion_pytorch_model-0000?-of-00006.safetensors 2>/dev/null | wc -l)
    inc_n=$(ls "$WAN"/.cache/huggingface/download/*/*.incomplete 2>/dev/null | wc -l)
    vae=$([ -f "$WAN/Wan2.1_VAE.pth" ] && echo 1 || echo 0)
    te_n=$(ls "$WAN"/models_t5_umt5-xxl-enc-bf16.pth "$WAN"/google/umt5-xxl/*.safetensors 2>/dev/null | wc -l)
    log "  hf_ready check: low_noise=$lo_n/6 high_noise=$hi_n/6 vae=$vae text_enc=$te_n incomplete=$inc_n"
    [ "$lo_n" -eq 6 ] && [ "$hi_n" -eq 6 ] && [ "$vae" -eq 1 ] && [ "$inc_n" -eq 0 ]
}

# ---------- 0. preflight ----------
set_phase "preflight"
log "torch: $(/usr/bin/python3 -c 'import torch;print(torch.__version__, torch.cuda.is_available())' 2>&1)"
log "gpu: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1)"

# ---------- 1. wait for HF download (content-based) ----------
set_phase "waiting_hf_download"
ATTEMPTS=0
MAX_HF_RESTARTS=3
RESTARTS=0
while ! hf_ready; do
    HF_PROC="$(pgrep -af 'huggingface-cli download Wan-AI' | head -1)"
    if [ -z "$HF_PROC" ]; then
        if [ "$RESTARTS" -ge "$MAX_HF_RESTARTS" ]; then
            mark_fail "hf_download_unrecoverable_after_${RESTARTS}_restarts"
            exit 1
        fi
        RESTARTS=$((RESTARTS+1))
        log "  no huggingface-cli process; auto-resuming (restart #$RESTARTS)"
        nohup /usr/local/bin/huggingface-cli download Wan-AI/Wan2.2-T2V-A14B \
            --local-dir "$WAN" \
            >> /workspace/hf_download.log 2>&1 &
        disown
        sleep 5
    fi
    sleep 90
    ATTEMPTS=$((ATTEMPTS+1))
    if [ $((ATTEMPTS % 5)) -eq 0 ]; then
        log "  hf wait: $((ATTEMPTS*90))s elapsed, $(du -sh $WAN 2>/dev/null | cut -f1) on disk"
    fi
done
log "Wan2.2-T2V-A14B verified complete"

# ---------- 2. wait for dataset build ----------
set_phase "waiting_dataset"
DS_TARGET=80   # accept "good enough" early so we don't burn budget
ATTEMPTS=0
while true; do
    N=$(wc -l < "$ROOT/data/manifests/ads_train.jsonl" 2>/dev/null || echo 0)
    DS_PROC="$(pgrep -af 'scripts.data.build_dataset' | head -1)"
    if [ "$N" -ge "$DS_TARGET" ]; then
        log "  dataset target reached ($N >= $DS_TARGET) — proceeding"
        break
    fi
    if [ -z "$DS_PROC" ]; then
        if [ "$N" -ge 30 ]; then
            log "  build process exited; manifest has $N clips (>=30) — proceeding"
            break
        else
            mark_fail "dataset_too_small ($N clips, no build process)"
            exit 1
        fi
    fi
    sleep 60
    ATTEMPTS=$((ATTEMPTS+1))
    if [ $((ATTEMPTS % 5)) -eq 0 ]; then
        log "  dataset wait: $N clips, build proc still running"
    fi
done
N=$(wc -l < "$ROOT/data/manifests/ads_train.jsonl" 2>/dev/null || echo 0)
log "dataset finalized: $N clips"

# ---------- 2b. presim physics keyframes (phase 5 prerequisite) ----------
set_phase "build_presim"
PRESIM_OUT="$ROOT/data/manifests/ads_train.presim.jsonl"
if [ -f "$PRESIM_OUT" ] && [ "$(wc -l < "$PRESIM_OUT")" -eq "$N" ]; then
    log "presim manifest already exists with $N entries — skipping rebuild"
else
    set +e
    /usr/bin/python3 "$RUNTIME/build_presim.py" \
        > "$RUNTIME/02b_build_presim.log" 2>&1
    RC=$?
    set -e
    if [ ! -f "$PRESIM_OUT" ]; then
        log "WARN: presim builder failed (rc=$RC); copying base manifest so phase 5 still has a file to open. Physics keyframes will all be zero (graceful)."
        cp "$ROOT/data/manifests/ads_train.jsonl" "$PRESIM_OUT"
    else
        PK_N=$(ls "$ROOT/data/physics_keyframes/"*.npz 2>/dev/null | wc -l)
        log "presim: $(wc -l < "$PRESIM_OUT") manifest entries, $PK_N physics_keyframes .npz files"
    fi
fi

# ---------- 3. validate keymap ----------
set_phase "validate_keymap"
set +e
/usr/bin/python3 -m scripts.retrofit.validate_keymap "$WAN" \
    --variant a14b --expert low_noise \
    --out-report "$RUNTIME/01_keymap_report.json" \
    > "$RUNTIME/01_validate_keymap.log" 2>&1
RC=$?
set -e
if [ "$RC" -ne 0 ]; then
    log "WARN: validate_keymap rc=$RC; see 01_keymap_report.json. Continuing — retrofit_import will hard-fail if dims are wrong."
fi

# ---------- 4. retrofit import ----------
set_phase "retrofit_import"
set +e
/usr/bin/python3 -c "from ogenti.cli import app; app(['retrofit', '$WAN', '--expert', 'low_noise', '--out', 'checkpoints/ogenti_a14b_retrofit_init'])" \
    > "$RUNTIME/02_retrofit_import.log" 2>&1
RC=$?
set -e
if [ ! -d checkpoints/ogenti_a14b_retrofit_init ]; then
    mark_fail "retrofit_import (rc=$RC)"
    exit 1
fi
log "retrofit init checkpoint ready"

# ---------- 5. BEFORE video (vanilla Wan2.2-A14B via retrofit init) ----------
set_phase "before_video"
set +e
/usr/bin/python3 -c "from ogenti.cli import app; app(['generate', '$PROMPT', '--ckpt', 'checkpoints/ogenti_a14b_retrofit_init', '--out', 'outputs/samples/before.mp4', '--steps', '30', '--guidance', '5.0', '--seed', '42'])" \
    > "$RUNTIME/03_before_video.log" 2>&1
set -e
if [ -f outputs/samples/before.mp4 ]; then
    log "BEFORE video saved"
else
    log "WARN: BEFORE video failed (see 03_before_video.log). Continuing."
fi

# ---------- 6. full curriculum ----------
set_phase "curriculum"
set +e
/usr/bin/python3 -c "from ogenti.cli import app; app(['curriculum', 'ogenti/configs/training/full_curriculum_a14b.yaml', '--init-from', 'checkpoints/ogenti_a14b_retrofit_init'])" \
    > "$RUNTIME/04_curriculum.log" 2>&1
RC=$?
set -e
log "curriculum exited rc=$RC"

# Discover deepest phase that completed
FINAL_CKPT=""
for ph in phase5_physics phase4_realism phase3_ocr_hardening phase2_subject_motion_anatomy phase1_entity_init; do
    d=outputs/runs/full_curriculum_a14b/$ph/final
    if [ -d "$d" ]; then
        FINAL_CKPT=$d
        log "deepest completed phase: $ph"
        break
    fi
done

# ---------- 7. AFTER video ----------
set_phase "after_video"
if [ -n "$FINAL_CKPT" ]; then
    set +e
    /usr/bin/python3 -c "from ogenti.cli import app; app(['generate', '$PROMPT', '--ckpt', '$FINAL_CKPT', '--out', 'outputs/samples/after.mp4', '--steps', '30', '--guidance', '5.0', '--seed', '42'])" \
        > "$RUNTIME/05_after_video.log" 2>&1
    set -e
    [ -f outputs/samples/after.mp4 ] && log "AFTER video saved" || log "WARN: AFTER video failed (see 05_after_video.log)"
else
    log "WARN: no curriculum checkpoint found; skipping AFTER video"
fi

# ---------- 8. side-by-side comparison ----------
set_phase "comparison"
if [ -f outputs/samples/before.mp4 ] && [ -f outputs/samples/after.mp4 ]; then
    set +e
    ffmpeg -y -i outputs/samples/before.mp4 -i outputs/samples/after.mp4 \
        -filter_complex "[0:v]drawtext=text='BEFORE (vanilla Wan2.2-A14B)':fontcolor=white:fontsize=20:box=1:boxcolor=black@0.5:boxborderw=5:x=10:y=10[a];[1:v]drawtext=text='AFTER (Ogenti curriculum)':fontcolor=white:fontsize=20:box=1:boxcolor=black@0.5:boxborderw=5:x=10:y=10[b];[a][b]hstack=inputs=2" \
        -c:v libx264 -crf 18 -preset fast \
        outputs/samples/comparison.mp4 \
        > "$RUNTIME/06_comparison.log" 2>&1
    set -e
    [ -f outputs/samples/comparison.mp4 ] && log "comparison saved" || log "WARN: comparison ffmpeg failed"
else
    log "WARN: missing before/after, skipping comparison"
fi

# ---------- DONE ----------
set_phase "DONE"
log "============================="
log "run_all.sh complete."
log "BEFORE:     outputs/samples/before.mp4 ($( [ -f outputs/samples/before.mp4 ] && du -h outputs/samples/before.mp4 | cut -f1 || echo MISSING ))"
log "AFTER:      outputs/samples/after.mp4  ($( [ -f outputs/samples/after.mp4 ]  && du -h outputs/samples/after.mp4  | cut -f1 || echo MISSING ))"
log "COMPARISON: outputs/samples/comparison.mp4 ($( [ -f outputs/samples/comparison.mp4 ] && du -h outputs/samples/comparison.mp4 | cut -f1 || echo MISSING ))"
log "FINAL CKPT: ${FINAL_CKPT:-NONE}"
log "============================="
