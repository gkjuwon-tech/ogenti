"""Pod-local presim builder. Uses the ACTUAL APIs in ogenti.physics
(`parse_prompt` + `simulate`) — the upstream `scripts/data/presim_manifest.py`
references names that do not exist (`parse_scene_from_prompt`,
`simulate_scene`), so it silently produces zero physics files.

This script does the same job correctly:
  - Reads <root>/manifests/ads_train.jsonl
  - For each entry: parse prompt -> SceneSpec, simulate -> KeyframeTrajectories
  - Packs into (K, T, D) descriptor + (K,) mask
  - Saves to <root>/physics_keyframes/<id>.npz
  - Writes <root>/manifests/ads_train.presim.jsonl with added
    `physics_keyframes` + `physics_realism_score` keys per entry

Entries with no physical interaction inferable from the caption pass through
without a physics_keyframes key — the dataloader returns zero descriptor +
mask, which the model treats as "no physics constraint".
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/workspace/ogenti")
sys.path.insert(0, str(ROOT))

from ogenti.physics.scene_parser import parse_prompt
from ogenti.physics.simulator import simulate
from ogenti.utils.logging import configure_root_logging, get_logger

configure_root_logging()
log = get_logger("ogenti.runtime.build_presim")


def _resample(arr: np.ndarray, n: int) -> np.ndarray:
    T = arr.shape[0]
    if T == 0:
        return np.zeros((n, arr.shape[1] if arr.ndim > 1 else 0))
    idx = np.linspace(0, T - 1, num=n).astype(np.int64)
    return arr[idx]


def pack_descriptor(traj, max_objects: int, tokens_per_object: int, descriptor_dim: int):
    desc = np.zeros((max_objects, tokens_per_object, descriptor_dim), dtype=np.float32)
    mask = np.zeros(max_objects, dtype=np.bool_)
    if traj is None:
        return desc, mask
    per_obj = getattr(traj, "per_object", None) or getattr(traj, "objects", None) or []
    for i, obj in enumerate(per_obj[:max_objects]):
        pos = np.asarray(getattr(obj, "positions", []), dtype=np.float32)
        rot = np.asarray(getattr(obj, "rotations", []), dtype=np.float32)
        vel = np.asarray(getattr(obj, "velocities", []), dtype=np.float32)
        if pos.size == 0 or pos.ndim < 2 or pos.shape[-1] < 3:
            continue
        cols = [_resample(pos[:, :3], tokens_per_object)]
        if rot.size and rot.ndim >= 2 and rot.shape[-1] >= 4:
            cols.append(_resample(rot[:, :4], tokens_per_object))
        else:
            cols.append(np.zeros((tokens_per_object, 4), dtype=np.float32))
        if vel.size and vel.ndim >= 2 and vel.shape[-1] >= 3:
            cols.append(_resample(vel[:, :3], tokens_per_object))
        else:
            cols.append(np.zeros((tokens_per_object, 3), dtype=np.float32))
        merged = np.concatenate(cols, axis=-1).astype(np.float32)[:, :descriptor_dim]
        desc[i, :, : merged.shape[-1]] = merged
        mask[i] = True
    return desc, mask


def main():
    data_root = ROOT / "data"
    src_manifest = data_root / "manifests" / "ads_train.jsonl"
    out_manifest = data_root / "manifests" / "ads_train.presim.jsonl"
    pk_dir = data_root / "physics_keyframes"
    pk_dir.mkdir(parents=True, exist_ok=True)

    max_objects = 8
    tokens_per_object = 16
    descriptor_dim = 10

    entries = []
    with open(src_manifest) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    log.info(f"presim: read {len(entries)} entries from {src_manifest}")
    written = 0
    skipped_no_scene = 0
    skipped_no_sim = 0
    t_start = time.time()

    out_lines = []
    for i, entry in enumerate(entries):
        prompt = entry.get("prompt", "") or ""
        # parse_prompt always returns a SceneSpec (uses default factory if no
        # match + no LLM), so we get coverage on every clip. The dataloader
        # will still gracefully handle zero descriptors for clips where the
        # simulator hands back an empty trajectory.
        try:
            spec = parse_prompt(prompt, use_llm_fallback=False)
        except Exception as e:
            log.debug(f"  parse_prompt failed for id={entry.get('id')}: {e}")
            spec = None
            skipped_no_scene += 1

        traj = None
        if spec is not None:
            try:
                traj = simulate(spec, backend="pybullet")
            except Exception as e:
                log.debug(f"  simulate failed for id={entry.get('id')}: {e}")
                skipped_no_sim += 1

        desc, mask = pack_descriptor(traj, max_objects, tokens_per_object, descriptor_dim)

        if mask.any():
            rel = Path("physics_keyframes") / f"{entry['id']}.npz"
            np.savez(data_root / rel, descriptor=desc, mask=mask)
            entry["physics_keyframes"] = str(rel)
            entry["physics_realism_score"] = float(getattr(traj, "realism_score", 1.0))
            written += 1
            if written % 10 == 0:
                dt = time.time() - t_start
                log.info(
                    f"  presim {written}/{len(entries)} ({dt:.0f}s elapsed, "
                    f"skipped_scene={skipped_no_scene} skipped_sim={skipped_no_sim})"
                )

        out_lines.append(json.dumps(entry))

    out_manifest.write_text("\n".join(out_lines) + "\n")
    dt = time.time() - t_start
    log.info(
        f"done: wrote {written} physics_keyframes files in {dt:.0f}s; "
        f"presim manifest -> {out_manifest}"
    )


if __name__ == "__main__":
    main()
