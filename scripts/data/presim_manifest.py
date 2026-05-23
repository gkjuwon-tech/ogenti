"""Pre-simulate physics keyframes for every entry in a manifest.

RFC-0005 §4 — physics is delegated to a deterministic simulator. To use
this during training we pre-run the simulator on each clip's inferred
SceneSpec and store the resulting trajectories per clip so the dataloader
can serve them without paying the simulation cost online.

Output per entry:
  - ``<root>/physics_keyframes/<id>.npz`` with::
        descriptor: (max_objects, tokens_per_object, descriptor_dim)
        mask:       (max_objects,) bool

The simulator backend is selected via ``ogenti.physics.simulator`` policies.
Entries lacking a parseable SceneSpec (no physical interaction implied by
the caption) are left untouched — the dataloader returns zero descriptor +
zero mask, which the model treats as "no physics constraint."
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import typer

from ogenti.utils.logging import configure_root_logging, get_logger

log = get_logger("ogenti.scripts.presim_manifest")

app = typer.Typer(pretty_exceptions_enable=False)


def _try_parse_scene(prompt: str):
    try:
        from ogenti.physics.scene_parser import parse_scene_from_prompt

        return parse_scene_from_prompt(prompt)
    except Exception as e:
        log.debug(f"scene parser failed: {e}")
        return None


def _try_simulate(spec, backend: str, duration: float, fps: int):
    try:
        from ogenti.physics.simulator import simulate_scene

        return simulate_scene(spec, backend=backend, duration=duration, fps=fps)
    except Exception as e:
        log.debug(f"simulator failed: {e}")
        return None


def _trajectories_to_descriptor(
    trajectories,
    max_objects: int,
    tokens_per_object: int,
    descriptor_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pack a ``KeyframeTrajectories`` into the (K, T, D) descriptor tensor.

    Descriptor channels: [x, y, z, qx, qy, qz, qw, vx, vy, vz][:descriptor_dim].
    """
    desc = np.zeros((max_objects, tokens_per_object, descriptor_dim), dtype=np.float32)
    mask = np.zeros(max_objects, dtype=np.bool_)
    if trajectories is None:
        return desc, mask
    per_obj = getattr(trajectories, "per_object", None) or getattr(trajectories, "objects", None)
    if not per_obj:
        return desc, mask
    for i, obj in enumerate(per_obj[:max_objects]):
        positions = np.asarray(getattr(obj, "positions", []), dtype=np.float32)
        rotations = np.asarray(getattr(obj, "rotations", []), dtype=np.float32)
        velocities = np.asarray(getattr(obj, "velocities", []), dtype=np.float32)
        T = positions.shape[0] if positions.ndim >= 2 else 0
        if T == 0:
            continue
        # Resample to tokens_per_object uniformly
        idx = np.linspace(0, T - 1, num=tokens_per_object).astype(np.int64)
        cols = []
        if positions.shape[-1] >= 3:
            cols.append(positions[idx, :3])
        else:
            cols.append(np.zeros((tokens_per_object, 3)))
        if rotations.size and rotations.shape[-1] >= 4:
            cols.append(rotations[idx, :4])
        else:
            cols.append(np.zeros((tokens_per_object, 4)))
        if velocities.size and velocities.shape[-1] >= 3:
            cols.append(velocities[idx, :3])
        else:
            cols.append(np.zeros((tokens_per_object, 3)))
        merged = np.concatenate(cols, axis=-1).astype(np.float32)
        merged = merged[:, :descriptor_dim]
        desc[i, :, : merged.shape[-1]] = merged
        mask[i] = True
    return desc, mask


@app.command()
def main(
    manifest: Path = typer.Argument(...),
    root: Path = typer.Option(Path("data/")),
    backend: str = typer.Option("pybullet"),
    duration: float = typer.Option(5.0),
    fps: int = typer.Option(24),
    max_objects: int = typer.Option(8),
    tokens_per_object: int = typer.Option(16),
    descriptor_dim: int = typer.Option(10),
    realism_threshold: float = typer.Option(0.0),
    limit: Optional[int] = typer.Option(None),
) -> None:
    configure_root_logging()
    out_dir = root / "physics_keyframes"
    out_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict] = []
    with open(manifest) as f:
        for line in f:
            if not line.strip():
                continue
            entries.append(json.loads(line))

    log.info(f"presim: {len(entries)} entries with backend={backend}")

    updated_lines: list[str] = []
    written = 0
    for i, entry in enumerate(entries):
        if limit is not None and i >= limit:
            updated_lines.append(json.dumps(entry))
            continue
        prompt = entry.get("prompt", "")
        spec = _try_parse_scene(prompt)
        if spec is None:
            updated_lines.append(json.dumps(entry))
            continue
        traj = _try_simulate(spec, backend, duration, fps)
        desc, mask = _trajectories_to_descriptor(traj, max_objects, tokens_per_object, descriptor_dim)
        if not mask.any():
            updated_lines.append(json.dumps(entry))
            continue
        out_rel = Path("physics_keyframes") / f"{entry['id']}.npz"
        np.savez(root / out_rel, descriptor=desc, mask=mask)
        entry["physics_keyframes"] = str(out_rel)
        entry["physics_realism_score"] = float(getattr(traj, "realism_score", 1.0))
        updated_lines.append(json.dumps(entry))
        written += 1
        if written % 25 == 0:
            log.info(f"  presim {written} clips")

    manifest.write_text("\n".join(updated_lines) + "\n")
    log.info(f"done: presim wrote {written} physics descriptor files")


if __name__ == "__main__":
    app()
