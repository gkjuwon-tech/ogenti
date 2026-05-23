"""Curriculum driver: run a multi-phase retrofit ladder in one process.

The curriculum YAML (see ``ogenti/configs/training/full_curriculum_a14b.yaml``)
encodes shared defaults at the top level and an ordered ``phases:`` list,
where each phase specifies the fields that change for that stage
(``optim.lr``, ``scheduler.total_steps``, ``loss_weights``,
``retrofit.unfreeze_modules``, optionally ``dataset.manifest_path`` and
``logging.run_dir``).

For each phase the driver:

1. Deep-merges ``curriculum_defaults <- phase_override`` into a flat training
   YAML (the same schema ``scripts/train/train.py:run_training`` expects).
2. Persists the materialized YAML under
   ``<curriculum_run_dir>/_phases/<phase_name>.yaml`` for reproducibility.
3. Invokes ``run_training(materialized_yaml, model_config, init_ckpt=<prev>)``.
4. Picks up the resulting ``final/`` checkpoint as the ``init_ckpt`` for the
   next phase.

The driver is deliberately a thin loop — no training logic lives here. This
keeps the per-stage code path identical to a manual ``ogenti train`` and
makes the curriculum auditable.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

from omegaconf import OmegaConf

from ogenti.utils.logging import configure_root_logging, get_logger
from scripts.train.train import run_training

log = get_logger("ogenti.curriculum")


_PHASE_OVERRIDE_KEYS = {
    "dataset",
    "dataloader",
    "flow",
    "backbone",
    "precision",
    "logging",
    "optim",
    "scheduler",
    "loss_weights",
    "retrofit",
    "seed",
    "stage_name",
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursive dict merge. ``override`` wins on leaf conflicts. Lists are
    replaced wholesale (so a phase that specifies ``unfreeze_modules`` fully
    replaces the inherited list, which is what we want)."""
    out = deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def _materialize_phase(
    defaults: dict[str, Any],
    phase: dict[str, Any],
    curriculum_run_dir: Path,
    phase_index: int,
) -> dict[str, Any]:
    """Build the flat training config dict for a single phase."""
    phase_name = phase.get("name", f"phase{phase_index + 1}")
    phase_override = {k: v for k, v in phase.items() if k in _PHASE_OVERRIDE_KEYS}

    merged = _deep_merge(defaults, phase_override)
    merged["stage_name"] = phase_name

    logging_cfg = merged.setdefault("logging", {})
    logging_cfg["run_dir"] = str(curriculum_run_dir / phase_name)
    return merged


def _persist_phase_yaml(phase_cfg: dict[str, Any], persist_dir: Path, phase_name: str) -> Path:
    persist_dir.mkdir(parents=True, exist_ok=True)
    target = persist_dir / f"{phase_name}.yaml"
    OmegaConf.save(config=OmegaConf.create(phase_cfg), f=str(target))
    return target


def _find_final_checkpoint(run_dir: Path) -> Optional[Path]:
    final = run_dir / "final"
    if final.exists():
        return final
    candidates = sorted(run_dir.glob("ckpt_step_*"), key=lambda p: p.name)
    if candidates:
        return candidates[-1]
    return None


def run_curriculum(
    curriculum_config: Path,
    model_config: Path,
    init_ckpt: Optional[Path] = None,
    only_phase: Optional[str] = None,
    skip_until: Optional[str] = None,
    curriculum_run_dir_override: Optional[Path] = None,
) -> None:
    configure_root_logging()
    log.info(f"loading curriculum from {curriculum_config}")
    full = OmegaConf.to_container(OmegaConf.load(str(curriculum_config)), resolve=True)
    if not isinstance(full, dict):
        raise ValueError("curriculum config must be a mapping at the top level")

    phases = full.pop("phases", None)
    if not isinstance(phases, list) or not phases:
        raise ValueError("curriculum config must define a non-empty `phases:` list")

    defaults = full

    stage_name = defaults.get("stage_name", "curriculum")
    if curriculum_run_dir_override is not None:
        curriculum_run_dir = curriculum_run_dir_override
    else:
        curriculum_run_dir = Path(
            defaults.get("logging", {}).get("run_dir", f"outputs/runs/{stage_name}")
        )
    curriculum_run_dir.mkdir(parents=True, exist_ok=True)

    materialized_dir = curriculum_run_dir / "_phases"
    state_path = curriculum_run_dir / "_state.json"
    state = (
        json.loads(state_path.read_text())
        if state_path.exists()
        else {"completed": [], "last_ckpt": str(init_ckpt) if init_ckpt else None}
    )

    prev_ckpt: Optional[Path] = (
        Path(state["last_ckpt"]) if state.get("last_ckpt") else init_ckpt
    )

    started = skip_until is None
    for idx, phase in enumerate(phases):
        if not isinstance(phase, dict) or "name" not in phase:
            raise ValueError(f"phase #{idx} must be a mapping with a `name` field")
        phase_name = phase["name"]

        if only_phase is not None and phase_name != only_phase:
            continue
        if not started:
            if phase_name == skip_until:
                started = True
            else:
                continue

        if phase_name in state["completed"] and only_phase is None:
            log.info(f"phase {phase_name} already complete (per _state.json) — skipping")
            phase_run_dir = curriculum_run_dir / phase_name
            final = _find_final_checkpoint(phase_run_dir)
            if final is not None:
                prev_ckpt = final
            continue

        log.info(f"=== curriculum phase {idx + 1}/{len(phases)}: {phase_name} ===")
        phase_cfg = _materialize_phase(defaults, phase, curriculum_run_dir, idx)
        phase_yaml = _persist_phase_yaml(phase_cfg, materialized_dir, phase_name)
        log.info(f"materialized phase config -> {phase_yaml}")

        if prev_ckpt is not None:
            log.info(f"init from previous checkpoint: {prev_ckpt}")
        else:
            log.info("no init checkpoint — training from raw model config")

        run_training(
            train_config=phase_yaml,
            model_config=model_config,
            init_ckpt=prev_ckpt,
        )

        phase_run_dir = curriculum_run_dir / phase_name
        final_ckpt = _find_final_checkpoint(phase_run_dir)
        if final_ckpt is None:
            raise RuntimeError(
                f"phase {phase_name} produced no checkpoint in {phase_run_dir}"
            )
        log.info(f"phase {phase_name} final checkpoint: {final_ckpt}")
        prev_ckpt = final_ckpt

        state["completed"].append(phase_name)
        state["last_ckpt"] = str(final_ckpt)
        state_path.write_text(json.dumps(state, indent=2))

        if only_phase is not None:
            break

    log.info("=== curriculum complete ===")
    if prev_ckpt is not None:
        log.info(f"final curriculum checkpoint: {prev_ckpt}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the full Ogenti retrofit curriculum.")
    p.add_argument("curriculum_config", type=Path)
    p.add_argument(
        "--model-config",
        type=Path,
        default=Path("ogenti/configs/model/ogenti_a14b.yaml"),
    )
    p.add_argument("--init-from", type=Path, default=None)
    p.add_argument("--only-phase", type=str, default=None,
                   help="Run a single named phase and exit.")
    p.add_argument("--skip-until", type=str, default=None,
                   help="Skip phases until the named one, then run from there.")
    p.add_argument("--run-dir", type=Path, default=None,
                   help="Override curriculum run directory.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    run_curriculum(
        curriculum_config=args.curriculum_config,
        model_config=args.model_config,
        init_ckpt=args.init_from,
        only_phase=args.only_phase,
        skip_until=args.skip_until,
        curriculum_run_dir_override=args.run_dir,
    )


if __name__ == "__main__":
    main()
