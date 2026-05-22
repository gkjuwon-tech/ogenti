"""Checkpoint I/O with safetensors backend and structural-retrofit awareness."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from safetensors.torch import load_file, save_file

from ogenti.utils.logging import get_logger

log = get_logger("ogenti.checkpoint")


def save_checkpoint(
    path: Path | str,
    *,
    model_state: dict[str, torch.Tensor],
    optimizer_state: Optional[dict[str, Any]] = None,
    scheduler_state: Optional[dict[str, Any]] = None,
    step: int = 0,
    epoch: int = 0,
    config: Any = None,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    save_file(model_state, str(path / "model.safetensors"))

    aux: dict[str, Any] = {"step": step, "epoch": epoch}
    if optimizer_state is not None:
        torch.save(optimizer_state, path / "optimizer.pt")
    if scheduler_state is not None:
        torch.save(scheduler_state, path / "scheduler.pt")
    if extra is not None:
        aux["extra"] = extra
    if config is not None:
        aux["config"] = asdict(config) if is_dataclass(config) else config

    (path / "meta.json").write_text(json.dumps(aux, indent=2, default=str))
    log.info(f"checkpoint saved: {path} (step={step})")


def load_model_state(path: Path | str) -> dict[str, torch.Tensor]:
    path = Path(path)
    if path.is_dir():
        path = path / "model.safetensors"
    return load_file(str(path))


def load_meta(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    if path.is_dir():
        path = path / "meta.json"
    return json.loads(path.read_text())


def strict_load_with_report(
    model: torch.nn.Module,
    state: dict[str, torch.Tensor],
    strict: bool = False,
) -> tuple[list[str], list[str]]:
    missing, unexpected = model.load_state_dict(state, strict=strict)
    if missing:
        log.warning(f"missing keys: {len(missing)} (showing first 10): {missing[:10]}")
    if unexpected:
        log.warning(f"unexpected keys: {len(unexpected)} (showing first 10): {unexpected[:10]}")
    return missing, unexpected
