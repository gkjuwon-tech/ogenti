"""Rich-backed structured logging with optional wandb/tensorboard sinks."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

from rich.console import Console
from rich.logging import RichHandler

_LOGGER_CACHE: dict[str, logging.Logger] = {}
_CONSOLE = Console(stderr=True)


def get_logger(name: str = "ogenti", level: int = logging.INFO) -> logging.Logger:
    if name in _LOGGER_CACHE:
        return _LOGGER_CACHE[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    handler = RichHandler(
        console=_CONSOLE,
        rich_tracebacks=True,
        tracebacks_show_locals=False,
        show_time=True,
        show_path=False,
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)

    _LOGGER_CACHE[name] = logger
    return logger


class MetricSink:
    """Unified metric sink — fans out to wandb + tensorboard if configured."""

    def __init__(
        self,
        run_dir: Path,
        wandb_project: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        enable_tensorboard: bool = True,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log = get_logger("ogenti.metrics")

        self._wandb = None
        if wandb_project and os.environ.get("WANDB_DISABLED", "").lower() != "true":
            try:
                import wandb

                self._wandb = wandb.init(
                    project=wandb_project,
                    name=wandb_run_name,
                    dir=str(self.run_dir),
                    config={},
                )
            except Exception as e:
                self.log.warning(f"wandb init failed, continuing without: {e}")

        self._tb = None
        if enable_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._tb = SummaryWriter(log_dir=str(self.run_dir / "tb"))
            except Exception as e:
                self.log.warning(f"tensorboard init failed: {e}")

    def log_scalars(self, metrics: dict[str, Any], step: int) -> None:
        if self._wandb is not None:
            self._wandb.log(metrics, step=step)
        if self._tb is not None:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self._tb.add_scalar(k, v, step)

    def log_video(self, tag: str, video: Any, step: int, fps: int = 8) -> None:
        if self._wandb is not None:
            try:
                import wandb

                self._wandb.log({tag: wandb.Video(video, fps=fps, format="mp4")}, step=step)
            except Exception as e:
                self.log.warning(f"wandb video log failed: {e}")
        if self._tb is not None:
            try:
                self._tb.add_video(tag, video, global_step=step, fps=fps)
            except Exception:
                pass

    def close(self) -> None:
        if self._wandb is not None:
            self._wandb.finish()
        if self._tb is not None:
            self._tb.close()


def configure_root_logging(level: str = "INFO") -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=lvl, stream=sys.stderr, format="%(message)s")
    get_logger("ogenti", lvl)
