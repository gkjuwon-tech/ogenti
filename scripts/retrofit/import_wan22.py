"""Standalone script: import Wan2.2 weights into an Ogenti checkpoint."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import typer
from omegaconf import OmegaConf

from ogenti.models.ogenti_transformer import OgentiTransformerConfig
from ogenti.retrofit.surgery.wan22_import import import_wan22_into_ogenti
from ogenti.utils.checkpoint import save_checkpoint
from ogenti.utils.logging import configure_root_logging, get_logger

app = typer.Typer()
log = get_logger("ogenti.scripts.retrofit")


@app.command()
def main(
    wan22_weights: Path = typer.Argument(...),
    model_config: Path = typer.Option("ogenti/configs/model/ogenti_5b.yaml"),
    output_dir: Path = typer.Option("checkpoints/ogenti_5b_retrofit_init"),
    no_verify: bool = typer.Option(False),
) -> None:
    configure_root_logging()
    raw = OmegaConf.load(str(model_config))
    raw_dict = OmegaConf.to_container(raw, resolve=True, throw_on_missing=False)
    raw_dict.pop("_target_", None)
    cfg = OgentiTransformerConfig(**raw_dict)

    model, report = import_wan22_into_ogenti(
        wan22_weights, cfg, verify_invariant=not no_verify
    )
    save_checkpoint(
        output_dir,
        model_state=model.state_dict(),
        step=0,
        config=asdict(cfg),
        extra={"retrofit_report": {k: len(v) for k, v in report.items()}},
    )
    log.info(f"wrote retrofit checkpoint to {output_dir}")


if __name__ == "__main__":
    app()
