"""Dry-run keymap validation — run BEFORE first retrofit to catch upstream key drift."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from omegaconf import OmegaConf

from ogenti.models.ogenti_transformer import OgentiTransformerConfig
from ogenti.retrofit.surgery.wan22_import import validate_keymap_dry_run
from ogenti.utils.logging import configure_root_logging, get_logger

app = typer.Typer()
log = get_logger("ogenti.scripts.validate_keymap")


@app.command()
def main(
    wan22_weights: Path = typer.Argument(...),
    model_config: Path = typer.Option("ogenti/configs/model/ogenti_5b.yaml"),
    variant: str = typer.Option(None, help="Force variant: official|diffusers"),
    out_report: Path = typer.Option("outputs/logs/keymap_report.json"),
) -> None:
    configure_root_logging()
    raw = OmegaConf.load(str(model_config))
    raw_dict = OmegaConf.to_container(raw, resolve=True, throw_on_missing=False)
    raw_dict.pop("_target_", None)
    cfg = OgentiTransformerConfig(**raw_dict)

    report = validate_keymap_dry_run(wan22_weights, cfg, force_variant=variant)

    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, indent=2))
    log.info(f"report written: {out_report}")

    fatal = len(report["shape_mismatch"]) > 0
    if fatal:
        log.error("SHAPE MISMATCHES FOUND — fix model config dims before training")
        raise typer.Exit(code=2)


if __name__ == "__main__":
    app()
