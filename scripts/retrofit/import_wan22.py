"""Standalone script: import Wan2.2 weights into an Ogenti checkpoint.

Defaults target the Wan2.2-T2V-A14B MoE backbone (RFC-0006). To retrofit from
the legacy 5B path instead, pass
``--model-config ogenti/configs/model/ogenti_5b.yaml``.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Optional

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
    wan22_weights: Path = typer.Argument(
        ...,
        help=(
            "Path to a Wan2.2 weights snapshot. For A14B point at the root "
            "containing high_noise_model/ + low_noise_model/."
        ),
    ),
    model_config: Path = typer.Option("ogenti/configs/model/ogenti_a14b.yaml"),
    output_dir: Path = typer.Option("checkpoints/ogenti_a14b_retrofit_init"),
    no_verify: bool = typer.Option(False),
    expert: Optional[str] = typer.Option(
        None,
        "--expert",
        help="Wan2.2-A14B MoE expert: low_noise (default) | high_noise.",
    ),
) -> None:
    configure_root_logging()
    raw = OmegaConf.load(str(model_config))
    raw_dict = OmegaConf.to_container(raw, resolve=True, throw_on_missing=False)
    raw_dict.pop("_target_", None)
    cfg = OgentiTransformerConfig(**raw_dict)

    model, report = import_wan22_into_ogenti(
        wan22_weights, cfg, verify_invariant=not no_verify, expert=expert
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
