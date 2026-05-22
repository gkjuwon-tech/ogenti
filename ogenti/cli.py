"""Ogenti command-line interface."""

from __future__ import annotations

from pathlib import Path
from typing import Optional  # noqa: F401  (used in optional typer args below)

import typer

from ogenti.utils.logging import configure_root_logging, get_logger

app = typer.Typer(no_args_is_help=True, help="Ogenti — ad-grade video generation.")
log = get_logger("ogenti.cli")


@app.command()
def retrofit(
    wan22_weights: Path = typer.Argument(..., help="Path to Wan2.2-TI2V-5B weights dir or .safetensors."),
    model_config: Path = typer.Option("ogenti/configs/model/ogenti_5b.yaml", "--config"),
    output_dir: Path = typer.Option("checkpoints/ogenti_5b_retrofit_init", "--out"),
    verify: bool = typer.Option(True, "--verify/--no-verify"),
) -> None:
    """Import a pretrained Wan2.2 transformer into an OgentiTransformer."""
    from omegaconf import OmegaConf

    from ogenti.models.ogenti_transformer import OgentiTransformerConfig
    from ogenti.retrofit.adapters.dit_to_ogenti import BackboneFamily, retrofit_from_backbone
    from ogenti.utils.checkpoint import save_checkpoint

    configure_root_logging()
    raw = OmegaConf.load(str(model_config))
    if "_target_" in raw:
        del raw["_target_"]
    schema = OmegaConf.structured(OgentiTransformerConfig)
    merged = OmegaConf.merge(schema, raw)
    cfg = OmegaConf.to_object(merged)

    model, report = retrofit_from_backbone(
        BackboneFamily.WAN22, wan22_weights, cfg, verify_invariant=verify
    )
    log.info(f"retrofit done: copied={len(report['copied'])} keys")

    save_checkpoint(
        output_dir,
        model_state=model.state_dict(),
        step=0,
        config=cfg,
        extra={"retrofit_report": report},
    )


@app.command()
def train(
    train_config: Path = typer.Argument(..., help="Path to training config YAML."),
    model_config: Path = typer.Option("ogenti/configs/model/ogenti_5b.yaml", "--model-config"),
    init_ckpt: Optional[Path] = typer.Option(None, "--init-from"),
) -> None:
    """Run a training stage."""
    configure_root_logging()
    from scripts.train.train import run_training

    run_training(train_config=train_config, model_config=model_config, init_ckpt=init_ckpt)


@app.command()
def generate(
    prompt: str = typer.Argument(...),
    ckpt: Path = typer.Option(...),
    out: Path = typer.Option("outputs/samples/gen.mp4", "--out"),
    steps: int = typer.Option(50),
    guidance: float = typer.Option(5.0),
    seed: int = typer.Option(0),
    camera_motion: Optional[str] = typer.Option(
        None, "--camera-motion",
        help="Camera preset: tripod|slider|gimbal|handheld|documentary|drone",
    ),
    subject_motion: Optional[str] = typer.Option(
        None, "--subject-motion",
        help="Subject preset: calm|walking|running|hesitant|explosive|browsing|dance",
    ),
    subject_tracks: int = typer.Option(1, "--subject-tracks"),
    lens_preset: Optional[str] = typer.Option(
        None, "--lens",
        help="Lens preset: clean|modern_prime|vintage|anamorphic|documentary",
    ),
    film_grain: Optional[float] = typer.Option(None, "--film-grain", min=0.0, max=1.0),
    no_micro_events: bool = typer.Option(False, "--no-micro-events"),
    physics_scene: Optional[str] = typer.Option(
        None, "--physics-scene",
        help="Free-form physics scene description; defaults to the prompt.",
    ),
    physics_backend: str = typer.Option(
        "auto", "--physics-backend",
        help="auto|pybullet|genesis|mujoco|brax|off (disable simulation)",
    ),
    negative_prompt: str = typer.Option("", "--negative-prompt"),
) -> None:
    """Generate a video from a prompt with full realism controls (incl. physics sim)."""
    configure_root_logging()
    from scripts.eval.generate_samples import generate_one

    generate_one(
        prompt=prompt, ckpt=ckpt, out=out, steps=steps, guidance=guidance, seed=seed,
        camera_motion=camera_motion,
        subject_motion=subject_motion,
        subject_motion_tracks=subject_tracks,
        lens_preset=lens_preset,
        film_grain_amount=film_grain,
        synthesize_micro_events=not no_micro_events,
        physics_scene=physics_scene,
        physics_backend=physics_backend,
        negative_prompt=negative_prompt,
    )


if __name__ == "__main__":
    app()
