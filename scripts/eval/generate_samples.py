"""Generate sample videos from a trained Ogenti checkpoint with full realism controls."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import typer

from ogenti.inference.pipelines.ogenti_pipeline import OgentiPipeline, OgentiPipelineConfig
from ogenti.inference.samplers.flow_euler import FlowEulerConfig
from ogenti.models.backbones.wan22_wrapper import Wan22Backbone, Wan22BackboneConfig
from ogenti.models.ogenti_transformer import OgentiTransformer, OgentiTransformerConfig
from ogenti.modules.conditioning.subject_motion import subject_preset_to_descriptor
from ogenti.modules.conditioning.micro_events import MicroEventConfig, synthesize_all
from ogenti.modules.postprocess.lens_artifacts import lens_preset_to_descriptor
from ogenti.utils.checkpoint import load_meta, load_model_state, strict_load_with_report
from ogenti.utils.logging import configure_root_logging, get_logger

app = typer.Typer()
log = get_logger("ogenti.scripts.eval")


def _save_video(pixels: torch.Tensor, out: Path, fps: int = 24) -> None:
    import imageio.v3 as iio

    video = pixels[0].clamp(-1, 1)
    video = ((video + 1.0) * 127.5).to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()
    out.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(str(out), video, fps=fps, codec="libx264")
    log.info(f"wrote {out}")


def generate_one(
    prompt: str,
    ckpt: Path,
    out: Path,
    steps: int = 50,
    guidance: float = 5.0,
    seed: int = 0,
    *,
    camera_motion: Optional[str] = None,
    subject_motion: Optional[str] = None,
    subject_motion_tracks: int = 1,
    lens_preset: Optional[str] = None,
    film_grain_amount: Optional[float] = None,
    synthesize_micro_events: bool = True,
    physics_scene: Optional[str] = None,
    physics_backend: str = "auto",
    negative_prompt: str = "",
) -> None:
    meta = load_meta(ckpt)
    cfg_dict = meta["config"]
    if isinstance(cfg_dict, dict):
        cfg_dict.pop("_target_", None)
    model_cfg = OgentiTransformerConfig(**cfg_dict) if isinstance(cfg_dict, dict) else cfg_dict

    model = OgentiTransformer(model_cfg)
    state = load_model_state(ckpt)
    strict_load_with_report(model, state, strict=False)

    backbone = Wan22Backbone(Wan22BackboneConfig()).load()

    pipe_cfg = OgentiPipelineConfig(
        sampler=FlowEulerConfig(num_inference_steps=steps, guidance_scale=guidance),
    )
    pipe = OgentiPipeline(model, backbone, pipe_cfg)

    sm_desc: Optional[torch.Tensor] = None
    sm_mask: Optional[torch.Tensor] = None
    if subject_motion is not None:
        d = subject_preset_to_descriptor(subject_motion, repeat_for_tracks=subject_motion_tracks)
        sm_desc = d.unsqueeze(0)
        sm_mask = torch.ones(1, subject_motion_tracks, dtype=torch.bool)
        log.info(f"subject motion preset: {subject_motion} x {subject_motion_tracks} tracks")

    lens_desc: Optional[torch.Tensor] = None
    if lens_preset is not None:
        lens_desc = lens_preset_to_descriptor(lens_preset).unsqueeze(0)
        log.info(f"lens preset: {lens_preset}")

    me_blink = me_breath = me_idle = None
    if synthesize_micro_events:
        signals = synthesize_all(num_frames=pipe_cfg.num_frames, fps=24.0, seed=seed)
        me_blink = signals["blink"].unsqueeze(0)
        me_breath = signals["breath"].unsqueeze(0)
        me_idle = signals["idle"].unsqueeze(0)
        log.info(f"synthesized micro-events: blinks={int(signals['blink'].sum())}")

    physics_scene_arg = physics_scene if physics_scene is not None else (
        prompt if physics_backend != "off" else None
    )

    pixels = pipe.generate(
        prompt=prompt,
        negative_prompt=negative_prompt,
        camera_motion=camera_motion,
        subject_motion=sm_desc,
        subject_motion_mask=sm_mask,
        micro_event_blink=me_blink,
        micro_event_breath=me_breath,
        micro_event_idle=me_idle,
        film_grain_amount=film_grain_amount,
        lens_descriptor=lens_desc,
        physics_scene=physics_scene_arg,
        physics_backend=physics_backend if physics_backend != "off" else "auto",
        seed=seed,
    )
    _save_video(pixels, out)


@app.command()
def main(
    prompt: str = typer.Argument(...),
    ckpt: Path = typer.Option(...),
    out: Path = typer.Option("outputs/samples/gen.mp4"),
    steps: int = typer.Option(50),
    guidance: float = typer.Option(5.0),
    seed: int = typer.Option(0),
    camera_motion: Optional[str] = typer.Option(
        None, "--camera-motion", help="tripod|slider|gimbal|handheld|documentary|drone"
    ),
    subject_motion: Optional[str] = typer.Option(
        None, "--subject-motion", help="calm|walking|running|hesitant|explosive|browsing|dance"
    ),
    subject_motion_tracks: int = typer.Option(1, "--subject-tracks"),
    lens_preset: Optional[str] = typer.Option(
        None, "--lens", help="clean|modern_prime|vintage|anamorphic|documentary"
    ),
    film_grain: Optional[float] = typer.Option(None, "--film-grain", min=0.0, max=1.0),
    no_micro_events: bool = typer.Option(False, "--no-micro-events"),
    physics_scene: Optional[str] = typer.Option(
        None, "--physics-scene", help="Free-form scene description for the simulator (defaults to prompt)."
    ),
    physics_backend: str = typer.Option(
        "auto", "--physics-backend", help="auto|pybullet|genesis|mujoco|brax|off"
    ),
    negative_prompt: str = typer.Option(""),
) -> None:
    configure_root_logging()
    generate_one(
        prompt=prompt, ckpt=ckpt, out=out, steps=steps, guidance=guidance, seed=seed,
        camera_motion=camera_motion,
        subject_motion=subject_motion,
        subject_motion_tracks=subject_motion_tracks,
        lens_preset=lens_preset,
        film_grain_amount=film_grain,
        synthesize_micro_events=not no_micro_events,
        physics_scene=physics_scene,
        physics_backend=physics_backend,
        negative_prompt=negative_prompt,
    )


if __name__ == "__main__":
    app()
