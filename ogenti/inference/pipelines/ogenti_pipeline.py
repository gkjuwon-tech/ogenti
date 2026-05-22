"""End-to-end Ogenti video generation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor

from ogenti.inference.samplers.flow_euler import FlowEulerConfig, FlowEulerSampler
from ogenti.models.backbones.wan22_wrapper import Wan22Backbone
from ogenti.models.ogenti_transformer import OgentiTransformer
from ogenti.modules.conditioning.camera_motion import preset_to_descriptor
from ogenti.physics.scene import SceneSpec
from ogenti.physics.scene_parser import parse_prompt as parse_physics_prompt
from ogenti.physics.simulator import simulate as simulate_physics
from ogenti.utils.logging import get_logger

log = get_logger("ogenti.pipeline")


@dataclass
class OgentiPipelineConfig:
    num_frames: int = 81
    height: int = 480
    width: int = 832
    vae_temporal_compression: int = 4
    vae_spatial_compression: int = 8
    latent_channels: int = 16
    dtype: torch.dtype = torch.bfloat16
    device: str = "cuda"
    sampler: FlowEulerConfig = field(default_factory=FlowEulerConfig)


class OgentiPipeline:
    def __init__(
        self,
        transformer: OgentiTransformer,
        backbone: Wan22Backbone,
        config: OgentiPipelineConfig,
    ) -> None:
        self.transformer = transformer.to(config.device).to(config.dtype).eval()
        self.backbone = backbone
        self.config = config
        self.sampler = FlowEulerSampler(config.sampler)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        glyph_crops: Optional[Tensor] = None,
        glyph_mask: Optional[Tensor] = None,
        camera_motion: Optional[str | Tensor] = None,
        subject_motion: Optional[Tensor] = None,
        subject_motion_mask: Optional[Tensor] = None,
        micro_event_blink: Optional[Tensor] = None,
        micro_event_breath: Optional[Tensor] = None,
        micro_event_idle: Optional[Tensor] = None,
        skin_mask: Optional[Tensor] = None,
        material_masks: Optional[Tensor] = None,
        film_grain_amount: Optional[float] = None,
        lens_descriptor: Optional[Tensor] = None,
        physics_scene: Optional[SceneSpec | str] = None,
        physics_backend: str = "auto",
        seed: Optional[int] = None,
    ) -> Tensor:
        cfg = self.config
        device = cfg.device
        if seed is not None:
            torch.manual_seed(seed)

        log.info(f"encoding prompt (len={len(prompt)})")
        text_emb = self.backbone.encode_text([prompt]).to(device).to(cfg.dtype)
        uncond_emb = self.backbone.encode_text([negative_prompt]).to(device).to(cfg.dtype)

        cm_desc: Optional[Tensor] = None
        if camera_motion is not None:
            if isinstance(camera_motion, str):
                cm_desc = preset_to_descriptor(camera_motion).unsqueeze(0).to(device).to(cfg.dtype)
            else:
                cm_desc = camera_motion.unsqueeze(0).to(device).to(cfg.dtype)
            log.info(f"camera motion conditioning: {camera_motion}")

        t_latent = cfg.num_frames // cfg.vae_temporal_compression + 1
        h_latent = cfg.height // cfg.vae_spatial_compression
        w_latent = cfg.width // cfg.vae_spatial_compression
        shape = (1, cfg.latent_channels, t_latent, h_latent, w_latent)

        if glyph_crops is not None:
            glyph_crops = glyph_crops.to(device).to(cfg.dtype)
            glyph_mask = glyph_mask.to(device)

        if subject_motion is not None: subject_motion = subject_motion.to(device).to(cfg.dtype)
        if subject_motion_mask is not None: subject_motion_mask = subject_motion_mask.to(device)
        if micro_event_blink is not None: micro_event_blink = micro_event_blink.to(device).to(cfg.dtype)
        if micro_event_breath is not None: micro_event_breath = micro_event_breath.to(device).to(cfg.dtype)
        if micro_event_idle is not None: micro_event_idle = micro_event_idle.to(device).to(cfg.dtype)

        pk_desc: Optional[Tensor] = None
        pk_mask: Optional[Tensor] = None
        if physics_scene is not None:
            if isinstance(physics_scene, str):
                log.info(f"parsing physics scene from prompt source: {physics_scene[:60]}")
                spec = parse_physics_prompt(physics_scene)
            else:
                spec = physics_scene
            log.info(f"simulating scene: {len(spec.objects)} objects @ {spec.fps}fps")
            traj = simulate_physics(spec, backend=physics_backend)
            desc_np, mask_np = traj.stack_descriptor(max_objects=8)
            pk_desc = torch.from_numpy(desc_np).unsqueeze(0).to(device).to(cfg.dtype)
            pk_mask = torch.from_numpy(mask_np).unsqueeze(0).to(device)
            log.info(f"physics keyframes prepared: shape={tuple(pk_desc.shape)} backend={traj.backend}")

        def v_cond(x: Tensor, t: Tensor) -> Tensor:
            return self.transformer(
                x, t, text_emb,
                glyph_crops=glyph_crops,
                glyph_mask=glyph_mask,
                camera_motion_descriptor=cm_desc,
                subject_motion_descriptors=subject_motion,
                subject_motion_mask=subject_motion_mask,
                micro_event_blink=micro_event_blink,
                micro_event_breath=micro_event_breath,
                micro_event_idle=micro_event_idle,
                physics_keyframe_descriptor=pk_desc,
                physics_keyframe_object_mask=pk_mask,
            )

        def v_uncond(x: Tensor, t: Tensor) -> Tensor:
            return self.transformer(x, t, uncond_emb)

        log.info(f"sampling {cfg.sampler.num_inference_steps} steps, shape={shape}")
        latents = self.sampler.sample(
            velocity_fn=v_cond,
            velocity_fn_uncond=v_uncond,
            latents_shape=shape,
            device=device,
            dtype=cfg.dtype,
        )

        log.info("decoding latents -> pixels")
        pixels = self.backbone.decode_latents(latents.float()).clamp(-1, 1)

        log.info("applying post-VAE refinement chain")
        if skin_mask is not None:
            skin_mask = skin_mask.to(device).to(pixels.dtype)
        if material_masks is not None:
            material_masks = material_masks.to(device).to(pixels.dtype)
        if lens_descriptor is not None:
            lens_descriptor = lens_descriptor.to(device).to(pixels.dtype)

        pixels = self.transformer.apply_full_postprocess(
            pixels,
            skin_mask=skin_mask,
            material_masks=material_masks,
            film_grain_amount=film_grain_amount,
            lens_descriptor=lens_descriptor,
        )

        return pixels
