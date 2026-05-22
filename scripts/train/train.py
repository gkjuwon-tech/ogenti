"""Main training loop. Multi-GPU via HuggingFace Accelerate."""

from __future__ import annotations

import fnmatch
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from ogenti.data.loaders.video_dataset import AdVideoDataset, VideoDatasetConfig, collate_fn
from ogenti.models.backbones.wan22_wrapper import Wan22Backbone, Wan22BackboneConfig
from ogenti.models.ogenti_transformer import OgentiTransformer, OgentiTransformerConfig
from ogenti.training.losses.diffusion import flow_matching_loss
from ogenti.training.losses.identity import IdentityLossConfig, compute_identity_loss
from ogenti.training.losses.anatomy import AnatomyLossConfig, AnatomyLoss
from ogenti.training.losses.ocr import OcrLossConfig, GlyphHeadLoss
from ogenti.training.losses.frequency import FrequencyLossConfig, skin_aware_frequency_loss
from ogenti.training.losses.temporal_realism import TemporalRealismConfig, temporal_realism_loss
from ogenti.training.losses.physics import PhysicsConfig, compute_physics_loss
from ogenti.training.losses.physics_keyframe import (
    PhysicsKeyframeLossConfig,
    compute_physics_keyframe_loss,
)
from ogenti.training.losses.motion_realism import MotionRealismConfig, compute_motion_realism_loss
from ogenti.training.losses.trajectory_tracker import TrackerConfig, boxes_from_trajectories, track_objects
from ogenti.training.schedulers.flow_matching import FlowMatchingConfig, FlowMatchingScheduler
from ogenti.utils.checkpoint import load_model_state, save_checkpoint, strict_load_with_report
from ogenti.utils.logging import MetricSink, configure_root_logging, get_logger

log = get_logger("ogenti.train")


def _apply_freeze(model: OgentiTransformer, unfreeze_patterns: list[str]) -> None:
    for n, p in model.named_parameters():
        p.requires_grad_(False)
        for pat in unfreeze_patterns:
            if fnmatch.fnmatch(n, pat) or pat in n:
                p.requires_grad_(True)
                break
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info(f"trainable: {trainable / 1e6:.1f}M / {total / 1e6:.1f}M total")


def _build_optimizer(model: OgentiTransformer, cfg: dict) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(
        params,
        lr=cfg["lr"],
        betas=tuple(cfg["betas"]),
        weight_decay=cfg["weight_decay"],
    )


def _build_lr_schedule(opt: torch.optim.Optimizer, cfg: dict) -> torch.optim.lr_scheduler.LambdaLR:
    warmup = cfg["warmup_steps"]
    total = cfg["total_steps"]
    min_lr = cfg["min_lr"]
    base_lr = opt.param_groups[0]["lr"]

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        cos = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159265)).item())
        return max(min_lr / base_lr, cos)

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def run_training(
    train_config: Path,
    model_config: Path,
    init_ckpt: Optional[Path] = None,
) -> None:
    from accelerate import Accelerator
    from accelerate.utils import ProjectConfiguration

    configure_root_logging()
    tcfg = OmegaConf.load(str(train_config))
    mcfg_raw = OmegaConf.load(str(model_config))

    torch.manual_seed(int(tcfg.seed))

    run_dir = Path(tcfg.logging.run_dir)
    proj_cfg = ProjectConfiguration(project_dir=str(run_dir), automatic_checkpoint_naming=False)

    grad_accum = int(tcfg.dataloader.get("gradient_accumulation_steps", 1))
    mixed_precision = "bf16" if tcfg.precision.dtype == "bfloat16" else "fp16"

    accelerator = Accelerator(
        gradient_accumulation_steps=grad_accum,
        mixed_precision=mixed_precision,
        project_config=proj_cfg,
        log_with=None,
    )

    is_main = accelerator.is_main_process
    if is_main:
        log.info(f"=== distributed init: world_size={accelerator.num_processes}, "
                 f"mixed_precision={mixed_precision}, grad_accum={grad_accum} ===")

    model_dict = OmegaConf.to_container(mcfg_raw, resolve=True, throw_on_missing=False)
    model_dict.pop("_target_", None)
    model_cfg = OgentiTransformerConfig(**model_dict)
    model = OgentiTransformer(model_cfg)

    if init_ckpt is not None:
        if is_main:
            log.info(f"loading init checkpoint from {init_ckpt}")
        state = load_model_state(init_ckpt)
        strict_load_with_report(model, state, strict=False)

    if tcfg.retrofit.freeze_backbone_weights:
        _apply_freeze(model, list(tcfg.retrofit.unfreeze_modules))

    if tcfg.precision.gradient_checkpointing:
        model.enable_gradient_checkpointing(True)

    backbone = Wan22Backbone(
        Wan22BackboneConfig(
            device=str(accelerator.device),
            vae_dtype=torch.float32,
            text_dtype=torch.bfloat16 if mixed_precision == "bf16" else torch.float16,
        )
    )
    backbone.load()
    backbone.vae.to(accelerator.device)
    backbone.text_encoder.to(accelerator.device)

    ds_dict = OmegaConf.to_container(tcfg.dataset, resolve=True, throw_on_missing=False)
    ds_dict.pop("_target_", None)
    ds_cfg = VideoDatasetConfig(**ds_dict)
    dataset = AdVideoDataset(ds_cfg)
    loader = DataLoader(
        dataset,
        batch_size=tcfg.dataloader.batch_size,
        shuffle=True,
        num_workers=tcfg.dataloader.num_workers,
        pin_memory=tcfg.dataloader.pin_memory,
        collate_fn=collate_fn,
        drop_last=True,
    )

    sched = FlowMatchingScheduler(
        FlowMatchingConfig(
            shift=float(tcfg.flow.shift),
            logit_mean=float(tcfg.flow.logit_mean),
            logit_std=float(tcfg.flow.logit_std),
        )
    )

    opt = _build_optimizer(model, dict(tcfg.optim))
    lr_sched = _build_lr_schedule(opt, dict(tcfg.scheduler))

    model, opt, loader, lr_sched = accelerator.prepare(model, opt, loader, lr_sched)

    sink = None
    if is_main:
        sink = MetricSink(
            run_dir=run_dir,
            wandb_project=tcfg.logging.get("wandb_project"),
        )

    # Initialize stateful loss modules
    anatomy_loss_fn = AnatomyLoss(AnatomyLossConfig()).to(accelerator.device)
    ocr_loss_fn = GlyphHeadLoss(OcrLossConfig(), num_classes=256) # dummy vocab size


    step = 0
    total_steps = int(tcfg.scheduler.total_steps)
    if is_main:
        log.info(f"=== starting training: {total_steps} steps ===")

    model.train()
    inner_model = accelerator.unwrap_model(model)

    while step < total_steps:
        for batch in loader:
            if step >= total_steps:
                break

            with accelerator.accumulate(model):
                video = batch["video"].to(accelerator.device, dtype=torch.float32)
                with torch.no_grad():
                    latents = backbone.encode_video(video)
                    text_emb = backbone.encode_text(list(batch["prompt"]))

                glyph_crops = batch.get("glyph_crops")
                glyph_mask = batch.get("glyph_mask")
                if glyph_crops is not None:
                    glyph_crops = glyph_crops.to(accelerator.device)
                    glyph_mask = glyph_mask.to(accelerator.device)

                cm_desc = batch.get("camera_motion_descriptor")
                if cm_desc is not None:
                    cm_desc = cm_desc.to(accelerator.device)
                
                sm_desc = batch.get("subject_motion_descriptors")
                sm_mask = batch.get("subject_motion_mask")
                if sm_desc is not None:
                    sm_desc = sm_desc.to(accelerator.device)
                    sm_mask = sm_mask.to(accelerator.device)

                me_blink = batch.get("micro_event_blink")
                me_breath = batch.get("micro_event_breath")
                me_idle = batch.get("micro_event_idle")
                if me_blink is not None: me_blink = me_blink.to(accelerator.device)
                if me_breath is not None: me_breath = me_breath.to(accelerator.device)
                if me_idle is not None: me_idle = me_idle.to(accelerator.device)

                pk_desc = batch.get("physics_keyframe_descriptor")
                pk_mask = batch.get("physics_keyframe_object_mask")
                pk_realism = batch.get("physics_realism_score")
                if pk_desc is not None:
                    pk_desc = pk_desc.to(accelerator.device)
                    pk_mask = pk_mask.to(accelerator.device) if pk_mask is not None else None

                noise = torch.randn_like(latents)
                t = sched.sample_timesteps(latents.shape[0], device=accelerator.device, dtype=torch.float32)
                x_t = sched.add_noise(latents.float(), noise.float(), t)
                v_target = sched.velocity_target(latents.float(), noise.float())

                v_pred = model(
                    x_t, sched.to_model_timesteps(t), text_emb,
                    glyph_crops=glyph_crops, glyph_mask=glyph_mask,
                    camera_motion_descriptor=cm_desc,
                    subject_motion_descriptors=sm_desc,
                    subject_motion_mask=sm_mask,
                    micro_event_blink=me_blink,
                    micro_event_breath=me_breath,
                    micro_event_idle=me_idle,
                    physics_keyframe_descriptor=pk_desc,
                    physics_keyframe_object_mask=pk_mask,
                )
                loss_diff = flow_matching_loss(v_pred.float(), v_target.float())

                loss_total = float(tcfg.loss_weights.diffusion) * loss_diff
                metrics = {"loss/diffusion": loss_diff.item()}

                if float(tcfg.loss_weights.identity) > 0 and inner_model.entity_bank is not None:
                    id_out = compute_identity_loss(
                        slots=inner_model.entity_bank.prototypes.unsqueeze(0),
                        attn_maps=None,
                        config=IdentityLossConfig(),
                    )
                    loss_total = loss_total + float(tcfg.loss_weights.identity) * id_out["id_total"]
                    metrics["loss/identity"] = id_out["id_total"].item()

                # ── Anatomy & OCR (Requires specific targets) ──
                if float(tcfg.loss_weights.get("anatomy", 0.0)) > 0 and "keypoints" in batch and step % 4 == 0:
                    # Approximate decoded pred for pixel-space losses (every 4 steps to save memory)
                    t_expanded = t.view(-1, 1, 1, 1, 1)
                    pred_latents = x_t + v_pred * (1.0 - t_expanded)
                    pixels_pred = backbone.decode_latents(pred_latents.to(backbone.vae.dtype)).float()
                    
                    kp = batch["keypoints"].to(accelerator.device)
                    kp_conf = batch["keypoint_conf"].to(accelerator.device)
                    anat_out = anatomy_loss_fn(pixels_pred, kp, kp_conf)
                    loss_total = loss_total + float(tcfg.loss_weights.anatomy) * anat_out["anatomy_total"]
                    metrics["loss/anatomy"] = anat_out["anatomy_total"].item()

                    # Frequency and Temporal Realism also use pixels_pred
                    if float(tcfg.loss_weights.get("frequency_skin", 0.0)) > 0 and "skin_mask" in batch:
                        skin_mask = batch["skin_mask"].to(accelerator.device)
                        freq_loss = skin_aware_frequency_loss(pixels_pred, video, skin_mask, FrequencyLossConfig())
                        loss_total = loss_total + float(tcfg.loss_weights.frequency_skin) * freq_loss
                        metrics["loss/frequency_skin"] = freq_loss.item()
                    
                    if float(tcfg.loss_weights.get("temporal_realism", 0.0)) > 0:
                        temp_out = temporal_realism_loss(pixels_pred, video, TemporalRealismConfig())
                        loss_total = loss_total + float(tcfg.loss_weights.temporal_realism) * temp_out["temporal_total"]
                        metrics["loss/temporal_realism"] = temp_out["temporal_total"].item()

                if float(tcfg.loss_weights.get("ocr", 0.0)) > 0 and "glyph_texts" in batch:
                    # Wiring placeholder: glyph branch should return char logits
                    # Since it doesn't yet, we just add a 0.0 loss to ensure the config path is active.
                    metrics["loss/ocr"] = 0.0
                
                # ── Physics & Motion Realism ──
                # Both losses require predicted trajectories. We derive them from the
                # decoded prediction via a differentiable centroid tracker (see
                # trajectory_tracker.py). pixels_pred is only computed when the anatomy
                # branch already decoded; we reuse it to avoid a second VAE pass.
                needs_physics = (
                    float(tcfg.loss_weights.get("physics", 0.0)) > 0
                    or float(tcfg.loss_weights.get("motion_realism", 0.0)) > 0
                ) and "object_trajectories" in batch

                if needs_physics:
                    gt_traj = batch["object_trajectories"].to(accelerator.device)
                    track_mask = batch.get("object_track_mask")
                    if track_mask is not None:
                        track_mask = track_mask.to(accelerator.device)
                    else:
                        track_mask = torch.ones(gt_traj.shape[:2], dtype=torch.bool, device=accelerator.device)

                    if "pixels_pred" not in locals() or pixels_pred is None:
                        t_expanded = t.view(-1, 1, 1, 1, 1)
                        pred_latents = x_t + v_pred * (1.0 - t_expanded)
                        pixels_pred = backbone.decode_latents(pred_latents.to(backbone.vae.dtype)).float()

                    _, _, _, ph, pw = pixels_pred.shape
                    init_boxes_norm = boxes_from_trajectories(gt_traj, default_box_size=0.1)
                    scale = torch.tensor([ph, pw, ph, pw], device=accelerator.device, dtype=init_boxes_norm.dtype)
                    init_boxes_px = init_boxes_norm * scale.view(1, 1, 4)

                    pred_traj = track_objects(pixels_pred, init_boxes_px, track_mask, TrackerConfig())

                    if float(tcfg.loss_weights.get("physics", 0.0)) > 0:
                        phys_out = compute_physics_loss(pred_traj, gt_traj, PhysicsConfig())
                        loss_total = loss_total + float(tcfg.loss_weights.physics) * phys_out["physics_total"]
                        metrics["loss/physics"] = phys_out["physics_total"].item()

                    if float(tcfg.loss_weights.get("motion_realism", 0.0)) > 0:
                        mr_out = compute_motion_realism_loss(pred_traj, gt_traj, MotionRealismConfig(), track_mask)
                        loss_total = loss_total + float(tcfg.loss_weights.motion_realism) * mr_out["motion_realism_total"]
                        metrics["loss/motion_realism"] = mr_out["motion_realism_total"].item()

                    if (
                        float(tcfg.loss_weights.get("physics_keyframe", 0.0)) > 0
                        and pk_desc is not None
                    ):
                        sim_pos = pk_desc[..., :3]
                        sim_xy = sim_pos[..., :2]
                        sim_xy_norm = sim_xy - sim_xy[:, :, :1]
                        scale = (sim_xy_norm.abs().amax(dim=(1, 2, 3), keepdim=True).clamp(min=1e-6))
                        sim_xy_norm = sim_xy_norm / scale * 0.5 + 0.5

                        k = min(pred_traj.shape[1], sim_xy_norm.shape[1])
                        t_min = min(pred_traj.shape[2], sim_xy_norm.shape[2])
                        weight = (pk_realism.to(accelerator.device).view(-1, 1, 1, 1)
                                  if pk_realism is not None else 1.0)

                        pkf_mask = pk_mask[:, :k] if pk_mask is not None else None
                        kf_out = compute_physics_keyframe_loss(
                            pred_traj[:, :k, :t_min] * weight,
                            sim_xy_norm[:, :k, :t_min] * weight,
                            PhysicsKeyframeLossConfig(),
                            object_mask=pkf_mask,
                        )
                        loss_total = loss_total + float(tcfg.loss_weights.physics_keyframe) * kf_out["physics_keyframe_total"]
                        metrics["loss/physics_keyframe"] = kf_out["physics_keyframe_total"].item()

                pixels_pred = None  # release for next step

                accelerator.backward(loss_total)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        float(tcfg.optim.grad_clip),
                    )
                opt.step()
                lr_sched.step()
                opt.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                metrics["loss/total"] = loss_total.item()
                metrics["lr"] = opt.param_groups[0]["lr"]

                if is_main and step % int(tcfg.logging.log_every) == 0:
                    log.info(f"step {step}/{total_steps} loss={loss_total.item():.4f}")
                    sink.log_scalars(metrics, step)

                if is_main and step > 0 and step % int(tcfg.logging.ckpt_every) == 0:
                    accelerator.wait_for_everyone()
                    unwrapped = accelerator.unwrap_model(model)
                    save_checkpoint(
                        run_dir / f"ckpt_step_{step:07d}",
                        model_state=unwrapped.state_dict(),
                        optimizer_state=opt.state_dict(),
                        step=step,
                        config=asdict(model_cfg),
                    )
                step += 1

    accelerator.wait_for_everyone()
    if is_main:
        unwrapped = accelerator.unwrap_model(model)
        save_checkpoint(
            run_dir / "final",
            model_state=unwrapped.state_dict(),
            step=step,
            config=asdict(model_cfg),
        )
        sink.close()
        log.info("training complete")
