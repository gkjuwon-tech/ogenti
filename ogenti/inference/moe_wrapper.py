"""Dual-expert MoE wrapper for Wan2.2-A14B retrofits.

Wan2.2-A14B ships two transformer weight sets (``high_noise_model`` and
``low_noise_model``) that share architecture but specialize on different
regions of the rectified-flow trajectory. Upstream Wan2.2 selects which
expert to evaluate at each sampling step based on a noise-level threshold.

After surgery, each expert becomes its own ``OgentiTransformer`` instance
(via ``import_wan22_into_ogenti(..., expert=...)``). At inference, this
wrapper holds both and forwards the call to whichever expert is active
for the current timestep, presenting the standard
``OgentiTransformer.forward`` signature so existing samplers
(``FlowEulerSampler``) don't need to know about MoE at all.

The default boundary is timestep ``0.5`` (in flow time, where 1.0 is pure
noise and 0.0 is clean). For timesteps ``t >= 0.5`` (early/noisy steps),
the ``high_noise`` expert is used; for ``t < 0.5`` (late/clean steps),
the ``low_noise`` expert is used. This matches the upstream Wan2.2-A14B
inference recipe.

Memory note: holding both 14B experts in GPU memory is ~56 GB in bf16. On
single-A100-80GB setups, run with ``offload_inactive_expert=True`` to keep
the non-active expert pinned in host memory and swap on the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from torch import Tensor, nn

from ogenti.models.ogenti_transformer import OgentiTransformer
from ogenti.utils.logging import get_logger

log = get_logger("ogenti.inference.moe")

ExpertRouting = Literal["timestep_threshold", "high_only", "low_only"]


@dataclass
class A14BMoEConfig:
    """Configuration for the dual-expert A14B inference wrapper."""

    # Timestep threshold in flow time. t in [0, 1] where 1 is pure noise.
    boundary_t: float = 0.5
    routing: ExpertRouting = "timestep_threshold"
    # If True, keep the inactive expert pinned in host memory and move it to
    # the device only at the boundary crossing. Halves peak GPU usage.
    offload_inactive_expert: bool = False
    # Internal: timesteps are passed in 0..1000 range (see flow_euler sampler).
    timestep_scale: float = 1000.0


class A14BMoEWrapper(nn.Module):
    """Holds two OgentiTransformer experts and routes forward() by timestep.

    Drop-in replacement for a single OgentiTransformer in
    :class:`ogenti.inference.pipelines.ogenti_pipeline.OgentiPipeline`.
    """

    def __init__(
        self,
        high_noise: OgentiTransformer,
        low_noise: OgentiTransformer,
        config: Optional[A14BMoEConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config or A14BMoEConfig()
        # Register as submodules so .to(device) / state_dict cover both.
        self.high_noise = high_noise
        self.low_noise = low_noise
        self._active: Optional[str] = None

    # The OgentiTransformer config is shared between experts (same arch).
    @property
    def shared_config(self):
        return self.low_noise.config

    def _select_expert(self, timesteps: Tensor) -> OgentiTransformer:
        if self.config.routing == "high_only":
            return self.high_noise
        if self.config.routing == "low_only":
            return self.low_noise

        # timestep_threshold: pick by mean batch timestep (samplers pass
        # a 1-D batch tensor; in practice all entries share the same t)
        t = float(timesteps.float().mean().item()) / self.config.timestep_scale
        return self.high_noise if t >= self.config.boundary_t else self.low_noise

    def _maybe_swap_device(self, target_name: str) -> None:
        if not self.config.offload_inactive_expert:
            return
        if self._active == target_name:
            return
        device = self.low_noise.patch_embed.proj.weight.device
        if device.type != "cuda":
            self._active = target_name
            return

        if target_name == "high_noise":
            self.high_noise.to(device)
            self.low_noise.to("cpu", non_blocking=True)
        else:
            self.low_noise.to(device)
            self.high_noise.to("cpu", non_blocking=True)
        self._active = target_name
        log.debug(f"swapped active expert -> {target_name}")

    def forward(
        self,
        latents: Tensor,
        timesteps: Tensor,
        text_embeddings: Tensor,
        **kwargs,
    ) -> Tensor:
        expert = self._select_expert(timesteps)
        name = "high_noise" if expert is self.high_noise else "low_noise"
        self._maybe_swap_device(name)
        return expert(latents, timesteps, text_embeddings, **kwargs)

    # ---- delegated post-VAE pixel-space refinement helpers ----
    # The pixel-space refinement heads (skin/material/grain/lens) live on the
    # OgentiTransformer instance, but they are identical in both experts after
    # a same-config retrofit. We delegate to ``low_noise`` since it dominates
    # final-step quality.
    def refine_skin_detail(self, *args, **kwargs):
        return self.low_noise.refine_skin_detail(*args, **kwargs)

    def refine_material_detail(self, *args, **kwargs):
        return self.low_noise.refine_material_detail(*args, **kwargs)

    def apply_motion_blur(self, *args, **kwargs):
        return self.low_noise.apply_motion_blur(*args, **kwargs)

    def apply_film_grain(self, *args, **kwargs):
        return self.low_noise.apply_film_grain(*args, **kwargs)

    def apply_lens_artifacts(self, *args, **kwargs):
        return self.low_noise.apply_lens_artifacts(*args, **kwargs)


def load_a14b_moe_from_retrofit_dirs(
    high_noise_ckpt: str,
    low_noise_ckpt: str,
    *,
    moe_config: Optional[A14BMoEConfig] = None,
) -> A14BMoEWrapper:
    """Convenience: load two retrofit checkpoints (one per expert) and wrap them.

    Each ckpt must have been produced by
    ``ogenti retrofit <wan22_a14b_dir> --expert high_noise|low_noise --out <path>``.
    """
    from ogenti.models.ogenti_transformer import OgentiTransformer, OgentiTransformerConfig
    from ogenti.utils.checkpoint import load_meta, load_model_state, strict_load_with_report

    def _load_one(path: str) -> OgentiTransformer:
        meta = load_meta(path)
        cfg_dict = meta["config"]
        if isinstance(cfg_dict, dict):
            cfg_dict.pop("_target_", None)
            cfg = OgentiTransformerConfig(**cfg_dict)
        else:
            cfg = cfg_dict
        model = OgentiTransformer(cfg)
        strict_load_with_report(model, load_model_state(path), strict=False)
        return model

    hi = _load_one(high_noise_ckpt)
    lo = _load_one(low_noise_ckpt)
    return A14BMoEWrapper(hi, lo, config=moe_config)
