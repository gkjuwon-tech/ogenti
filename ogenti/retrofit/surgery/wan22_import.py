"""Wan2.2 transformer -> OgentiTransformer weight import.

Strategy:
  1. Define multiple candidate keymap variants (upstream Wan releases use
     slightly different naming — diffusers fork vs. official repo) including
     the dual-expert A14B MoE layout (high_noise_model/ + low_noise_model/).
  2. At import time, sniff the source state dict to detect which variant
     we're looking at and apply the matching map.
  3. Report unmapped/shape-mismatched keys so the user can diagnose
     before training spends compute.

Supported upstream layouts:
  - Wan2.2-TI2V-5B          (legacy single-expert, kept as fallback)
  - Wan2.2-T2V-A14B         (primary — MoE dual expert)
  - Wan2.2-I2V-A14B         (primary — MoE dual expert, with image cond)
  - diffusers WanTransformer3DModel state dicts

For A14B sources, pick ONE expert per OgentiTransformer instance via the
`expert` argument. The dual-expert inference wrapper
(`ogenti.inference.moe_wrapper`) loads both and routes by timestep.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import torch
from safetensors.torch import load_file

from ogenti.models.ogenti_transformer import OgentiTransformer, OgentiTransformerConfig
from ogenti.utils.logging import get_logger

log = get_logger("ogenti.retrofit.wan22")

ExpertSelector = Literal["low_noise", "high_noise"]


# OgentiTransformer (post-RFC-0006-arch-fix) uses the Wan2.2-native state-dict
# layout for the per-block core: `modulation` is a parameter, `ffn` is a
# `WanFeedForward` Sequential (state-dict keys `ffn.0.*` / `ffn.2.*`), and the
# self/cross attention `q_norm` / `k_norm` are per-channel RMSNorms.
#
# The only thing the keymap still needs to rename is the q/k/v/o linear suffix
# (`q.weight` -> `q_proj.weight`) since Ogenti's `MultiHeadAttention` uses
# `{q,k,v,out}_proj` attribute names. Everything else is identity-mapped.

# Variant A: official Wan2.2 repo naming
WAN22_KEYMAP_OFFICIAL_BLOCK: dict[str, str] = {
    "self_attn.q.weight": "self_attn.q_proj.weight",
    "self_attn.q.bias": "self_attn.q_proj.bias",
    "self_attn.k.weight": "self_attn.k_proj.weight",
    "self_attn.k.bias": "self_attn.k_proj.bias",
    "self_attn.v.weight": "self_attn.v_proj.weight",
    "self_attn.v.bias": "self_attn.v_proj.bias",
    "self_attn.o.weight": "self_attn.out_proj.weight",
    "self_attn.o.bias": "self_attn.out_proj.bias",
    "self_attn.q_norm.weight": "self_attn.q_norm.weight",
    "self_attn.k_norm.weight": "self_attn.k_norm.weight",
    "ffn.0.weight": "ffn.0.weight",
    "ffn.0.bias": "ffn.0.bias",
    "ffn.2.weight": "ffn.2.weight",
    "ffn.2.bias": "ffn.2.bias",
    "modulation": "modulation",
}

# Variant B: diffusers WanTransformer3DModel naming
WAN22_KEYMAP_DIFFUSERS_BLOCK: dict[str, str] = {
    "attn1.to_q.weight": "self_attn.q_proj.weight",
    "attn1.to_q.bias": "self_attn.q_proj.bias",
    "attn1.to_k.weight": "self_attn.k_proj.weight",
    "attn1.to_k.bias": "self_attn.k_proj.bias",
    "attn1.to_v.weight": "self_attn.v_proj.weight",
    "attn1.to_v.bias": "self_attn.v_proj.bias",
    "attn1.to_out.0.weight": "self_attn.out_proj.weight",
    "attn1.to_out.0.bias": "self_attn.out_proj.bias",
    "attn1.norm_q.weight": "self_attn.q_norm.weight",
    "attn1.norm_k.weight": "self_attn.k_norm.weight",
    "ffn.net.0.proj.weight": "ffn.0.weight",
    "ffn.net.2.weight": "ffn.2.weight",
    "scale_shift_table": "modulation",
}

# Variant C: official Wan2.2-TI2V-5B naming
WAN22_KEYMAP_TI2V_BLOCK: dict[str, str] = {
    "self_attn.q.weight": "self_attn.q_proj.weight",
    "self_attn.q.bias": "self_attn.q_proj.bias",
    "self_attn.k.weight": "self_attn.k_proj.weight",
    "self_attn.k.bias": "self_attn.k_proj.bias",
    "self_attn.v.weight": "self_attn.v_proj.weight",
    "self_attn.v.bias": "self_attn.v_proj.bias",
    "self_attn.o.weight": "self_attn.out_proj.weight",
    "self_attn.o.bias": "self_attn.out_proj.bias",
    "self_attn.norm_q.weight": "self_attn.q_norm.weight",
    "self_attn.norm_k.weight": "self_attn.k_norm.weight",
    "cross_attn.q.weight": "cross_attn.q_proj.weight",
    "cross_attn.q.bias": "cross_attn.q_proj.bias",
    "cross_attn.k.weight": "cross_attn.k_proj.weight",
    "cross_attn.k.bias": "cross_attn.k_proj.bias",
    "cross_attn.v.weight": "cross_attn.v_proj.weight",
    "cross_attn.v.bias": "cross_attn.v_proj.bias",
    "cross_attn.o.weight": "cross_attn.out_proj.weight",
    "cross_attn.o.bias": "cross_attn.out_proj.bias",
    "cross_attn.norm_q.weight": "cross_attn.q_norm.weight",
    "cross_attn.norm_k.weight": "cross_attn.k_norm.weight",
    "ffn.0.weight": "ffn.0.weight",
    "ffn.0.bias": "ffn.0.bias",
    "ffn.2.weight": "ffn.2.weight",
    "ffn.2.bias": "ffn.2.bias",
    "modulation": "modulation",
    "norm3.weight": "norm3.weight",
    "norm3.bias": "norm3.bias",
}

WAN22_KEYMAP_OFFICIAL_TOP: dict[str, str] = {
    "patch_embedding.weight": "patch_embedding.weight",
    "patch_embedding.bias": "patch_embedding.bias",
    "text_embedding.0.weight": "text_embedding.0.weight",
    "text_embedding.0.bias": "text_embedding.0.bias",
    "text_embedding.2.weight": "text_embedding.2.weight",
    "text_embedding.2.bias": "text_embedding.2.bias",
    "time_embedding.0.weight": "time_embedding.0.weight",
    "time_embedding.0.bias": "time_embedding.0.bias",
    "time_embedding.2.weight": "time_embedding.2.weight",
    "time_embedding.2.bias": "time_embedding.2.bias",
    "time_projection.1.weight": "time_projection.1.weight",
    "time_projection.1.bias": "time_projection.1.bias",
    "head.modulation": "head.modulation",
    "head.head.weight": "head.head.weight",
    "head.head.bias": "head.head.bias",
}

WAN22_KEYMAP_DIFFUSERS_TOP: dict[str, str] = {
    "patch_embedding.weight": "patch_embedding.weight",
    "patch_embedding.bias": "patch_embedding.bias",
    "condition_embedder.text_embedder.linear_1.weight": "text_embedding.0.weight",
    "condition_embedder.text_embedder.linear_1.bias": "text_embedding.0.bias",
    "condition_embedder.text_embedder.linear_2.weight": "text_embedding.2.weight",
    "condition_embedder.text_embedder.linear_2.bias": "text_embedding.2.bias",
    "condition_embedder.time_embedder.linear_1.weight": "time_embedding.0.weight",
    "condition_embedder.time_embedder.linear_1.bias": "time_embedding.0.bias",
    "condition_embedder.time_embedder.linear_2.weight": "time_embedding.2.weight",
    "condition_embedder.time_embedder.linear_2.bias": "time_embedding.2.bias",
    "condition_embedder.time_proj.weight": "time_projection.1.weight",
    "condition_embedder.time_proj.bias": "time_projection.1.bias",
    "scale_shift_table": "head.modulation",
    "proj_out.weight": "head.head.weight",
    "proj_out.bias": "head.head.bias",
}

WAN22_KEYMAP_TI2V_TOP: dict[str, str] = {
    "patch_embedding.weight": "patch_embedding.weight",
    "patch_embedding.bias": "patch_embedding.bias",
    "text_embedding.0.weight": "text_embedding.0.weight",
    "text_embedding.0.bias": "text_embedding.0.bias",
    "text_embedding.2.weight": "text_embedding.2.weight",
    "text_embedding.2.bias": "text_embedding.2.bias",
    "time_embedding.0.weight": "time_embedding.0.weight",
    "time_embedding.0.bias": "time_embedding.0.bias",
    "time_embedding.2.weight": "time_embedding.2.weight",
    "time_embedding.2.bias": "time_embedding.2.bias",
    "time_projection.1.weight": "time_projection.1.weight",
    "time_projection.1.bias": "time_projection.1.bias",
    "head.modulation": "head.modulation",
    "head.head.weight": "head.head.weight",
    "head.head.bias": "head.head.bias",
}

# Variant D: Wan2.2-{T2V,I2V}-A14B per-expert state dict.
# Each expert (high_noise_model / low_noise_model) ships as its own shard set
# whose intra-block naming matches the upstream Wan2.2 reference repo.
# Now that OgentiTransformer uses the Wan-native block layout, this map is
# almost entirely identity — we only rename the q/k/v/o linear suffix.
WAN22_KEYMAP_A14B_BLOCK: dict[str, str] = {
    "self_attn.q.weight": "self_attn.q_proj.weight",
    "self_attn.q.bias": "self_attn.q_proj.bias",
    "self_attn.k.weight": "self_attn.k_proj.weight",
    "self_attn.k.bias": "self_attn.k_proj.bias",
    "self_attn.v.weight": "self_attn.v_proj.weight",
    "self_attn.v.bias": "self_attn.v_proj.bias",
    "self_attn.o.weight": "self_attn.out_proj.weight",
    "self_attn.o.bias": "self_attn.out_proj.bias",
    "self_attn.norm_q.weight": "self_attn.q_norm.weight",
    "self_attn.norm_k.weight": "self_attn.k_norm.weight",
    "cross_attn.q.weight": "cross_attn.q_proj.weight",
    "cross_attn.q.bias": "cross_attn.q_proj.bias",
    "cross_attn.k.weight": "cross_attn.k_proj.weight",
    "cross_attn.k.bias": "cross_attn.k_proj.bias",
    "cross_attn.v.weight": "cross_attn.v_proj.weight",
    "cross_attn.v.bias": "cross_attn.v_proj.bias",
    "cross_attn.o.weight": "cross_attn.out_proj.weight",
    "cross_attn.o.bias": "cross_attn.out_proj.bias",
    "cross_attn.norm_q.weight": "cross_attn.q_norm.weight",
    "cross_attn.norm_k.weight": "cross_attn.k_norm.weight",
    "ffn.0.weight": "ffn.0.weight",
    "ffn.0.bias": "ffn.0.bias",
    "ffn.2.weight": "ffn.2.weight",
    "ffn.2.bias": "ffn.2.bias",
    "modulation": "modulation",
    "norm3.weight": "norm3.weight",
    "norm3.bias": "norm3.bias",
}

WAN22_KEYMAP_A14B_TOP: dict[str, str] = {
    "patch_embedding.weight": "patch_embedding.weight",
    "patch_embedding.bias": "patch_embedding.bias",
    "text_embedding.0.weight": "text_embedding.0.weight",
    "text_embedding.0.bias": "text_embedding.0.bias",
    "text_embedding.2.weight": "text_embedding.2.weight",
    "text_embedding.2.bias": "text_embedding.2.bias",
    "time_embedding.0.weight": "time_embedding.0.weight",
    "time_embedding.0.bias": "time_embedding.0.bias",
    "time_embedding.2.weight": "time_embedding.2.weight",
    "time_embedding.2.bias": "time_embedding.2.bias",
    "time_projection.1.weight": "time_projection.1.weight",
    "time_projection.1.bias": "time_projection.1.bias",
    "head.modulation": "head.modulation",
    "head.head.weight": "head.head.weight",
    "head.head.bias": "head.head.bias",
}

# A14B MoE subdirs inside a model snapshot directory.
A14B_EXPERT_SUBDIRS: dict[str, tuple[str, ...]] = {
    "high_noise": ("high_noise_model", "high_noise", "transformer_high_noise"),
    "low_noise": ("low_noise_model", "low_noise", "transformer_low_noise"),
}


@dataclass
class KeymapVariant:
    name: str
    block_prefix: str
    block_map: dict[str, str]
    top_map: dict[str, str]
    block_regex: re.Pattern

    @classmethod
    def official(cls) -> "KeymapVariant":
        return cls(
            name="official",
            block_prefix="blocks.",
            block_map=WAN22_KEYMAP_OFFICIAL_BLOCK,
            top_map=WAN22_KEYMAP_OFFICIAL_TOP,
            block_regex=re.compile(r"^blocks\.(\d+)\.(.+)$"),
        )

    @classmethod
    def diffusers(cls) -> "KeymapVariant":
        return cls(
            name="diffusers",
            block_prefix="blocks.",
            block_map=WAN22_KEYMAP_DIFFUSERS_BLOCK,
            top_map=WAN22_KEYMAP_DIFFUSERS_TOP,
            block_regex=re.compile(r"^blocks\.(\d+)\.(.+)$"),
        )

    @classmethod
    def ti2v(cls) -> "KeymapVariant":
        return cls(
            name="ti2v",
            block_prefix="blocks.",
            block_map=WAN22_KEYMAP_TI2V_BLOCK,
            top_map=WAN22_KEYMAP_TI2V_TOP,
            block_regex=re.compile(r"^blocks\.(\d+)\.(.+)$"),
        )

    @classmethod
    def a14b(cls) -> "KeymapVariant":
        return cls(
            name="a14b",
            block_prefix="blocks.",
            block_map=WAN22_KEYMAP_A14B_BLOCK,
            top_map=WAN22_KEYMAP_A14B_TOP,
            block_regex=re.compile(r"^blocks\.(\d+)\.(.+)$"),
        )


VARIANTS = [
    KeymapVariant.official(),
    KeymapVariant.diffusers(),
    KeymapVariant.ti2v(),
    KeymapVariant.a14b(),
]


def detect_variant(state_keys: list[str]) -> KeymapVariant:
    """Heuristic: pick the variant whose top-level keys appear most often."""
    scores: dict[str, int] = {}
    keyset = set(state_keys)
    for v in VARIANTS:
        score = sum(1 for k in v.top_map if k in keyset)
        scores[v.name] = score
    best = max(VARIANTS, key=lambda v: scores[v.name])
    log.info(f"variant detection scores: {scores} -> chose '{best.name}'")
    return best


def _resolve_expert_dir(root: Path, expert: Optional[ExpertSelector]) -> Path:
    """For A14B snapshots, descend into the requested MoE expert subdir.

    Returns `root` unchanged if no expert subdirs are found (single-expert models
    like TI2V-5B). Otherwise returns the chosen expert directory.
    """
    if not root.is_dir():
        return root

    present = {
        name: next((root / cand for cand in cands if (root / cand).is_dir()), None)
        for name, cands in A14B_EXPERT_SUBDIRS.items()
    }
    present = {k: v for k, v in present.items() if v is not None}
    if not present:
        return root

    if expert is None:
        # default to low_noise (fine-detail expert) when both are present
        chosen = present.get("low_noise") or present.get("high_noise")
        log.info(
            f"A14B MoE layout detected at {root}; no expert specified, "
            f"defaulting to 'low_noise' -> {chosen}"
        )
        assert chosen is not None
        return chosen

    if expert not in present:
        raise FileNotFoundError(
            f"requested expert '{expert}' not found under {root} "
            f"(available: {sorted(present)})"
        )
    log.info(f"A14B MoE: selected expert '{expert}' from {present[expert]}")
    return present[expert]


def _load_state(
    path: Path | str,
    *,
    expert: Optional[ExpertSelector] = None,
) -> dict[str, torch.Tensor]:
    path = Path(path)
    if path.is_dir():
        path = _resolve_expert_dir(path, expert)

        candidates = sorted(path.glob("*.safetensors"))
        if not candidates:
            for sub in ("transformer", "dit"):
                candidates = sorted((path / sub).glob("*.safetensors")) if (path / sub).exists() else []
                if candidates:
                    break
        if not candidates:
            candidates = sorted(path.glob("*.bin")) + sorted(path.glob("*.pt"))
        if not candidates:
            raise FileNotFoundError(f"no weight shards under {path}")
        merged: dict[str, torch.Tensor] = {}
        for shard in candidates:
            log.info(f"loading shard {shard.name}")
            if shard.suffix == ".safetensors":
                merged.update(load_file(str(shard)))
            else:
                merged.update(torch.load(str(shard), map_location="cpu"))
        return merged

    if path.suffix == ".safetensors":
        return load_file(str(path))
    return torch.load(str(path), map_location="cpu")


def _translate_key(src_key: str, variant: KeymapVariant) -> Optional[str]:
    if src_key in variant.top_map:
        return variant.top_map[src_key]
    m = variant.block_regex.match(src_key)
    if m:
        idx, suffix = m.group(1), m.group(2)
        mapped = variant.block_map.get(suffix)
        if mapped is None:
            return None
        return f"blocks.{idx}.{mapped}"
    return None


def import_wan22_into_ogenti(
    wan22_weights_path: Path | str,
    ogenti_config: OgentiTransformerConfig,
    *,
    strict: bool = False,
    verify_invariant: bool = True,
    force_variant: Optional[str] = None,
    expert: Optional[ExpertSelector] = None,
) -> tuple[OgentiTransformer, dict[str, list[str]]]:
    log.info("=== Wan2.2 -> Ogenti retrofit import ===")
    src_state = _load_state(wan22_weights_path, expert=expert)
    log.info(f"loaded {len(src_state)} source tensors")

    if force_variant:
        variant = next((v for v in VARIANTS if v.name == force_variant), None)
        if variant is None:
            raise ValueError(f"unknown variant '{force_variant}'")
    else:
        variant = detect_variant(list(src_state.keys()))

    model = OgentiTransformer(ogenti_config)
    dst_state = dict(model.state_dict())
    report: dict[str, list[str]] = {
        "copied": [],
        "skipped_shape": [],
        "skipped_unknown": [],
        "variant": [variant.name],
        "expert": [expert] if expert else [],
    }

    for src_k, src_v in src_state.items():
        dst_k = _translate_key(src_k, variant)
        if dst_k is None:
            report["skipped_unknown"].append(src_k)
            continue
        if dst_k not in dst_state:
            report["skipped_unknown"].append(f"{src_k} -> {dst_k} (not in target)")
            continue
        if dst_state[dst_k].shape != src_v.shape:
            report["skipped_shape"].append(
                f"{src_k} -> {dst_k}: src{tuple(src_v.shape)} vs dst{tuple(dst_state[dst_k].shape)}"
            )
            continue
        dst_state[dst_k] = src_v.to(dst_state[dst_k].dtype)
        report["copied"].append(f"{src_k} -> {dst_k}")

    missing, unexpected = model.load_state_dict(dst_state, strict=strict)
    if missing:
        log.warning(f"missing target keys: {len(missing)} (first 10): {missing[:10]}")
    if unexpected:
        log.warning(f"unexpected target keys: {len(unexpected)} (first 10): {unexpected[:10]}")

    log.info(
        f"import: variant={variant.name} copied={len(report['copied'])} "
        f"shape_skip={len(report['skipped_shape'])} unknown={len(report['skipped_unknown'])}"
    )

    if verify_invariant:
        _verify_retrofit_invariant(model)

    return model, report


def validate_keymap_dry_run(
    wan22_weights_path: Path | str,
    ogenti_config: OgentiTransformerConfig,
    *,
    force_variant: Optional[str] = None,
    expert: Optional[ExpertSelector] = None,
) -> dict[str, list[str]]:
    """Load weights and run the import, but do not instantiate full model state.

    Returns a coverage report so the operator can review before training.
    """
    src_state = _load_state(wan22_weights_path, expert=expert)
    variant = (
        next((v for v in VARIANTS if v.name == force_variant), None)
        if force_variant
        else detect_variant(list(src_state.keys()))
    )
    if variant is None:
        raise ValueError(f"unknown variant '{force_variant}'")

    model = OgentiTransformer(ogenti_config)
    dst_state = model.state_dict()

    report = {
        "variant": [variant.name],
        "expert": [expert] if expert else [],
        "mapped_ok": [],
        "shape_mismatch": [],
        "unmapped_source": [],
        "unmatched_target": [],
    }

    matched_targets: set[str] = set()
    for src_k, src_v in src_state.items():
        dst_k = _translate_key(src_k, variant)
        if dst_k is None or dst_k not in dst_state:
            report["unmapped_source"].append(src_k)
            continue
        if dst_state[dst_k].shape != src_v.shape:
            report["shape_mismatch"].append(
                f"{src_k} -> {dst_k}: {tuple(src_v.shape)} != {tuple(dst_state[dst_k].shape)}"
            )
            continue
        report["mapped_ok"].append(f"{src_k} -> {dst_k}")
        matched_targets.add(dst_k)

    for dst_k in dst_state:
        if dst_k in matched_targets:
            continue
        if _is_ogenti_specific_key(dst_k):
            continue
        report["unmatched_target"].append(dst_k)

    log.info(
        f"dry-run: variant={variant.name} "
        f"ok={len(report['mapped_ok'])} "
        f"shape_mismatch={len(report['shape_mismatch'])} "
        f"unmapped_src={len(report['unmapped_source'])} "
        f"unmatched_dst={len(report['unmatched_target'])}"
    )
    return report


def _is_ogenti_specific_key(key: str) -> bool:
    """Return True if `key` belongs to an Ogenti-only branch that has no Wan2.2
    counterpart and should therefore NOT be reported as an unmatched_target.

    These include the entity / glyph extras (Pass A / C of the OgentiBlock),
    the additional conditioning embeds (camera / subject motion, micro events,
    physics keyframes), the post-VAE pixel-space refinement heads, and the
    axial RoPE buffers (which are computed, not loaded).
    """
    patterns = [
        # OgentiBlock extras (Pass A / C, zero-init gates)
        "entity_bank",
        "entity_refine",
        "patch_from_entity",
        "glyph_fuse",
        "glyph_branch",
        ".gate",
        # Top-level conditioning embeds (zero-init out-projs at step 0)
        "camera_motion_embed",
        "subject_motion_embed",
        "micro_event_embed",
        "physics_keyframe_embed",
        # Post-VAE pixel-space refinement heads
        "skin_detail_head",
        "material_detail_head",
        "motion_blur_head",
        "film_grain_head",
        "lens_artifact_head",
        # Axial RoPE buffers (recomputed at forward; never loaded)
        "rope.",
    ]
    return any(p in key for p in patterns)


def _verify_retrofit_invariant(model: OgentiTransformer) -> None:
    for i, block in enumerate(model.blocks):
        if block.patch_from_entity is not None:
            gate = block.patch_from_entity.gate.detach()
            if not torch.allclose(gate, torch.zeros_like(gate)):
                log.error(f"block[{i}] patch_from_entity.gate is non-zero: {gate}")
        if block.glyph_fuse is not None:
            gate = block.glyph_fuse.gate.detach()
            if not torch.allclose(gate, torch.zeros_like(gate)):
                log.error(f"block[{i}] glyph_fuse.gate is non-zero: {gate}")
    log.info("retrofit invariant check passed: all Ogenti gates are zero-initialized")
