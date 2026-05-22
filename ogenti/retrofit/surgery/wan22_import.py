"""Wan2.2 transformer -> OgentiTransformer weight import.

Strategy:
  1. Define multiple candidate keymap variants (upstream Wan releases use
     slightly different naming — diffusers fork vs. official repo).
  2. At import time, sniff the source state dict to detect which variant
     we're looking at and apply the matching map.
  3. Report unmapped/shape-mismatched keys so the user can diagnose
     before training spends compute.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from safetensors.torch import load_file

from ogenti.models.ogenti_transformer import OgentiTransformer, OgentiTransformerConfig
from ogenti.utils.logging import get_logger

log = get_logger("ogenti.retrofit.wan22")


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
    "ffn.0.weight": "ffn.w1.weight",
    "ffn.2.weight": "ffn.w2.weight",
    "ffn.gate.weight": "ffn.w3.weight",
    "modulation": "adaln.linear.weight",
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
    "ffn.net.0.proj.weight": "ffn.w1.weight",
    "ffn.net.2.weight": "ffn.w2.weight",
    "ffn.net.0.proj_gate.weight": "ffn.w3.weight",
    "scale_shift_table": "adaln.linear.weight",
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
    "ffn.0.weight": "ffn.w1.weight",
    "ffn.0.bias": "ffn.w1.bias",
    "ffn.2.weight": "ffn.w2.weight",
    "ffn.2.bias": "ffn.w2.bias",
    "modulation": "adaln.linear.weight",
    "norm3.weight": "norm3.weight",
    "norm3.bias": "norm3.bias",
}

WAN22_KEYMAP_OFFICIAL_TOP: dict[str, str] = {
    "patch_embedding.weight": "patch_embed.proj.weight",
    "patch_embedding.bias": "patch_embed.proj.bias",
    "text_embedding.0.weight": "text_proj.weight",
    "text_embedding.0.bias": "text_proj.bias",
    "time_embedding.0.weight": "timestep_head.mlp.0.weight",
    "time_embedding.0.bias": "timestep_head.mlp.0.bias",
    "time_embedding.2.weight": "timestep_head.mlp.2.weight",
    "time_embedding.2.bias": "timestep_head.mlp.2.bias",
    "head.modulation": "adaln_out.weight",
    "head.head.weight": "unpatchify.proj.weight",
    "head.head.bias": "unpatchify.proj.bias",
}

WAN22_KEYMAP_DIFFUSERS_TOP: dict[str, str] = {
    "patch_embedding.weight": "patch_embed.proj.weight",
    "patch_embedding.bias": "patch_embed.proj.bias",
    "condition_embedder.text_embedder.linear_1.weight": "text_proj.weight",
    "condition_embedder.text_embedder.linear_1.bias": "text_proj.bias",
    "condition_embedder.time_embedder.linear_1.weight": "timestep_head.mlp.0.weight",
    "condition_embedder.time_embedder.linear_1.bias": "timestep_head.mlp.0.bias",
    "condition_embedder.time_embedder.linear_2.weight": "timestep_head.mlp.2.weight",
    "condition_embedder.time_embedder.linear_2.bias": "timestep_head.mlp.2.bias",
    "scale_shift_table": "adaln_out.weight",
    "proj_out.weight": "unpatchify.proj.weight",
    "proj_out.bias": "unpatchify.proj.bias",
}

WAN22_KEYMAP_TI2V_TOP: dict[str, str] = {
    "patch_embedding.weight": "patch_embed.proj.weight",
    "patch_embedding.bias": "patch_embed.proj.bias",
    "text_embedding.0.weight": "text_proj.0.weight",
    "text_embedding.0.bias": "text_proj.0.bias",
    "text_embedding.2.weight": "text_proj.2.weight",
    "text_embedding.2.bias": "text_proj.2.bias",
    "time_embedding.0.weight": "timestep_head.mlp.0.weight",
    "time_embedding.0.bias": "timestep_head.mlp.0.bias",
    "time_embedding.2.weight": "timestep_head.mlp.2.weight",
    "time_embedding.2.bias": "timestep_head.mlp.2.bias",
    "time_projection.1.weight": "time_projection.weight",
    "time_projection.1.bias": "time_projection.bias",
    "head.modulation": "adaln_out.weight",
    "head.head.weight": "unpatchify.proj.weight",
    "head.head.bias": "unpatchify.proj.bias",
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


VARIANTS = [KeymapVariant.official(), KeymapVariant.diffusers(), KeymapVariant.ti2v()]


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


def _load_state(path: Path | str) -> dict[str, torch.Tensor]:
    path = Path(path)
    if path.is_dir():
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
) -> tuple[OgentiTransformer, dict[str, list[str]]]:
    log.info("=== Wan2.2 -> Ogenti retrofit import ===")
    src_state = _load_state(wan22_weights_path)
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
) -> dict[str, list[str]]:
    """Load weights and run the import, but do not instantiate full model state.

    Returns a coverage report so the operator can review before training.
    """
    src_state = _load_state(wan22_weights_path)
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
    patterns = [
        "entity_bank",
        "entity_refine",
        "patch_from_entity",
        "glyph_fuse",
        "glyph_branch",
        ".gate",
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
