"""OgentiBlock — drop-in retrofit replacement for a DiT transformer block.

Three-pass design (see RFC-0001):

    Pass A (Entity refinement): slots cross-attend to patches
    Pass B (Patch self-attn with AdaLN-Zero + entity anchoring cross-attn)
    Pass C (optional glyph fusion, gated)

Retrofit invariant: with `entity_tokens=None` and `glyph_stream=None`, AND the
entity/glyph submodule out-projections at their zero initialization, this block
reduces EXACTLY to a vanilla DiT block at step 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn

from ogenti.modules.attention.gated_cross_attn import GatedCrossAttention
from ogenti.modules.attention.multihead import FeedForward, MultiHeadAttention
from ogenti.modules.attention.rope import AxialRoPE3D
from ogenti.modules.conditioning.timestep import AdaLNZero, modulate
from ogenti.modules.identity.entity_tokens import EntityRefineLayer


@dataclass
class OgentiBlockConfig:
    dim: int = 3072
    num_heads: int = 24
    head_dim: int = 128
    mlp_ratio: float = 4.0
    qk_norm: bool = True
    cond_dim: int = 3072

    enable_entity_attention: bool = False
    entity_dim: Optional[int] = None
    entity_attn_heads: int = 16
    entity_attn_head_dim: int = 64

    enable_glyph_fusion: bool = False
    glyph_kv_dim: int = 3072
    glyph_heads: int = 16
    glyph_head_dim: int = 64

    enable_cross_attn: bool = False
    cross_attn_dim: Optional[int] = None

    source_block_name: Optional[str] = None


class OgentiBlock(nn.Module):
    def __init__(self, config: OgentiBlockConfig) -> None:
        super().__init__()
        self.config = config
        self._use_gradient_checkpointing: bool = False
        d = config.dim

        # --- DiT-equivalent core (warm-start target) ---
        self.adaln = AdaLNZero(cond_dim=config.cond_dim, target_dim=d)
        self.norm1 = nn.LayerNorm(d, elementwise_affine=False)
        self.self_attn = MultiHeadAttention(
            dim=d,
            num_heads=config.num_heads,
            head_dim=config.head_dim,
            qk_norm=config.qk_norm,
        )
        self.norm2 = nn.LayerNorm(d, elementwise_affine=False)
        self.ffn = FeedForward(d, mlp_ratio=config.mlp_ratio)

        # --- Pass A: entity refinement (slots <- patches) ---
        if config.enable_entity_attention:
            ent_dim = config.entity_dim or d
            self.entity_refine = EntityRefineLayer(
                dim=ent_dim,
                num_heads=config.entity_attn_heads,
                head_dim=config.entity_attn_head_dim,
            )
            # --- Pass B addendum: patches <- entities, zero-init delta ---
            self.patch_from_entity = GatedCrossAttention(
                dim=d,
                kv_dim=ent_dim,
                num_heads=config.entity_attn_heads,
                head_dim=config.entity_attn_head_dim,
                gate_init=0.0,
            )
        else:
            self.entity_refine = None
            self.patch_from_entity = None

        # --- Pass C: glyph fusion ---
        if config.enable_glyph_fusion:
            self.glyph_fuse = GatedCrossAttention(
                dim=d,
                kv_dim=config.glyph_kv_dim,
                num_heads=config.glyph_heads,
                head_dim=config.glyph_head_dim,
                gate_init=0.0,
            )
        else:
            self.glyph_fuse = None

        # --- TI2V Cross Attention ---
        if config.enable_cross_attn:
            self.norm3 = nn.LayerNorm(d, elementwise_affine=True)
            self.cross_attn = MultiHeadAttention(
                dim=d,
                num_heads=config.num_heads,
                head_dim=config.head_dim,
                qk_norm=config.qk_norm,
                kv_dim=config.cross_attn_dim or d,
            )
        else:
            self.norm3 = None
            self.cross_attn = None

    def forward(
        self,
        patch_tokens: Tensor,
        timestep_emb: Tensor,
        entity_tokens: Optional[Tensor] = None,
        glyph_stream: Optional[Tensor] = None,
        glyph_mask: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        rope_freqs: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Optional[Tensor]]:
        if self._use_gradient_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint

            def run(p, t, e, g, gm, c, rf, am):
                return self._forward_impl(p, t, e, g, gm, c, rf, am)

            return checkpoint(
                run,
                patch_tokens, timestep_emb, entity_tokens,
                glyph_stream, glyph_mask, context, rope_freqs, attn_mask,
                use_reentrant=False,
            )
        return self._forward_impl(
            patch_tokens, timestep_emb, entity_tokens,
            glyph_stream, glyph_mask, context, rope_freqs, attn_mask,
        )

    def _forward_impl(
        self,
        patch_tokens: Tensor,
        timestep_emb: Tensor,
        entity_tokens: Optional[Tensor],
        glyph_stream: Optional[Tensor],
        glyph_mask: Optional[Tensor],
        context: Optional[Tensor],
        rope_freqs: Optional[Tensor],
        attn_mask: Optional[Tensor],
    ) -> tuple[Tensor, Optional[Tensor]]:
        shift_msa, scale_msa, gate_msa, shift_ffn, scale_ffn, gate_ffn = self.adaln(timestep_emb)

        if self.entity_refine is not None and entity_tokens is not None:
            entity_tokens = self.entity_refine(entity_tokens, patch_tokens)

        h = modulate(self.norm1(patch_tokens), shift_msa, scale_msa)
        attn_out = self.self_attn(
            h,
            rope_apply_fn=AxialRoPE3D.apply if rope_freqs is not None else None,
            rope_freqs_q=rope_freqs,
            rope_freqs_k=rope_freqs,
            attn_mask=attn_mask,
        )
        patch_tokens = patch_tokens + gate_msa.unsqueeze(1) * attn_out

        if self.cross_attn is not None and context is not None:
            patch_tokens = patch_tokens + self.cross_attn(self.norm3(patch_tokens), kv_in=context)

        if self.patch_from_entity is not None and entity_tokens is not None:
            patch_tokens = self.patch_from_entity(patch_tokens, entity_tokens)

        if self.glyph_fuse is not None and glyph_stream is not None:
            mask = None
            if glyph_mask is not None:
                mask = glyph_mask[:, None, None, :].to(patch_tokens.dtype)
                mask = mask.masked_fill(glyph_mask[:, None, None, :] == 0, float("-inf"))
                mask = mask.masked_fill(glyph_mask[:, None, None, :] != 0, 0.0)
            patch_tokens = self.glyph_fuse(patch_tokens, glyph_stream, attn_mask=mask)

        h = modulate(self.norm2(patch_tokens), shift_ffn, scale_ffn)
        patch_tokens = patch_tokens + gate_ffn.unsqueeze(1) * self.ffn(h)

        return patch_tokens, entity_tokens

    @classmethod
    def from_dit_block(
        cls,
        dit_block_state: dict[str, Tensor],
        config: OgentiBlockConfig,
        key_map: Optional[dict[str, str]] = None,
    ) -> "OgentiBlock":
        block = cls(config)
        own_state = block.state_dict()
        mapped = {}

        for src_k, src_v in dit_block_state.items():
            dst_k = key_map.get(src_k, src_k) if key_map else src_k
            if dst_k in own_state and own_state[dst_k].shape == src_v.shape:
                mapped[dst_k] = src_v

        missing = set(own_state.keys()) - set(mapped.keys())
        new_state = {**own_state, **mapped}
        block.load_state_dict(new_state)
        block._retrofit_missing_keys = sorted(missing)
        return block
