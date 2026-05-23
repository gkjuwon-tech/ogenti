"""OgentiBlock — Wan2.2-A14B compatible transformer block + Ogenti extras.

Layout (Wan-native core):
    blocks.N.modulation                                   # (1,6,dim) param
    blocks.N.self_attn.{q,k,v,o}.{weight,bias}            # via MultiHeadAttention
    blocks.N.self_attn.norm_q.weight                      # per-channel RMSNorm(dim)
    blocks.N.self_attn.norm_k.weight
    blocks.N.norm3.{weight,bias}                          # affine LayerNorm pre cross-attn
    blocks.N.cross_attn.{q,k,v,o}.{weight,bias}           # text cross-attention
    blocks.N.cross_attn.norm_q.weight, norm_k.weight
    blocks.N.ffn.0.{weight,bias}                          # Linear(dim, ffn_dim)
    blocks.N.ffn.2.{weight,bias}                          # Linear(ffn_dim, dim)

Ogenti extras (Wan-invariant; preserved as a strict no-op at step 0):
    blocks.N.entity_refine.*                              # slots <- patches
    blocks.N.patch_from_entity.*  (zero-init gate)        # patches <- entities
    blocks.N.glyph_fuse.*         (zero-init gate)        # patches <- glyph tokens

Retrofit invariant:
    - With `entity_tokens=None` and `glyph_stream=None`, AND `patch_from_entity` /
      `glyph_fuse` at their zero-init gates, this block reduces EXACTLY to a vanilla
      Wan2.2-A14B block at step 0.
    - The Wan-style cross_attn is always-on (when enabled) because every Wan A14B
      block has it; without it we would have nowhere to load the cross_attn shards.

See RFC-0006 (`docs/rfcs/RFC-0006-promoting-a14b.md`) for the decision record.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn

from ogenti.modules.attention.gated_cross_attn import GatedCrossAttention
from ogenti.modules.attention.multihead import MultiHeadAttention, WanFeedForward
from ogenti.modules.attention.rope import AxialRoPE3D
from ogenti.modules.conditioning.timestep import modulate
from ogenti.modules.identity.entity_tokens import EntityRefineLayer


@dataclass
class OgentiBlockConfig:
    dim: int = 3072
    num_heads: int = 24
    head_dim: int = 128
    mlp_ratio: float = 4.0
    ffn_dim: Optional[int] = None  # if None: dim * mlp_ratio (rounded). Wan A14B: 13824.
    qk_norm: bool = True
    qk_norm_mode: str = "per_channel"  # Wan2.2 native; legacy 5B configs may override

    enable_entity_attention: bool = False
    entity_dim: Optional[int] = None
    entity_attn_heads: int = 16
    entity_attn_head_dim: int = 64

    enable_glyph_fusion: bool = False
    glyph_kv_dim: int = 3072
    glyph_heads: int = 16
    glyph_head_dim: int = 64

    enable_cross_attn: bool = True
    cross_attn_dim: Optional[int] = None

    source_block_name: Optional[str] = None


class OgentiBlock(nn.Module):
    def __init__(self, config: OgentiBlockConfig) -> None:
        super().__init__()
        self.config = config
        self._use_gradient_checkpointing: bool = False
        d = config.dim
        ffn_dim = config.ffn_dim if config.ffn_dim is not None else int(d * config.mlp_ratio)

        # --- per-block AdaLN modulation (Wan2.2 native) ---
        # State dict key: `blocks.N.modulation` (raw parameter, no submodule wrapper).
        self.modulation = nn.Parameter(torch.empty(1, 6, d))
        nn.init.normal_(self.modulation, std=1.0 / math.sqrt(d))

        # --- self attention with per-channel q/k norm (Wan2.2 native) ---
        self.norm1 = nn.LayerNorm(d, elementwise_affine=False)
        self.self_attn = MultiHeadAttention(
            dim=d,
            num_heads=config.num_heads,
            head_dim=config.head_dim,
            qk_norm=config.qk_norm,
            qk_norm_mode=config.qk_norm_mode,
        )

        # --- text cross-attention (Wan2.2 native — always present for A14B) ---
        if config.enable_cross_attn:
            self.norm3 = nn.LayerNorm(d, elementwise_affine=True)
            self.cross_attn = MultiHeadAttention(
                dim=d,
                num_heads=config.num_heads,
                head_dim=config.head_dim,
                qk_norm=config.qk_norm,
                qk_norm_mode=config.qk_norm_mode,
                kv_dim=config.cross_attn_dim or d,
            )
        else:
            self.norm3 = None
            self.cross_attn = None

        # --- ffn (Wan2.2 native: 2-linear + GELU + bias) ---
        self.norm2 = nn.LayerNorm(d, elementwise_affine=False)
        self.ffn = WanFeedForward(dim=d, ffn_dim=ffn_dim)

        # --- Pass A: entity refinement (slots <- patches) ---
        if config.enable_entity_attention:
            ent_dim = config.entity_dim or d
            self.entity_refine = EntityRefineLayer(
                dim=ent_dim,
                num_heads=config.entity_attn_heads,
                head_dim=config.entity_attn_head_dim,
            )
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

    def forward(
        self,
        patch_tokens: Tensor,
        e6: Tensor,
        entity_tokens: Optional[Tensor] = None,
        glyph_stream: Optional[Tensor] = None,
        glyph_mask: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        rope_freqs: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Optional[Tensor]]:
        """Forward pass.

        Args:
            patch_tokens: (B, N, dim)
            e6: (B, 6, dim) — global time-conditioned AdaLN params from `time_projection`.
                Added to this block's `modulation` parameter before chunking.
            entity_tokens / glyph_stream / glyph_mask: Ogenti extras (optional)
            context: (B, L, dim) — Wan-native text cross-attn KV (T5 embeddings post text_embedding)
            rope_freqs: axial RoPE freqs for self-attention
            attn_mask: optional attention mask for self-attention
        """
        if self._use_gradient_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint

            def run(p, e, ent, g, gm, c, rf, am):
                return self._forward_impl(p, e, ent, g, gm, c, rf, am)

            return checkpoint(
                run,
                patch_tokens, e6, entity_tokens,
                glyph_stream, glyph_mask, context, rope_freqs, attn_mask,
                use_reentrant=False,
            )
        return self._forward_impl(
            patch_tokens, e6, entity_tokens,
            glyph_stream, glyph_mask, context, rope_freqs, attn_mask,
        )

    def _forward_impl(
        self,
        patch_tokens: Tensor,
        e6: Tensor,
        entity_tokens: Optional[Tensor],
        glyph_stream: Optional[Tensor],
        glyph_mask: Optional[Tensor],
        context: Optional[Tensor],
        rope_freqs: Optional[Tensor],
        attn_mask: Optional[Tensor],
    ) -> tuple[Tensor, Optional[Tensor]]:
        # Each chunk is (B, 1, dim) — broadcast over sequence dim automatically.
        # modulation: (1, 6, dim) broadcast added to e6: (B, 6, dim) -> (B, 6, dim).
        mod = self.modulation + e6
        shift_msa, scale_msa, gate_msa, shift_ffn, scale_ffn, gate_ffn = mod.chunk(6, dim=1)

        # ── Pass A: entity refinement (slots <- patches) ───────────────────
        if self.entity_refine is not None and entity_tokens is not None:
            entity_tokens = self.entity_refine(entity_tokens, patch_tokens)

        # ── self-attention with AdaLN-zero modulation ──────────────────────
        h = modulate(self.norm1(patch_tokens), shift_msa, scale_msa)
        attn_out = self.self_attn(
            h,
            rope_apply_fn=AxialRoPE3D.apply if rope_freqs is not None else None,
            rope_freqs_q=rope_freqs,
            rope_freqs_k=rope_freqs,
            attn_mask=attn_mask,
        )
        patch_tokens = patch_tokens + gate_msa * attn_out

        # ── Wan-native text cross-attention (always on for A14B) ──────────
        if self.cross_attn is not None and context is not None:
            patch_tokens = patch_tokens + self.cross_attn(self.norm3(patch_tokens), kv_in=context)

        # ── Ogenti extras (zero-init gates at step 0 → no-op) ─────────────
        if self.patch_from_entity is not None and entity_tokens is not None:
            patch_tokens = self.patch_from_entity(patch_tokens, entity_tokens)

        if self.glyph_fuse is not None and glyph_stream is not None:
            mask = None
            if glyph_mask is not None:
                mask = glyph_mask[:, None, None, :].to(patch_tokens.dtype)
                mask = mask.masked_fill(glyph_mask[:, None, None, :] == 0, float("-inf"))
                mask = mask.masked_fill(glyph_mask[:, None, None, :] != 0, 0.0)
            patch_tokens = self.glyph_fuse(patch_tokens, glyph_stream, attn_mask=mask)

        # ── ffn with AdaLN-zero modulation ─────────────────────────────────
        h = modulate(self.norm2(patch_tokens), shift_ffn, scale_ffn)
        patch_tokens = patch_tokens + gate_ffn * self.ffn(h)

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
