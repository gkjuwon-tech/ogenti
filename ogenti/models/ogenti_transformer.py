"""Full Ogenti video transformer — orchestrates patch embed, blocks, entity bank, glyph branch.

Phase 4-RE / 4-RE2 modules:
  - CameraMotionEmbed, SubjectMotionEmbed, MicroEventEmbed → conditioning stream
  - SkinDetailResidualHead, UniversalMaterialDetailHead → post-VAE texture refinement
  - MotionBlurHead, FilmGrainHead, LensArtifactHead → post-VAE optical simulation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor, nn

from einops import rearrange

from ogenti.models.blocks.ogenti_block import OgentiBlock, OgentiBlockConfig
from ogenti.modules.attention.rope import AxialRoPE3D, RopeConfig
from ogenti.modules.conditioning.camera_motion import (
    CAMERA_MOTION_DIM,
    CameraMotionEmbed,
    CameraMotionEmbedConfig,
)
from ogenti.modules.conditioning.micro_events import MicroEventConfig, MicroEventEmbed
from ogenti.modules.conditioning.physics_keyframes import (
    PhysicsKeyframeEmbed,
    PhysicsKeyframeEmbedConfig,
)
from ogenti.modules.conditioning.subject_motion import (
    SubjectMotionEmbed,
    SubjectMotionEmbedConfig,
)
from ogenti.modules.conditioning.timestep import (
    SinusoidalTimestepEmbedding,
    WanOutputHead,
    WanTextEmbedding,
    WanTimeEmbedding,
    WanTimeProjection,
)
from ogenti.modules.glyph.glyph_branch import GlyphBranch, GlyphBranchConfig
from ogenti.modules.identity.entity_tokens import EntityBankConfig, EntityTokenBank
from ogenti.modules.postprocess.film_grain import FilmGrainConfig, FilmGrainHead
from ogenti.modules.postprocess.lens_artifacts import LensArtifactConfig, LensArtifactHead
from ogenti.modules.postprocess.motion_blur import MotionBlurConfig, MotionBlurHead, estimate_flow_from_frames
from ogenti.modules.texture.material_detail_head import UMDHConfig, UniversalMaterialDetailHead
from ogenti.modules.texture.skin_detail_head import SDRHConfig, SkinDetailResidualHead
from ogenti.modules.tokenizers.patch_embed import PatchEmbed3DConfig


@dataclass
class OgentiTransformerConfig:
    in_channels: int = 16
    out_channels: int = 16
    dim: int = 3072
    num_blocks: int = 30
    num_heads: int = 24
    head_dim: int = 128
    mlp_ratio: float = 4.0
    ffn_dim: Optional[int] = None  # if None: derived from mlp_ratio. Wan A14B: 13824.
    text_dim: int = 4096
    freq_dim: int = 256             # sinusoidal timestep dim fed into time_embedding
    cond_dim: int = 3072
    qk_norm_mode: str = "per_channel"  # Wan2.2 native

    patch: PatchEmbed3DConfig = field(default_factory=PatchEmbed3DConfig)
    rope: RopeConfig = field(default_factory=lambda: RopeConfig(head_dim=128, t_split=32, h_split=48, w_split=48))
    entity: EntityBankConfig = field(default_factory=EntityBankConfig)
    glyph: GlyphBranchConfig = field(default_factory=GlyphBranchConfig)

    enable_entity_attention: bool = True
    enable_glyph_fusion: bool = True
    glyph_fusion_start_block: int = 20

    enable_ti2v_mode: bool = False

    enable_camera_motion_cond: bool = True
    camera_motion: CameraMotionEmbedConfig = field(
        default_factory=lambda: CameraMotionEmbedConfig(out_dim=3072)
    )

    enable_skin_detail_head: bool = True
    skin_detail: SDRHConfig = field(default_factory=SDRHConfig)

    enable_subject_motion_cond: bool = True
    subject_motion: SubjectMotionEmbedConfig = field(
        default_factory=lambda: SubjectMotionEmbedConfig(out_dim=3072)
    )

    enable_micro_events_cond: bool = True
    micro_events: MicroEventConfig = field(
        default_factory=lambda: MicroEventConfig(cond_dim=3072)
    )

    enable_physics_keyframes_cond: bool = True
    physics_keyframes: PhysicsKeyframeEmbedConfig = field(
        default_factory=lambda: PhysicsKeyframeEmbedConfig(out_dim=3072)
    )

    enable_material_detail_head: bool = True
    material_detail: UMDHConfig = field(default_factory=UMDHConfig)

    enable_motion_blur_head: bool = True
    motion_blur: MotionBlurConfig = field(default_factory=MotionBlurConfig)

    enable_film_grain_head: bool = True
    film_grain: FilmGrainConfig = field(default_factory=FilmGrainConfig)

    enable_lens_artifacts: bool = True
    lens_artifacts: LensArtifactConfig = field(default_factory=LensArtifactConfig)

    def __post_init__(self):
        if isinstance(self.patch, dict): self.patch = PatchEmbed3DConfig(**self.patch)
        if isinstance(self.rope, dict): self.rope = RopeConfig(**self.rope)
        if isinstance(self.entity, dict): self.entity = EntityBankConfig(**self.entity)
        if isinstance(self.glyph, dict): self.glyph = GlyphBranchConfig(**self.glyph)
        if isinstance(self.camera_motion, dict): self.camera_motion = CameraMotionEmbedConfig(**self.camera_motion)
        if isinstance(self.skin_detail, dict): self.skin_detail = SDRHConfig(**self.skin_detail)
        if isinstance(self.subject_motion, dict): self.subject_motion = SubjectMotionEmbedConfig(**self.subject_motion)
        if isinstance(self.micro_events, dict): self.micro_events = MicroEventConfig(**self.micro_events)
        if isinstance(self.physics_keyframes, dict): self.physics_keyframes = PhysicsKeyframeEmbedConfig(**self.physics_keyframes)
        if isinstance(self.material_detail, dict): self.material_detail = UMDHConfig(**self.material_detail)
        if isinstance(self.motion_blur, dict): self.motion_blur = MotionBlurConfig(**self.motion_blur)
        if isinstance(self.film_grain, dict): self.film_grain = FilmGrainConfig(**self.film_grain)
        if isinstance(self.lens_artifacts, dict): self.lens_artifacts = LensArtifactConfig(**self.lens_artifacts)


class OgentiTransformer(nn.Module):
    def __init__(self, config: OgentiTransformerConfig) -> None:
        super().__init__()
        self.config = config

        # ── patch embed (Wan-native: flat `patch_embedding` Conv3d) ──
        pt, ph, pw = config.patch.patch_size
        self.patch_embedding = nn.Conv3d(
            config.in_channels,
            config.dim,
            kernel_size=(pt, ph, pw),
            stride=(pt, ph, pw),
        )

        # ── text / time embeddings (Wan-native 2-layer MLPs) ─────────
        self.text_embedding = WanTextEmbedding(text_dim=config.text_dim, dim=config.dim)
        self.sinusoidal_t = SinusoidalTimestepEmbedding(config.freq_dim)
        self.time_embedding = WanTimeEmbedding(freq_dim=config.freq_dim, dim=config.dim)
        self.time_projection = WanTimeProjection(dim=config.dim)

        # ── conditioning embeds ──────────────────────────────────────
        self.camera_motion_embed: Optional[CameraMotionEmbed] = None
        if config.enable_camera_motion_cond:
            cm_cfg = CameraMotionEmbedConfig(
                descriptor_dim=config.camera_motion.descriptor_dim,
                embed_dim=config.camera_motion.embed_dim,
                out_dim=config.cond_dim,
                zero_init=True,
            )
            self.camera_motion_embed = CameraMotionEmbed(cm_cfg)

        self.subject_motion_embed: Optional[SubjectMotionEmbed] = None
        if config.enable_subject_motion_cond:
            sm_cfg = SubjectMotionEmbedConfig(
                descriptor_dim=config.subject_motion.descriptor_dim,
                max_tracks=config.subject_motion.max_tracks,
                embed_dim=config.subject_motion.embed_dim,
                out_dim=config.cond_dim,
                zero_init=True,
            )
            self.subject_motion_embed = SubjectMotionEmbed(sm_cfg)

        self.micro_event_embed: Optional[MicroEventEmbed] = None
        if config.enable_micro_events_cond:
            me_cfg = MicroEventConfig(
                embed_dim=config.micro_events.embed_dim,
                cond_dim=config.cond_dim,
                zero_init=True,
            )
            self.micro_event_embed = MicroEventEmbed(me_cfg)

        self.physics_keyframe_embed: Optional[PhysicsKeyframeEmbed] = None
        if config.enable_physics_keyframes_cond:
            pk_cfg = PhysicsKeyframeEmbedConfig(
                max_objects=config.physics_keyframes.max_objects,
                tokens_per_object=config.physics_keyframes.tokens_per_object,
                descriptor_per_token=config.physics_keyframes.descriptor_per_token,
                embed_dim=config.physics_keyframes.embed_dim,
                out_dim=config.cond_dim,
                zero_init=True,
            )
            self.physics_keyframe_embed = PhysicsKeyframeEmbed(pk_cfg)

        # ── post-VAE texture heads ───────────────────────────────────
        self.skin_detail_head: Optional[SkinDetailResidualHead] = None
        if config.enable_skin_detail_head:
            self.skin_detail_head = SkinDetailResidualHead(config.skin_detail)

        self.material_detail_head: Optional[UniversalMaterialDetailHead] = None
        if config.enable_material_detail_head:
            self.material_detail_head = UniversalMaterialDetailHead(config.material_detail)

        # ── post-VAE optical heads ───────────────────────────────────
        self.motion_blur_head: Optional[MotionBlurHead] = None
        if config.enable_motion_blur_head:
            self.motion_blur_head = MotionBlurHead(config.motion_blur)

        self.film_grain_head: Optional[FilmGrainHead] = None
        if config.enable_film_grain_head:
            self.film_grain_head = FilmGrainHead(config.film_grain)

        self.lens_artifact_head: Optional[LensArtifactHead] = None
        if config.enable_lens_artifacts:
            self.lens_artifact_head = LensArtifactHead(config.lens_artifacts)

        # ── RoPE ─────────────────────────────────────────────────────
        self.rope = AxialRoPE3D(config.rope)

        # ── entity bank ──────────────────────────────────────────────
        self.entity_bank: Optional[EntityTokenBank] = None
        if config.enable_entity_attention:
            self.entity_bank = EntityTokenBank(
                EntityBankConfig(
                    num_slots=config.entity.num_slots,
                    dim=config.dim,
                    init_scale=config.entity.init_scale,
                )
            )

        # ── glyph branch ────────────────────────────────────────────
        self.glyph_branch: Optional[GlyphBranch] = None
        if config.enable_glyph_fusion:
            gcfg = GlyphBranchConfig(
                in_channels=config.glyph.in_channels,
                embed_dim=config.glyph.embed_dim,
                patch_size=config.glyph.patch_size,
                num_blocks=config.glyph.num_blocks,
                num_heads=config.glyph.num_heads,
                head_dim=config.glyph.head_dim,
                out_dim=config.dim,
                max_regions=config.glyph.max_regions,
                tokens_per_region=config.glyph.tokens_per_region,
            )
            self.glyph_branch = GlyphBranch(gcfg)

        # ── transformer blocks (Wan-native + Ogenti extras) ──────────
        self.blocks = nn.ModuleList()
        for i in range(config.num_blocks):
            block_cfg = OgentiBlockConfig(
                dim=config.dim,
                num_heads=config.num_heads,
                head_dim=config.head_dim,
                mlp_ratio=config.mlp_ratio,
                ffn_dim=config.ffn_dim,
                qk_norm_mode=config.qk_norm_mode,
                enable_entity_attention=config.enable_entity_attention,
                enable_glyph_fusion=(
                    config.enable_glyph_fusion and i >= config.glyph_fusion_start_block
                ),
                # Wan2.2-A14B always has per-block text cross-attention; we keep it
                # always-on so we have somewhere to land the cross_attn shards.
                enable_cross_attn=True,
                cross_attn_dim=config.dim,
                source_block_name=f"transformer_blocks.{i}",
            )
            self.blocks.append(OgentiBlock(block_cfg))

        # ── output head (Wan-native: `head.modulation` Param + `head.head` Linear) ─
        out_dim = config.out_channels * pt * ph * pw
        self.head = WanOutputHead(dim=config.dim, out_dim=out_dim)

    # ────────────────────────────────────────────────────────────────
    # Forward — latent space
    # ────────────────────────────────────────────────────────────────

    def forward(
        self,
        latents: Tensor,
        timesteps: Tensor,
        text_embeddings: Tensor,
        glyph_crops: Optional[Tensor] = None,
        glyph_mask: Optional[Tensor] = None,
        camera_motion_descriptor: Optional[Tensor] = None,
        subject_motion_descriptors: Optional[Tensor] = None,
        subject_motion_mask: Optional[Tensor] = None,
        micro_event_blink: Optional[Tensor] = None,
        micro_event_breath: Optional[Tensor] = None,
        micro_event_idle: Optional[Tensor] = None,
        physics_keyframe_descriptor: Optional[Tensor] = None,
        physics_keyframe_object_mask: Optional[Tensor] = None,
    ) -> Tensor:
        b = latents.shape[0]
        d = self.config.dim
        pt, ph, pw = self.config.patch.patch_size

        # ── patch embed (B, C, T, H, W) -> (B, N, dim) ──────────────
        x_emb = self.patch_embedding(latents)
        _, _, t_grid, h_grid, w_grid = x_emb.shape
        patch_tokens = rearrange(x_emb, "b d t h w -> b (t h w) d")

        rope_freqs = self.rope.freqs(t_grid, h_grid, w_grid, device=latents.device)

        # ── text / time conditioning (Wan-native) ────────────────────
        context = self.text_embedding(text_embeddings)              # (B, L, dim)
        t_emb = self.time_embedding(self.sinusoidal_t(timesteps))   # (B, dim)

        cond = t_emb
        if self.camera_motion_embed is not None:
            cond = cond + self.camera_motion_embed(
                camera_motion_descriptor, batch_size=b, device=latents.device, dtype=cond.dtype
            )
        if self.subject_motion_embed is not None:
            cond = cond + self.subject_motion_embed(
                subject_motion_descriptors, subject_motion_mask,
                batch_size=b, device=latents.device, dtype=cond.dtype,
            )
        if self.micro_event_embed is not None:
            cond = cond + self.micro_event_embed(
                micro_event_blink, micro_event_breath, micro_event_idle,
                batch_size=b, device=latents.device, dtype=cond.dtype,
            )
        if self.physics_keyframe_embed is not None:
            cond = cond + self.physics_keyframe_embed(
                physics_keyframe_descriptor, physics_keyframe_object_mask,
                batch_size=b, device=latents.device, dtype=cond.dtype,
            )

        # Wan2.2 per-step AdaLN params: time_projection(cond) -> (B, 6*dim) -> (B, 6, dim).
        e6 = self.time_projection(cond).reshape(b, 6, d)

        # ── entity tokens ────────────────────────────────────────────
        entity_tokens = None
        if self.entity_bank is not None:
            entity_tokens = self.entity_bank.expand(b)

        # ── glyph branch ────────────────────────────────────────────
        glyph_stream = None
        glyph_token_mask = None
        if self.glyph_branch is not None and glyph_crops is not None and glyph_mask is not None:
            glyph_stream, glyph_token_mask = self.glyph_branch(glyph_crops, glyph_mask)

        # ── transformer blocks ───────────────────────────────────────
        for block in self.blocks:
            patch_tokens, entity_tokens = block(
                patch_tokens=patch_tokens,
                e6=e6,
                entity_tokens=entity_tokens,
                glyph_stream=glyph_stream,
                glyph_mask=glyph_token_mask,
                context=context,
                rope_freqs=rope_freqs,
            )

        # ── output projection (Wan-native head: norm + AdaLN + Linear) ─
        x = self.head(patch_tokens, t_emb)

        # ── unpatchify back to (B, C_out, T, H, W) ──────────────────
        return rearrange(
            x,
            "b (t h w) (c pt ph pw) -> b c (t pt) (h ph) (w pw)",
            t=t_grid, h=h_grid, w=w_grid,
            pt=pt, ph=ph, pw=pw, c=self.config.out_channels,
        )

    # ────────────────────────────────────────────────────────────────
    # Post-VAE-decode refinement methods (pixel space)
    # ────────────────────────────────────────────────────────────────

    def refine_skin_detail(
        self,
        decoded_video: Tensor,
        skin_mask: Tensor,
    ) -> Tensor:
        """Post-VAE-decode HF residual refinement for skin regions.
        Returns unchanged video if SDRH disabled."""
        if self.skin_detail_head is None:
            return decoded_video
        return self.skin_detail_head(decoded_video, skin_mask)

    def refine_material_detail(
        self,
        decoded_video: Tensor,
        material_masks: Tensor,
    ) -> Tensor:
        """Post-VAE-decode HF residual refinement for all material classes.
        Returns unchanged video if UMDH disabled."""
        if self.material_detail_head is None:
            return decoded_video
        return self.material_detail_head(decoded_video, material_masks)

    def apply_motion_blur(
        self,
        decoded_video: Tensor,
        flow: Optional[Tensor] = None,
    ) -> Tensor:
        """Apply velocity-conditioned motion blur. If no flow is provided,
        estimates it from frame differences. Returns unchanged video if disabled."""
        if self.motion_blur_head is None:
            return decoded_video
        if flow is None:
            flow = estimate_flow_from_frames(decoded_video)
        return self.motion_blur_head(decoded_video, flow)

    def apply_film_grain(
        self,
        decoded_video: Tensor,
        amount: Tensor | float | None = None,
        grain_size: float | None = None,
        seed: int | None = None,
    ) -> Tensor:
        """Add parameterized film grain overlay.
        Returns unchanged video if disabled."""
        if self.film_grain_head is None:
            return decoded_video
        return self.film_grain_head(decoded_video, amount=amount, grain_size=grain_size, seed=seed)

    def apply_lens_artifacts(
        self,
        decoded_video: Tensor,
        descriptor: Optional[Tensor] = None,
    ) -> Tensor:
        """Apply chromatic aberration, vignette, and glow.
        Returns unchanged video if disabled."""
        if self.lens_artifact_head is None:
            return decoded_video
        return self.lens_artifact_head(decoded_video, descriptor=descriptor)

    def apply_full_postprocess(
        self,
        decoded_video: Tensor,
        skin_mask: Optional[Tensor] = None,
        material_masks: Optional[Tensor] = None,
        flow: Optional[Tensor] = None,
        film_grain_amount: Tensor | float | None = None,
        film_grain_size: float | None = None,
        film_grain_seed: int | None = None,
        lens_descriptor: Optional[Tensor] = None,
    ) -> Tensor:
        """Run the full post-VAE pixel-space refinement chain.

        Order matters: texture heads first (restore HF detail), then optical
        simulation heads (which would otherwise blur restored detail).

        1. Skin detail (SDRH)  — restore pore-level texture
        2. Material detail (UMDH) — restore per-material HF
        3. Motion blur — simulate shutter integration
        4. Film grain — add sensor/film noise
        5. Lens artifacts — CA, vignette, glow
        """
        video = decoded_video

        if skin_mask is not None:
            video = self.refine_skin_detail(video, skin_mask)

        if material_masks is not None:
            video = self.refine_material_detail(video, material_masks)

        video = self.apply_motion_blur(video, flow=flow)
        video = self.apply_film_grain(video, amount=film_grain_amount,
                                      grain_size=film_grain_size, seed=film_grain_seed)
        video = self.apply_lens_artifacts(video, descriptor=lens_descriptor)

        return video

    # ────────────────────────────────────────────────────────────────
    # Utility
    # ────────────────────────────────────────────────────────────────

    def num_parameters(self, trainable_only: bool = False) -> int:
        return sum(p.numel() for p in self.parameters() if (p.requires_grad or not trainable_only))

    def enable_gradient_checkpointing(self, enabled: bool = True) -> None:
        for block in self.blocks:
            block._use_gradient_checkpointing = enabled
