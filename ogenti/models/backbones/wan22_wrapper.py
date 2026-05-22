"""Thin wrapper around the Wan2.2 family of base models.

Responsibilities:
  - Load the pretrained VAE (frozen) and text encoder (frozen).
  - Provide a stable interface for the rest of Ogenti regardless of upstream
    Wan2.2 API churn.

As of RFC-0006 the default backbone is **Wan2.2-T2V-A14B** (14B-param MoE).
The legacy TI2V-5B path is still supported by passing
``Wan22BackboneConfig(model_id="Wan-AI/Wan2.2-TI2V-5B")`` but is no longer the
default — empirically the 5B variant cannot hold form past ~3s, mangles glyphs,
and ignores complex prompts (see RFC-0006 for the post-mortem).

Note: the actual Wan2.2 transformer is *not* used directly at inference time
in Ogenti — it is dissected by ogenti/retrofit/surgery/wan22_import.py into
OgentiBlocks. This wrapper exists for VAE encode/decode + text embedding,
both of which are identical across the Wan2.2 5B / A14B / I2V variants
(same T5 text encoder, same 16-channel VAE with 4× temporal / 8× spatial
compression).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor, nn

from ogenti.utils.logging import get_logger

log = get_logger("ogenti.wan22")


# Canonical model ids per Wan2.2 sub-variant.
WAN22_MODEL_IDS: dict[str, str] = {
    "t2v_a14b": "Wan-AI/Wan2.2-T2V-A14B",
    "i2v_a14b": "Wan-AI/Wan2.2-I2V-A14B",
    "ti2v_5b": "Wan-AI/Wan2.2-TI2V-5B",
}


@dataclass
class Wan22BackboneConfig:
    # Default to the 14B MoE T2V model (RFC-0006).
    model_id: str = "Wan-AI/Wan2.2-T2V-A14B"
    weights_dir: Optional[str] = None
    vae_dtype: torch.dtype = torch.float32
    text_dtype: torch.dtype = torch.bfloat16
    device: str = "cuda"


class Wan22Backbone(nn.Module):
    """VAE + text encoder reuse from Wan2.2. Transformer not exposed (see surgery)."""

    def __init__(self, config: Wan22BackboneConfig) -> None:
        super().__init__()
        self.config = config
        self.vae: Optional[nn.Module] = None
        self.text_encoder: Optional[nn.Module] = None
        self.tokenizer = None
        self._loaded = False

    def load(self) -> "Wan22Backbone":
        try:
            from diffusers import AutoencoderKLWan
            from transformers import AutoTokenizer, T5EncoderModel
        except ImportError as e:
            raise ImportError(
                "diffusers>=0.31 and transformers>=4.46 required for Wan2.2 loading"
            ) from e

        src = self.config.weights_dir or self.config.model_id
        log.info(f"loading Wan2.2 VAE from {src}")
        self.vae = AutoencoderKLWan.from_pretrained(
            src, subfolder="vae", torch_dtype=self.config.vae_dtype
        )
        self.vae.eval().requires_grad_(False)

        log.info(f"loading Wan2.2 text encoder from {src}")
        self.text_encoder = T5EncoderModel.from_pretrained(
            src, subfolder="text_encoder", torch_dtype=self.config.text_dtype
        )
        self.text_encoder.eval().requires_grad_(False)
        self.tokenizer = AutoTokenizer.from_pretrained(src, subfolder="tokenizer")

        self._loaded = True
        return self

    @torch.no_grad()
    def encode_text(self, prompts: list[str], max_length: int = 512) -> Tensor:
        assert self._loaded, "call .load() first"
        toks = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.text_encoder.device)
        out = self.text_encoder(input_ids=toks.input_ids, attention_mask=toks.attention_mask)
        return out.last_hidden_state

    @torch.no_grad()
    def encode_video(self, pixels: Tensor) -> Tensor:
        """pixels: (B, C=3, T, H, W) in [-1, 1] -> latents (B, C_z, T', H', W')."""
        assert self._loaded
        return self.vae.encode(pixels.to(self.config.vae_dtype)).latent_dist.sample()

    @torch.no_grad()
    def decode_latents(self, latents: Tensor) -> Tensor:
        assert self._loaded
        return self.vae.decode(latents.to(self.config.vae_dtype)).sample
