"""Higher-level adapter for DiT-family backbones.

Currently dispatches to Wan2.2 importer. Will fan out to CogVideoX, HunyuanVideo
adapters as we add fallback support (see RFC-0002, RFC-0006).
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

from ogenti.models.ogenti_transformer import OgentiTransformer, OgentiTransformerConfig
from ogenti.retrofit.surgery.wan22_import import ExpertSelector, import_wan22_into_ogenti
from ogenti.utils.logging import get_logger

log = get_logger("ogenti.retrofit")


class BackboneFamily(str, Enum):
    WAN22 = "wan22"
    COGVIDEOX = "cogvideox"
    HUNYUAN = "hunyuan"


def retrofit_from_backbone(
    family: BackboneFamily,
    weights_path: Path | str,
    config: OgentiTransformerConfig,
    *,
    expert: Optional[ExpertSelector] = None,
    **kwargs,
) -> tuple[OgentiTransformer, dict]:
    """Run weight surgery for the given backbone family.

    Args:
        family: which upstream DiT family the weights came from.
        weights_path: path to weights (directory or single safetensors file).
            For Wan2.2-A14B this can be the snapshot root containing
            ``high_noise_model/`` and ``low_noise_model/`` subdirs; the
            specific expert is selected by ``expert``.
        config: Ogenti model config to instantiate.
        expert: for Wan2.2-A14B MoE checkpoints, pick which expert to load
            (``"low_noise"`` default for fine-detail, ``"high_noise"`` for the
            early-timestep coarse-structure expert).
    """
    if family == BackboneFamily.WAN22:
        return import_wan22_into_ogenti(weights_path, config, expert=expert, **kwargs)
    raise NotImplementedError(f"retrofit for {family} not implemented; current target: WAN22")
