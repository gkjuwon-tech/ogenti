"""Higher-level adapter for DiT-family backbones.

Currently dispatches to Wan2.2 importer. Will fan out to CogVideoX, HunyuanVideo
adapters as we add fallback support (see RFC-0002).
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from ogenti.models.ogenti_transformer import OgentiTransformer, OgentiTransformerConfig
from ogenti.retrofit.surgery.wan22_import import import_wan22_into_ogenti
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
    **kwargs,
) -> tuple[OgentiTransformer, dict]:
    if family == BackboneFamily.WAN22:
        return import_wan22_into_ogenti(weights_path, config, **kwargs)
    raise NotImplementedError(f"retrofit for {family} not implemented; current target: WAN22")
