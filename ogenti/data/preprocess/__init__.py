"""Preprocessing utilities consumed by ``scripts/data/build_dataset.py`` to
materialize the per-clip signals that ``AdVideoDataset`` later loads.

All extractors degrade gracefully (return zeros) when their underlying
dependency (MediaPipe / easyocr / OpenCV) is unavailable — the retrofit
invariant tolerates zero conditioning by construction.
"""

from ogenti.data.preprocess.camera_motion import (
    CameraMotionConfig,
    compute_camera_motion,
)
from ogenti.data.preprocess.glyph_regions import GlyphConfig, extract_glyph_regions
from ogenti.data.preprocess.keypoints import extract_keypoints
from ogenti.data.preprocess.micro_events import extract_micro_events
from ogenti.data.preprocess.skin_masks import extract_skin_masks
from ogenti.data.preprocess.subject_motion import extract_subject_motion

__all__ = [
    "CameraMotionConfig",
    "compute_camera_motion",
    "GlyphConfig",
    "extract_glyph_regions",
    "extract_keypoints",
    "extract_micro_events",
    "extract_skin_masks",
    "extract_subject_motion",
]
