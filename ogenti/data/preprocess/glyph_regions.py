"""Glyph region extraction (RFC-0001 §3.2, Dedicated Glyph Branch supervision).

Pipeline:
  1. Run easyocr on a sparse set of keyframes (every Nth frame).
  2. Pick top-K boxes by area × confidence.
  3. Propagate each box across the clip via centroid optical-flow tracking
     (cheap proxy — sufficient for crops at training time).
  4. Crop and save each region to disk.

Returns a list of dicts compatible with the manifest schema::

    {"bbox": [x, y, w, h], "text": "COCA-COLA", "crop": "glyphs/<id>.jpg"}
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from ogenti.utils.logging import get_logger

log = get_logger("ogenti.data.preprocess.glyph_regions")

_OCR = None


def _get_ocr(langs: tuple[str, ...] = ("en", "ko")):
    global _OCR
    if _OCR is not None:
        return _OCR
    try:
        import easyocr

        _OCR = easyocr.Reader(list(langs), gpu=False, verbose=False)
    except Exception as e:
        log.debug(f"easyocr init failed: {e}")
        _OCR = False
    return _OCR


@dataclass
class GlyphConfig:
    max_regions: int = 16
    keyframe_stride: int = 8
    min_confidence: float = 0.3
    min_area_pixels: int = 400
    crop_size: int = 128


def extract_glyph_regions(
    frames: np.ndarray,
    out_dir: Path,
    clip_id: str,
    config: GlyphConfig | None = None,
) -> list[dict[str, object]]:
    """Detect top text regions and write crops to ``out_dir``.

    ``frames``: (T, H, W, 3) uint8 RGB.
    Returns list of region dicts (paths are relative to ``out_dir.parent.parent``).
    """
    if config is None:
        config = GlyphConfig()
    reader = _get_ocr()
    if not reader:
        return []

    T, H, W = frames.shape[:3]
    candidates: list[tuple[float, tuple[int, int, int, int], str, int]] = []
    for t in range(0, T, config.keyframe_stride):
        try:
            res = reader.readtext(frames[t])
        except Exception as e:
            log.debug(f"ocr frame {t} failed: {e}")
            continue
        for poly, text, conf in res:
            if conf < config.min_confidence:
                continue
            poly_arr = np.asarray(poly, dtype=np.float32)
            x = int(poly_arr[:, 0].min())
            y = int(poly_arr[:, 1].min())
            x2 = int(poly_arr[:, 0].max())
            y2 = int(poly_arr[:, 1].max())
            w = max(1, x2 - x)
            h = max(1, y2 - y)
            if w * h < config.min_area_pixels:
                continue
            score = float(conf) * float(w * h)
            candidates.append((score, (x, y, w, h), str(text), t))

    # Deduplicate by IoU keeping highest-scoring.
    candidates.sort(reverse=True, key=lambda c: c[0])
    kept: list[tuple[float, tuple[int, int, int, int], str, int]] = []
    for cand in candidates:
        if len(kept) >= config.max_regions:
            break
        if all(_iou(cand[1], k[1]) < 0.5 for k in kept):
            kept.append(cand)

    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import cv2
    except ImportError:
        return []

    regions: list[dict[str, object]] = []
    for i, (_, bbox, text, t_idx) in enumerate(kept):
        x, y, w, h = bbox
        crop = frames[t_idx, max(0, y) : min(H, y + h), max(0, x) : min(W, x + w)]
        if crop.size == 0:
            continue
        crop_resized = cv2.resize(crop, (config.crop_size, config.crop_size), interpolation=cv2.INTER_AREA)
        crop_bgr = cv2.cvtColor(crop_resized, cv2.COLOR_RGB2BGR)
        rel_path = f"glyphs/{clip_id}_{i:02d}.jpg"
        abs_path = out_dir.parent / rel_path
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(abs_path), crop_bgr)
        regions.append(
            {
                "bbox": [x / W, y / H, w / W, h / H],
                "text": text,
                "crop": rel_path,
                "keyframe": int(t_idx),
            }
        )
    return regions


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter + 1e-6
    return float(inter / union)
