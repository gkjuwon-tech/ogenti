"""Skin segmentation masks (RFC-0003 §3, SDRH supervision target).

Pipeline:
  1. MediaPipe SelfieSegmentation gives a person mask.
  2. HSV chroma gate refines to skin pixels (YCrCb skin range).

Output: (T, H, W) float32 mask in [0, 1].
"""

from __future__ import annotations

import numpy as np

from ogenti.utils.logging import get_logger

log = get_logger("ogenti.data.preprocess.skin_masks")

_SEG = None


def _get_seg():
    global _SEG
    if _SEG is not None:
        return _SEG
    try:
        import mediapipe as mp

        _SEG = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
    except Exception as e:
        log.debug(f"mediapipe selfie segmentation init failed: {e}")
        _SEG = False
    return _SEG


def _ycrcb_skin_gate(frame_rgb: np.ndarray) -> np.ndarray:
    try:
        import cv2

        ycrcb = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2YCrCb)
        cr = ycrcb[..., 1]
        cb = ycrcb[..., 2]
        skin = ((cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)).astype(np.float32)
        return skin
    except Exception:
        return np.zeros(frame_rgb.shape[:2], dtype=np.float32)


def extract_skin_masks(frames: np.ndarray, target_hw: tuple[int, int] | None = None) -> np.ndarray:
    """Returns (T, H, W) float32. ``target_hw`` resizes; default = frames' resolution."""
    seg = _get_seg()
    T, H, W = frames.shape[:3]
    if target_hw is None:
        target_hw = (H, W)
    out = np.zeros((T, target_hw[0], target_hw[1]), dtype=np.float32)
    try:
        import cv2
    except ImportError:
        return out

    for t in range(T):
        person = np.ones((H, W), dtype=np.float32)
        if seg:
            try:
                res = seg.process(frames[t])
                if getattr(res, "segmentation_mask", None) is not None:
                    person = res.segmentation_mask
            except Exception as e:
                log.debug(f"selfie seg frame {t} failed: {e}")
        skin = _ycrcb_skin_gate(frames[t])
        mask = (skin * np.clip(person, 0.0, 1.0)).astype(np.float32)
        if (H, W) != target_hw:
            mask = cv2.resize(mask, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_AREA)
        out[t] = mask
    return out
