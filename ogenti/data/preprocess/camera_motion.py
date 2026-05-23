"""Camera motion descriptor extraction (RFC-0003 §3).

Each clip is reduced to an 8-dim descriptor:

    [shake_xy_x, shake_xy_y, shake_rot, zoom_drift,
     entropy, handheld_logit, gimbal_logit, locked_logit]

Pipeline:
  1. Compute per-frame Farneback dense optical flow.
  2. Fit RANSAC affine (translation + rotation + scale) between consecutive frames.
  3. From the per-frame (tx, ty, theta, s) signal, derive:
       - high-frequency shake (std after detrending) → ``shake_xy_*``
       - rotational HF shake → ``shake_rot``
       - low-frequency scale drift → ``zoom_drift``
       - motion-spectrum entropy → ``entropy``
       - 3-way soft logit (handheld / gimbal / locked) from spectral fingerprint.

When OpenCV is unavailable the extractor returns zeros (which match the
``CameraMotionEmbed`` zero-init invariant — model treats it as "no signal").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ogenti.utils.logging import get_logger

log = get_logger("ogenti.data.preprocess.camera_motion")


@dataclass
class CameraMotionConfig:
    flow_pyr_scale: float = 0.5
    flow_levels: int = 3
    flow_winsize: int = 21
    flow_iterations: int = 3
    flow_poly_n: int = 5
    flow_poly_sigma: float = 1.2
    ransac_reproj_threshold: float = 3.0
    max_frames: int = 96
    target_size: tuple[int, int] = (240, 320)


def _resize_for_flow(frames: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    import cv2

    th, tw = size
    out = np.empty((frames.shape[0], th, tw), dtype=np.uint8)
    for i, f in enumerate(frames):
        gray = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
        out[i] = cv2.resize(gray, (tw, th), interpolation=cv2.INTER_AREA)
    return out


def _affine_from_flow(prev: np.ndarray, curr: np.ndarray, cfg: CameraMotionConfig) -> tuple[float, float, float, float]:
    """Return ``(tx, ty, theta, scale)`` for the global affine fit between two frames."""
    import cv2

    flow = cv2.calcOpticalFlowFarneback(
        prev,
        curr,
        None,
        cfg.flow_pyr_scale,
        cfg.flow_levels,
        cfg.flow_winsize,
        cfg.flow_iterations,
        cfg.flow_poly_n,
        cfg.flow_poly_sigma,
        0,
    )
    h, w = prev.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    src = np.stack([xs.reshape(-1), ys.reshape(-1)], axis=1)
    dst = src + flow.reshape(-1, 2)
    # Subsample for RANSAC speed
    idx = np.linspace(0, src.shape[0] - 1, num=min(8000, src.shape[0]), dtype=np.int64)
    M, _ = cv2.estimateAffinePartial2D(
        src[idx],
        dst[idx],
        method=cv2.RANSAC,
        ransacReprojThreshold=cfg.ransac_reproj_threshold,
        maxIters=2000,
        confidence=0.99,
    )
    if M is None:
        return 0.0, 0.0, 0.0, 1.0
    a, b, tx = M[0]
    c, d, ty = M[1]
    scale = float(np.sqrt(a * a + c * c))
    theta = float(np.arctan2(c, a))
    return float(tx) / w, float(ty) / h, theta, scale


def compute_camera_motion(frames: np.ndarray, config: CameraMotionConfig | None = None) -> np.ndarray:
    """Return an 8-dim float32 camera motion descriptor.

    ``frames``: (T, H, W, 3) uint8 in RGB. If ``T < 2`` returns zeros.
    """
    if config is None:
        config = CameraMotionConfig()
    try:
        import cv2  # noqa: F401
    except ImportError:
        log.debug("opencv unavailable; returning zero camera motion descriptor")
        return np.zeros(8, dtype=np.float32)

    if frames.shape[0] < 2:
        return np.zeros(8, dtype=np.float32)
    if frames.shape[0] > config.max_frames:
        idx = np.linspace(0, frames.shape[0] - 1, num=config.max_frames, dtype=np.int64)
        frames = frames[idx]

    try:
        gray = _resize_for_flow(frames, config.target_size)
    except Exception as e:
        log.debug(f"flow preproc failed: {e}; zeros")
        return np.zeros(8, dtype=np.float32)

    series = np.zeros((gray.shape[0] - 1, 4), dtype=np.float32)
    for i in range(gray.shape[0] - 1):
        try:
            tx, ty, theta, scale = _affine_from_flow(gray[i], gray[i + 1], config)
        except Exception:
            tx = ty = theta = 0.0
            scale = 1.0
        series[i] = (tx, ty, theta, scale - 1.0)

    # HF shake = std of detrended signal (subtract running mean window=5)
    def _detrend_std(x: np.ndarray) -> float:
        if x.size < 4:
            return 0.0
        k = min(5, x.size)
        kernel = np.ones(k) / k
        trend = np.convolve(x, kernel, mode="same")
        return float(np.std(x - trend))

    shake_x = _detrend_std(series[:, 0])
    shake_y = _detrend_std(series[:, 1])
    shake_rot = _detrend_std(series[:, 2])
    zoom_drift = float(np.mean(series[:, 3]))

    # Motion entropy via FFT power spectrum
    spec = np.abs(np.fft.rfft(series[:, 0])) ** 2 + np.abs(np.fft.rfft(series[:, 1])) ** 2
    p = spec / (spec.sum() + 1e-9)
    entropy = float(-(p * np.log(p + 1e-12)).sum() / np.log(max(2, p.size)))

    # 3-way fingerprint logits (informal, supervisable later).
    hf_energy = float(shake_x + shake_y + shake_rot)
    lf_energy = float(np.std(series[:, 0]) + np.std(series[:, 1])) - hf_energy
    locked_logit = -math_log1p(hf_energy + abs(lf_energy))
    handheld_logit = math_log1p(hf_energy * 4.0)
    gimbal_logit = math_log1p(max(0.0, lf_energy * 2.0 - hf_energy))

    return np.array(
        [shake_x, shake_y, shake_rot, zoom_drift, entropy, handheld_logit, gimbal_logit, locked_logit],
        dtype=np.float32,
    )


def math_log1p(x: float) -> float:
    """``log1p`` with float fallback so this file imports without ``math`` import noise."""
    import math

    return float(math.log1p(max(0.0, x)))
