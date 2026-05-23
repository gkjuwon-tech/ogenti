"""Per-subject motion descriptors (RFC-0003 §3, RFC-0004 M-class).

For each tracked subject we compute an 8-dim descriptor::

    [center_drift_x, center_drift_y, accel_mag, vel_std,
     dwell_fraction, micro_bounce, contact_proxy, weightlessness_proxy]

This is a low-cost proxy: we use MediaPipe Pose centroid to track up to N
subjects. When MediaPipe is unavailable we fall back to a single global
foreground centroid derived from optical flow.
"""

from __future__ import annotations

import numpy as np

from ogenti.utils.logging import get_logger

log = get_logger("ogenti.data.preprocess.subject_motion")


def _pose_centroids(frames: np.ndarray) -> np.ndarray | None:
    try:
        import mediapipe as mp

        pose = mp.solutions.pose.Pose(
            static_image_mode=False, model_complexity=0, min_detection_confidence=0.3
        )
    except Exception as e:
        log.debug(f"mediapipe pose unavailable: {e}")
        return None
    T = frames.shape[0]
    centroids = np.full((T, 2), np.nan, dtype=np.float32)
    for t in range(T):
        try:
            res = pose.process(frames[t])
            lm = getattr(res, "pose_landmarks", None)
            if lm is None:
                continue
            pts = [(p.x, p.y) for p in lm.landmark]
            arr = np.asarray(pts, dtype=np.float32)
            centroids[t] = arr.mean(axis=0)
        except Exception:
            continue
    return centroids


def _flow_centroid(frames: np.ndarray) -> np.ndarray:
    try:
        import cv2
    except ImportError:
        return np.full((frames.shape[0], 2), np.nan, dtype=np.float32)

    T, H, W = frames.shape[:3]
    centroids = np.full((T, 2), 0.5, dtype=np.float32)
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
    for t in range(1, T):
        curr_gray = cv2.cvtColor(frames[t], cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 21, 3, 5, 1.2, 0)
        mag = np.linalg.norm(flow, axis=-1)
        if mag.sum() <= 1e-6:
            centroids[t] = centroids[t - 1]
        else:
            ys, xs = np.mgrid[0:H, 0:W]
            cx = float((xs * mag).sum() / mag.sum()) / W
            cy = float((ys * mag).sum() / mag.sum()) / H
            centroids[t] = (cx, cy)
        prev_gray = curr_gray
    return centroids


def _series_to_descriptor(centroids: np.ndarray) -> np.ndarray:
    valid = ~np.isnan(centroids).any(axis=-1)
    if valid.sum() < 4:
        return np.zeros(8, dtype=np.float32)
    pts = centroids[valid]
    center_drift = pts[-1] - pts[0]
    vel = np.diff(pts, axis=0)
    vel_std = float(vel.std())
    accel = np.diff(vel, axis=0)
    accel_mag = float(np.linalg.norm(accel, axis=-1).mean())
    dwell = float((np.linalg.norm(vel, axis=-1) < 0.005).mean())
    bounce = float(np.abs(np.diff(vel[:, 1])).mean()) if vel.shape[0] >= 2 else 0.0
    contact_proxy = float(np.clip(1.0 - accel_mag * 50.0, 0.0, 1.0))
    weightless = float(np.clip(accel_mag * 100.0 - 0.2, 0.0, 1.0))
    return np.array(
        [
            float(center_drift[0]),
            float(center_drift[1]),
            accel_mag,
            vel_std,
            dwell,
            bounce,
            contact_proxy,
            weightless,
        ],
        dtype=np.float32,
    )


def extract_subject_motion(frames: np.ndarray, max_tracks: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Returns ``(descriptors (K, 8), mask (K,) bool)``."""
    descriptors = np.zeros((max_tracks, 8), dtype=np.float32)
    mask = np.zeros(max_tracks, dtype=np.bool_)
    pose_centroids = _pose_centroids(frames)
    if pose_centroids is not None and (~np.isnan(pose_centroids)).any():
        descriptors[0] = _series_to_descriptor(pose_centroids)
        mask[0] = True
        return descriptors, mask
    flow_centroids = _flow_centroid(frames)
    descriptors[0] = _series_to_descriptor(flow_centroids)
    mask[0] = bool((descriptors[0] != 0).any())
    return descriptors, mask
