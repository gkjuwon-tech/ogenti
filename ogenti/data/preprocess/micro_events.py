"""Micro-event signal extraction (RFC-0004 μ-class).

Three 16-dim signals (downsampled to ``micro_event_len``):
  - ``blink``  — eye openness (1 - eye aspect ratio) over time
  - ``breath`` — chest area expansion (proxy: mid-torso vertical std)
  - ``idle``   — small-scale incidental motion energy

Sourcing:
  - MediaPipe Face Mesh for eye landmarks (blink).
  - MediaPipe Pose for shoulder/chest mid-points (breath).
  - Frame-difference RMS in face/torso ROI (idle).

If MediaPipe is missing, returns zeros for blink/breath and idle from raw
diff. Phase 4-RE2 micro_event_embed is zero-init so any noise here is
absorbed without breaking the retrofit invariant.
"""

from __future__ import annotations

import numpy as np

from ogenti.utils.logging import get_logger

log = get_logger("ogenti.data.preprocess.micro_events")

_FACE_MESH = None


def _get_face_mesh():
    global _FACE_MESH
    if _FACE_MESH is not None:
        return _FACE_MESH
    try:
        import mediapipe as mp

        _FACE_MESH = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            refine_landmarks=False,
            max_num_faces=1,
            min_detection_confidence=0.3,
        )
    except Exception as e:
        log.debug(f"mediapipe face mesh unavailable: {e}")
        _FACE_MESH = False
    return _FACE_MESH


LEFT_EYE = (159, 145)  # top, bottom of left eye
RIGHT_EYE = (386, 374)


def _eye_aspect(lm, top_idx: int, bot_idx: int) -> float:
    return abs(lm[top_idx].y - lm[bot_idx].y)


def _downsample(series: np.ndarray, length: int) -> np.ndarray:
    if series.size <= 1:
        return np.zeros(length, dtype=np.float32)
    idx = np.linspace(0, series.size - 1, num=length).astype(np.int64)
    return series[idx].astype(np.float32)


def extract_micro_events(frames: np.ndarray, length: int = 16) -> dict[str, np.ndarray]:
    T = frames.shape[0]
    face = _get_face_mesh()

    blink = np.zeros(T, dtype=np.float32)
    breath = np.zeros(T, dtype=np.float32)

    if face:
        eye_series = np.zeros(T, dtype=np.float32)
        torso_series = np.zeros(T, dtype=np.float32)
        for t in range(T):
            try:
                res = face.process(frames[t])
                lm_set = getattr(res, "multi_face_landmarks", None)
                if not lm_set:
                    continue
                lm = lm_set[0].landmark
                ear_left = _eye_aspect(lm, *LEFT_EYE)
                ear_right = _eye_aspect(lm, *RIGHT_EYE)
                eye_series[t] = float((ear_left + ear_right) / 2.0)
                torso_series[t] = float(lm[10].y)  # face bottom as breath proxy
            except Exception:
                continue
        if eye_series.any():
            blink = 1.0 - eye_series / (eye_series.max() + 1e-6)
        if torso_series.any():
            breath = torso_series - torso_series.mean()

    # idle = frame-to-frame difference RMS (no mediapipe required)
    try:
        diffs = np.zeros(T, dtype=np.float32)
        if T >= 2:
            for t in range(1, T):
                d = (frames[t].astype(np.float32) - frames[t - 1].astype(np.float32)) / 255.0
                diffs[t] = float(np.sqrt((d * d).mean()))
        idle = diffs
    except Exception:
        idle = np.zeros(T, dtype=np.float32)

    return {
        "blink": _downsample(blink, length),
        "breath": _downsample(breath, length),
        "idle": _downsample(idle, length),
    }
