"""MediaPipe Pose-based keypoint extraction (RFC-0001 §3, RFC-0004 M-class).

Output shapes:
  - ``keypoints``     (T, 33, 2)  normalized to [0, 1]
  - ``keypoint_conf`` (T, 33)     in [0, 1]

When MediaPipe is unavailable we degrade to zeros so the dataset loader stays
healthy. ``anatomy_loss`` weight is 0 in smoke configs so zero supervision is
a no-op early in the curriculum.
"""

from __future__ import annotations

import numpy as np

from ogenti.utils.logging import get_logger

log = get_logger("ogenti.data.preprocess.keypoints")

_POSE = None


def _get_pose():
    global _POSE
    if _POSE is not None:
        return _POSE
    try:
        import mediapipe as mp

        _POSE = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
        )
    except Exception as e:
        log.debug(f"mediapipe pose init failed: {e}")
        _POSE = False
    return _POSE


def extract_keypoints(frames: np.ndarray, num_joints: int = 33) -> tuple[np.ndarray, np.ndarray]:
    """Returns ``(keypoints (T, J, 2), conf (T, J))`` in normalized coords."""
    pose = _get_pose()
    T = frames.shape[0]
    kp = np.zeros((T, num_joints, 2), dtype=np.float32)
    conf = np.zeros((T, num_joints), dtype=np.float32)
    if not pose:
        return kp, conf

    for t in range(T):
        try:
            res = pose.process(frames[t])
            lm = getattr(res, "pose_landmarks", None)
            if lm is None:
                continue
            for j, p in enumerate(lm.landmark[:num_joints]):
                kp[t, j, 0] = float(p.x)
                kp[t, j, 1] = float(p.y)
                conf[t, j] = float(getattr(p, "visibility", 1.0))
        except Exception as e:
            log.debug(f"pose frame {t} failed: {e}")
    return kp, conf
