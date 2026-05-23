"""AdVideoDataset — manifest-driven video dataset for Ogenti training.

The training loop in ``scripts/train/train.py`` expects this module to expose:

- :class:`VideoDatasetConfig` — dataclass referenced from training YAML configs
- :class:`AdVideoDataset` — PyTorch :class:`torch.utils.data.Dataset`
- :func:`collate_fn` — batch collator that handles optional / variable-length keys

The manifest is a JSONL file (``data/manifests/ads_train.jsonl`` by default).
Each line is a JSON object describing one shot, with paths relative to ``root``.

Minimum fields per entry::

    {"video": "videos/abcd.mp4", "prompt": "A bottle of cola falls onto marble."}

Optional precomputed signals (only loaded if the corresponding ``load_*`` flag
is True; missing files degrade gracefully to zero tensors so smoke runs that
only need ``video + prompt`` never break):

- ``camera_motion``: ``.npy`` of shape (8,)   — Phase 4-RE
- ``subject_motion``: ``.npy`` (max_tracks, 8) — Phase 4-RE2
- ``subject_motion_mask``: ``.npy`` bool (max_tracks,)
- ``micro_events``: ``.npz`` with keys {blink, breath, idle} each (T,)
- ``keypoints``: ``.npy`` (T, J, 2)
- ``keypoint_conf``: ``.npy`` (T, J)
- ``skin_mask``: ``.npy`` (T, H, W) uint8 / bool
- ``material_mask``: ``.npy`` (T, H, W) int (per-pixel material class)
- ``object_trajectories``: ``.npy`` (max_obj, T, 4) — [x, y, w, h] in normalized image coords
- ``object_track_mask``: ``.npy`` bool (max_obj,)
- ``glyph_regions``: list of {bbox: [x,y,w,h], text: str, crop: relpath}
- ``physics_keyframes``: ``.npz`` with arrays {descriptor (K, T, 10), mask (K,)}
- ``physics_realism_score``: float in [0, 1]
- ``tanghulu_score``: float — RFC-0003 skin-retouched score (used to down-weight)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from ogenti.utils.logging import get_logger

log = get_logger("ogenti.data.video_dataset")


@dataclass
class VideoDatasetConfig:
    """Configuration consumed by training YAMLs (see ``ogenti/configs/training/*.yaml``)."""

    manifest_path: str = "data/manifests/ads_train.jsonl"
    root: str = "data/"

    target_frames: int = 81
    target_height: int = 480
    target_width: int = 832
    target_fps: int = 24

    load_keypoints: bool = False
    load_glyph_regions: bool = False
    load_skin_masks: bool = False
    load_camera_motion: bool = False
    load_subject_motion: bool = False
    load_material_masks: bool = False
    load_object_trajectories: bool = False
    load_micro_events: bool = False
    load_physics_scenes: bool = False

    max_glyph_regions: int = 0
    glyph_crop_size: int = 64

    max_subject_tracks: int = 4
    max_physics_objects: int = 8
    physics_tokens_per_object: int = 16
    physics_descriptor_per_token: int = 10

    micro_event_len: int = 16
    keypoint_joints: int = 33
    weight_by_tanghulu_score: bool = True

    pixel_range: tuple[float, float] = (-1.0, 1.0)
    fail_on_missing: bool = False


def _resolve(root: Path, rel: Optional[str]) -> Optional[Path]:
    if not rel:
        return None
    p = Path(rel)
    return p if p.is_absolute() else (root / p)


def _read_video_frames(
    path: Path,
    target_frames: int,
    target_h: int,
    target_w: int,
    target_fps: int,
) -> tuple[np.ndarray, float]:
    """Read ``target_frames`` frames at ``target_fps``, resize to ``(target_h, target_w)``.

    Returns ``(frames, source_fps)`` where ``frames`` is a (T, H, W, 3) uint8 array.
    Tries ``decord`` first (fast), falls back to ``imageio`` then ``opencv``.
    """
    try:
        import decord  # type: ignore

        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(str(path), width=target_w, height=target_h)
        src_fps = float(vr.get_avg_fps() or target_fps)
        stride = max(1, round(src_fps / target_fps))
        idx = list(range(0, len(vr), stride))[:target_frames]
        if len(idx) < target_frames and len(vr) > 0:
            idx = idx + [idx[-1]] * (target_frames - len(idx))
        arr = vr.get_batch(idx).asnumpy()
        return arr.astype(np.uint8), src_fps
    except Exception:  # pragma: no cover - decord unavailable on Windows etc.
        pass

    try:
        import imageio.v3 as iio

        meta = iio.immeta(str(path))
        src_fps = float(meta.get("fps", target_fps))
        stride = max(1, round(src_fps / target_fps))
        out: list[np.ndarray] = []
        for i, frame in enumerate(iio.imiter(str(path))):
            if i % stride != 0:
                continue
            if frame.ndim == 2:
                frame = np.stack([frame] * 3, axis=-1)
            if frame.shape[-1] == 4:
                frame = frame[..., :3]
            out.append(frame)
            if len(out) >= target_frames:
                break
        if not out:
            raise RuntimeError(f"no frames decoded from {path}")
        while len(out) < target_frames:
            out.append(out[-1])
        arr = np.stack(out, axis=0)
        if arr.shape[1] != target_h or arr.shape[2] != target_w:
            import cv2

            arr = np.stack(
                [cv2.resize(f, (target_w, target_h), interpolation=cv2.INTER_AREA) for f in arr],
                axis=0,
            )
        return arr.astype(np.uint8), src_fps
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"failed to read video {path}: {e}") from e


def _video_to_tensor(frames: np.ndarray, pixel_range: tuple[float, float]) -> Tensor:
    """Convert (T, H, W, 3) uint8 -> (C, T, H, W) float in ``pixel_range``."""
    lo, hi = pixel_range
    arr = frames.astype(np.float32) / 255.0
    arr = arr * (hi - lo) + lo
    t = torch.from_numpy(arr).permute(3, 0, 1, 2).contiguous()
    return t


def _safe_load_npy(path: Optional[Path], shape: tuple[int, ...], dtype: Any) -> np.ndarray:
    if path is not None and path.exists():
        try:
            arr = np.load(str(path))
            return arr.astype(dtype)
        except Exception as e:
            log.debug(f"failed to load {path}: {e}; using zeros {shape}")
    return np.zeros(shape, dtype=dtype)


class AdVideoDataset(Dataset):
    """Manifest-driven video dataset that yields the full Ogenti training payload."""

    def __init__(self, config: VideoDatasetConfig) -> None:
        super().__init__()
        self.config = config
        self.root = Path(config.root)
        manifest_path = Path(config.manifest_path)
        if not manifest_path.is_absolute():
            # Allow manifest_path to be relative to either CWD or root.
            if not manifest_path.exists():
                manifest_path = self.root / manifest_path.name
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"manifest not found: {manifest_path} (root={self.root}). "
                "Build a manifest with `python -m scripts.data.build_dataset`."
            )
        with open(manifest_path) as f:
            self.entries = [json.loads(line) for line in f if line.strip()]
        log.info(f"loaded {len(self.entries)} entries from {manifest_path}")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        cfg = self.config
        entry = self.entries[idx]

        video_rel = entry["video"]
        video_path = _resolve(self.root, video_rel)
        if video_path is None or not video_path.exists():
            if cfg.fail_on_missing:
                raise FileNotFoundError(video_path)
            log.warning(f"missing video {video_path}; substituting next entry")
            return self.__getitem__((idx + 1) % len(self.entries))

        frames, _ = _read_video_frames(
            video_path,
            cfg.target_frames,
            cfg.target_height,
            cfg.target_width,
            cfg.target_fps,
        )
        video = _video_to_tensor(frames, cfg.pixel_range)

        sample: dict[str, Any] = {
            "video": video,
            "prompt": entry.get("prompt", entry.get("caption", "")),
            "video_id": entry.get("id", str(idx)),
        }

        if cfg.load_camera_motion:
            cm_rel = entry.get("camera_motion") or entry.get("camera_motion_path")
            arr = _safe_load_npy(_resolve(self.root, cm_rel), (8,), np.float32)
            sample["camera_motion_descriptor"] = torch.from_numpy(arr)

        if cfg.load_subject_motion:
            sm_rel = entry.get("subject_motion") or entry.get("subject_motion_path")
            sm = _safe_load_npy(
                _resolve(self.root, sm_rel),
                (cfg.max_subject_tracks, 8),
                np.float32,
            )
            sm_mask_rel = entry.get("subject_motion_mask")
            mask = _safe_load_npy(
                _resolve(self.root, sm_mask_rel),
                (cfg.max_subject_tracks,),
                np.bool_,
            )
            if not mask.any():
                # Default: first track active when descriptor is non-zero.
                mask[0] = bool((sm[0] != 0).any())
            sample["subject_motion_descriptors"] = torch.from_numpy(sm)
            sample["subject_motion_mask"] = torch.from_numpy(mask)

        if cfg.load_micro_events:
            me_rel = entry.get("micro_events") or entry.get("micro_events_path")
            blink, breath, idle = self._load_micro_events(
                _resolve(self.root, me_rel), cfg.micro_event_len
            )
            sample["micro_event_blink"] = torch.from_numpy(blink)
            sample["micro_event_breath"] = torch.from_numpy(breath)
            sample["micro_event_idle"] = torch.from_numpy(idle)

        if cfg.load_keypoints:
            kp_rel = entry.get("keypoints")
            kp = _safe_load_npy(
                _resolve(self.root, kp_rel),
                (cfg.target_frames, cfg.keypoint_joints, 2),
                np.float32,
            )
            conf_rel = entry.get("keypoint_conf")
            conf = _safe_load_npy(
                _resolve(self.root, conf_rel),
                (cfg.target_frames, cfg.keypoint_joints),
                np.float32,
            )
            sample["keypoints"] = torch.from_numpy(kp)
            sample["keypoint_conf"] = torch.from_numpy(conf)

        if cfg.load_skin_masks:
            sk_rel = entry.get("skin_mask")
            sk = _safe_load_npy(
                _resolve(self.root, sk_rel),
                (cfg.target_frames, cfg.target_height, cfg.target_width),
                np.float32,
            )
            sample["skin_mask"] = torch.from_numpy(sk)

        if cfg.load_material_masks:
            mt_rel = entry.get("material_mask")
            mt = _safe_load_npy(
                _resolve(self.root, mt_rel),
                (cfg.target_frames, cfg.target_height, cfg.target_width),
                np.int64,
            )
            sample["material_mask"] = torch.from_numpy(mt)

        if cfg.load_object_trajectories:
            tr_rel = entry.get("object_trajectories")
            tr = _safe_load_npy(
                _resolve(self.root, tr_rel),
                (cfg.max_physics_objects, cfg.target_frames, 4),
                np.float32,
            )
            tm_rel = entry.get("object_track_mask")
            tm = _safe_load_npy(
                _resolve(self.root, tm_rel),
                (cfg.max_physics_objects,),
                np.bool_,
            )
            sample["object_trajectories"] = torch.from_numpy(tr)
            sample["object_track_mask"] = torch.from_numpy(tm)

        if cfg.load_physics_scenes:
            pk = self._load_physics_keyframes(entry, cfg)
            if pk is not None:
                sample.update(pk)

        if cfg.load_glyph_regions and cfg.max_glyph_regions > 0:
            crops, mask, texts = self._load_glyph_regions(entry, cfg)
            sample["glyph_crops"] = crops
            sample["glyph_mask"] = mask
            sample["glyph_texts"] = texts

        if cfg.weight_by_tanghulu_score and "tanghulu_score" in entry:
            # RFC-0003: down-weight heavily-retouched skin shots.
            sample["sample_weight"] = float(max(0.0, 1.0 - float(entry["tanghulu_score"])))

        return sample

    def _load_micro_events(
        self, path: Optional[Path], length: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if path is None or not path.exists():
            zeros = np.zeros(length, dtype=np.float32)
            return zeros.copy(), zeros.copy(), zeros.copy()
        try:
            data = np.load(str(path))
            blink = np.asarray(data.get("blink", np.zeros(length)), dtype=np.float32)
            breath = np.asarray(data.get("breath", np.zeros(length)), dtype=np.float32)
            idle = np.asarray(data.get("idle", np.zeros(length)), dtype=np.float32)
            return self._pad_to(blink, length), self._pad_to(breath, length), self._pad_to(idle, length)
        except Exception as e:
            log.debug(f"micro events load failed for {path}: {e}")
            zeros = np.zeros(length, dtype=np.float32)
            return zeros.copy(), zeros.copy(), zeros.copy()

    @staticmethod
    def _pad_to(arr: np.ndarray, length: int) -> np.ndarray:
        if arr.shape[0] >= length:
            return arr[:length]
        out = np.zeros(length, dtype=arr.dtype)
        out[: arr.shape[0]] = arr
        return out

    def _load_physics_keyframes(
        self, entry: dict[str, Any], cfg: VideoDatasetConfig
    ) -> Optional[dict[str, Tensor]]:
        path = _resolve(self.root, entry.get("physics_keyframes"))
        desc_shape = (cfg.max_physics_objects, cfg.physics_tokens_per_object, cfg.physics_descriptor_per_token)
        mask_shape = (cfg.max_physics_objects,)
        if path is None or not path.exists():
            return {
                "physics_keyframe_descriptor": torch.zeros(desc_shape, dtype=torch.float32),
                "physics_keyframe_object_mask": torch.zeros(mask_shape, dtype=torch.bool),
                "physics_realism_score": torch.tensor(
                    float(entry.get("physics_realism_score", 1.0)), dtype=torch.float32
                ),
            }
        try:
            data = np.load(str(path))
            desc = np.asarray(data["descriptor"], dtype=np.float32)
            mask = np.asarray(data.get("mask", np.zeros(cfg.max_physics_objects)), dtype=np.bool_)
            desc = self._fit_to_shape(desc, desc_shape)
            mask = self._fit_to_shape(mask, mask_shape)
            return {
                "physics_keyframe_descriptor": torch.from_numpy(desc),
                "physics_keyframe_object_mask": torch.from_numpy(mask),
                "physics_realism_score": torch.tensor(
                    float(entry.get("physics_realism_score", 1.0)), dtype=torch.float32
                ),
            }
        except Exception as e:
            log.debug(f"physics keyframes load failed for {path}: {e}")
            return {
                "physics_keyframe_descriptor": torch.zeros(desc_shape, dtype=torch.float32),
                "physics_keyframe_object_mask": torch.zeros(mask_shape, dtype=torch.bool),
                "physics_realism_score": torch.tensor(1.0, dtype=torch.float32),
            }

    @staticmethod
    def _fit_to_shape(arr: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
        out = np.zeros(shape, dtype=arr.dtype)
        sl = tuple(slice(0, min(a, b)) for a, b in zip(arr.shape, shape))
        out[sl] = arr[sl]
        return out

    def _load_glyph_regions(
        self, entry: dict[str, Any], cfg: VideoDatasetConfig
    ) -> tuple[Tensor, Tensor, list[str]]:
        regions = entry.get("glyph_regions", []) or []
        regions = regions[: cfg.max_glyph_regions]
        crops_arr = np.zeros(
            (cfg.max_glyph_regions, 3, cfg.glyph_crop_size, cfg.glyph_crop_size),
            dtype=np.float32,
        )
        mask = np.zeros(cfg.max_glyph_regions, dtype=np.bool_)
        texts: list[str] = []
        for i, region in enumerate(regions):
            crop_rel = region.get("crop")
            crop_path = _resolve(self.root, crop_rel)
            if crop_path is None or not crop_path.exists():
                continue
            try:
                import cv2

                img = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                img = cv2.resize(img, (cfg.glyph_crop_size, cfg.glyph_crop_size), interpolation=cv2.INTER_AREA)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
                crops_arr[i] = img.transpose(2, 0, 1)
                mask[i] = True
                texts.append(str(region.get("text", "")))
            except Exception as e:
                log.debug(f"glyph crop load failed for {crop_path}: {e}")
        while len(texts) < cfg.max_glyph_regions:
            texts.append("")
        return torch.from_numpy(crops_arr), torch.from_numpy(mask), texts


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Variable-length-tolerant collator.

    Stacks tensors, list-collates strings/lists. Missing keys in some entries
    are filled with zeros matching the shape of the first present entry — this
    matches the training loop's per-key ``batch.get(...)`` access pattern.
    """
    if not batch:
        return {}
    keys: set[str] = set()
    for item in batch:
        keys.update(item.keys())

    out: dict[str, Any] = {}
    for key in keys:
        values = [b.get(key) for b in batch]
        present = [v for v in values if v is not None]
        if not present:
            continue
        first = present[0]
        if isinstance(first, Tensor):
            normalized: list[Tensor] = []
            ref_shape = first.shape
            for v in values:
                if v is None:
                    normalized.append(torch.zeros(ref_shape, dtype=first.dtype))
                else:
                    normalized.append(v)
            try:
                out[key] = torch.stack(normalized, dim=0)
            except RuntimeError:
                # Heterogeneous shapes — fall back to list (caller decides).
                out[key] = normalized
        elif isinstance(first, str):
            out[key] = [v if isinstance(v, str) else "" for v in values]
        elif isinstance(first, (int, float)):
            out[key] = torch.tensor([float(v) if v is not None else 0.0 for v in values])
        elif isinstance(first, list):
            out[key] = [v if isinstance(v, list) else [] for v in values]
        else:
            out[key] = values
    return out


__all__ = ["VideoDatasetConfig", "AdVideoDataset", "collate_fn"]
