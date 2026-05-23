"""End-to-end dataset builder for Ogenti smoke / retrofit training.

Pipeline per source:
  1. Search & fetch clips matching ad-creative-style queries.
  2. Trim to ``--clip-seconds`` (default ≈ A100-GUIDE clip length).
  3. Caption (Qwen2-VL / BLIP-2 / template fallback).
  4. Run preprocessing extractors (camera_motion, optionally pose/glyph/skin).
  5. Write manifest JSONL at ``<root>/manifests/ads_train.jsonl``.

Sources (all opt-in, all support per-source ``--<src>-max`` caps):
  - Pexels API           — needs ``PEXELS_API_KEY`` env var (free signup).
  - Pixabay API          — needs ``PIXABAY_API_KEY`` env var (free signup).
  - Internet Archive     — anonymous, public domain commercials.
  - Wikimedia Commons    — anonymous, CC-BY/CC0 video files.

Robustness:
  - Each source is wrapped in try/except so one outage doesn't kill the run.
  - Missing API keys silently skip that source.
  - Failed downloads / decode errors are logged and skipped.
  - Manifest is appended incrementally so partial runs are recoverable.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import quote

import numpy as np
import typer

from ogenti.utils.logging import configure_root_logging, get_logger

log = get_logger("ogenti.scripts.build_dataset")

app = typer.Typer(pretty_exceptions_enable=False)


# Ad-creative-style search queries. These steer the dataset toward the
# domains where Ogenti has to win (product hero shots, beauty, food, fashion,
# automotive, lifestyle). Mix concrete subjects + cinematography terms.
DEFAULT_QUERIES: list[str] = [
    "luxury watch close up",
    "perfume bottle product shot",
    "cosmetic skincare model",
    "lipstick application macro",
    "soda bottle pour slow motion",
    "coffee pour latte art",
    "burger ingredients falling slow motion",
    "running shoe close up athletic",
    "sports car driving city",
    "fashion model walking studio",
    "smartphone product unboxing",
    "headphones lifestyle minimal",
    "ice cream dessert macro",
    "wine glass pour slow motion",
    "denim jeans fashion model",
    "skincare cream texture macro",
    "athlete training gym workout",
    "coffee beans roasting close up",
    "watch movement gears macro",
    "chocolate melting macro",
]


@dataclass
class BuildConfig:
    root: Path
    clip_seconds: float = 4.0
    fps: int = 24
    target_height: int = 480
    target_width: int = 832
    pexels_max: int = 1000
    pixabay_max: int = 400
    archive_max: int = 50
    wikimedia_max: int = 50
    skip_archive: bool = False
    skip_wikimedia: bool = True
    skip_pexels: bool = False
    skip_pixabay: bool = False
    caption_backend: str = "template"
    caption_device: str = "cuda"
    queries: list[str] = field(default_factory=lambda: list(DEFAULT_QUERIES))
    per_query_cap: int = 60
    run_keypoints: bool = False
    run_skin: bool = False
    run_glyphs: bool = False
    run_subject_motion: bool = False
    run_micro_events: bool = False
    user_agent: str = "ogenti-data-builder/0.1 (research, contact: noreply@example.com)"
    seed: int = 42


def _hash_id(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:14]


def _safe_request_json(url: str, headers: dict[str, str] | None = None, timeout: float = 30.0) -> Optional[dict]:
    import requests

    try:
        r = requests.get(url, headers=headers or {}, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.debug(f"GET {url[:80]}... failed: {e}")
        return None


def _download_file(url: str, out_path: Path, timeout: float = 120.0) -> bool:
    import requests

    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
        return True
    except Exception as e:
        log.warning(f"download failed {url[:80]}: {e}")
        if out_path.exists():
            out_path.unlink(missing_ok=True)
        return False


# ----------------------------- source: Pexels -----------------------------


def search_pexels(query: str, per_page: int) -> list[dict]:
    key = os.environ.get("PEXELS_API_KEY")
    if not key:
        return []
    url = f"https://api.pexels.com/videos/search?query={quote(query)}&per_page={per_page}&size=medium"
    data = _safe_request_json(url, headers={"Authorization": key})
    if not data:
        return []
    out: list[dict] = []
    for v in data.get("videos", []):
        files = v.get("video_files", []) or []
        # Prefer mid-resolution mp4 to keep download small.
        files = [f for f in files if f.get("file_type", "").endswith("mp4")]
        files.sort(key=lambda f: abs(int(f.get("height") or 0) - 720))
        if not files:
            continue
        chosen = files[0]
        out.append(
            {
                "source": "pexels",
                "url": chosen["link"],
                "id": f"pexels-{v['id']}",
                "duration": v.get("duration", 0),
                "width": v.get("width"),
                "height": v.get("height"),
                "query": query,
                "user": (v.get("user") or {}).get("name", ""),
                "page_url": v.get("url"),
            }
        )
    return out


# ----------------------------- source: Pixabay -----------------------------


def search_pixabay(query: str, per_page: int) -> list[dict]:
    key = os.environ.get("PIXABAY_API_KEY")
    if not key:
        return []
    url = (
        f"https://pixabay.com/api/videos/?key={key}"
        f"&q={quote(query)}&per_page={min(200, max(3, per_page))}&safesearch=true"
    )
    data = _safe_request_json(url)
    if not data:
        return []
    out: list[dict] = []
    for hit in data.get("hits", []):
        videos = hit.get("videos", {}) or {}
        # tier preference: large > medium > small > tiny
        for tier in ("large", "medium", "small", "tiny"):
            vt = videos.get(tier) or {}
            if vt.get("url"):
                out.append(
                    {
                        "source": "pixabay",
                        "url": vt["url"],
                        "id": f"pixabay-{hit['id']}",
                        "duration": hit.get("duration", 0),
                        "width": vt.get("width"),
                        "height": vt.get("height"),
                        "query": query,
                        "user": hit.get("user", ""),
                        "page_url": hit.get("pageURL"),
                    }
                )
                break
    return out


# -------------------------- source: Internet Archive --------------------------


def search_internet_archive(query: str, per_page: int) -> list[dict]:
    """Search the Internet Archive's classic_tv_commercials + adverts collections."""
    url = (
        "https://archive.org/advancedsearch.php?q="
        + quote(f'({query}) AND mediatype:movies AND collection:(classic_tv_commercials OR adverts OR ephemera)')
        + f"&fl[]=identifier&fl[]=title&fl[]=description&fl[]=year&rows={per_page}&output=json"
    )
    data = _safe_request_json(url)
    if not data:
        return []
    out: list[dict] = []
    for doc in data.get("response", {}).get("docs", []):
        identifier = doc.get("identifier")
        if not identifier:
            continue
        out.append(
            {
                "source": "internet_archive",
                "id": f"archive-{identifier}",
                "identifier": identifier,
                "title": doc.get("title", ""),
                "description": (doc.get("description", "") or "")[:512],
                "year": doc.get("year"),
                "query": query,
            }
        )
    return out


def resolve_internet_archive_url(identifier: str) -> Optional[str]:
    meta = _safe_request_json(f"https://archive.org/metadata/{identifier}")
    if not meta:
        return None
    for f in meta.get("files", []):
        name = f.get("name", "")
        if name.lower().endswith(".mp4"):
            return f"https://archive.org/download/{identifier}/{name}"
    # fall back to source file if no derivative
    for f in meta.get("files", []):
        name = f.get("name", "")
        if name.lower().endswith((".mov", ".m4v", ".mpg", ".mpeg")):
            return f"https://archive.org/download/{identifier}/{name}"
    return None


# -------------------------- source: Wikimedia Commons --------------------------


def search_wikimedia(query: str, per_page: int, user_agent: str) -> list[dict]:
    import requests

    url = (
        "https://commons.wikimedia.org/w/api.php?action=query&format=json&list=search&srlimit="
        + str(per_page)
        + "&srnamespace=6&srsearch="
        + quote(f"{query} filetype:video")
    )
    try:
        r = requests.get(url, headers={"User-Agent": user_agent}, timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log.debug(f"wikimedia search failed: {e}")
        return []
    out: list[dict] = []
    for item in data.get("query", {}).get("search", []):
        title = item.get("title", "")
        if not title.startswith("File:"):
            continue
        meta_url = (
            "https://commons.wikimedia.org/w/api.php?action=query&format=json&prop=videoinfo|imageinfo"
            "&viprop=url&iiprop=url&titles=" + quote(title)
        )
        try:
            r = requests.get(meta_url, headers={"User-Agent": user_agent}, timeout=30)
            mdata = r.json()
        except Exception:
            continue
        pages = (mdata.get("query") or {}).get("pages") or {}
        for _, page in pages.items():
            info = (page.get("videoinfo") or page.get("imageinfo") or [{}])[0]
            url_v = info.get("url")
            if url_v and url_v.lower().endswith((".webm", ".ogv", ".mp4")):
                out.append(
                    {
                        "source": "wikimedia",
                        "id": f"wikimedia-{_hash_id(title)}",
                        "url": url_v,
                        "title": title,
                        "query": query,
                    }
                )
    return out


# ----------------------------- caption backends -----------------------------


def _template_caption(meta: dict) -> str:
    query = meta.get("query", "")
    title = meta.get("title", "")
    parts: list[str] = []
    if query:
        parts.append(query.strip())
    if title:
        parts.append(f"({title.strip()[:120]})")
    if not parts:
        parts.append("Cinematic commercial product shot.")
    return " — ".join(parts)


class _CaptionBackend:
    def __init__(self, backend: str, device: str) -> None:
        self.backend = backend
        self.device = device
        self._model = None
        self._processor = None

    def caption(self, frame: np.ndarray, meta: dict) -> str:
        if self.backend == "template":
            return _template_caption(meta)
        if self.backend == "blip2":
            return self._blip2(frame, meta)
        if self.backend == "qwen2vl":
            return self._qwen2vl(frame, meta)
        return _template_caption(meta)

    def _blip2(self, frame: np.ndarray, meta: dict) -> str:
        try:
            from PIL import Image
            from transformers import Blip2ForConditionalGeneration, Blip2Processor

            if self._model is None:
                self._processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
                import torch

                self._model = Blip2ForConditionalGeneration.from_pretrained(
                    "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
                ).to(self.device)
            img = Image.fromarray(frame)
            inputs = self._processor(images=img, return_tensors="pt").to(self.device)
            import torch

            with torch.no_grad():
                out = self._model.generate(**inputs, max_new_tokens=40)
            text = self._processor.tokenizer.decode(out[0], skip_special_tokens=True).strip()
            if not text:
                return _template_caption(meta)
            return text
        except Exception as e:
            log.debug(f"blip2 caption failed: {e}")
            return _template_caption(meta)

    def _qwen2vl(self, frame: np.ndarray, meta: dict) -> str:
        try:
            from PIL import Image
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

            if self._model is None:
                model_id = "Qwen/Qwen2-VL-2B-Instruct"
                self._processor = AutoProcessor.from_pretrained(model_id)
                import torch

                self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_id, torch_dtype=torch.bfloat16
                ).to(self.device)
            img = Image.fromarray(frame)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {
                            "type": "text",
                            "text": "Describe this advertising still for use as a text-to-video prompt. "
                            "Include subject, action, lighting, lens, mood. One sentence.",
                        },
                    ],
                }
            ]
            text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self._processor(text=[text], images=[img], padding=True, return_tensors="pt").to(self.device)
            import torch

            with torch.no_grad():
                out = self._model.generate(**inputs, max_new_tokens=80)
            decoded = self._processor.batch_decode(out, skip_special_tokens=True)[0]
            # Strip the prompt prefix
            caption = decoded.split("assistant\n")[-1].strip()
            return caption or _template_caption(meta)
        except Exception as e:
            log.debug(f"qwen2vl caption failed: {e}")
            return _template_caption(meta)


# ----------------------------- ffmpeg trim/normalize -----------------------------


def trim_and_normalize(src: Path, dst: Path, cfg: BuildConfig) -> bool:
    """Use ffmpeg to trim to ``cfg.clip_seconds`` and resize to target HxW @ target fps."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        "0",
        "-i",
        str(src),
        "-t",
        str(cfg.clip_seconds),
        "-vf",
        f"scale={cfg.target_width}:{cfg.target_height}:force_original_aspect_ratio=increase,"
        f"crop={cfg.target_width}:{cfg.target_height},fps={cfg.fps}",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        str(dst),
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            check=False,
            timeout=120,
        )
        if result.returncode != 0:
            log.debug(f"ffmpeg failed for {src.name}: {result.stderr.decode(errors='ignore')[-200:]}")
            return False
        return dst.exists() and dst.stat().st_size > 0
    except Exception as e:
        log.debug(f"ffmpeg exception: {e}")
        return False


# ----------------------------- per-clip processing -----------------------------


def _read_for_preprocess(clip_path: Path, cfg: BuildConfig) -> Optional[np.ndarray]:
    """Decode a normalized clip into (T, H, W, 3) uint8 RGB."""
    try:
        import imageio.v3 as iio

        frames: list[np.ndarray] = []
        for frame in iio.imiter(str(clip_path)):
            if frame.ndim == 2:
                frame = np.stack([frame] * 3, axis=-1)
            if frame.shape[-1] == 4:
                frame = frame[..., :3]
            frames.append(frame)
        if not frames:
            return None
        return np.stack(frames, axis=0)
    except Exception as e:
        log.debug(f"decode failed for {clip_path.name}: {e}")
        return None


def process_clip(
    raw_path: Path,
    meta: dict,
    cfg: BuildConfig,
    captioner: _CaptionBackend,
) -> Optional[dict]:
    clip_id = meta["id"]
    clip_rel = Path("videos") / f"{clip_id}.mp4"
    clip_abs = cfg.root / clip_rel
    if not trim_and_normalize(raw_path, clip_abs, cfg):
        return None

    frames = _read_for_preprocess(clip_abs, cfg)
    if frames is None or frames.shape[0] < 4:
        clip_abs.unlink(missing_ok=True)
        return None

    entry: dict = {
        "id": clip_id,
        "video": str(clip_rel),
        "source": meta.get("source", "unknown"),
        "query": meta.get("query", ""),
        "duration": cfg.clip_seconds,
        "fps": cfg.fps,
        "width": cfg.target_width,
        "height": cfg.target_height,
        "license": _license_for(meta.get("source", "")),
        "attribution": {
            "user": meta.get("user", ""),
            "page_url": meta.get("page_url") or meta.get("identifier"),
        },
    }

    # ------- caption -------
    keyframe = frames[frames.shape[0] // 2]
    entry["prompt"] = captioner.caption(keyframe, meta).strip()

    # ------- camera motion (always run; cheap and needed for smoke) -------
    try:
        from ogenti.data.preprocess import compute_camera_motion

        cm = compute_camera_motion(frames)
        cm_rel = Path("camera_motion") / f"{clip_id}.npy"
        (cfg.root / cm_rel).parent.mkdir(parents=True, exist_ok=True)
        np.save(cfg.root / cm_rel, cm)
        entry["camera_motion"] = str(cm_rel)
    except Exception as e:
        log.debug(f"camera motion failed for {clip_id}: {e}")

    if cfg.run_keypoints:
        try:
            from ogenti.data.preprocess import extract_keypoints

            kp, conf = extract_keypoints(frames)
            kp_rel = Path("keypoints") / f"{clip_id}.npy"
            conf_rel = Path("keypoints") / f"{clip_id}_conf.npy"
            (cfg.root / kp_rel).parent.mkdir(parents=True, exist_ok=True)
            np.save(cfg.root / kp_rel, kp)
            np.save(cfg.root / conf_rel, conf)
            entry["keypoints"] = str(kp_rel)
            entry["keypoint_conf"] = str(conf_rel)
        except Exception as e:
            log.debug(f"keypoints failed for {clip_id}: {e}")

    if cfg.run_skin:
        try:
            from ogenti.data.preprocess import extract_skin_masks

            sk = extract_skin_masks(frames, target_hw=(cfg.target_height, cfg.target_width))
            sk_rel = Path("skin_masks") / f"{clip_id}.npy"
            (cfg.root / sk_rel).parent.mkdir(parents=True, exist_ok=True)
            np.save(cfg.root / sk_rel, sk)
            entry["skin_mask"] = str(sk_rel)
        except Exception as e:
            log.debug(f"skin masks failed for {clip_id}: {e}")

    if cfg.run_glyphs:
        try:
            from ogenti.data.preprocess import extract_glyph_regions

            glyph_dir = cfg.root / "glyphs"
            regions = extract_glyph_regions(frames, glyph_dir, clip_id)
            if regions:
                entry["glyph_regions"] = regions
        except Exception as e:
            log.debug(f"glyphs failed for {clip_id}: {e}")

    if cfg.run_subject_motion:
        try:
            from ogenti.data.preprocess import extract_subject_motion

            sm, mask = extract_subject_motion(frames)
            sm_rel = Path("subject_motion") / f"{clip_id}.npy"
            sm_mask_rel = Path("subject_motion") / f"{clip_id}_mask.npy"
            (cfg.root / sm_rel).parent.mkdir(parents=True, exist_ok=True)
            np.save(cfg.root / sm_rel, sm)
            np.save(cfg.root / sm_mask_rel, mask)
            entry["subject_motion"] = str(sm_rel)
            entry["subject_motion_mask"] = str(sm_mask_rel)
        except Exception as e:
            log.debug(f"subject motion failed for {clip_id}: {e}")

    if cfg.run_micro_events:
        try:
            from ogenti.data.preprocess import extract_micro_events

            ev = extract_micro_events(frames)
            ev_rel = Path("micro_events") / f"{clip_id}.npz"
            (cfg.root / ev_rel).parent.mkdir(parents=True, exist_ok=True)
            np.savez(cfg.root / ev_rel, **ev)
            entry["micro_events"] = str(ev_rel)
        except Exception as e:
            log.debug(f"micro events failed for {clip_id}: {e}")

    return entry


def _license_for(source: str) -> str:
    return {
        "pexels": "Pexels License (free for commercial, no attribution required)",
        "pixabay": "Pixabay Content License (free for commercial, no attribution required)",
        "internet_archive": "Public domain / Internet Archive (verify per item)",
        "wikimedia": "Creative Commons (CC-BY / CC-BY-SA / CC0, verify per file)",
    }.get(source, "unknown")


# ----------------------------- main pipeline -----------------------------


def _collect_candidates(cfg: BuildConfig) -> list[dict]:
    """Walk all sources and collect (still-undownloaded) candidate descriptors."""
    candidates: list[dict] = []
    log.info("collecting candidates across sources...")

    if not cfg.skip_pexels:
        per_query = max(1, cfg.pexels_max // max(1, len(cfg.queries)))
        for q in cfg.queries:
            hits = search_pexels(q, min(80, max(per_query, cfg.per_query_cap)))
            log.debug(f"  pexels '{q}' → {len(hits)} hits")
            candidates.extend(hits[: per_query + 5])

    if not cfg.skip_pixabay:
        per_query = max(1, cfg.pixabay_max // max(1, len(cfg.queries)))
        for q in cfg.queries:
            hits = search_pixabay(q, min(80, max(per_query, cfg.per_query_cap)))
            log.debug(f"  pixabay '{q}' → {len(hits)} hits")
            candidates.extend(hits[: per_query + 5])

    if not cfg.skip_archive:
        per_query = max(1, cfg.archive_max // max(1, len(cfg.queries)))
        for q in cfg.queries:
            hits = search_internet_archive(q, per_page=per_query + 5)
            log.debug(f"  archive '{q}' → {len(hits)} hits")
            candidates.extend(hits[: per_query + 2])

    if not cfg.skip_wikimedia:
        per_query = max(1, cfg.wikimedia_max // max(1, len(cfg.queries)))
        for q in cfg.queries:
            hits = search_wikimedia(q, per_page=per_query + 5, user_agent=cfg.user_agent)
            log.debug(f"  wikimedia '{q}' → {len(hits)} hits")
            candidates.extend(hits[: per_query + 2])

    # Deduplicate by id
    seen: set[str] = set()
    unique: list[dict] = []
    for c in candidates:
        if c["id"] in seen:
            continue
        seen.add(c["id"])
        unique.append(c)
    log.info(f"  candidates: total={len(candidates)}, unique={len(unique)}")
    return unique


def _apply_caps(candidates: list[dict], cfg: BuildConfig) -> list[dict]:
    by_source: dict[str, list[dict]] = {}
    for c in candidates:
        by_source.setdefault(c["source"], []).append(c)
    caps = {
        "pexels": cfg.pexels_max,
        "pixabay": cfg.pixabay_max,
        "internet_archive": cfg.archive_max,
        "wikimedia": cfg.wikimedia_max,
    }
    out: list[dict] = []
    for src, items in by_source.items():
        cap = caps.get(src, len(items))
        out.extend(items[:cap])
    return out


def _resolve_download_url(meta: dict) -> Optional[str]:
    if meta.get("url"):
        return meta["url"]
    if meta["source"] == "internet_archive":
        return resolve_internet_archive_url(meta["identifier"])
    return None


@app.command()
def main(
    root: Path = typer.Option(Path("data/")),
    clip_seconds: float = typer.Option(4.0),
    fps: int = typer.Option(24),
    target_height: int = typer.Option(480),
    target_width: int = typer.Option(832),
    pexels_max: int = typer.Option(1000),
    pixabay_max: int = typer.Option(400),
    archive_max: int = typer.Option(50),
    wikimedia_max: int = typer.Option(50),
    skip_archive: bool = typer.Option(False),
    skip_wikimedia: bool = typer.Option(True),
    skip_pexels: bool = typer.Option(False),
    skip_pixabay: bool = typer.Option(False),
    caption_backend: str = typer.Option("template", help="template|blip2|qwen2vl"),
    caption_device: str = typer.Option("cuda"),
    queries_file: Optional[Path] = typer.Option(None, help="Newline-separated query strings."),
    run_keypoints: bool = typer.Option(False),
    run_skin: bool = typer.Option(False),
    run_glyphs: bool = typer.Option(False),
    run_subject_motion: bool = typer.Option(False),
    run_micro_events: bool = typer.Option(False),
    target_count: Optional[int] = typer.Option(None, help="Hard cap on total clips written."),
    seed: int = typer.Option(42),
) -> None:
    configure_root_logging()

    queries = list(DEFAULT_QUERIES)
    if queries_file is not None and queries_file.exists():
        queries = [ln.strip() for ln in queries_file.read_text().splitlines() if ln.strip()]

    cfg = BuildConfig(
        root=root,
        clip_seconds=clip_seconds,
        fps=fps,
        target_height=target_height,
        target_width=target_width,
        pexels_max=pexels_max,
        pixabay_max=pixabay_max,
        archive_max=archive_max,
        wikimedia_max=wikimedia_max,
        skip_archive=skip_archive,
        skip_wikimedia=skip_wikimedia,
        skip_pexels=skip_pexels,
        skip_pixabay=skip_pixabay,
        caption_backend=caption_backend,
        caption_device=caption_device,
        queries=queries,
        run_keypoints=run_keypoints,
        run_skin=run_skin,
        run_glyphs=run_glyphs,
        run_subject_motion=run_subject_motion,
        run_micro_events=run_micro_events,
        seed=seed,
    )

    np.random.seed(cfg.seed)

    if shutil.which("ffmpeg") is None:
        log.error("ffmpeg not found on PATH; install with `apt-get install -y ffmpeg`")
        raise typer.Exit(code=2)

    raw_dir = cfg.root / "_raw"
    manifest_dir = cfg.root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "ads_train.jsonl"

    # Resume support: track already-processed ids.
    seen_ids: set[str] = set()
    if manifest_path.exists():
        with open(manifest_path) as f:
            for line in f:
                try:
                    seen_ids.add(json.loads(line)["id"])
                except Exception:
                    continue
        log.info(f"resuming: {len(seen_ids)} clips already in manifest")

    candidates = _collect_candidates(cfg)
    candidates = _apply_caps(candidates, cfg)
    np.random.shuffle(candidates)  # interleave sources for diversity

    captioner = _CaptionBackend(cfg.caption_backend, cfg.caption_device)

    written = 0
    skipped = 0
    started = time.time()
    cap_total = target_count or sum(
        v
        for k, v in {
            "pexels": cfg.pexels_max if not cfg.skip_pexels else 0,
            "pixabay": cfg.pixabay_max if not cfg.skip_pixabay else 0,
            "internet_archive": cfg.archive_max if not cfg.skip_archive else 0,
            "wikimedia": cfg.wikimedia_max if not cfg.skip_wikimedia else 0,
        }.items()
    )

    with open(manifest_path, "a") as out_f:
        for i, meta in enumerate(candidates):
            if meta["id"] in seen_ids:
                continue
            url = _resolve_download_url(meta)
            if not url:
                skipped += 1
                continue
            raw_path = raw_dir / f"{meta['id']}{_ext_for_url(url)}"
            if not raw_path.exists() and not _download_file(url, raw_path):
                skipped += 1
                continue
            entry = process_clip(raw_path, meta, cfg, captioner)
            raw_path.unlink(missing_ok=True)  # save disk; only keep normalized clip
            if entry is None:
                skipped += 1
                continue
            out_f.write(json.dumps(entry) + "\n")
            out_f.flush()
            written += 1
            seen_ids.add(meta["id"])
            if written % 25 == 0 or written == 1:
                elapsed = time.time() - started
                log.info(
                    f"  built {written} clips ({skipped} skipped) "
                    f"[{elapsed:.0f}s, src={meta.get('source')}]"
                )
            if written >= cap_total:
                break

    log.info(f"done: wrote {written} entries to {manifest_path} ({skipped} skipped)")


def _ext_for_url(url: str) -> str:
    for ext in (".mp4", ".webm", ".ogv", ".mov", ".m4v", ".mpg", ".mpeg"):
        if url.lower().split("?")[0].endswith(ext):
            return ext
    return ".mp4"


if __name__ == "__main__":
    app()
