"""Perception artifact saving utilities.

This module is deliberately implemented as a thin :class:`PerceptionBackend`
wrapper so the core perception adapters do not need to know anything about
visualization, video encoding, or debug output. It can wrap the toy backend,
the DA3 + YOLO-World composed backend, or any future backend that returns
``ChunkPerception``.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from directme.perception.base import ChunkPerception, FramePerception, PerceptionBackend, VideoFrame


def _track_color(track_id: str | None) -> tuple[int, int, int]:
    if not track_id:
        return (255, 64, 64)
    h = abs(hash(track_id))
    return (64 + h % 160, 64 + (h // 97) % 160, 64 + (h // 7919) % 160)


def _load_frame_rgb(frame: VideoFrame) -> Image.Image:
    if frame.image_path:
        return Image.open(frame.image_path).convert("RGB")
    if frame.image is None:
        raise ValueError(f"Frame {frame.index} has neither image_path nor image data")
    arr = np.asarray(frame.image)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr[..., :3]).convert("RGB")


def _depth_to_rgb(depth: np.ndarray) -> Image.Image:
    arr = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(arr) & (arr > 0)
    if not valid.any():
        gray = np.zeros(arr.shape, dtype=np.uint8)
    else:
        lo, hi = np.percentile(arr[valid], [2, 98])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(arr[valid].min()), float(arr[valid].max())
        if hi <= lo:
            hi = lo + 1.0
        gray = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
        gray = (gray * 255).astype(np.uint8)
    try:
        import cv2  # type: ignore

        colored = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        return Image.fromarray(colored)
    except Exception:
        return Image.fromarray(np.stack([gray, gray, gray], axis=-1), mode="RGB")


def _draw_tracking(fp: FramePerception) -> Image.Image:
    img = _load_frame_rgb(fp.frame)
    draw = ImageDraw.Draw(img, "RGBA")
    for obj in fp.objects:
        if obj.bbox_xyxy is None:
            continue
        x1, y1, x2, y2 = [float(v) for v in obj.bbox_xyxy]
        color = _track_color(obj.track_id)
        rgba = (*color, 230)
        draw.rectangle([x1, y1, x2, y2], outline=rgba, width=3)
        label = f"{obj.label}"
        if obj.track_id:
            label += f" | {obj.track_id}"
        if obj.score is not None:
            label += f" | {obj.score:.2f}"
        try:
            bbox = draw.textbbox((x1, y1), label)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            tw, th = 8 * len(label), 14
        y_text = max(0, y1 - th - 4)
        draw.rectangle([x1, y_text, x1 + tw + 6, y_text + th + 4], fill=(0, 0, 0, 160))
        draw.text((x1 + 3, y_text + 2), label, fill=(255, 255, 255, 255))
    return img


def _write_video(frames: list[Path], out_path: Path, fps: float) -> None:
    if not frames:
        return
    try:
        import imageio.v2 as imageio  # type: ignore
    except Exception as exc:
        warnings.warn(f"imageio is unavailable; skip video {out_path}: {exc}")
        return

    try:
        images: list[np.ndarray] = []
        first_size: tuple[int, int] | None = None
        for path in frames:
            with Image.open(path) as im:
                im = im.convert("RGB")
                if first_size is None:
                    first_size = im.size
                elif im.size != first_size:
                    im = im.resize(first_size)
                images.append(np.asarray(im))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(str(out_path), images, fps=fps, macro_block_size=1)
    except Exception as exc:
        warnings.warn(f"failed to write video {out_path}: {exc}")


@dataclass
class PerceptionArtifactBackend(PerceptionBackend):
    """Wrap a perception backend and persist depth/tracking artifacts per chunk.

    The wrapper keeps perception chunking semantics intact: each call to
    :meth:`process_chunk` delegates to the wrapped backend once, then saves the
    returned ``FramePerception`` objects under chunk-specific directories.

    Output layout::

        artifact_dir/
          manifest.jsonl
          depth_frames/chunk_000000/frame_000001_depth.png
          tracking_frames/chunk_000000/frame_000001_tracking.png
          videos/depth_chunk_000000.mp4
          videos/tracking_chunk_000000.mp4
          videos/depth_all.mp4
          videos/tracking_all.mp4
    """

    backend: PerceptionBackend
    artifact_dir: str | Path
    video_fps: float = 1.0
    write_images: bool = True
    write_chunk_videos: bool = True
    write_full_videos: bool = True
    depth_frame_paths: list[Path] = field(default_factory=list, init=False)
    tracking_frame_paths: list[Path] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.artifact_dir = Path(self.artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        (self.artifact_dir / "depth_frames").mkdir(parents=True, exist_ok=True)
        (self.artifact_dir / "tracking_frames").mkdir(parents=True, exist_ok=True)
        (self.artifact_dir / "videos").mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.artifact_dir / "manifest.jsonl"

    def process_chunk(self, frames: list[VideoFrame], chunk_id: int) -> ChunkPerception:
        perception = self.backend.process_chunk(frames=frames, chunk_id=chunk_id)
        self._save_chunk(perception)
        return perception

    def _save_chunk(self, perception: ChunkPerception) -> None:
        chunk_name = f"chunk_{perception.chunk_id:06d}"
        depth_dir = self.artifact_dir / "depth_frames" / chunk_name
        tracking_dir = self.artifact_dir / "tracking_frames" / chunk_name
        if self.write_images:
            depth_dir.mkdir(parents=True, exist_ok=True)
            tracking_dir.mkdir(parents=True, exist_ok=True)

        chunk_depth_paths: list[Path] = []
        chunk_tracking_paths: list[Path] = []
        manifest_records: list[dict[str, Any]] = []

        for fp in perception.frames:
            stem = f"frame_{fp.frame.index:06d}"
            depth_path: Path | None = None
            tracking_path: Path | None = None

            if self.write_images and fp.depth is not None:
                depth_path = depth_dir / f"{stem}_depth.png"
                _depth_to_rgb(fp.depth).save(depth_path)
                chunk_depth_paths.append(depth_path)
                self.depth_frame_paths.append(depth_path)

            if self.write_images:
                try:
                    tracking_img = _draw_tracking(fp)
                except Exception as exc:
                    warnings.warn(f"failed to draw tracking frame {fp.frame.index}: {exc}")
                    tracking_img = None
                if tracking_img is not None:
                    tracking_path = tracking_dir / f"{stem}_tracking.png"
                    tracking_img.save(tracking_path)
                    chunk_tracking_paths.append(tracking_path)
                    self.tracking_frame_paths.append(tracking_path)

            manifest_records.append(
                {
                    "chunk_id": perception.chunk_id,
                    "frame_index": fp.frame.index,
                    "timestamp": fp.frame.timestamp,
                    "n_objects": len(fp.objects),
                    "depth_path": str(depth_path) if depth_path else None,
                    "tracking_path": str(tracking_path) if tracking_path else None,
                    "objects": [
                        {
                            "label": obj.label,
                            "track_id": obj.track_id,
                            "score": obj.score,
                            "bbox_xyxy": list(obj.bbox_xyxy) if obj.bbox_xyxy is not None else None,
                        }
                        for obj in fp.objects
                    ],
                }
            )

        with self.manifest_path.open("a", encoding="utf-8") as f:
            for rec in manifest_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if self.write_chunk_videos:
            _write_video(chunk_depth_paths, self.artifact_dir / "videos" / f"depth_{chunk_name}.mp4", self.video_fps)
            _write_video(chunk_tracking_paths, self.artifact_dir / "videos" / f"tracking_{chunk_name}.mp4", self.video_fps)

    def finalize(self) -> None:
        """Write full-run videos after the last chunk has been processed."""
        if not self.write_full_videos:
            return
        _write_video(self.depth_frame_paths, self.artifact_dir / "videos" / "depth_all.mp4", self.video_fps)
        _write_video(self.tracking_frame_paths, self.artifact_dir / "videos" / "tracking_all.mp4", self.video_fps)
