#!/usr/bin/env python3
"""Full DirectMe perception-to-mapping test.

This script verifies the real pipeline:

  SCAL3R depth/pose/intrinsics
  + YOLO-World detections
  + optional SAM2 masks
  + tracking IDs
  + DirectMe OfflineMappingEngine
  + 3D scene graph node
  + semantic point cloud PLY export

Default input:
  /data/ywang/dataset/SpatialMemory/data_frames_1fps/scene0804_00-0
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from PIL import Image

from directme.perception.color_attributes import (
        dominant_hsv_color,
        hsv_histogram_from_image_mask,
    )

import numpy as np


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _numeric_sort_key(path: Path):
    nums = re.findall(r"\d+", path.stem)
    if nums:
        return int(nums[-1]), path.name
    return 10**12, path.name


def _list_images(image_dir: str | Path, max_images: int | None = None) -> list[Path]:
    image_dir = Path(image_dir).resolve()
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")

    paths = [
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
    ]
    paths.sort(key=_numeric_sort_key)

    if max_images and max_images > 0:
        paths = paths[:max_images]

    if not paths:
        raise FileNotFoundError(f"No image files found in: {image_dir}")

    return paths


def _load_classes_from_yaml(path: str | Path, class_limit: int = 100) -> list[str]:
    import yaml

    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Classes YAML does not exist: {path}")

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    names = data["names"]

    if isinstance(names, dict):
        classes = [
            str(v).strip()
            for _, v in sorted(names.items(), key=lambda kv: int(kv[0]))
        ]
    elif isinstance(names, list):
        classes = [str(v).strip() for v in names]
    else:
        raise ValueError(f"Unsupported names format in {path}")

    classes = [c for c in classes if c]

    if class_limit and class_limit > 0:
        classes = classes[:class_limit]

    return classes


def _make_video_frames(image_paths: list[Path]):
    from directme.perception.base import VideoFrame

    return [
        VideoFrame(
            index=i,
            timestamp=float(i),
            image_path=str(path),
            image=None,
            metadata={},
        )
        for i, path in enumerate(image_paths)
    ]


def _resolve_scal3r_result_dir(path: str | Path | None) -> Path | None:
    """Accept either an exact SCAL3R output dir or a parent containing one mat.txt."""
    if not path:
        return None

    root = Path(path).resolve()
    if not root.exists():
        raise FileNotFoundError(f"SCAL3R result root does not exist: {root}")

    if (root / "mat.txt").exists():
        return root

    mats = sorted(root.rglob("mat.txt"))
    if len(mats) == 1:
        return mats[0].parent

    if len(mats) > 1:
        msg = "\n".join(str(p) for p in mats[:20])
        raise RuntimeError(
            f"Multiple mat.txt files found under {root}. "
            f"Please pass the exact result directory.\nCandidates:\n{msg}"
        )

    raise FileNotFoundError(f"No mat.txt found under SCAL3R result root: {root}")


def _count_mat_rows(result_dir: Path) -> int:
    mat = np.loadtxt(result_dir / "mat.txt")
    if mat.ndim == 1:
        return 1
    return int(mat.shape[0])


def _se3_to_matrix(se3: Any) -> np.ndarray:
    """Robustly convert DirectMe SE3 to 4x4 matrix."""
    if hasattr(se3, "matrix"):
        mat = getattr(se3, "matrix")
        mat = mat() if callable(mat) else mat
        mat = np.asarray(mat, dtype=np.float32)
        if mat.shape == (4, 4):
            return mat

    for name in ["as_matrix", "to_matrix"]:
        if hasattr(se3, name):
            mat = np.asarray(getattr(se3, name)(), dtype=np.float32)
            if mat.shape == (4, 4):
                return mat

    if hasattr(se3, "rotation") and hasattr(se3, "translation"):
        mat = np.eye(4, dtype=np.float32)
        mat[:3, :3] = np.asarray(se3.rotation, dtype=np.float32)
        mat[:3, 3] = np.asarray(se3.translation, dtype=np.float32)
        return mat

    arr = np.asarray(se3, dtype=np.float32)
    if arr.shape == (4, 4):
        return arr

    raise TypeError(f"Cannot convert SE3 object to 4x4 matrix: {type(se3)}")


def _color_from_string(text: str) -> tuple[int, int, int]:
    """Deterministic color from string.

    Do not use Python's built-in hash(), because it is randomized between
    processes by PYTHONHASHSEED.
    """
    import hashlib

    digest = hashlib.md5(text.encode("utf-8")).digest()
    return (
        50 + digest[0] % 180,
        50 + digest[1] % 180,
        50 + digest[2] % 180,
    )


def _mask_or_bbox_to_depth_mask(
    obj: Any,
    depth_hw: tuple[int, int],
    image_hw: tuple[int, int],
) -> np.ndarray | None:
    """Convert object mask/bbox from image resolution to depth resolution.

    SAM2 masks are usually in original image resolution, while SCAL3R depth is
    usually resized. This function resizes masks/bboxes to the depth map size.
    """
    depth_h, depth_w = depth_hw
    image_h, image_w = image_hw

    if depth_h <= 0 or depth_w <= 0:
        return None

    # Prefer SAM2 mask if available.
    if getattr(obj, "mask", None) is not None:
        mask = np.asarray(obj.mask)

        # Robustly handle shapes like (1, H, W), (H, W, 1), or (H, W).
        mask = np.squeeze(mask)

        if mask.ndim != 2:
            return None

        mask = mask.astype(bool)

        if mask.shape[:2] != (depth_h, depth_w):
            import cv2

            mask = cv2.resize(
                mask.astype(np.uint8),
                (depth_w, depth_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        if mask.sum() <= 0:
            return None

        return mask

    # Fallback to bbox if no mask exists.
    bbox = getattr(obj, "bbox_xyxy", None)
    if bbox is None:
        return None

    x1, y1, x2, y2 = [float(v) for v in bbox]

    # bbox is in original image resolution; scale to SCAL3R depth resolution.
    sx = depth_w / max(float(image_w), 1.0)
    sy = depth_h / max(float(image_h), 1.0)

    x1 = int(round(x1 * sx))
    x2 = int(round(x2 * sx))
    y1 = int(round(y1 * sy))
    y2 = int(round(y2 * sy))

    x1 = max(0, min(depth_w - 1, x1))
    x2 = max(0, min(depth_w, x2))
    y1 = max(0, min(depth_h - 1, y1))
    y2 = max(0, min(depth_h, y2))

    if x2 <= x1 or y2 <= y1:
        return None

    mask = np.zeros((depth_h, depth_w), dtype=bool)
    mask[y1:y2, x1:x2] = True

    if mask.sum() <= 0:
        return None

    return mask

def _mask_or_bbox_to_image_mask(
    obj: Any,
    image_hw: tuple[int, int],
) -> tuple[np.ndarray, str] | None:
    """Convert object mask/bbox to original image-resolution mask.

    Return:
      (mask, "mask") if valid SAM2 mask is used.
      (mask, "bbox") if bbox fallback is used.
    """
    image_h, image_w = image_hw

    if image_h <= 0 or image_w <= 0:
        return None

    raw_mask = getattr(obj, "mask", None)
    if raw_mask is not None:
        try:
            mask = np.asarray(raw_mask)
            mask = np.squeeze(mask)

            if mask.ndim == 2:
                mask = mask.astype(bool)

                if mask.shape[:2] != (image_h, image_w):
                    import cv2

                    mask = cv2.resize(
                        mask.astype(np.uint8),
                        (image_w, image_h),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)

                if mask.sum() > 0:
                    return mask, "mask"
        except Exception:
            pass

    bbox = getattr(obj, "bbox_xyxy", None)
    if bbox is None:
        return None

    x1, y1, x2, y2 = [float(v) for v in bbox]

    x1 = max(0, min(image_w - 1, x1))
    x2 = max(0, min(image_w, x2))
    y1 = max(0, min(image_h - 1, y1))
    y2 = max(0, min(image_h, y2))

    if x2 <= x1 or y2 <= y1:
        return None

    # 最小改动：bbox 取中心区域，减少背景污染。
    # 对没有 SAM2 mask 的情况，颜色会明显更稳。
    shrink = 0.15
    bw = x2 - x1
    bh = y2 - y1

    if bw > 6 and bh > 6:
        x1 = x1 + bw * shrink
        x2 = x2 - bw * shrink
        y1 = y1 + bh * shrink
        y2 = y2 - bh * shrink

    x1 = int(round(max(0, min(image_w - 1, x1))))
    x2 = int(round(max(0, min(image_w, x2))))
    y1 = int(round(max(0, min(image_h - 1, y1))))
    y2 = int(round(max(0, min(image_h, y2))))

    if x2 <= x1 or y2 <= y1:
        return None

    mask = np.zeros((image_h, image_w), dtype=bool)
    mask[y1:y2, x1:x2] = True

    if mask.sum() <= 0:
        return None

    return mask, "bbox"



def _attach_color_attributes_to_chunk(chunk: Any) -> dict[str, Any]:
    """Attach color attributes to each detected object in the perception chunk.

    Adds these attributes to each object:
      - obj.color_name: coarse color name, e.g. red / blue / white / black
      - obj.color_histogram: HSV hue histogram list[float], length 12
      - obj.color_source: "mask", "bbox", or "none"
    """


    n_objects = 0
    n_colored_objects = 0
    n_mask_color = 0
    n_bbox_color = 0
    n_failed = 0

    image_cache: dict[str, np.ndarray] = {}

    for fp in chunk.frames:
        image_path = getattr(fp.frame, "image_path", None)
        if not image_path or not Path(image_path).exists():
            for obj in fp.objects:
                n_objects += 1
                setattr(obj, "color_name", "unknown")
                setattr(obj, "color_histogram", None)
                setattr(obj, "color_source", "none")
                n_failed += 1
            continue

        image_path = str(Path(image_path).resolve())

        if image_path not in image_cache:
            with Image.open(image_path) as im:
                image_cache[image_path] = np.asarray(im.convert("RGB"))

        image_rgb = image_cache[image_path]
        image_h, image_w = image_rgb.shape[:2]

        for obj in fp.objects:
            n_objects += 1

            mask_result = _mask_or_bbox_to_image_mask(
                obj,
                image_hw=(image_h, image_w),
            )

            if mask_result is None:
                setattr(obj, "color_name", "unknown")
                setattr(obj, "color_histogram", None)
                setattr(obj, "color_source", "none")
                n_failed += 1
                continue

            image_mask, color_source = mask_result

            try:
                color_name = dominant_hsv_color(
                    image_rgb,
                    image_mask,
                )
                color_histogram = hsv_histogram_from_image_mask(
                    image_rgb,
                    image_mask,
                    bins=12,
                )

                setattr(obj, "color_name", color_name)
                setattr(obj, "color_histogram", color_histogram)
                setattr(obj, "color_source", color_source)

                if color_name != "unknown":
                    n_colored_objects += 1

                if color_source == "mask":
                    n_mask_color += 1
                else:
                    n_bbox_color += 1

            except Exception:
                setattr(obj, "color_name", "unknown")
                setattr(obj, "color_histogram", None)
                setattr(obj, "color_source", "none")
                n_failed += 1

    return {
        "n_objects": int(n_objects),
        "n_colored_objects": int(n_colored_objects),
        "n_mask_color": int(n_mask_color),
        "n_bbox_color": int(n_bbox_color),
        "n_failed": int(n_failed),
    }

def _majority_vote(
    values: list[Any],
    default: str = "unknown",
    ignore_values: set[str] | None = None,
) -> str:
    if ignore_values is None:
        ignore_values = {"", "unknown", "none", "null", "nan"}

    counts: dict[str, int] = {}

    for v in values:
        if v is None:
            continue

        s = str(v).strip().lower()
        if s in ignore_values:
            continue

        counts[s] = counts.get(s, 0) + 1

    if not counts:
        return default

    return max(counts.items(), key=lambda kv: kv[1])[0]


def _mean_histogram(values: list[Any]) -> list[float] | None:
    arrays = []

    for v in values:
        if v is None:
            continue

        arr = np.asarray(v, dtype=np.float32)
        if arr.ndim == 1 and arr.size > 0:
            arrays.append(arr)

    if not arrays:
        return None

    mean = np.mean(np.stack(arrays, axis=0), axis=0)
    total = float(mean.sum())

    if total > 0:
        mean = mean / total

    return mean.astype(float).tolist()

def export_semantic_pointcloud_ply(
    chunk: Any,
    out_path: str | Path,
    *,
    max_points_per_object: int = 3000,
    max_points_per_fused_object: int = 80000,
    depth_min: float = 0.05,
    depth_max: float = 20.0,
    random_seed: int = 2025,
) -> dict[str, Any]:
    """Export observation-level and fused object-level semantic point clouds.

    Outputs:
      1. semantic_pointcloud.ply
         Observation-level point cloud. Each object observation in each frame
         has its own observation_id.

      2. semantic_pointcloud_fused.ply
         Fused object-level point cloud. Observations are conservatively fused
         by (label, track_id).

      3. semantic_objects_observations.json
         Per-frame object observation metadata, including global centroid/bbox.

      4. semantic_objects_fused.json
         Cross-frame fused object metadata, including global centroid/bbox.

    Fusion rule:
      - If track_id is valid, fuse by (label, track_id).
      - If track_id is missing/None, do not fuse across frames; keep that
        observation as its own fused object.

    This avoids wrongly merging nearby repeated objects of the same category,
    e.g. multiple power outlets or multiple picture frames.
    """
    from PIL import Image

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(random_seed)

    obs_vertices: list[tuple[float, float, float, int, int, int, int]] = []
    obs_records: list[dict[str, Any]] = []
    obs_points: list[np.ndarray] = []
    group_to_obs_ids: dict[str, list[int]] = {}

    def _valid_track_id(value: Any) -> str | None:
        if value is None:
            return None
        s = str(value).strip()
        if not s or s.lower() in {"none", "null", "nan"}:
            return None
        return s

    def _to_float_list(arr: np.ndarray) -> list[float]:
        return [float(x) for x in np.asarray(arr, dtype=np.float64).reshape(-1).tolist()]

    def _write_ascii_ply(
        ply_path: Path,
        vertices: list[tuple[float, float, float, int, int, int, int]],
        comments: list[str],
        id_property_name: str,
    ) -> None:
        ply_path.parent.mkdir(parents=True, exist_ok=True)
        with ply_path.open("w", encoding="utf-8") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            for comment in comments:
                f.write(f"comment {comment}\n")
            f.write(f"element vertex {len(vertices)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write(f"property int {id_property_name}\n")
            f.write("end_header\n")

            for x, y, z, r, g, b, sid in vertices:
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b} {sid}\n")

    # ---------------------------------------------------------------------
    # 1. Build observation-level 3D points.
    # ---------------------------------------------------------------------
    for fp in chunk.frames:
        if fp.depth is None or fp.intrinsics is None or fp.local_pose is None:
            continue

        depth = np.asarray(fp.depth, dtype=np.float32)
        if depth.ndim != 2:
            continue

        depth_h, depth_w = depth.shape
        K = np.asarray(fp.intrinsics, dtype=np.float32)

        if K.shape != (3, 3):
            continue

        fx = float(K[0, 0])
        fy = float(K[1, 1])
        cx = float(K[0, 2])
        cy = float(K[1, 2])

        if fx <= 0 or fy <= 0:
            continue

        T_world_from_cam = _se3_to_matrix(fp.local_pose)

        image_path = getattr(fp.frame, "image_path", None)
        if image_path and Path(image_path).exists():
            with Image.open(image_path) as im:
                image_w, image_h = im.size
        else:
            image_h, image_w = depth_h, depth_w

        for obj in fp.objects:
            label = str(getattr(obj, "label", "unknown"))
            track_id = _valid_track_id(getattr(obj, "track_id", None))
            score = float(getattr(obj, "score", 1.0))

            mask = _mask_or_bbox_to_depth_mask(
                obj,
                depth_hw=(depth_h, depth_w),
                image_hw=(image_h, image_w),
            )
            if mask is None:
                continue

            ys, xs = np.where(mask)
            if len(xs) == 0:
                continue

            z = depth[ys, xs]
            valid = np.isfinite(z) & (z > depth_min) & (z < depth_max)

            xs = xs[valid]
            ys = ys[valid]
            z = z[valid]

            if len(xs) == 0:
                continue

            if max_points_per_object and len(xs) > max_points_per_object:
                choice = rng.choice(
                    len(xs),
                    size=max_points_per_object,
                    replace=False,
                )
                xs = xs[choice]
                ys = ys[choice]
                z = z[choice]

            x_cam = (xs.astype(np.float32) - cx) / fx * z
            y_cam = (ys.astype(np.float32) - cy) / fy * z

            pts_cam_h = np.stack(
                [
                    x_cam,
                    y_cam,
                    z,
                    np.ones_like(z, dtype=np.float32),
                ],
                axis=1,
            )

            pts_world = (T_world_from_cam @ pts_cam_h.T).T[:, :3].astype(np.float32)

            if pts_world.shape[0] <= 0:
                continue

            observation_id = len(obs_records)

            # Conservative fusion key.
            # Valid track_id: fuse same label + same track.
            # No track_id: keep this observation separate.
            if track_id is not None:
                group_key = f"label={label}|track={track_id}"
            else:
                group_key = f"label={label}|observation={observation_id}"

            centroid = pts_world.mean(axis=0)
            bbox_min = pts_world.min(axis=0)
            bbox_max = pts_world.max(axis=0)

            bbox_xyxy = getattr(obj, "bbox_xyxy", None)
            bbox_xyxy_list = (
                [float(v) for v in bbox_xyxy]
                if bbox_xyxy is not None
                else None
            )

            obs_records.append(
                {
                    "observation_id": int(observation_id),
                    "fused_id": None,  # filled after fusion
                    "group_key": group_key,
                    "label": label,
                    "track_id": track_id,
                    "frame_index": int(fp.frame.index),
                    "score": score,
                    "has_mask": getattr(obj, "mask", None) is not None,
                    "bbox_xyxy": bbox_xyxy_list,
                    "color_name": getattr(obj, "color_name", None),
                    "color_histogram": getattr(obj, "color_histogram", None),
                    "color_source": getattr(obj, "color_source", None),
                    "n_points": int(pts_world.shape[0]),
                    "centroid_world": _to_float_list(centroid),

                    "bbox_world_min": _to_float_list(bbox_min),
                    "bbox_world_max": _to_float_list(bbox_max),
                    "depth_shape": [int(depth_h), int(depth_w)],
                    "image_shape": [int(image_h), int(image_w)],
                }
            )

            obs_points.append(pts_world)
            group_to_obs_ids.setdefault(group_key, []).append(observation_id)

            color = _color_from_string(f"observation|{group_key}")
            for p in pts_world:
                obs_vertices.append(
                    (
                        float(p[0]),
                        float(p[1]),
                        float(p[2]),
                        int(color[0]),
                        int(color[1]),
                        int(color[2]),
                        int(observation_id),
                    )
                )

    # ---------------------------------------------------------------------
    # 2. Build fused object-level points.
    # ---------------------------------------------------------------------
    fused_vertices: list[tuple[float, float, float, int, int, int, int]] = []
    fused_records: list[dict[str, Any]] = []

    # Preserve insertion order from group_to_obs_ids. This keeps fused_id stable
    # for the same processing order.
    for fused_id, (group_key, observation_ids) in enumerate(group_to_obs_ids.items()):
        point_blocks = [obs_points[i] for i in observation_ids]
        pts_all = np.concatenate(point_blocks, axis=0)

        if pts_all.shape[0] <= 0:
            continue

        first_obs = obs_records[observation_ids[0]]
        label = str(first_obs["label"])
        track_id = first_obs["track_id"]

        frames = sorted({int(obs_records[i]["frame_index"]) for i in observation_ids})
        scores = [float(obs_records[i]["score"]) for i in observation_ids]

        color_names = [
            obs_records[i].get("color_name")
            for i in observation_ids
        ]
        color_sources = [
            obs_records[i].get("color_source")
            for i in observation_ids
        ]
        color_histograms = [
            obs_records[i].get("color_histogram")
            for i in observation_ids
        ]

        fused_color_name = _majority_vote(color_names, default="unknown")
        fused_color_source = _majority_vote(color_sources, default="none")
        fused_color_histogram = _mean_histogram(color_histograms)

        centroid = pts_all.mean(axis=0)

        bbox_min = pts_all.min(axis=0)
        bbox_max = pts_all.max(axis=0)

        # Fill fused_id back into observation records.
        for obs_id in observation_ids:
            obs_records[obs_id]["fused_id"] = int(fused_id)

        # Limit written fused points for file size, while keeping full stats.
        pts_write = pts_all
        if (
            max_points_per_fused_object
            and pts_write.shape[0] > max_points_per_fused_object
        ):
            choice = rng.choice(
                pts_write.shape[0],
                size=max_points_per_fused_object,
                replace=False,
            )
            pts_write = pts_write[choice]

        color = _color_from_string(f"fused|{group_key}")
        for p in pts_write:
            fused_vertices.append(
                (
                    float(p[0]),
                    float(p[1]),
                    float(p[2]),
                    int(color[0]),
                    int(color[1]),
                    int(color[2]),
                    int(fused_id),
                )
            )

        fused_records.append(
            {
                "fused_id": int(fused_id),
                "group_key": group_key,
                "label": label,
                "track_id": track_id,
                "frames": frames,
                "observation_ids": [int(i) for i in observation_ids],
                "n_observations": int(len(observation_ids)),
                "n_points_total": int(pts_all.shape[0]),
                "n_points_written": int(pts_write.shape[0]),
                "score_mean": float(np.mean(scores)) if scores else 0.0,
                "score_max": float(np.max(scores)) if scores else 0.0,
                "color_name": fused_color_name,
                "color_histogram": fused_color_histogram,
                "color_source": fused_color_source,
                "centroid_world": _to_float_list(centroid),

                "bbox_world_min": _to_float_list(bbox_min),
                "bbox_world_max": _to_float_list(bbox_max),
            }
        )

    # ---------------------------------------------------------------------
    # 3. Write files.
    # ---------------------------------------------------------------------
    obs_comments = [
        (
            f"observation_id={r['observation_id']},"
            f"fused_id={r['fused_id']},"
            f"label={r['label']},"
            f"track_id={r['track_id']},"
            f"frame={r['frame_index']}"
        )
        for r in obs_records
    ]

    fused_comments = [
        (
            f"fused_id={r['fused_id']},"
            f"label={r['label']},"
            f"track_id={r['track_id']},"
            f"n_observations={r['n_observations']}"
        )
        for r in fused_records
    ]

    observation_ply_path = out_path
    fused_ply_path = out_path.with_name("semantic_pointcloud_fused.ply")
    observations_json_path = out_path.with_name("semantic_objects_observations.json")
    fused_objects_json_path = out_path.with_name("semantic_objects_fused.json")

    _write_ascii_ply(
        observation_ply_path,
        obs_vertices,
        obs_comments,
        id_property_name="observation_id",
    )

    _write_ascii_ply(
        fused_ply_path,
        fused_vertices,
        fused_comments,
        id_property_name="fused_id",
    )

    observations_json_path.write_text(
        json.dumps(obs_records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    fused_objects_json_path.write_text(
        json.dumps(fused_records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Keep old keys for compatibility with your current main().
    return {
        "ply_path": str(observation_ply_path),
        "n_points": int(len(obs_vertices)),
        "n_semantic_instances": int(len(obs_records)),

        "observation_ply_path": str(observation_ply_path),
        "fused_ply_path": str(fused_ply_path),
        "observations_json_path": str(observations_json_path),
        "fused_objects_json_path": str(fused_objects_json_path),

        "n_observation_points": int(len(obs_vertices)),
        "n_observations": int(len(obs_records)),
        "n_fused_points": int(len(fused_vertices)),
        "n_fused_objects": int(len(fused_records)),
    }



@dataclass
class CachedBackend:
    """A tiny backend wrapper so OfflineMappingEngine reuses one perception pass."""

    chunk: Any

    def process_chunk(self, frames, chunk_id: int):
        from directme.perception.base import ChunkPerception

        return ChunkPerception(chunk_id=chunk_id, frames=self.chunk.frames)


def _summarize_chunk(chunk: Any) -> dict[str, Any]:
    frame_summaries = []
    total_objects = 0
    total_masks = 0

    for fp in chunk.frames:
        n_obj = len(fp.objects)
        n_mask = sum(1 for obj in fp.objects if getattr(obj, "mask", None) is not None)
        total_objects += n_obj
        total_masks += n_mask

        frame_summaries.append(
            {
                "frame_index": fp.frame.index,
                "has_depth": fp.depth is not None,
                "depth_shape": None if fp.depth is None else list(np.asarray(fp.depth).shape),
                "has_intrinsics": fp.intrinsics is not None,
                "has_pose": fp.local_pose is not None,
                "n_objects": n_obj,
                "n_masks": n_mask,
                "scene_tag": getattr(fp, "scene_tag", None),
                "objects_preview": [
                    {
                        "label": obj.label,
                        "track_id": obj.track_id,
                        "score": float(obj.score),
                        "has_mask": obj.mask is not None,
                        "color_name": getattr(obj, "color_name", None),
                        "color_source": getattr(obj, "color_source", None),
                    }
                    for obj in fp.objects[:10]
                ],

            }
        )

    return {
        "n_frames": len(chunk.frames),
        "total_objects": total_objects,
        "total_masks": total_masks,
        "frames": frame_summaries,
    }


def _summarize_graph(graph: Any) -> dict[str, Any]:
    nodes = getattr(graph, "nodes", {})
    edges = getattr(graph, "edges", {})

    if isinstance(nodes, dict):
        node_items = list(nodes.items())
    else:
        try:
            node_items = list(enumerate(nodes))
        except Exception:
            node_items = []

    preview = []
    for node_id, node in node_items[:20]:
        item = {
            "node_id": str(node_id),
            "repr": repr(node)[:300],
        }
        for attr in [
            "label",
            "track_id",
            "color_name",
            "color_source",
            "position",
            "centroid",
            "world_position",
        ]:

            if hasattr(node, attr):
                value = getattr(node, attr)
                if hasattr(value, "tolist"):
                    value = value.tolist()
                item[attr] = value
        preview.append(item)

    return {
        "n_nodes": len(nodes) if hasattr(nodes, "__len__") else None,
        "n_edges": len(edges) if hasattr(edges, "__len__") else None,
        "nodes_preview": preview,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Full SCAL3R + YOLO-World + SAM2 + DirectMe mapping test."
    )

    parser.add_argument(
        "--image-dir",
        type=str,
        default="/data/ywang/dataset/SpatialMemory/data_frames_1fps/scene0804_00-0",
    )
    parser.add_argument("--max-images", type=int, default=30)

    parser.add_argument(
        "--work-dir",
        type=str,
        default="/data/ywang/my_projects/VideoUnderstanding/Directme/tmp/directme_scal3r_full_pipeline",
    )
    parser.add_argument(
        "--precomputed-scal3r-root",
        type=str,
        default="",
        help="Existing SCAL3R result dir containing mat.txt/intri.yml/depths. "
             "Can also be a parent dir if it contains exactly one mat.txt.",
    )

    parser.add_argument(
        "--scal3r-config",
        type=str,
        default="/data/ywang/my_projects/VideoUnderstanding/Directme/configs/scal3r/scal3r.yaml",
    )
    parser.add_argument(
        "--scal3r-checkpoint",
        type=str,
        default="/data/ywang/my_projects/VideoUnderstanding/Directme/ckpts/scal3r/scal3r.pt",
    )
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument(
        "--yolo-weights",
        type=str,
        default="/data/ywang/my_projects/VideoUnderstanding/Directme/ckpts/yolo/yolov8m-worldv2.pt",
    )
    parser.add_argument(
        "--classes-file",
        type=str,
        default="/data/ywang/my_projects/VideoUnderstanding/Directme/directme/perception/adapters/Object.yaml",
    )
    parser.add_argument("--class-limit", type=int, default=300)
    parser.add_argument("--score-threshold", type=float, default=0.15)
    parser.add_argument("--detection-stride", type=int, default=1)

    parser.add_argument("--use-sam2", action="store_true", default=True)
    parser.add_argument(
        "--sam2-checkpoint",
        type=str,
        default="/data/ywang/my_projects/VideoUnderstanding/Directme/ckpts/sam2/sam2.1_hiera_base_plus.pt",
    )
    parser.add_argument(
        "--sam2-config",
        type=str,
        default="configs/sam2.1/sam2.1_hiera_b+.yaml",
    )

    parser.add_argument("--max-points-per-object", type=int, default=5000)
    parser.add_argument("--require-graph-nodes", action="store_true", default=False)
    parser.add_argument("--require-masks", action="store_true", default=False)

    args = parser.parse_args()

    from directme.config import DirectMeConfig
    from directme.mapping.offline_engine import OfflineMappingEngine
    from directme.perception.adapters.open_vocab_tracking import (
        OpenVocabularyTrackingAdapter,
        Sam2MaskRefiner,
        SimpleIoUAppearanceTracker,
        YoloWorldDetector,
    )
    from directme.perception.adapters.scal3r import (
        Scal3RComposedBackend,
        Scal3RDepthPoseAdapter,
        Scal3RRunner,
    )

    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    precomputed_root = _resolve_scal3r_result_dir(args.precomputed_scal3r_root)

    image_paths = _list_images(args.image_dir, args.max_images)

    if precomputed_root is not None:
        n_poses = _count_mat_rows(precomputed_root)
        if len(image_paths) != n_poses:
            print(
                f"[WARN] image count ({len(image_paths)}) != SCAL3R pose count ({n_poses}); "
                f"using first {n_poses} images."
            )
            image_paths = _list_images(args.image_dir, n_poses)

    frames = _make_video_frames(image_paths)

    print(f"[INFO] image_dir: {Path(args.image_dir).resolve()}")
    print(f"[INFO] n_frames: {len(frames)}")
    print(f"[INFO] work_dir: {work_dir}")
    print(f"[INFO] precomputed_scal3r_root: {precomputed_root}")
    print(f"[INFO] use_sam2: {args.use_sam2}")

    for p in image_paths[:5]:
        print(f"  - {p}")
    if len(image_paths) > 5:
        print("  ...")

    classes = _load_classes_from_yaml(args.classes_file, class_limit=args.class_limit)
    print(f"[INFO] classes count: {len(classes)}")
    print(f"[INFO] classes preview: {classes[:20]}")

    depth_pose = Scal3RDepthPoseAdapter(
        runner=Scal3RRunner(
            config=args.scal3r_config,
            checkpoint=args.scal3r_checkpoint,
            device=args.device,
            save_dpt=1,
            save_xyz=0,
        ),
        precomputed_root=precomputed_root,
        work_dir=work_dir / "scal3r_work",
        keep_work_dir=True,
    )

    detector = YoloWorldDetector(
        weights=args.yolo_weights,
        classes=classes,
        score_threshold=args.score_threshold,
        device=args.device,
    )

    segmenter = None
    if args.use_sam2:
        segmenter = Sam2MaskRefiner(
            checkpoint=args.sam2_checkpoint,
            config=args.sam2_config,
            device=args.device,
        )

    tracker = OpenVocabularyTrackingAdapter(
        detector=detector,
        segmenter=segmenter,
        tracker=SimpleIoUAppearanceTracker(),
        detection_stride=args.detection_stride,
    )

    backend = Scal3RComposedBackend(
        depth_pose=depth_pose,
        tracker=tracker,
    )

    print("[INFO] Running one perception pass...")
    chunk = backend.process_chunk(frames, chunk_id=0)

    print("[INFO] Extracting object color attributes...")
    color_summary = _attach_color_attributes_to_chunk(chunk)
    print(f"[INFO] color_summary: {color_summary}")

    chunk_summary = _summarize_chunk(chunk)


    if chunk_summary["total_objects"] <= 0:
        raise RuntimeError(
            "No objects were detected. Try lowering --score-threshold or increasing --class-limit."
        )

    if args.require_masks and chunk_summary["total_masks"] <= 0:
        raise RuntimeError(
            "No SAM2 masks were produced. Check --use-sam2 and SAM2 checkpoint/config paths."
        )

    ply_summary = export_semantic_pointcloud_ply(
        chunk,
        work_dir / "semantic_pointcloud.ply",
        max_points_per_object=args.max_points_per_object,
    )

    if ply_summary["n_points"] <= 0:
        raise RuntimeError(
            "Semantic point cloud has 0 points. Check depth, intrinsics, masks/bboxes, and scale alignment."
        )

    print("[INFO] Running OfflineMappingEngine with cached perception chunk...")
    mapping_run_dir = work_dir / "directme_mapping_run"
    config = DirectMeConfig(run_dir=str(mapping_run_dir))
    engine = OfflineMappingEngine(
        backend=CachedBackend(chunk),
        config=config,
    )
    events = engine.process_chunk(frames, chunk_id=0)

    graph_summary = _summarize_graph(engine.graph)

    if args.require_graph_nodes:
        n_nodes = graph_summary.get("n_nodes")
        if not n_nodes or n_nodes <= 0:
            raise RuntimeError("OfflineMappingEngine produced 0 graph nodes.")

    summary = {
        "status": "ok",
        "n_frames": len(frames),
        "used_sam2": bool(args.use_sam2),
        "used_precomputed_scal3r": precomputed_root is not None,
        "color_attributes": color_summary,
        "chunk": chunk_summary,
        "semantic_pointcloud": ply_summary,
        "mapping": {
            "n_events": len(events),
            "run_dir": str(mapping_run_dir),
            "scene_graph_json": str(mapping_run_dir / "scene_graph.json"),
            "graph": graph_summary,
        },
    }


    summary_path = work_dir / "full_pipeline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("[OK] Full SCAL3R + YOLO-World + SAM2 + DirectMe mapping test passed.")
    print(f"[INFO] Summary: {summary_path}")
    print(f"[INFO] Semantic point cloud: {ply_summary['ply_path']}")
    print(f"[INFO] Observation point cloud: {ply_summary['observation_ply_path']}")
    print(f"[INFO] Fused point cloud: {ply_summary['fused_ply_path']}")
    print(f"[INFO] Observation objects JSON: {ply_summary['observations_json_path']}")
    print(f"[INFO] Fused objects JSON: {ply_summary['fused_objects_json_path']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
