from __future__ import annotations

from typing import Any

import numpy as np


def backproject_pixel(u: float, v: float, depth: float, intrinsics: np.ndarray) -> np.ndarray:
    """Back-project a pixel into camera coordinates.

    Camera convention: x right, y down, z forward.
    """
    k = np.asarray(intrinsics, dtype=float)
    if k.shape != (3, 3):
        raise ValueError("intrinsics must have shape (3, 3)")
    if depth <= 0 or not np.isfinite(depth):
        raise ValueError("depth must be a positive finite number")
    fx, fy = k[0, 0], k[1, 1]
    cx, cy = k[0, 2], k[1, 2]
    x = (float(u) - cx) * depth / fx
    y = (float(v) - cy) * depth / fy
    return np.array([x, y, depth], dtype=float)


def _mask_to_bool(mask: Any) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError("mask must be a 2D array")
    return arr.astype(bool)


def robust_mask_depth(mask: Any, depth_map: np.ndarray, percentile_clip: tuple[float, float] = (5, 95)) -> float:
    mask_bool = _mask_to_bool(mask)
    depth = np.asarray(depth_map, dtype=float)
    if depth.shape != mask_bool.shape:
        raise ValueError("depth_map and mask must have the same HxW shape")
    values = depth[mask_bool]
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        raise ValueError("mask contains no valid positive depth values")
    lo, hi = np.percentile(values, percentile_clip)
    clipped = values[(values >= lo) & (values <= hi)]
    if clipped.size == 0:
        clipped = values
    return float(np.median(clipped))


def mask_centroid(mask: Any) -> tuple[float, float]:
    mask_bool = _mask_to_bool(mask)
    ys, xs = np.nonzero(mask_bool)
    if xs.size == 0:
        raise ValueError("empty mask")
    return float(xs.mean()), float(ys.mean())


def unproject_mask_centroid(mask: Any, depth_map: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """Use a mask centroid and robust median object depth to estimate object center in camera frame."""
    u, v = mask_centroid(mask)
    z = robust_mask_depth(mask, depth_map)
    return backproject_pixel(u, v, z, intrinsics)



def _clip_bbox_xyxy(bbox_xyxy: tuple[float, float, float, float], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    x1, y1, x2, y2 = bbox_xyxy
    ix1 = max(0, min(w - 1, int(np.floor(x1))))
    iy1 = max(0, min(h - 1, int(np.floor(y1))))
    ix2 = max(ix1 + 1, min(w, int(np.ceil(x2))))
    iy2 = max(iy1 + 1, min(h, int(np.ceil(y2))))
    return ix1, iy1, ix2, iy2


def robust_bbox_depth(
    bbox_xyxy: tuple[float, float, float, float],
    depth_map: np.ndarray,
    inner_ratio: float = 0.5,
    percentile_clip: tuple[float, float] = (5, 95),
) -> float:
    """Estimate object depth from a bbox when a segmentation mask is unavailable.

    The central window is used by default because detector boxes often include
    background. This is less precise than mask-based unprojection, but it keeps
    DirectMe usable when SAM 2 is not installed.
    """
    depth = np.asarray(depth_map, dtype=float)
    if depth.ndim != 2:
        raise ValueError("depth_map must have shape (H, W)")
    x1, y1, x2, y2 = _clip_bbox_xyxy(bbox_xyxy, depth.shape)
    if inner_ratio <= 0 or inner_ratio > 1:
        raise ValueError("inner_ratio must be in (0, 1]")
    if inner_ratio < 1:
        bw = x2 - x1
        bh = y2 - y1
        shrink_x = int(round(bw * (1 - inner_ratio) / 2.0))
        shrink_y = int(round(bh * (1 - inner_ratio) / 2.0))
        x1 += shrink_x
        x2 -= shrink_x
        y1 += shrink_y
        y2 -= shrink_y
        x2 = max(x2, x1 + 1)
        y2 = max(y2, y1 + 1)
    values = depth[y1:y2, x1:x2].reshape(-1)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        raise ValueError("bbox contains no valid positive depth values")
    lo, hi = np.percentile(values, percentile_clip)
    clipped = values[(values >= lo) & (values <= hi)]
    if clipped.size == 0:
        clipped = values
    return float(np.median(clipped))


def bbox_center(bbox_xyxy: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox_xyxy
    return float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)


def unproject_bbox_center(
    bbox_xyxy: tuple[float, float, float, float],
    depth_map: np.ndarray,
    intrinsics: np.ndarray,
    inner_ratio: float = 0.5,
) -> np.ndarray:
    """Back-project a detector bbox center using robust depth in the bbox."""
    u, v = bbox_center(bbox_xyxy)
    z = robust_bbox_depth(bbox_xyxy, depth_map, inner_ratio=inner_ratio)
    return backproject_pixel(u, v, z, intrinsics)
