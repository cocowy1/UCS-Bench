"""Color attributes for object identity matching.

Two layers:
 1. Discrete color name normalization (multilingual aliases) — fast, coarse.
 2. HSV hue histogram with cosine similarity — fine-grained, robust to lighting.

The mapping engine uses (1) as a hard gate when both labels carry an unambiguous
color name, and (2) as a soft score that can break ties for same-label/no-color
candidates.
"""

from __future__ import annotations

import colorsys
import math
from collections.abc import Sequence
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# 1. Discrete color names (multilingual aliases)
# ---------------------------------------------------------------------------

_COLOR_ALIASES: dict[str, str] = {
    "红": "red", "红色": "red",
    "蓝": "blue", "蓝色": "blue",
    "绿": "green", "绿色": "green",
    "黄": "yellow", "黄色": "yellow",
    "黑": "black", "黑色": "black",
    "白": "white", "白色": "white",
    "灰": "gray", "灰色": "gray", "grey": "gray",
    "橙": "orange", "橙色": "orange",
    "紫": "purple", "紫色": "purple",
    "粉": "pink", "粉色": "pink",
    "棕": "brown", "棕色": "brown", "褐": "brown",
}


def normalize_color_name(color: str | None) -> str | None:
    if not color:
        return None
    c = str(color).strip().lower()
    if not c:
        return None
    return _COLOR_ALIASES.get(c, c)


# ---------------------------------------------------------------------------
# 2. HSV hue histogram features
# ---------------------------------------------------------------------------

# Hue centers (in [0, 1]) for coarse named-color classification.
_HUE_BINS = {
    "red":    (0.95, 0.05),  # wraps around 0
    "orange": (0.05, 0.10),
    "yellow": (0.10, 0.18),
    "green":  (0.20, 0.45),
    "cyan":   (0.45, 0.55),
    "blue":   (0.55, 0.72),
    "purple": (0.72, 0.83),
    "pink":   (0.83, 0.95),
}


def rgb_to_hsv_histogram(
    rgb_pixels: Sequence[tuple[int, int, int]], bins: int = 12
) -> list[float]:
    """Pure-python helper retained for backward compat tests."""
    hist = [0.0] * bins
    if not rgb_pixels:
        return hist
    for r, g, b in rgb_pixels:
        h, _s, _v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
        idx = min(int(h * bins), bins - 1)
        hist[idx] += 1.0
    total = sum(hist) or 1.0
    return [v / total for v in hist]


def hsv_histogram_from_image_mask(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    bins: int = 12,
    saturation_min: float = 0.15,
    value_min: float = 0.10,
) -> list[float]:
    """Compute a normalized hue histogram for masked pixels.

    Saturation/value gates filter out near-grayscale pixels which produce
    meaningless hues. Falls back to a uniform histogram for empty masks.
    """
    arr = np.asarray(image_rgb)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("image_rgb must be a (H, W, 3) RGB array")
    m = np.asarray(mask, dtype=bool)
    if m.shape != arr.shape[:2]:
        raise ValueError("mask shape must match image HxW")

    pixels = arr[m].astype(np.float32) / 255.0
    if pixels.size == 0:
        return [1.0 / bins] * bins

    r, g, b = pixels[:, 0], pixels[:, 1], pixels[:, 2]
    maxc = np.max(pixels, axis=1)
    minc = np.min(pixels, axis=1)
    v = maxc
    delta = maxc - minc
    s = np.where(maxc > 0, delta / np.maximum(maxc, 1e-8), 0.0)

    # Vectorized HSV hue computation.
    h = np.zeros_like(maxc)
    nonzero = delta > 1e-8
    rc = np.where(nonzero, (maxc - r) / np.maximum(delta, 1e-8), 0.0)
    gc = np.where(nonzero, (maxc - g) / np.maximum(delta, 1e-8), 0.0)
    bc = np.where(nonzero, (maxc - b) / np.maximum(delta, 1e-8), 0.0)
    h_red = (bc - gc)
    h_grn = (2.0 + rc - bc)
    h_blu = (4.0 + gc - rc)
    h = np.where(maxc == r, h_red, np.where(maxc == g, h_grn, h_blu))
    h = (h / 6.0) % 1.0

    keep = (s >= saturation_min) & (v >= value_min)
    if not keep.any():
        return [1.0 / bins] * bins
    h = h[keep]

    indices = np.minimum((h * bins).astype(int), bins - 1)
    counts = np.bincount(indices, minlength=bins).astype(np.float32)
    total = counts.sum()
    if total <= 0:
        return [1.0 / bins] * bins
    return (counts / total).tolist()


def histogram_cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two normalized histograms in [0, 1]."""
    av = np.asarray(a, dtype=np.float32)
    bv = np.asarray(b, dtype=np.float32)
    if av.shape != bv.shape or av.size == 0:
        return 0.0
    na = float(np.linalg.norm(av))
    nb = float(np.linalg.norm(bv))
    if na <= 0 or nb <= 0:
        return 0.0
    return float(np.clip(np.dot(av, bv) / (na * nb), 0.0, 1.0))


def histogram_chi_square_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Symmetric chi-square distance, lower is more similar."""
    av = np.asarray(a, dtype=np.float32)
    bv = np.asarray(b, dtype=np.float32)
    denom = av + bv + 1e-8
    return float(0.5 * np.sum((av - bv) ** 2 / denom))


def dominant_hsv_color(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    bins: int = 36,
    saturation_min: float = 0.18,
    value_min: float = 0.15,
) -> str:
    """Map masked pixels to one of the named hue bins, or to gray/white/black.

    This is intentionally coarse so it composes well with multilingual aliases
    in :func:`normalize_color_name`.
    """
    arr = np.asarray(image_rgb)
    m = np.asarray(mask, dtype=bool)
    pixels = arr[m].astype(np.float32) / 255.0
    if pixels.size == 0:
        return "unknown"

    maxc = np.max(pixels, axis=1)
    minc = np.min(pixels, axis=1)
    v = maxc
    delta = maxc - minc
    s = np.where(maxc > 0, delta / np.maximum(maxc, 1e-8), 0.0)

    # Achromatic case.
    chromatic_mask = (s >= saturation_min) & (v >= value_min)
    chromatic_ratio = float(chromatic_mask.mean())
    if chromatic_ratio < 0.20:
        avg_v = float(v.mean())
        if avg_v < 0.20:
            return "black"
        if avg_v > 0.80:
            return "white"
        return "gray"

    chromatic_pixels = pixels[chromatic_mask]
    r, g, b = chromatic_pixels[:, 0], chromatic_pixels[:, 1], chromatic_pixels[:, 2]
    maxc_c = np.max(chromatic_pixels, axis=1)
    delta_c = maxc_c - np.min(chromatic_pixels, axis=1)
    rc = (maxc_c - r) / np.maximum(delta_c, 1e-8)
    gc = (maxc_c - g) / np.maximum(delta_c, 1e-8)
    bc = (maxc_c - b) / np.maximum(delta_c, 1e-8)
    h_red = bc - gc
    h_grn = 2.0 + rc - bc
    h_blu = 4.0 + gc - rc
    h = np.where(maxc_c == r, h_red, np.where(maxc_c == g, h_grn, h_blu))
    h = (h / 6.0) % 1.0
    mean_h = float(np.median(h))

    for name, (lo, hi) in _HUE_BINS.items():
        if lo > hi:
            if mean_h >= lo or mean_h < hi:
                return name
        else:
            if lo <= mean_h < hi:
                return name
    return "unknown"
