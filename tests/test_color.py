"""Tests for HSV histogram and color-name fusion."""

import numpy as np

from directme.perception.color import (
    dominant_hsv_color,
    histogram_chi_square_distance,
    histogram_cosine_similarity,
    hsv_histogram_from_image_mask,
    normalize_color_name,
)


def test_normalize_color_name_aliases():
    assert normalize_color_name("红") == "red"
    assert normalize_color_name("RED") == "red"
    assert normalize_color_name("绿色") == "green"
    assert normalize_color_name(None) is None
    assert normalize_color_name("") is None
    assert normalize_color_name("teal") == "teal"


def _solid_color_image(rgb, h=8, w=8):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = rgb[0]
    img[:, :, 1] = rgb[1]
    img[:, :, 2] = rgb[2]
    return img


def test_hsv_histogram_red_concentrates_in_first_bin():
    img = _solid_color_image((220, 20, 20))
    mask = np.ones((8, 8), dtype=bool)
    hist = hsv_histogram_from_image_mask(img, mask, bins=12)
    assert sum(hist) > 0.99
    # Red hue is near 0, so bin 0 (and possibly bin 11) should dominate.
    assert hist[0] + hist[11] >= 0.9


def test_hsv_histogram_blue_concentrates_in_blue_bins():
    img = _solid_color_image((30, 30, 220))
    mask = np.ones((8, 8), dtype=bool)
    hist = hsv_histogram_from_image_mask(img, mask, bins=12)
    # Blue hue ~0.667, with 12 bins that's bin 8 (or 7 with rounding).
    assert hist[7] + hist[8] >= 0.9


def test_histogram_cosine_similarity_extremes():
    a = [1.0, 0.0, 0.0, 0.0]
    b = [1.0, 0.0, 0.0, 0.0]
    c = [0.0, 0.0, 1.0, 0.0]
    assert histogram_cosine_similarity(a, b) == 1.0
    assert histogram_cosine_similarity(a, c) == 0.0
    assert histogram_chi_square_distance(a, b) == 0.0
    assert histogram_chi_square_distance(a, c) > 0.5


def test_dominant_hsv_color_classification():
    img_red = _solid_color_image((220, 30, 30))
    img_blue = _solid_color_image((30, 30, 220))
    img_gray = _solid_color_image((128, 128, 128))
    img_black = _solid_color_image((10, 10, 10))
    img_white = _solid_color_image((245, 245, 245))
    mask = np.ones((8, 8), dtype=bool)
    assert dominant_hsv_color(img_red, mask) == "red"
    assert dominant_hsv_color(img_blue, mask) == "blue"
    assert dominant_hsv_color(img_gray, mask) == "gray"
    assert dominant_hsv_color(img_black, mask) == "black"
    assert dominant_hsv_color(img_white, mask) == "white"
