"""Tests for the discrete egocentric relation classifier and reachability."""

from directme.retrieval.egocentric import (
    DEFAULT_REACHABLE_RADIUS_M,
    EGO_RELATION_LABELS,
    classify_egocentric_relation,
    compute_ego_relation,
    natural_language_location,
)


def test_relation_label_set_is_stable():
    assert set(EGO_RELATION_LABELS) == {
        "front", "behind", "left", "right",
        "front_left", "front_right", "behind_left", "behind_right",
    }


def test_classify_pure_front_and_behind():
    assert classify_egocentric_relation([0.0, 0.0, 1.0]) == "front"
    assert classify_egocentric_relation([0.0, 0.0, -1.0]) == "behind"


def test_classify_lateral_offsets():
    # Strongly to the right while moving forward.
    assert classify_egocentric_relation([2.0, 0.0, 1.0]) == "front_right"
    assert classify_egocentric_relation([-2.0, 0.0, 1.0]) == "front_left"
    assert classify_egocentric_relation([2.0, 0.0, -1.0]) == "behind_right"
    assert classify_egocentric_relation([-2.0, 0.0, -1.0]) == "behind_left"


def test_centered_cone_collapses_to_pure_front():
    # |x| = 0.1 with z = 5: tolerance = 0.20 * 5 = 1.0, so 0.1 is centered.
    assert classify_egocentric_relation([0.1, 0.0, 5.0]) == "front"
    # Tighten the cone and it flips to front_right.
    assert classify_egocentric_relation([0.1, 0.0, 5.0], lateral_tolerance_ratio=0.005) == "front_right"


def test_compute_ego_relation_reachable_within_radius():
    rel = compute_ego_relation([0.3, 0.0, 0.4], reachable_radius_m=DEFAULT_REACHABLE_RADIUS_M)
    assert rel.reachable is True
    assert rel.distance_m < DEFAULT_REACHABLE_RADIUS_M


def test_compute_ego_relation_out_of_reach_beyond_radius():
    rel = compute_ego_relation([0.0, 0.0, 6.0], reachable_radius_m=5.0)
    assert rel.reachable is False
    assert rel.distance_m > 5.0


def test_compute_ego_relation_distance_uses_euclidean_norm_including_height():
    # 3-4-5 triangle: sqrt(3^2 + 4^2) = 5
    rel = compute_ego_relation([3.0, 0.0, 4.0], reachable_radius_m=5.0)
    assert abs(rel.distance_m - 5.0) < 1e-9
    # Tied at the boundary, classified as reachable (<=).
    assert rel.reachable is True


def test_natural_language_localization_zh_and_en():
    zh = natural_language_location("front_right", 0.4, True, language="zh")
    en = natural_language_location("front_right", 0.4, True, language="en")
    assert "右前方" in zh and "伸手可及" in zh
    assert "front-right" in en and "within reach" in en

    zh_far = natural_language_location("behind", 12.3, False, language="zh")
    assert "正后方" in zh_far and "不可及" in zh_far


def test_compute_ego_relation_reports_consistent_p_cam():
    rel = compute_ego_relation([1.0, -0.5, 2.0])
    assert rel.p_cam == (1.0, -0.5, 2.0)


def test_classify_pure_left_and_right_when_abreast():
    assert classify_egocentric_relation([1.0, 0.0, 0.0]) == "right"
    assert classify_egocentric_relation([-1.0, 0.0, 0.0]) == "left"
