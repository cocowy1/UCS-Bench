"""Tests for directme.retrieval.reliability."""

from __future__ import annotations

import numpy as np

from directme.retrieval.reliability import keyframes_as_of, node_reliability, position_as_of
from graph_factories import make_node, make_obs


def test_reliability_high_for_consistent_static_node():
    obs = [
        make_obs(t, i, p_world=(1.0, 0.0, 2.0), confidence=0.9, track_id="t1", label="Cup")
        for t, i in zip(range(0, 20), range(20))
    ]
    node = make_node(observations=obs)
    r = node_reliability(node)
    assert r["reliability"] > 0.8
    assert r["track_stability"] == 1.0
    assert r["position_consistency"] > 0.9
    assert r["semantic_consistency"] == 1.0


def test_reliability_low_for_fragmented_jumpy_node():
    pts = [(float(i % 5), 0.0, 0.0) for i in range(20)]
    obs = [
        make_obs(t, i, p_world=pts[i], confidence=0.25, track_id=f"t{i}", label=f"L{i % 3}")
        for t, i in zip(range(20), range(20))
    ]
    node = make_node(observations=obs)
    r = node_reliability(node)
    assert r["reliability"] < 0.5
    assert r["track_stability"] < 0.1


def test_reliability_causal_as_of_filters_future():
    obs = [make_obs(t, i, p_world=(t * 0.01, 0, 0), confidence=0.9, track_id="t1") for t, i in zip(range(10), range(10))]
    node = make_node(observations=obs)
    r_all = node_reliability(node)
    r_early = node_reliability(node, as_of_timestamp=0.0)
    assert r_all["n_observations"] == 10
    assert r_early["n_observations"] == 1
    assert r_early["temporal_persistence"] < r_all["temporal_persistence"]


def test_reliability_neutral_when_no_signals():
    node = make_node(observations=[])
    assert node_reliability(node)["reliability"] == 0.0
    # Single toy observation without confidence signal still yields a score.
    node = make_node(observations=[make_obs(0, 0, confidence=1.0)])
    r = node_reliability(node)
    assert 0.0 <= r["reliability"] <= 1.0


def test_position_as_of_ignores_future_and_weights():
    obs = [
        make_obs(0.0, 0, p_world=(0.0, 0.0, 0.0), confidence=1.0, pose_confidence=1.0),
        make_obs(1.0, 1, p_world=(1.0, 0.0, 0.0), confidence=1.0, pose_confidence=1.0),
        make_obs(2.0, 2, p_world=(100.0, 0.0, 0.0), confidence=1.0, pose_confidence=1.0),
    ]
    node = make_node(observations=obs)
    p = position_as_of(node, as_of_timestamp=1.5)
    assert p is not None
    # causal: future obs excluded; with recency decay the anchor sits between
    # the two causal observations, slightly favouring the recent one.
    assert 0.0 < p[0] < 1.0
    assert abs(p[0] - 0.5) < 0.1
    assert position_as_of(node, as_of_timestamp=-1.0) is None


def test_position_as_of_recency_decay_favours_recent():
    """Early drifted observations must not drag the causal anchor."""
    obs = [
        make_obs(0.0, 0, p_world=(0.0, 0.0, 0.0), confidence=1.0, pose_confidence=1.0),
        make_obs(100.0, 100, p_world=(2.0, 0.0, 0.0), confidence=1.0, pose_confidence=1.0),
    ]
    node = make_node(observations=obs)
    # tau much smaller than the 100 s gap → the recent observation dominates.
    p = position_as_of(node, as_of_timestamp=100.0, recency_tau_s=10.0)
    assert p[0] > 1.8
    # Uniform causal mean (no decay) would sit at 1.0.
    p_uniform = position_as_of(node, as_of_timestamp=100.0, recency_tau_s=None)
    assert abs(p_uniform[0] - 1.0) < 1e-6


def test_position_as_of_movable_takes_last():
    obs = [
        make_obs(0.0, 0, p_world=(0.0, 0.0, 0.0)),
        make_obs(1.0, 1, p_world=(5.0, 0.0, 0.0)),
    ]
    node = make_node(observations=obs, is_movable=True)
    p = position_as_of(node, as_of_timestamp=1.0)
    assert np.allclose(p, [5.0, 0.0, 0.0])


def test_keyframes_as_of():
    obs = [
        make_obs(0.0, 0, keyframe_path="frame_000000.png"),
        make_obs(1.0, 1, keyframe_path="frame_000001.png"),
        make_obs(2.0, 2, keyframe_path="frame_000002.png"),
    ]
    node = make_node(observations=obs)
    kf = keyframes_as_of(node, as_of_timestamp=1.0)
    assert kf == ["frame_000000.png", "frame_000001.png"]
