"""Tests for causal pose lookup and scene-graph schema round-trip."""

from __future__ import annotations

from directme.geometry.poses import SE3
from directme.mapping.scene_graph import SceneGraph
from directme.retrieval.pose_lookup import pose_from_graph_timeline, pose_record_from_se3
from graph_factories import make_graph, make_node, make_obs


def _timeline(translations):
    graph = make_graph([])
    for i, t in enumerate(translations):
        pose = SE3.identity()
        pose.matrix[0, 3] = t[0]
        pose.matrix[1, 3] = t[1]
        pose.matrix[2, 3] = t[2]
        graph.metadata.setdefault("ego_pose_timeline", []).append(
            pose_record_from_se3(pose, timestamp=float(i), chunk_id=0, frame_index=i)
        )
    return graph


def test_nearest_before_never_returns_future_pose():
    graph = _timeline([(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)])
    pose = pose_from_graph_timeline(graph, timestamp=1.6)
    assert abs(pose.translation[0] - 1.0) < 1e-6


def test_nearest_before_before_timeline_returns_earliest():
    graph = _timeline([(5, 0, 0), (6, 0, 0)])
    pose = pose_from_graph_timeline(graph, timestamp=0.0)
    assert abs(pose.translation[0] - 5.0) < 1e-6


def test_allow_future_restores_legacy_behaviour():
    graph = _timeline([(0, 0, 0), (10, 0, 0)])
    pose = pose_from_graph_timeline(graph, timestamp=9.5, allow_future=True)
    assert abs(pose.translation[0] - 10.0) < 1e-6


def test_latest_pose_when_no_timestamp():
    graph = _timeline([(0, 0, 0), (7, 0, 0)])
    pose = pose_from_graph_timeline(graph, timestamp=None)
    assert abs(pose.translation[0] - 7.0) < 1e-6


# ── schema round-trip (label / mask_score / first_seen_timestamp) ───────────


def test_observation_record_new_fields_roundtrip(tmp_path):
    obs = make_obs(1.5, 3, p_world=(1, 2, 3), label="Couch", mask_score=0.87)
    node = make_node("entity_001", "Couch", [obs], first_seen=1.5)
    graph = make_graph([node])
    p = tmp_path / "g.json"
    graph.save_json(p)
    loaded = SceneGraph.load_json(p)

    n = loaded.nodes["entity_001"]
    assert n.observations[0].label == "Couch"
    assert abs(n.observations[0].mask_score - 0.87) < 1e-6
    assert n.first_seen_timestamp == 1.5
    assert n.first_seen == 1.5


def test_old_schema_graph_defaults():
    """Graphs saved before v0.6 must load with None label/mask_score."""
    node = make_node("entity_001", "Cup", [make_obs(0.0, 0)])
    node.observations[0].label = None
    node.observations[0].mask_score = None
    node.first_seen_timestamp = None
    assert node.first_seen == 0.0  # falls back to created_at

    rec = {
        "timestamp": 0.0, "frame_index": 0, "track_id": "t", "p_world": [0, 0, 0],
    }
    from directme.mapping.scene_graph import ObservationRecord

    o = ObservationRecord.from_dict(rec)
    assert o.label is None and o.mask_score is None


def test_first_seen_survives_observation_truncation():
    """After 128-obs truncation, first_seen must not drift (old bug)."""
    obs = [make_obs(float(t), t, p_world=(0, 0, 0)) for t in range(300)]
    node = make_node("entity_001", "Cup", obs, first_seen=0.0)
    # Simulate the truncation that happens in add_observation.
    node.observations = node.observations[-128:]
    assert node.first_seen == 0.0  # persisted value wins over observations[0]
    assert node.observations[0].timestamp == 172.0
