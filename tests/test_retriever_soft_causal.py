"""Tests for soft scoring, strict-causal retrieval, and fallback behaviour."""

from __future__ import annotations

from directme.retrieval.retriever import GraphRetriever


def _build_graph():
    """Three nodes: a Cup (seen 0-10 s), a Couch (seen 5-15 s), a table (20-30 s)."""
    from graph_factories import make_graph, make_node, make_obs

    cup_obs = [make_obs(t, i, p_world=(0.5, 0.0, 1.0), track_id="t1", label="Cup", keyframe_path=f"f{i}.png")
               for t, i in zip(range(0, 11), range(11))]
    couch_obs = [make_obs(t, i, p_world=(2.0, 0.0, 3.0), track_id="t2", label="Couch", keyframe_path=f"g{i}.png")
                 for t, i in zip(range(5, 16), range(5, 16))]
    table_obs = [make_obs(t, i, p_world=(-1.0, 0.0, 2.0), track_id="t3", label="Dining Table", keyframe_path=f"h{i}.png")
                 for t, i in zip(range(20, 31), range(20, 31))]
    return make_graph([
        make_node("entity_001", "Cup", cup_obs, color="red"),
        make_node("entity_002", "Couch", couch_obs, color="gray"),
        make_node("entity_003", "Dining Table", table_obs, color="brown"),
    ])


def test_legacy_hard_gate_zero_retrieval_on_vocab_mismatch(identity_pose):
    graph = _build_graph()
    legacy = GraphRetriever(graph, strict_causal=False, soft_scoring=False)
    ctx = legacy.retrieve("Where is the sofa?", identity_pose, top_k=8, as_of_timestamp=50.0)
    # "sofa" intent never matches "Couch" labels → legacy returns nothing.
    assert ctx.items == []
    assert ctx.count == 0


def test_soft_scoring_returns_weak_candidates_on_vocab_mismatch(identity_pose):
    graph = _build_graph()
    improved = GraphRetriever(graph, strict_causal=True, soft_scoring=True)
    ctx = improved.retrieve("Where is the sofa?", identity_pose, top_k=8, as_of_timestamp=50.0)
    # Weak candidates are returned for evidence …
    assert len(ctx.items) > 0
    assert all(it.match_type == "fallback" for it in ctx.items)
    # … but never counted as matches (COUNT semantics preserved).
    assert ctx.count == 0


def test_positive_match_ranked_above_weak(identity_pose):
    graph = _build_graph()
    improved = GraphRetriever(graph, strict_causal=True, soft_scoring=True)
    ctx = improved.retrieve("Where is the cup?", identity_pose, top_k=8, as_of_timestamp=50.0)
    assert len(ctx.items) >= 1
    assert ctx.items[0].match_type == "matched"
    assert ctx.items[0].node.semantic_label == "Cup"
    assert ctx.count == 1


def test_strict_causal_excludes_future_nodes(identity_pose):
    graph = _build_graph()
    improved = GraphRetriever(graph, strict_causal=True, soft_scoring=True)
    # At t_q = 2 s the table (first seen at 20 s) must not exist yet.
    ctx = improved.retrieve("Where is the dining table?", identity_pose, top_k=8, as_of_timestamp=2.0)
    assert all(it.node.semantic_label != "Dining Table" for it in ctx.items)
    # And the Couch (first seen at 5 s) must not exist at t_q = 2 s either.
    assert all(it.node.semantic_label != "Couch" for it in ctx.items)


def test_strict_causal_position_uses_asof_anchor(identity_pose):
    """A moved movable object must render its position as of t_q, not the final one."""
    from directme.geometry.poses import SE3
    from graph_factories import make_graph, make_node, make_obs

    obs = [
        make_obs(0.0, 0, p_world=(0.0, 0.0, 1.0), track_id="t1"),
        make_obs(10.0, 10, p_world=(10.0, 0.0, 1.0), track_id="t1"),
    ]
    graph = make_graph([make_node("entity_001", "Cup", obs, is_movable=True)])
    graph.nodes["entity_001"].attributes["_spatial_absolute"]["p_world"] = [10.0, 0.0, 1.0]

    improved = GraphRetriever(graph, strict_causal=True, soft_scoring=True)
    ctx = improved.retrieve("Where is the cup?", SE3.identity(), top_k=8, as_of_timestamp=5.0)
    item = next(it for it in ctx.items if it.node.node_id == "entity_001")
    # Distance from world origin at t_q=5 s must reflect the early position
    # (~1 m), not the final anchor (10 m).
    assert item.egocentric["distance_m"] < 2.0

    legacy = GraphRetriever(graph, strict_causal=False, soft_scoring=False)
    ctx2 = legacy.retrieve("Where is the cup?", SE3.identity(), top_k=8, as_of_timestamp=5.0)
    item2 = next(it for it in ctx2.items if it.node.node_id == "entity_001")
    assert item2.egocentric["distance_m"] > 9.0  # legacy leaks the final anchor


def test_causal_keyframes_do_not_leak_future(identity_pose):
    graph = _build_graph()
    improved = GraphRetriever(graph, strict_causal=True, soft_scoring=True)
    ctx = improved.retrieve("Where is the cup?", identity_pose, top_k=8, as_of_timestamp=3.0)
    item = next(it for it in ctx.items if it.node.semantic_label == "Cup")
    timestamps = [o.timestamp for o in item.node.observations]
    assert max(timestamps) > 3.0  # the node *has* future observations
    # …but the keyframes handed out are all from observations ≤ t_q.
    assert len(item.keyframes) == 4  # frames 0..3


def test_empty_intent_count_is_zero_not_whole_graph(identity_pose):
    graph = _build_graph()
    improved = GraphRetriever(graph, strict_causal=True, soft_scoring=True)
    # "lamp" is not in the alias table → empty intent → no positive matches.
    ctx = improved.retrieve("How many lamps are there?", identity_pose, top_k=8, as_of_timestamp=50.0)
    assert ctx.count == 0
    assert len(ctx.items) > 0  # weak evidence still returned


def test_timelines_truncated_under_strict_causal(identity_pose):
    graph = _build_graph()
    graph.metadata["ego_pose_timeline"] = [
        {"timestamp": float(t), "chunk_id": 0, "translation": [0, 0, 0]} for t in range(0, 50)
    ]
    improved = GraphRetriever(graph, strict_causal=True, soft_scoring=True)
    ctx = improved.retrieve("Where is the cup?", identity_pose, top_k=8, as_of_timestamp=10.0)
    assert len(ctx.ego_pose_timeline) == 11
    legacy = GraphRetriever(graph, strict_causal=False, soft_scoring=False)
    ctx2 = legacy.retrieve("Where is the cup?", identity_pose, top_k=8, as_of_timestamp=10.0)
    assert len(ctx2.ego_pose_timeline) == 50
