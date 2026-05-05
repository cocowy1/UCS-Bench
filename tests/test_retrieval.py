from directme.geometry.poses import SE3
from directme.mapping.scene_graph import SceneGraph
from directme.retrieval.retriever import EGO_NODE_ID, GraphRetriever


def test_retrieve_red_cups_count_and_egocentric():
    graph = SceneGraph(merge_threshold_m=0.5)
    graph.upsert_object("cup", [2, 0, 3], 0, 0, attributes={"color": "red"})
    graph.upsert_object("cup", [7.3, 0, 0.4], 20, 3, attributes={"color": "red"})
    current_pose = SE3.from_translation([7, 0, 0])
    ctx = GraphRetriever(graph).retrieve(
        "我身边有几个红杯子？在哪？", current_pose, language="zh"
    )
    assert ctx.count == 2
    assert len(ctx.items) == 2

    # Cup at world (7.3, 0, 0.4) is right in front of the user → front_right.
    # Cup at world (2.0, 0, 3.0) lies forward AND well to the left of the user.
    by_x = sorted(ctx.items, key=lambda i: i.node.spatial_absolute["p_world"][0])
    far, near = by_x[0], by_x[1]
    assert near.egocentric["relation"] == "front_right"
    assert far.egocentric["relation"] == "front_left"


def test_ego_edges_emitted_with_relation_and_reachability():
    graph = SceneGraph()
    # Object directly in front of the user, 0.4 m forward, easily reachable.
    graph.upsert_object("cup", [0.0, 0.0, 0.4], 0, 0, attributes={"color": "red"})
    # Object 8 m forward, well past the 5 m reachability threshold.
    graph.upsert_object("cup", [0.0, 0.0, 8.0], 0, 0, attributes={"color": "red"})

    current_pose = SE3.identity()
    ctx = GraphRetriever(graph, reachable_radius_m=5.0).retrieve(
        "我身边的红杯子在哪？我够得着吗？", current_pose, language="zh"
    )
    # Two ego edges, one per matched node.
    assert len(ctx.ego_edges) == 2
    for edge in ctx.ego_edges:
        assert edge["source"] == EGO_NODE_ID
        assert edge["relation"] in {"front", "front_left", "front_right"}
        assert edge["reference_frame"] == "Current_Camera"

    edges_by_target = {e["target"]: e for e in ctx.ego_edges}
    near_edge = next(e for e in ctx.ego_edges if e["distance_m"] < 1.0)
    far_edge = next(e for e in ctx.ego_edges if e["distance_m"] > 5.0)
    assert near_edge["reachable"] is True
    assert far_edge["reachable"] is False
    assert ctx.reachable_count == 1


def test_relation_labels_cover_all_octants():
    graph = SceneGraph()
    placements = {
        "front":         [0.0, 0.0,  3.0],
        "behind":        [0.0, 0.0, -3.0],
        "front_left":    [-2.0, 0.0,  3.0],
        "front_right":   [ 2.0, 0.0,  3.0],
        "behind_left":   [-2.0, 0.0, -3.0],
        "behind_right":  [ 2.0, 0.0, -3.0],
    }
    for label, p in placements.items():
        graph.upsert_object(label.replace("_", "-"), p, 0, 0)

    current_pose = SE3.identity()
    ctx = GraphRetriever(graph).retrieve("show me everything", current_pose, language="en")
    relations = {it.node.semantic_label.replace("-", "_"): it.egocentric["relation"]
                 for it in ctx.items}
    for expected_label, p in placements.items():
        assert relations.get(expected_label) == expected_label, (expected_label, relations)


def test_reachability_question_uses_reachability_phrasing():
    from directme.qa.generator import RuleBasedAnswerGenerator

    graph = SceneGraph()
    graph.upsert_object("cup", [0.0, 0.0, 0.5], 0, 0, attributes={"color": "red"})
    ctx = GraphRetriever(graph, reachable_radius_m=5.0).retrieve(
        "我能拿到杯子吗？", SE3.identity(), language="zh"
    )
    answer = RuleBasedAnswerGenerator().answer(ctx)
    assert ctx.intent.wants_reachability
    assert "可及" in answer or "伸手可及" in answer


def test_count_is_not_truncated_by_top_k():
    """Regression test for v0.2.x bug: ``ctx.count`` used to sum only the
    top_k retrieved items, so a high-density scene with N >> top_k objects
    was silently under-counted. v0.3 captures the full match set before
    truncation."""
    graph = SceneGraph()
    for i in range(12):
        graph.upsert_object("chair", [float(i), 0, 1], 0, i)

    ctx = GraphRetriever(graph).retrieve(
        "how many chairs are there", SE3.identity(), top_k=8
    )
    assert ctx.count == 12, "ctx.count must reflect total matches, not top_k"
    assert len(ctx.items) == 8, "items must still respect top_k for display"
    assert ctx.total_matched_count == 12
    assert len(ctx.total_matched_node_ids) == 12


def test_retrieve_does_not_mutate_graph_node():
    """Regression test for v0.2.x bug: render_egocentric() used to write
    the per-query camera-frame snapshot back onto the persistent node,
    causing query-time state to leak into saved graph files. v0.3 keeps
    the egocentric state strictly on the per-query RetrievedItem."""
    graph = SceneGraph()
    graph.upsert_object("cup", [0, 0, 3], 0, 0, attributes={"color": "red"})
    node = next(iter(graph.nodes.values()))

    # The node should NOT have a `spatial_egocentric_dynamic` attribute at all.
    assert not hasattr(node, "spatial_egocentric_dynamic")

    ctx = GraphRetriever(graph).retrieve("cup", SE3.identity())
    # Egocentric state lives on the per-query item, not the node.
    assert ctx.items[0].egocentric["relation"] == "front"
    assert not hasattr(node, "spatial_egocentric_dynamic")

    # The serialized graph must NOT carry a `spatial_egocentric_dynamic`
    # field for any node.
    payload = graph.to_dict()
    for n in payload["nodes"]:
        assert "spatial_egocentric_dynamic" not in n
