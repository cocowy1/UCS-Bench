import numpy as np

from directme.mapping.scene_graph import SceneGraph


def test_graph_merges_near_same_label_and_spawns_far():
    graph = SceneGraph(merge_threshold_m=0.5)
    a, action_a = graph.upsert_object("cup", [0, 0, 1], 0, 0, attributes={"color": "red"})
    b, action_b = graph.upsert_object("cup", [0.1, 0, 1.1], 1, 1, attributes={"color": "red"})
    c, action_c = graph.upsert_object("cup", [3, 0, 1], 2, 2, attributes={"color": "red"})

    assert action_a == "spawn"
    assert action_b == "merge"
    assert action_c == "spawn"
    assert a.node_id == b.node_id
    assert c.node_id != a.node_id
    assert len(graph.nodes) == 2


def test_color_gate_prevents_wrong_merge():
    graph = SceneGraph(merge_threshold_m=0.5)
    graph.upsert_object("cup", [0, 0, 1], 0, 0, attributes={"color": "red"})
    graph.upsert_object("cup", [0.1, 0, 1.1], 1, 1, attributes={"color": "blue"})
    assert len(graph.nodes) == 2


def test_same_track_far_static_object_spawns_new_node():
    graph = SceneGraph(merge_threshold_m=0.5)
    graph.upsert_object("cup", [0, 0, 1], 0, 0, track_id="track_1")
    _, action = graph.upsert_object("cup", [5, 0, 1], 10, 10, track_id="track_1")
    assert action == "spawn"
    assert len(graph.nodes) == 2


def test_same_track_continuous_movable_object_can_move():
    graph = SceneGraph(merge_threshold_m=0.5, track_match_max_gap_frames=5, track_match_max_jump_m=2.0)
    node, action1 = graph.upsert_object(
        "phone", [0, 0, 1], 0, 0, track_id="track_phone", attributes={"is_movable": True}
    )
    node2, action2 = graph.upsert_object(
        "phone", [1.5, 0, 1], 1, 1, track_id="track_phone", attributes={"is_movable": True}
    )
    assert action1 == "spawn"
    assert action2 == "merge"
    assert node.node_id == node2.node_id
