"""Tests for v0.6 retrieval and evaluator additions:

* :func:`GraphRetriever._score_node` boosts nodes whose ``scene_tag`` /
  ``place_id`` matches an ``intent.rooms`` token (Bug #2 from the audit).
* :func:`score_position_orientation` resolves the queried target via
  ``expected_target_node_id`` / ``expected_target_place`` before falling
  back to label substring (Bug #2 / #3).
* :class:`UCSBenchEvaluator` honours ``expected_query_pose`` (Bug #3).
"""

from __future__ import annotations

import numpy as np

from directme.datasets.ucsbench import UCSQuestion
from directme.eval.ucsbench import UCSBenchEvaluator, score_position_orientation
from directme.geometry.poses import SE3
from directme.mapping.scene_graph import EntityNode, ObservationRecord, SceneGraph
from directme.retrieval.retriever import GraphRetriever


def _two_cup_graph() -> SceneGraph:
    graph = SceneGraph()
    living = EntityNode(
        node_id="entity_001",
        semantic_label="cup",
        attributes={"color": "red", "scene_tag": "living room", "is_movable": True},
        observations=[ObservationRecord(timestamp=0.0, frame_index=0, track_id="t1",
                                         p_world=[2.0, 0.0, 3.0])],
        place_id="place_living",
    )
    living.attributes["_spatial_absolute"] = {
        "reference_frame": graph.reference_frame,
        "p_world": [2.0, 0.0, 3.0],
        "observation_count": 1,
        "last_seen_timestamp": 0.0,
    }
    kitchen = EntityNode(
        node_id="entity_002",
        semantic_label="cup",
        attributes={"color": "red", "scene_tag": "kitchen", "is_movable": True},
        observations=[ObservationRecord(timestamp=20.0, frame_index=3, track_id="t2",
                                         p_world=[7.3, 0.0, 0.4])],
        place_id="place_kitchen",
    )
    kitchen.attributes["_spatial_absolute"] = {
        "reference_frame": graph.reference_frame,
        "p_world": [7.3, 0.0, 0.4],
        "observation_count": 1,
        "last_seen_timestamp": 20.0,
    }
    graph.nodes[living.node_id] = living
    graph.nodes[kitchen.node_id] = kitchen
    return graph


def test_room_soft_match_prefers_living_room_cup() -> None:
    graph = _two_cup_graph()
    retriever = GraphRetriever(graph)
    ctx = retriever.retrieve(
        "客厅那个红杯子相对于我现在在哪个方位？",
        current_pose=SE3.from_translation([7.0, 0.0, 0.0]),
        top_k=2,
    )
    # Both cups are returned, but the living-room cup outscores the kitchen one.
    assert [it.node.node_id for it in ctx.items][0] == "entity_001"


def test_position_orientation_uses_expected_target_place() -> None:
    graph = _two_cup_graph()
    retriever = GraphRetriever(graph)
    ctx = retriever.retrieve(
        "红杯子在哪？",
        current_pose=SE3.from_translation([7.0, 0.0, 0.0]),
        top_k=2,
    )
    q = UCSQuestion(
        video_uid="demo", query_timestamp=0.0,
        question="客厅那个红杯子在哪？",
        options={
            "expected_relation": "front_left",
            "expected_target_label": "cup",
            "expected_target_place": "living_room",
        },
    )
    correct, predicted, expected, _ = score_position_orientation(q, ctx)
    assert correct is True
    assert predicted["target_node"] == "entity_001"


def test_expected_query_pose_overrides_timeline() -> None:
    graph = _two_cup_graph()
    # Plant a misleading timeline pose at t=0 (identity).
    graph.metadata["ego_pose_timeline"] = [
        {"timestamp": 0.0, "T_world_from_camera": SE3.identity().to_list()},
    ]
    pose_at_kitchen = SE3.from_translation([7.0, 0.0, 0.0])
    q = UCSQuestion(
        video_uid="demo", query_timestamp=0.0,
        question="客厅那个红杯子相对于我现在在哪个方位？",
        options={
            "expected_relation": "front_left",
            "expected_target_label": "cup",
            "expected_target_place": "living_room",
            "expected_query_pose": pose_at_kitchen.to_list(),
        },
    )
    evaluator = UCSBenchEvaluator(graph=graph)
    pred = evaluator.evaluate_one(q)
    assert pred.correct is True
    assert pred.predicted["relation"] == "front_left"


def test_room_soft_match_does_not_filter_when_no_room_in_query() -> None:
    """A query without any room token should not be biased by scene_tag."""
    graph = _two_cup_graph()
    retriever = GraphRetriever(graph)
    ctx = retriever.retrieve("红杯子在哪？", current_pose=SE3.identity(), top_k=2)
    # Both cups are still returned (the room boost is additive, not a filter).
    ids = sorted(it.node.node_id for it in ctx.items)
    assert ids == ["entity_001", "entity_002"]
