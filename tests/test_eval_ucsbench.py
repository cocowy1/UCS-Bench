"""Tests for the UCS-Bench 4-dimension evaluator."""

import json
import tempfile
from pathlib import Path

from directme.datasets.ucsbench import UCSQuestion, load_ucs_questions
from directme.eval import UCSBenchEvaluator, UCSDimension, classify_dimension
from directme.geometry.poses import SE3
from directme.mapping.scene_graph import SceneGraph


def _two_cup_graph() -> SceneGraph:
    """Mirror the toy living-room→kitchen demo's final state."""
    g = SceneGraph(merge_threshold_m=0.5)
    g.upsert_object("cup", [2.0, 0.0, 3.0], 0, 0, attributes={"color": "red"})
    g.upsert_object("cup", [7.3, 0.0, 0.4], 20, 3, attributes={"color": "red"})
    return g


def _pose_at_kitchen() -> SE3:
    return SE3.from_translation([7.0, 0.0, 0.0])


# -------- classify_dimension -----------------------------------------------


def test_classify_dimension_uses_explicit_field():
    q = UCSQuestion(
        video_uid="x", query_timestamp=0.0,
        question="anything",
        options={"dimension": UCSDimension.PROXIMITY_REACHABILITY},
    )
    assert classify_dimension(q) == UCSDimension.PROXIMITY_REACHABILITY


def test_classify_dimension_falls_back_to_intent_parsing():
    q1 = UCSQuestion("v", 0.0, "几个红杯子？")
    q2 = UCSQuestion("v", 0.0, "我能拿到杯子吗？")
    q3 = UCSQuestion("v", 0.0, "杯子在我哪个方位？")
    q4 = UCSQuestion("v", 0.0, "我之前走过哪些房间？")
    assert classify_dimension(q1) == UCSDimension.CATEGORY_QUANTITY
    assert classify_dimension(q2) == UCSDimension.PROXIMITY_REACHABILITY
    assert classify_dimension(q3) == UCSDimension.POSITION_ORIENTATION
    assert classify_dimension(q4) == UCSDimension.TRAJECTORY_MOVEMENT


# -------- per-dimension scoring --------------------------------------------


def test_category_quantity_scores_count_and_labels():
    g = _two_cup_graph()
    questions = [
        UCSQuestion(
            "demo", 0.0, "几个红杯子？",
            options={
                "dimension": "category_quantity",
                "expected_count": 2,
                "expected_labels": ["cup"],
            },
        ),
        UCSQuestion(
            "demo", 0.0, "几个红杯子？",
            options={
                "dimension": "category_quantity",
                "expected_count": 5,  # wrong on purpose
                "expected_labels": ["cup"],
            },
        ),
    ]
    report = UCSBenchEvaluator(graph=g, pose_lookup={("demo", 0.0): _pose_at_kitchen()}).run(questions)
    cq = report.by_dimension[UCSDimension.CATEGORY_QUANTITY]
    assert cq.n_total == 2 and cq.n_scored == 2 and cq.n_correct == 1
    assert cq.accuracy == 0.5


def test_position_orientation_picks_first_matching_target():
    g = _two_cup_graph()
    q = UCSQuestion(
        "demo", 0.0, "客厅那个红杯子在我哪个方位？",
        options={
            "dimension": "position_orientation",
            "expected_relation": "front_left",
            "expected_target_label": "cup",
        },
    )
    report = UCSBenchEvaluator(graph=g, pose_lookup={("demo", 0.0): _pose_at_kitchen()}).run([q])
    po = report.by_dimension[UCSDimension.POSITION_ORIENTATION]
    assert po.n_correct == 1
    pred = report.predictions[0]
    assert pred.predicted["relation"] == "front_left"


def test_position_orientation_rejects_unknown_relation():
    g = _two_cup_graph()
    q = UCSQuestion(
        "demo", 0.0, "杯子在哪？",
        options={"dimension": "position_orientation", "expected_relation": "above_me"},
    )
    report = UCSBenchEvaluator(graph=g).run([q])
    pred = report.predictions[0]
    assert pred.correct is None
    assert pred.skipped_reason == "invalid_expected_relation"


def test_proximity_reachability_true_and_false_branches():
    g = _two_cup_graph()
    qs = [
        UCSQuestion(
            "demo", 0.0, "我能拿到红杯子吗？",
            options={
                "dimension": "proximity_reachability",
                "expected_reachable": True,
                "expected_target_label": "cup",
            },
        ),
        UCSQuestion(
            "demo", 0.0, "我能拿到手机吗？",
            options={
                "dimension": "proximity_reachability",
                "expected_reachable": False,
                "expected_target_label": "phone",  # not in graph
            },
        ),
    ]
    report = UCSBenchEvaluator(
        graph=g, pose_lookup={("demo", 0.0): _pose_at_kitchen()}
    ).run(qs)
    pr = report.by_dimension[UCSDimension.PROXIMITY_REACHABILITY]
    assert pr.n_total == 2 and pr.n_scored == 2 and pr.n_correct == 2


def test_trajectory_movement_partial_only_when_gold_missing():
    """A T&M question without ``expected_path_labels`` is skipped with the
    canonical ``trajectory_movement_partial_only`` reason. v0.3 only scores
    place-visit-sequence questions; richer trajectory queries fall back."""
    g = _two_cup_graph()
    q = UCSQuestion(
        "demo", 0.0, "我刚才走的具体路径是什么？",
        options={"dimension": "trajectory_movement"},  # no expected_path_labels
    )
    report = UCSBenchEvaluator(graph=g).run([q])
    tm = report.by_dimension[UCSDimension.TRAJECTORY_MOVEMENT]
    assert tm.n_total == 1
    assert tm.n_scored == 0
    assert tm.skipped_reason == "trajectory_movement_partial_only"
    assert tm.accuracy is None


def test_trajectory_movement_scores_against_place_visit_timeline():
    """When the graph carries a place_visit_timeline in metadata, T&M is
    scored against ``expected_path_labels``."""
    g = _two_cup_graph()
    g.metadata["place_visit_timeline"] = [
        {"chunk_id": 0, "timestamp": 0.0, "scene_tag": "living room"},
        {"chunk_id": 1, "timestamp": 10.0, "scene_tag": "kitchen"},
    ]
    q = UCSQuestion(
        "demo", 0.0, "我之前走过哪些房间？",
        options={
            "dimension": "trajectory_movement",
            "expected_path_labels": ["living_room", "kitchen"],
        },
    )
    report = UCSBenchEvaluator(graph=g).run([q])
    tm = report.by_dimension[UCSDimension.TRAJECTORY_MOVEMENT]
    assert tm.n_correct == 1 and tm.n_scored == 1


def test_trajectory_movement_false_when_visits_disagree():
    g = _two_cup_graph()
    g.metadata["place_visit_timeline"] = [
        {"chunk_id": 0, "timestamp": 0.0, "scene_tag": "office"},
    ]
    q = UCSQuestion(
        "demo", 0.0, "我之前走过哪些房间？",
        options={
            "dimension": "trajectory_movement",
            "expected_path_labels": ["living_room", "kitchen"],
        },
    )
    report = UCSBenchEvaluator(graph=g).run([q])
    tm = report.by_dimension[UCSDimension.TRAJECTORY_MOVEMENT]
    assert tm.n_scored == 1 and tm.n_correct == 0


# -------- end-to-end with the shipped sample dataset -----------------------


def test_sample_jsonl_round_trip(tmp_path):
    src = Path(__file__).resolve().parent.parent / "directme" / "eval" / "sample_questions.jsonl"
    questions = load_ucs_questions(src)
    assert len(questions) == 6
    dims = {classify_dimension(q) for q in questions}
    assert UCSDimension.CATEGORY_QUANTITY in dims
    assert UCSDimension.POSITION_ORIENTATION in dims
    assert UCSDimension.PROXIMITY_REACHABILITY in dims
    assert UCSDimension.TRAJECTORY_MOVEMENT in dims


def test_sample_dataset_against_toy_graph_reaches_full_score():
    """The shipped sample dataset is calibrated so DirectMe scores 100 % on
    all four UCS-Bench dimensions when the graph carries the toy
    place_visit_timeline that the offline engine produces."""
    g = _two_cup_graph()
    # The OfflineMappingEngine writes this automatically for real chunks; for
    # this unit test we inject the equivalent metadata directly so the T&M
    # scorer has something to score against.
    g.metadata["place_visit_timeline"] = [
        {"chunk_id": 0, "timestamp": 0.0, "scene_tag": "living room"},
        {"chunk_id": 1, "timestamp": 10.0, "scene_tag": "kitchen"},
    ]
    src = Path(__file__).resolve().parent.parent / "directme" / "eval" / "sample_questions.jsonl"
    questions = load_ucs_questions(src)
    report = UCSBenchEvaluator(
        graph=g, pose_lookup={("demo", 0.0): _pose_at_kitchen()}
    ).run(questions)

    cq = report.by_dimension[UCSDimension.CATEGORY_QUANTITY]
    po = report.by_dimension[UCSDimension.POSITION_ORIENTATION]
    pr = report.by_dimension[UCSDimension.PROXIMITY_REACHABILITY]
    tm = report.by_dimension[UCSDimension.TRAJECTORY_MOVEMENT]
    assert cq.accuracy == 1.0
    assert po.accuracy == 1.0
    assert pr.accuracy == 1.0
    assert tm.accuracy == 1.0  # v0.3: T&M is now scored


def test_report_serializes_to_clean_json():
    g = _two_cup_graph()
    q = UCSQuestion(
        "demo", 0.0, "几个红杯子？",
        options={"dimension": "category_quantity", "expected_count": 2,
                 "expected_labels": ["cup"]},
    )
    report = UCSBenchEvaluator(graph=g).run([q])
    payload = report.to_dict()
    text = json.dumps(payload, ensure_ascii=False)
    # Must round-trip and keep the predictions list.
    again = json.loads(text)
    assert again["overall_accuracy"] == 1.0
    assert len(again["predictions"]) == 1


def test_evaluator_raises_when_graph_missing_for_uid():
    q = UCSQuestion("missing_video", 0.0, "几个杯子？",
                    options={"dimension": "category_quantity", "expected_count": 1,
                             "expected_labels": ["cup"]})
    ev = UCSBenchEvaluator(graph_lookup={})
    try:
        ev.run([q])
    except KeyError as exc:
        assert "missing_video" in str(exc)
    else:
        raise AssertionError("expected KeyError for missing video_uid")
