"""UCS-Bench evaluation along the four task dimensions.

UCS-Bench (paper §3) groups questions into four task dimensions:

* **Position & Orientation** — where is X relative to me right now?
* **Trajectory & Movement**  — what path did I / X take? *(v0.3: a
  minimum-viable place-visit scorer is provided. Questions whose gold
  field is ``expected_path_labels`` are now scored against the
  ``place_visit_timeline`` recorded by the offline engine. Other
  trajectory question types remain ``"trajectory_movement_partial_only"``
  and are excluded from headline accuracy.)*
* **Proximity & Reachability** — can I reach X from here?
* **Category & Quantity** — how many X have I seen, and what kind?

Each :class:`UCSQuestion` carries gold annotations specific to its dimension:

==================================  ==============================================
Dimension                           Gold fields consumed
==================================  ==============================================
position_orientation                ``expected_relation``, ``expected_target_label``
trajectory_movement                 ``expected_path_labels`` (ordered scene tags)
proximity_reachability              ``expected_reachable``, optional
                                    ``expected_target_label``
category_quantity                   ``expected_count``, ``expected_labels``
==================================  ==============================================

The evaluator is **answer-format-agnostic**: it scores against the structured
:class:`~directme.retrieval.retriever.RetrievedContext` rather than parsing the
generator's free-form text. This avoids brittle string matching and isolates
the scene-graph quality from the QA generator quality.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from directme.datasets.ucsbench import UCSQuestion
from directme.geometry.poses import SE3
from directme.mapping.scene_graph import SceneGraph
from directme.retrieval.egocentric import EGO_RELATION_LABELS
from directme.retrieval.pose_lookup import pose_from_graph_timeline
from directme.retrieval.query_parser import parse_query
from directme.retrieval.retriever import GraphRetriever, RetrievedContext


# ---------------------------------------------------------------------------
# Dimension labels
# ---------------------------------------------------------------------------


class UCSDimension:
    """String constants for the four UCS-Bench task dimensions."""

    POSITION_ORIENTATION = "position_orientation"
    TRAJECTORY_MOVEMENT = "trajectory_movement"
    PROXIMITY_REACHABILITY = "proximity_reachability"
    CATEGORY_QUANTITY = "category_quantity"

    ALL: tuple[str, ...] = (
        POSITION_ORIENTATION,
        TRAJECTORY_MOVEMENT,
        PROXIMITY_REACHABILITY,
        CATEGORY_QUANTITY,
    )

    SKIPPED: frozenset[str] = frozenset({TRAJECTORY_MOVEMENT})


def classify_dimension(question: UCSQuestion) -> str:
    """Return the dimension for a question.

    Uses ``question.options['dimension']`` or a top-level ``dimension``
    annotation if present; otherwise infers from the question text via
    :func:`directme.retrieval.query_parser.parse_query`.
    """
    explicit = None
    if isinstance(question.options, Mapping):
        explicit = question.options.get("dimension")
    if explicit:
        return str(explicit)

    intent = parse_query(question.question)
    if intent.wants_trajectory:
        return UCSDimension.TRAJECTORY_MOVEMENT
    if intent.wants_reachability:
        return UCSDimension.PROXIMITY_REACHABILITY
    if intent.wants_count:
        return UCSDimension.CATEGORY_QUANTITY
    if intent.wants_location:
        return UCSDimension.POSITION_ORIENTATION
    return UCSDimension.POSITION_ORIENTATION  # safe default


# ---------------------------------------------------------------------------
# Per-question prediction record (so callers can dump for analysis)
# ---------------------------------------------------------------------------


@dataclass
class QuestionPrediction:
    question: UCSQuestion
    dimension: str
    score: float | None       # None ⇔ skipped or unscoreable
    correct: bool | None      # None ⇔ skipped
    skipped_reason: str | None = None
    predicted: dict[str, Any] = field(default_factory=dict)
    expected: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_uid": self.question.video_uid,
            "query_timestamp": self.question.query_timestamp,
            "question": self.question.question,
            "dimension": self.dimension,
            "score": self.score,
            "correct": self.correct,
            "skipped_reason": self.skipped_reason,
            "predicted": self.predicted,
            "expected": self.expected,
        }


# ---------------------------------------------------------------------------
# Per-dimension result + overall report
# ---------------------------------------------------------------------------


@dataclass
class DimensionResult:
    dimension: str
    n_total: int = 0
    n_scored: int = 0
    n_correct: int = 0
    skipped_reason: str | None = None

    @property
    def accuracy(self) -> float | None:
        if self.n_scored == 0:
            return None
        return self.n_correct / self.n_scored

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "n_total": self.n_total,
            "n_scored": self.n_scored,
            "n_correct": self.n_correct,
            "accuracy": self.accuracy,
            "skipped_reason": self.skipped_reason,
        }


@dataclass
class EvaluationReport:
    by_dimension: dict[str, DimensionResult]
    predictions: list[QuestionPrediction] = field(default_factory=list)

    @property
    def n_total(self) -> int:
        return sum(r.n_total for r in self.by_dimension.values())

    @property
    def n_scored(self) -> int:
        return sum(r.n_scored for r in self.by_dimension.values())

    @property
    def n_correct(self) -> int:
        return sum(r.n_correct for r in self.by_dimension.values())

    @property
    def overall_accuracy(self) -> float | None:
        return self.n_correct / self.n_scored if self.n_scored > 0 else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "by_dimension": {k: v.to_dict() for k, v in self.by_dimension.items()},
            "n_total": self.n_total,
            "n_scored": self.n_scored,
            "n_correct": self.n_correct,
            "overall_accuracy": self.overall_accuracy,
            "predictions": [p.to_dict() for p in self.predictions],
        }

    def render_summary(self) -> str:
        lines = ["=== DirectMe UCS-Bench Evaluation ==="]
        lines.append(f"Total questions:    {self.n_total}")
        lines.append(f"Scored questions:   {self.n_scored}")
        lines.append(f"Correct answers:    {self.n_correct}")
        if self.overall_accuracy is not None:
            lines.append(f"Overall accuracy:   {self.overall_accuracy * 100:.1f}%")
        else:
            lines.append("Overall accuracy:   N/A (no scoreable questions)")
        lines.append("")
        lines.append("Per-dimension breakdown:")
        for dim in UCSDimension.ALL:
            r = self.by_dimension.get(dim)
            if r is None:
                continue
            if r.n_scored > 0 and r.accuracy is not None:
                # We did score some questions in this dimension. Show the
                # score; if some questions were also partial-skipped, append
                # that reason as a parenthetical.
                line = (
                    f"  {dim:<26} {r.n_total:>3} q | "
                    f"{r.n_correct}/{r.n_scored} correct | "
                    f"{r.accuracy * 100:.1f}%"
                )
                if r.skipped_reason and r.n_scored < r.n_total:
                    line += f" | partial: {r.skipped_reason}"
                lines.append(line)
            elif r.skipped_reason:
                lines.append(
                    f"  {dim:<26} {r.n_total:>3} q | skipped ({r.skipped_reason})"
                )
            else:
                lines.append(f"  {dim:<26} {r.n_total:>3} q | n/a")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Per-dimension scorers
# ---------------------------------------------------------------------------


def _expected(question: UCSQuestion, key: str, default: Any = None) -> Any:
    """Read ``question.options[key]`` (where the dataset stores gold)."""
    if isinstance(question.options, Mapping) and key in question.options:
        return question.options[key]
    return default


def score_position_orientation(
    question: UCSQuestion, ctx: RetrievedContext
) -> tuple[bool | None, dict[str, Any], dict[str, Any], str | None]:
    """Score a P&O question. Returns (correct, predicted, expected, reason).

    Disambiguation priority for the queried target:
      1. ``expected_target_node_id`` — exact node id (strongest).
      2. ``expected_target_place`` — match against ``node.place_id`` /
         ``node.attributes['scene_tag']`` after normalization.
      3. ``expected_target_label`` — first label substring match (legacy).
      4. Otherwise the top-1 retrieval item.
    """
    expected_relation = _expected(question, "expected_relation")
    if expected_relation is None:
        return None, {}, {}, "missing_expected_relation"
    if expected_relation not in EGO_RELATION_LABELS:
        return None, {}, {"expected_relation": expected_relation}, "invalid_expected_relation"

    expected_node_id = _expected(question, "expected_target_node_id")
    expected_place = _expected(question, "expected_target_place")
    expected_label = _expected(question, "expected_target_label")

    chosen = None
    if expected_node_id:
        for item in ctx.items:
            if item.node.node_id == expected_node_id:
                chosen = item
                break
    if chosen is None and expected_place:
        wanted = _normalize_place_name(expected_place)
        for item in ctx.items:
            node_place = _normalize_place_name(item.node.place_id or "")
            node_scene = _normalize_place_name(item.node.attributes.get("scene_tag") or "")
            if wanted and (wanted == node_place or wanted == node_scene):
                chosen = item
                break
    if chosen is None and expected_label:
        for item in ctx.items:
            if expected_label.lower() in item.node.semantic_label.lower():
                chosen = item
                break
    if chosen is None and ctx.items:
        chosen = ctx.items[0]
    if chosen is None:
        return False, {"relation": None, "target": None}, \
               {
                   "expected_relation": expected_relation,
                   "expected_target_label": expected_label,
                   "expected_target_node_id": expected_node_id,
                   "expected_target_place": expected_place,
               }, None

    predicted_relation = chosen.egocentric.get("relation")
    correct = predicted_relation == expected_relation
    return (
        correct,
        {
            "relation": predicted_relation,
            "target_node": chosen.node.node_id,
            "target_label": chosen.node.semantic_label,
            "target_place": chosen.node.place_id
            or chosen.node.attributes.get("scene_tag"),
        },
        {
            "expected_relation": expected_relation,
            "expected_target_label": expected_label,
            "expected_target_node_id": expected_node_id,
            "expected_target_place": expected_place,
        },
        None,
    )


def score_proximity_reachability(
    question: UCSQuestion, ctx: RetrievedContext
) -> tuple[bool | None, dict[str, Any], dict[str, Any], str | None]:
    """Score a P&R question."""
    expected = _expected(question, "expected_reachable")
    if expected is None:
        return None, {}, {}, "missing_expected_reachable"
    expected_bool = bool(expected)

    expected_label = _expected(question, "expected_target_label")
    if expected_label:
        items = [
            it for it in ctx.items
            if expected_label.lower() in it.node.semantic_label.lower()
        ]
    else:
        items = list(ctx.items)

    if not items:
        # No matched object → not reachable.
        predicted_bool = False
    else:
        predicted_bool = any(it.egocentric.get("reachable") for it in items)

    correct = predicted_bool == expected_bool
    return (
        correct,
        {
            "reachable": predicted_bool,
            "matched_node_ids": [it.node.node_id for it in items],
            "reachable_radius_m": ctx.reachable_radius_m,
        },
        {
            "expected_reachable": expected_bool,
            "expected_target_label": expected_label,
        },
        None,
    )


def score_category_quantity(
    question: UCSQuestion, ctx: RetrievedContext
) -> tuple[bool | None, dict[str, Any], dict[str, Any], str | None]:
    """Score a C&Q question. Counts and label-set both must match.

    Counting uses the FULL match set captured before top_k truncation, so a
    high-density scene (e.g. 12 chairs, top_k=8) is still counted correctly.
    """
    expected_count = _expected(question, "expected_count")
    expected_labels = _expected(question, "expected_labels") or []

    if expected_count is None and not expected_labels:
        return None, {}, {}, "missing_expected_count_and_labels"

    # ctx.count returns the full match count when count_all_matches is on
    # (the default), else falls back to the top_k items.
    predicted_count = ctx.count
    if ctx.total_matched_labels:
        predicted_labels = list(ctx.total_matched_labels)
    else:
        predicted_labels = sorted({item.node.semantic_label.lower() for item in ctx.items})
    if ctx.total_matched_node_ids:
        predicted_node_ids = list(ctx.total_matched_node_ids)
    else:
        predicted_node_ids = [it.node.node_id for it in ctx.items]
    expected_labels_norm = sorted({str(s).lower() for s in expected_labels})

    count_ok = (expected_count is None) or (predicted_count == int(expected_count))
    labels_ok = (not expected_labels_norm) or set(predicted_labels) == set(expected_labels_norm)
    correct = count_ok and labels_ok

    return (
        correct,
        {
            "count": predicted_count,
            "labels": predicted_labels,
            "node_ids": predicted_node_ids,
        },
        {
            "expected_count": expected_count,
            "expected_labels": expected_labels_norm,
        },
        None,
    )


def _normalize_place_name(s: str) -> str:
    """Make scene-tag comparisons robust to spaces / underscores / case."""
    return str(s).strip().lower().replace(" ", "_").replace("-", "_")


def score_trajectory_movement(
    question: UCSQuestion, ctx: RetrievedContext
) -> tuple[bool | None, dict[str, Any], dict[str, Any], str | None]:
    """Score a Trajectory & Movement question.

    v0.3 supports a *minimum-viable* T&M signal: the offline engine records
    a ``place_visit_timeline`` keyed by ``scene_tag``. We score against
    ``expected_path_labels`` (a list of place names visited in order).

    We accept the prediction if:

      * the predicted visit sequence (deduplicated, in order) starts with
        the expected sequence, OR
      * the set of visited places equals the expected set (order-independent
        fall-back for graphs where chunk ordering is unreliable).

    Place names are compared after normalization (``"living room"``,
    ``"living-room"`` and ``"living_room"`` are equivalent).

    More elaborate trajectory queries (object movement, route summaries,
    turn-by-turn paths) remain out of scope for v0.3 and return
    ``"trajectory_movement_partial_only"``.
    """
    expected_path = _expected(question, "expected_path_labels")
    if not expected_path:
        return None, {}, {}, "trajectory_movement_partial_only"

    visits = ctx.place_visit_timeline or []
    expected_seq = [_normalize_place_name(s) for s in expected_path]

    if not visits:
        return (
            False,
            {"visited_places": [], "ego_pose_timeline_len": len(ctx.ego_pose_timeline)},
            {"expected_path_labels": expected_seq},
            None,
        )

    # Deduplicate consecutive identical scene_tags while preserving order.
    visited_seq: list[str] = []
    for v in visits:
        tag = _normalize_place_name(v.get("scene_tag") or "")
        if not tag:
            continue
        if not visited_seq or visited_seq[-1] != tag:
            visited_seq.append(tag)

    prefix_match = visited_seq[: len(expected_seq)] == expected_seq
    set_match = set(visited_seq) == set(expected_seq)
    correct = bool(prefix_match or set_match)

    return (
        correct,
        {
            "visited_places": visited_seq,
            "ego_pose_timeline_len": len(ctx.ego_pose_timeline),
        },
        {"expected_path_labels": expected_seq},
        None,
    )


# ---------------------------------------------------------------------------
# The evaluator
# ---------------------------------------------------------------------------


@dataclass
class UCSBenchEvaluator:
    """Run a :class:`GraphRetriever` against a list of UCSQuestion objects.

    Scene graphs may be supplied either as a single ``graph`` (used for every
    question) or via ``graph_lookup`` for multi-video evaluation::

        evaluator = UCSBenchEvaluator(graph=graph, retriever=retriever)
        report = evaluator.run(questions)

        evaluator = UCSBenchEvaluator(graph_lookup={"vid01": g1, "vid02": g2})
        report = evaluator.run(questions)

    A pose lookup may also be supplied keyed by ``(video_uid, query_timestamp)``;
    if absent, evaluation uses the nearest pose from ``graph.metadata['ego_pose_timeline']``
    and falls back to identity only when no timeline is available.
    """

    graph: SceneGraph | None = None
    retriever: GraphRetriever | None = None
    graph_lookup: Mapping[str, SceneGraph] | None = None
    pose_lookup: Mapping[tuple[str, float], SE3] | None = None
    reachable_radius_m: float = 5.0
    lateral_tolerance_ratio: float = 0.20
    top_k: int = 8
    language: str = "zh"

    def _graph_for(self, video_uid: str) -> SceneGraph:
        graph = self.graph
        if graph is None and self.graph_lookup is not None:
            graph = self.graph_lookup.get(video_uid)
        if graph is None:
            raise KeyError(
                f"No graph available for video_uid={video_uid!r}. Pass `graph=` "
                "or include the uid in `graph_lookup`."
            )
        return graph

    def _retriever_for(self, video_uid: str) -> GraphRetriever:
        if self.retriever is not None:
            return self.retriever
        graph = self._graph_for(video_uid)
        return GraphRetriever(
            graph,
            reachable_radius_m=self.reachable_radius_m,
            lateral_tolerance_ratio=self.lateral_tolerance_ratio,
        )

    def _pose_for(self, question: UCSQuestion, graph: SceneGraph) -> SE3:
        # Priority 1: an explicit query-time pose written into the dataset.
        # This makes test cases reproducible without depending on the noisy
        # mapping of `query_timestamp` to the recorded ego-pose timeline.
        if isinstance(question.options, Mapping):
            pose_payload = question.options.get("expected_query_pose")
            if pose_payload is not None:
                try:
                    return SE3.from_list(pose_payload)
                except (TypeError, ValueError):
                    pass  # fall through to the timeline / lookup paths
        # Priority 2: an external pose lookup table (e.g. from a live tracker).
        key = (question.video_uid, float(question.query_timestamp))
        if self.pose_lookup is not None and key in self.pose_lookup:
            return self.pose_lookup[key]
        # Priority 3: the offline engine's recorded ego-pose timeline.
        return pose_from_graph_timeline(graph, timestamp=float(question.query_timestamp))

    def evaluate_one(self, question: UCSQuestion) -> QuestionPrediction:
        dim = classify_dimension(question)

        graph = self._graph_for(question.video_uid)
        retriever = self._retriever_for(question.video_uid)
        ctx = retriever.retrieve(
            question.question,
            current_pose=self._pose_for(question, graph),
            top_k=self.top_k,
            language=self.language,
        )

        if dim == UCSDimension.POSITION_ORIENTATION:
            correct, predicted, expected, reason = score_position_orientation(question, ctx)
        elif dim == UCSDimension.PROXIMITY_REACHABILITY:
            correct, predicted, expected, reason = score_proximity_reachability(question, ctx)
        elif dim == UCSDimension.CATEGORY_QUANTITY:
            correct, predicted, expected, reason = score_category_quantity(question, ctx)
        elif dim == UCSDimension.TRAJECTORY_MOVEMENT:
            correct, predicted, expected, reason = score_trajectory_movement(question, ctx)
        else:
            correct, predicted, expected, reason = None, {}, {}, f"unknown_dimension:{dim}"

        return QuestionPrediction(
            question=question,
            dimension=dim,
            score=None if correct is None else float(correct),
            correct=correct,
            skipped_reason=reason,
            predicted=predicted,
            expected=expected,
        )

    def run(self, questions: Sequence[UCSQuestion]) -> EvaluationReport:
        per_dim = {dim: DimensionResult(dimension=dim) for dim in UCSDimension.ALL}
        predictions: list[QuestionPrediction] = []

        for q in questions:
            pred = self.evaluate_one(q)
            predictions.append(pred)
            r = per_dim.setdefault(pred.dimension, DimensionResult(dimension=pred.dimension))
            r.n_total += 1
            if pred.correct is None:
                # Record the first skip reason seen for this dimension, so
                # report.render_summary can surface it instead of "n/a".
                if r.skipped_reason is None and pred.skipped_reason:
                    r.skipped_reason = pred.skipped_reason
                continue
            r.n_scored += 1
            if pred.correct:
                r.n_correct += 1

        return EvaluationReport(by_dimension=per_dim, predictions=predictions)


# ---------------------------------------------------------------------------
# Convenience: load + evaluate
# ---------------------------------------------------------------------------


def evaluate_jsonl(
    dataset_path: str | Path,
    graph: SceneGraph,
    *,
    reachable_radius_m: float = 5.0,
    lateral_tolerance_ratio: float = 0.20,
    top_k: int = 8,
    language: str = "zh",
) -> EvaluationReport:
    """Convenience wrapper for the CLI: load questions and run all dimensions."""
    from directme.datasets.ucsbench import load_ucs_questions

    questions = load_ucs_questions(dataset_path)
    evaluator = UCSBenchEvaluator(
        graph=graph,
        reachable_radius_m=reachable_radius_m,
        lateral_tolerance_ratio=lateral_tolerance_ratio,
        top_k=top_k,
        language=language,
    )
    return evaluator.run(questions)


def write_report(report: EvaluationReport, out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
                   encoding="utf-8")
    return out
