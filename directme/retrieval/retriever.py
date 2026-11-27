from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from directme.geometry.poses import SE3
from directme.mapping.scene_graph import EntityNode, SceneGraph
from directme.perception.color_attributes import normalize_color_name
from directme.retrieval.egocentric import (
    DEFAULT_LATERAL_TOLERANCE_RATIO,
    DEFAULT_REACHABLE_RADIUS_M,
    render_egocentric,
)
from directme.retrieval.query_parser import QueryIntent, parse_query
from directme.retrieval.reliability import keyframes_as_of, node_reliability, position_as_of


EGO_NODE_ID: str = "ego"

# Soft-scoring weights (v0.6). Instead of zeroing a node when its label/color
# does not match the parsed intent, we down-weight it by these factors and let
# it stay in the ranked candidate list. This is what fixes the "22.1% of
# questions retrieve zero keyframes" failure mode: an alias-table vocabulary
# mismatch (query "sofa" vs detector label "Couch") now yields weak-but-usable
# candidates rather than nothing at all.
LABEL_MISMATCH_WEIGHT: float = 0.15
COLOR_MISMATCH_WEIGHT: float = 0.40


@dataclass
class RetrievedItem:
    node: EntityNode
    score: float
    egocentric: dict[str, Any]
    # "matched" → label/color gates agree with the parsed question intent.
    # "fallback" → no node matched the intent; this is a weak generic
    # candidate returned so the MLLM still has *some* graph evidence. Fallback
    # items are excluded from ``total_matched_count`` (COUNT semantics).
    match_type: str = "matched"
    reliability: dict[str, Any] = field(default_factory=dict)
    # Causal keyframe candidates for this item (see RetrievedContext.keyframes).
    keyframes: list[str] = field(default_factory=list)


@dataclass
class RetrievedContext:
    """The minimal subgraph + supporting keyframes returned to online QA."""

    question: str
    intent: QueryIntent
    current_pose: SE3
    items: list[RetrievedItem] = field(default_factory=list)
    ego_edges: list[dict[str, Any]] = field(default_factory=list)
    reachable_radius_m: float = DEFAULT_REACHABLE_RADIUS_M

    total_matched_count: int = 0
    total_matched_node_ids: list[str] = field(default_factory=list)
    total_matched_labels: list[str] = field(default_factory=list)
    ego_pose_timeline: list[dict[str, Any]] = field(default_factory=list)
    place_visit_timeline: list[dict[str, Any]] = field(default_factory=list)

    @property
    def count(self) -> int:
        if self.total_matched_count:
            return self.total_matched_count
        return sum(
            1 for item in self.items if item.match_type == "matched"
        )

    @property
    def reachable_count(self) -> int:
        return sum(1 for item in self.items if item.egocentric.get("reachable"))

    @property
    def keyframes(self) -> list[str]:
        paths: list[str] = []
        for item in self.items:
            item_paths = item.keyframes or item.node.keyframes
            for path in item_paths:
                if path and path not in paths:
                    paths.append(path)
        return paths


def _contains_label_token(text: str, label: str) -> bool:
    label_norm = label.lower().strip()
    if not label_norm:
        return False
    if re.search(r"[一-鿿]", label_norm):
        return label_norm in text
    pieces = [re.escape(p) for p in re.split(r"\s+", label_norm) if p]
    if not pieces:
        return False
    pattern = r"(?<![a-z0-9])" + r"\s+".join(pieces) + r"(?![a-z0-9])"
    return re.search(pattern, text) is not None


def _normalize_room_token(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def _truncate_timeline(timeline: list[dict[str, Any]], as_of_timestamp: float) -> list[dict[str, Any]]:
    """Keep only records at/before the question time (causal semantics)."""
    out = []
    for rec in timeline:
        if not isinstance(rec, dict):
            continue
        try:
            ts = float(rec.get("timestamp", 0.0))
        except (TypeError, ValueError):
            continue
        if ts <= as_of_timestamp:
            out.append(rec)
    return out


class GraphRetriever:
    def __init__(
        self,
        graph: SceneGraph,
        reachable_radius_m: float = DEFAULT_REACHABLE_RADIUS_M,
        lateral_tolerance_ratio: float = DEFAULT_LATERAL_TOLERANCE_RATIO,
        count_all_matches: bool = True,
        allow_unlabeled_fallback: bool = False,   # Change A: whole-graph fallback disabled by default
        strict_causal: bool = True,
        soft_scoring: bool = True,
    ):
        self.graph = graph
        self.reachable_radius_m = reachable_radius_m
        self.lateral_tolerance_ratio = lateral_tolerance_ratio
        self.count_all_matches = count_all_matches
        self.allow_unlabeled_fallback = allow_unlabeled_fallback
        # strict_causal=True (v0.6 default): the question timestamp filters
        # *everything* — node existence, the position used for egocentric
        # rendering, keyframe candidates, and the ego/place timelines. The
        # legacy mode only filtered node existence and rendered the node's
        # final (post-t_q) anchor, leaking future observations.
        self.strict_causal = strict_causal
        # soft_scoring=True (v0.6 default): label/color mismatches down-weight
        # instead of zeroing candidates (see LABEL_MISMATCH_WEIGHT).
        self.soft_scoring = soft_scoring

    def retrieve(
        self,
        question: str,
        current_pose: SE3,
        top_k: int = 8,
        language: str | None = None,
        as_of_timestamp: float | None = None,     # Change A: timestamp filtering for online QA
    ) -> RetrievedContext:
        """Retrieve from the scene graph.

        Parameters
        ----------
        as_of_timestamp : float | None
            If provided, only information observed at/before this time is
            used. Under ``strict_causal`` this covers node existence
            (``first_seen``), positions, keyframes, and timelines; otherwise
            (legacy) it only filters node existence. ``None`` degrades to
            offline retrieval (all nodes, full history).
        allow_unlabeled_fallback : bool
            Whether to fall back to all candidate nodes when both
            ``intent.labels`` and ``intent.colors`` are empty. Defaults to
            False: an empty intent yields an empty match, instead of treating
            all 128 graph nodes as "matched".
        """
        intent = parse_query(question, language=language)

        # ── Change A: filter the candidate node pool by timestamp ──
        if as_of_timestamp is not None:
            t_q = float(as_of_timestamp)
            if self.strict_causal:
                candidate_nodes = [
                    n for n in self.graph.nodes.values()
                    if n.first_seen is not None and float(n.first_seen) <= t_q
                ]
            else:
                candidate_nodes = [
                    n for n in self.graph.nodes.values()
                    if n.observations and float(n.observations[0].timestamp) <= t_q
                ]
        else:
            t_q = None
            candidate_nodes = list(self.graph.nodes.values())
        # ─────────────────────────────────────────────────────────────────────

        scored: list[tuple[float, EntityNode, bool]] = []  # (score, node, is_match)
        for node in candidate_nodes:
            score, is_match = self._score_node(node, intent, as_of_timestamp=t_q)
            if score > 0:
                scored.append((score, node, is_match))

        # ── Change A: disable whole-graph fallback (the old code returned every node, breaking COUNT) ──
        if not scored and not intent.labels and not intent.colors and self.allow_unlabeled_fallback:
            scored = [(0.1, node, False) for node in candidate_nodes]
        # ─────────────────────────────────────────────────────────────────────

        scored_sorted = sorted(scored, key=lambda x: x[0], reverse=True)
        matched = [(s, n) for s, n, m in scored_sorted if m]
        weak = [(s, n) for s, n, m in scored_sorted if not m]

        if self.count_all_matches:
            total_matched_count = len(matched)
            total_matched_node_ids = [node.node_id for _s, node in matched]
            total_matched_labels = sorted({node.semantic_label.lower() for _s, node in matched})
        else:
            total_matched_count = 0
            total_matched_node_ids = []
            total_matched_labels = []

        # v0.6: ranked selection. Prefer intent-matched nodes; if the intent
        # matched nothing (vocabulary mismatch / empty graph region), fall
        # back to the weak-but-nonzero candidates so the MLLM still receives
        # graph evidence *and* keyframes instead of answering blind. Fallback
        # items never contribute to total_matched_count.
        selected = matched[:top_k] if matched else weak[:top_k]

        items: list[RetrievedItem] = []
        ego_edges: list[dict[str, Any]] = []
        for score, node in selected:
            reliability = node_reliability(node, as_of_timestamp=t_q)
            match_type = "matched" if matched else "fallback"

            if t_q is not None and self.strict_causal:
                p_override = position_as_of(node, t_q)
                item_keyframes = keyframes_as_of(node, t_q)
            else:
                p_override = None
                item_keyframes = list(node.keyframes)

            ego = render_egocentric(
                node,
                current_pose,
                language=intent.language,
                reachable_radius_m=self.reachable_radius_m,
                lateral_tolerance_ratio=self.lateral_tolerance_ratio,
                p_world_override=p_override,
            )
            items.append(
                RetrievedItem(
                    node=node,
                    score=score,
                    egocentric=ego,
                    match_type=match_type,
                    reliability=reliability,
                    keyframes=item_keyframes,
                )
            )
            ego_edges.append(
                {
                    "source": EGO_NODE_ID,
                    "target": node.node_id,
                    "relation": ego["relation"],
                    "distance_m": ego["distance_m"],
                    "reachable": ego["reachable"],
                    "reference_frame": "Current_Camera",
                }
            )

        ego_pose_timeline = list(self.graph.metadata.get("ego_pose_timeline", []))
        place_visit_timeline = list(self.graph.metadata.get("place_visit_timeline", []))
        if t_q is not None and self.strict_causal:
            ego_pose_timeline = _truncate_timeline(ego_pose_timeline, t_q)
            place_visit_timeline = _truncate_timeline(place_visit_timeline, t_q)

        return RetrievedContext(
            question=question,
            intent=intent,
            current_pose=current_pose,
            items=items,
            ego_edges=ego_edges,
            reachable_radius_m=self.reachable_radius_m,
            total_matched_count=total_matched_count,
            total_matched_node_ids=total_matched_node_ids,
            total_matched_labels=total_matched_labels,
            ego_pose_timeline=ego_pose_timeline,
            place_visit_timeline=place_visit_timeline,
        )

    def _score_node(
        self, node: EntityNode, intent: QueryIntent, as_of_timestamp: float | None = None
    ) -> tuple[float, bool]:
        """Score one node against the parsed intent.

        Returns ``(score, is_match)`` where ``is_match`` reports whether the
        label/color gates *positively* agree with the intent (used for COUNT
        semantics and match_type tagging). Under soft scoring a mismatching
        node keeps a small non-zero score (down-weighted, never dropped).
        """
        label_text = f"{node.semantic_label} {' '.join(node.aliases)}".lower()
        score = 0.0

        label_ok = True
        if intent.labels:
            if any(_contains_label_token(label_text, label) for label in intent.labels):
                score += 2.0
            else:
                label_ok = False
                if self.soft_scoring:
                    score += 0.1  # weak base; multiplied down below
                else:
                    return 0.0, False

        color_ok = True
        if intent.colors:
            node_color = (
                normalize_color_name(str(node.attributes.get("color")))
                if node.attributes.get("color")
                else None
            )
            if node_color in intent.colors:
                score += 1.5
            else:
                color_ok = False
                if self.soft_scoring:
                    pass  # no bonus; multiplied down below
                else:
                    return 0.0, False

        if not intent.labels and not intent.colors:
            score += 0.1

        if intent.rooms:
            node_scene_tag = _normalize_room_token(node.attributes.get("scene_tag"))
            node_place_id = _normalize_room_token(node.place_id)
            wanted = {_normalize_room_token(r) for r in intent.rooms}
            wanted.discard("")
            if wanted and (
                (node_scene_tag and node_scene_tag in wanted)
                or (node_place_id and node_place_id in wanted)
            ):
                score += 1.0

        if as_of_timestamp is not None:
            n_obs = sum(
                1 for o in node.observations if float(o.timestamp) <= float(as_of_timestamp)
            )
        else:
            n_obs = len(node.observations)
        score += min(n_obs, 5) * 0.05

        # Soft down-weighting (v0.6): mismatches stay rankable but sink below
        # every positively-matched node; reliability further reorders within
        # each tier (down-weight only — never a hard filter).
        if self.soft_scoring:
            if not label_ok:
                score *= LABEL_MISMATCH_WEIGHT
            if not color_ok:
                score *= COLOR_MISMATCH_WEIGHT
            rel = node_reliability(node, as_of_timestamp=as_of_timestamp)["reliability"]
            score *= 0.5 + 0.5 * float(rel)

        # A node only counts as a *positive* match when the intent actually
        # named something (label, color, or room). With a completely
        # unparseable intent every node is a generic candidate — returned as
        # weak evidence, but never counted (COUNT would otherwise sum the
        # whole graph).
        is_match = (
            label_ok
            and color_ok
            and bool(intent.labels or intent.colors or intent.rooms)
        )
        return score, is_match

    @staticmethod
    def render_summary(context: RetrievedContext) -> str:
        lines = [
            "[Graph Summary]",
            f"Question: {context.question}",
            f"Matched physical count: {context.count} (reachable: {context.reachable_count})",
            f"Reachable radius: {context.reachable_radius_m:.1f} m",
        ]
        for item in context.items:
            node = item.node
            ego = item.egocentric
            color = node.attributes.get("color", "unknown")
            rel = item.reliability.get("reliability") if item.reliability else None
            rel_txt = f"{rel:.2f}" if isinstance(rel, float) else "n/a"
            match_txt = "" if item.match_type == "matched" else " [weak match]"
            lines.append(
                "obj={node_id} | label={label} | color={color} | rel={rel} | dist={dist:.2f}m | "
                "reachable={reach} | reliability={relia}{weak} | where={where}".format(
                    node_id=node.node_id,
                    label=node.semantic_label,
                    color=color,
                    rel=ego["relation"],
                    dist=float(ego["distance_m"]),
                    reach=ego["reachable"],
                    relia=rel_txt,
                    weak=match_txt,
                    where=ego["natural_language"],
                )
            )
        if context.ego_edges:
            lines.append("[Ego edges]")
            for e in context.ego_edges:
                lines.append(
                    "  ({src}) --[{rel}, {d:.2f}m, reachable={r}]--> ({tgt})".format(
                        src=e["source"], rel=e["relation"], d=e["distance_m"],
                        r=e["reachable"], tgt=e["target"],
                    )
                )
        return "\n".join(lines)
