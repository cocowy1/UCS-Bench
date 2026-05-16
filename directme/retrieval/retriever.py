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


# Conventional id used for the (virtual) ego node referenced by ego_edges.
EGO_NODE_ID: str = "ego"


@dataclass
class RetrievedItem:
    node: EntityNode
    score: float
    egocentric: dict[str, Any]


@dataclass
class RetrievedContext:
    """The minimal subgraph + supporting keyframes returned to online QA."""

    question: str
    intent: QueryIntent
    current_pose: SE3
    items: list[RetrievedItem] = field(default_factory=list)
    ego_edges: list[dict[str, Any]] = field(default_factory=list)
    reachable_radius_m: float = DEFAULT_REACHABLE_RADIUS_M

    # Full match set, recorded BEFORE top_k truncation. ``items`` only
    # contains the displayed top-k for keyframe / prompt assembly; counting
    # questions ("how many cups have I seen") must use these full-match
    # fields so a high-density scene with N >> top_k objects is not silently
    # under-counted.
    total_matched_count: int = 0
    total_matched_node_ids: list[str] = field(default_factory=list)
    total_matched_labels: list[str] = field(default_factory=list)
    # Optional ego-pose / place-visit history surfaced for trajectory queries.
    # Populated from ``graph.metadata`` if present; absent on bare graphs.
    ego_pose_timeline: list[dict[str, Any]] = field(default_factory=list)
    place_visit_timeline: list[dict[str, Any]] = field(default_factory=list)

    @property
    def count(self) -> int:
        """Total physical count of matched objects (NOT truncated by top_k).

        This is what UCS-Bench Category & Quantity questions consume. If the
        retriever was constructed with ``count_all_matches=True`` (default),
        this returns the count over all matched nodes, summing each node's
        ``count_contribution``. Otherwise it falls back to the truncated
        ``items`` for backward compatibility.
        """
        if self.total_matched_count:
            return self.total_matched_count
        return sum(int(item.node.attributes.get("count_contribution", 1)) for item in self.items)

    @property
    def reachable_count(self) -> int:
        return sum(1 for item in self.items if item.egocentric.get("reachable"))

    @property
    def keyframes(self) -> list[str]:
        paths: list[str] = []
        for item in self.items:
            for path in item.node.keyframes:
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
    """Normalize a room-like token so ``"living room"``, ``"living-room"`` and
    ``"living_room"`` compare equal. Empty / None inputs return ``""``."""
    if value is None:
        return ""
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


class GraphRetriever:
    def __init__(
        self,
        graph: SceneGraph,
        reachable_radius_m: float = DEFAULT_REACHABLE_RADIUS_M,
        lateral_tolerance_ratio: float = DEFAULT_LATERAL_TOLERANCE_RATIO,
        count_all_matches: bool = True,
    ):
        self.graph = graph
        self.reachable_radius_m = reachable_radius_m
        self.lateral_tolerance_ratio = lateral_tolerance_ratio
        self.count_all_matches = count_all_matches

    def retrieve(
        self,
        question: str,
        current_pose: SE3,
        top_k: int = 8,
        language: str | None = None,
    ) -> RetrievedContext:
        intent = parse_query(question, language=language)
        scored: list[tuple[float, EntityNode]] = []
        for node in self.graph.nodes.values():
            score = self._score_node(node, intent)
            if score > 0:
                scored.append((score, node))

        # If the query has no recognizable object/color terms, fall back to all nodes.
        if not scored and not intent.labels and not intent.colors:
            scored = [(0.1, node) for node in self.graph.nodes.values()]

        scored_sorted = sorted(scored, key=lambda x: x[0], reverse=True)

        # Snapshot ALL matched nodes BEFORE truncation so counting questions
        # are correct even when top_k is small. ``items`` below is only the
        # displayed top-k; ``total_matched_*`` fields below are the universe.
        if self.count_all_matches:
            total_matched_count = sum(
                int(node.attributes.get("count_contribution", 1)) for _s, node in scored_sorted
            )
            total_matched_node_ids = [node.node_id for _s, node in scored_sorted]
            total_matched_labels = sorted({node.semantic_label.lower() for _s, node in scored_sorted})
        else:
            total_matched_count = 0
            total_matched_node_ids = []
            total_matched_labels = []

        items: list[RetrievedItem] = []
        ego_edges: list[dict[str, Any]] = []
        for score, node in scored_sorted[:top_k]:
            ego = render_egocentric(
                node,
                current_pose,
                language=intent.language,
                reachable_radius_m=self.reachable_radius_m,
                lateral_tolerance_ratio=self.lateral_tolerance_ratio,
            )
            items.append(RetrievedItem(node=node, score=score, egocentric=ego))
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

        # Pull trajectory / visit memory from graph metadata if present. The
        # offline engine writes these incrementally; bare graphs simply have
        # empty lists, which is fine for trajectory queries to fall back on.
        ego_pose_timeline = list(self.graph.metadata.get("ego_pose_timeline", []))
        place_visit_timeline = list(self.graph.metadata.get("place_visit_timeline", []))

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

    @staticmethod
    def _score_node(node: EntityNode, intent: QueryIntent) -> float:
        label_text = f"{node.semantic_label} {' '.join(node.aliases)}".lower()
        score = 0.0

        if intent.labels:
            if any(_contains_label_token(label_text, label) for label in intent.labels):
                score += 2.0
            else:
                return 0.0

        if intent.colors:
            node_color = (
                normalize_color_name(str(node.attributes.get("color")))
                if node.attributes.get("color")
                else None
            )
            if node_color in intent.colors:
                score += 1.5
            else:
                return 0.0

        if not intent.labels and not intent.colors:
            score += 0.1

        # Room / place soft-match. Questions like "客厅那个红杯子" carry
        # ``intent.rooms == ["living_room"]``; we boost (but do not require)
        # nodes whose ``scene_tag`` or assigned ``place_id`` aligns. Soft
        # matching keeps recall high when the scene classifier is noisy.
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

        # Prefer nodes with more observations.
        score += min(len(node.observations), 5) * 0.05
        return score

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
            lines.append(
                "obj={node_id} | label={label} | color={color} | rel={rel} | "
                "dist={dist:.2f}m | reachable={reach} | where={where}".format(
                    node_id=node.node_id,
                    label=node.semantic_label,
                    color=color,
                    rel=ego["relation"],
                    dist=float(ego["distance_m"]),
                    reach=ego["reachable"],
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
