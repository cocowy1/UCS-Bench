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

    total_matched_count: int = 0
    total_matched_node_ids: list[str] = field(default_factory=list)
    total_matched_labels: list[str] = field(default_factory=list)
    ego_pose_timeline: list[dict[str, Any]] = field(default_factory=list)
    place_visit_timeline: list[dict[str, Any]] = field(default_factory=list)

    @property
    def count(self) -> int:
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
        allow_unlabeled_fallback: bool = False,   # 改动 A：默认关闭全图 fallback
    ):
        self.graph = graph
        self.reachable_radius_m = reachable_radius_m
        self.lateral_tolerance_ratio = lateral_tolerance_ratio
        self.count_all_matches = count_all_matches
        self.allow_unlabeled_fallback = allow_unlabeled_fallback

    def retrieve(
        self,
        question: str,
        current_pose: SE3,
        top_k: int = 8,
        language: str | None = None,
        as_of_timestamp: float | None = None,     # 改动 A：在线 QA 时间戳过滤
    ) -> RetrievedContext:
        """检索场景图。

        Parameters
        ----------
        as_of_timestamp : float | None
            若提供，则仅保留首次观测时间 <= as_of_timestamp 的节点，
            确保不使用"未来"观测回答当前时刻的问题（UCS-Bench 在线 QA 语义）。
            None 时退化为离线检索（使用全部节点）。
        allow_unlabeled_fallback : bool
            当 intent.labels 和 intent.colors 均为空时，是否回退到全体候选节点。
            默认 False：空意图时返回空匹配，避免把全图 128 个节点当成"匹配"。
        """
        intent = parse_query(question, language=language)

        # ── 改动 A：按时间戳过滤候选节点池 ──────────────────────────────────
        if as_of_timestamp is not None:
            candidate_nodes = [
                n for n in self.graph.nodes.values()
                if n.observations and float(n.observations[0].timestamp) <= float(as_of_timestamp)
            ]
        else:
            candidate_nodes = list(self.graph.nodes.values())
        # ─────────────────────────────────────────────────────────────────────

        scored: list[tuple[float, EntityNode]] = []
        for node in candidate_nodes:
            score = self._score_node(node, intent)
            if score > 0:
                scored.append((score, node))

        # ── 改动 A：关闭全图 fallback（原代码把全部节点都返回，导致 COUNT 题错误）──
        if not scored and not intent.labels and not intent.colors:
            if self.allow_unlabeled_fallback:
                scored = [(0.1, node) for node in candidate_nodes]
            # else: 保持 scored=[]，ctx.count=0，QA 端可识别"未检索到目标"
        # ─────────────────────────────────────────────────────────────────────

        scored_sorted = sorted(scored, key=lambda x: x[0], reverse=True)

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
