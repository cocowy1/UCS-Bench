"""Pose-anchored spatial scene graph with multi-cue fusion.

Upgrades over v1:
  * HSV hue histogram is now part of the matching cost (cosine similarity).
  * Per-label adaptive merge thresholds (small objects strict, big objects loose).
  * Pose-confidence-weighted EMA: low-confidence observations contribute less.
  * Optional motion detection switches between dynamic-overwrite and static-EMA.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from directme.perception.color import (
    histogram_cosine_similarity,
    normalize_color_name,
)


def _as_point(p: Any) -> np.ndarray:
    arr = np.asarray(p, dtype=float).reshape(3)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"invalid 3D point: {p}")
    return arr


def _norm_label(label: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]+", " ", label.lower())).strip()


def _token_set(label: str) -> set[str]:
    return set(_norm_label(label).split())


def _semantic_compatible(a: str, b: str) -> bool:
    na, nb = _norm_label(a), _norm_label(b)
    if not na or not nb:
        return True
    if na == nb or na in nb or nb in na:
        return True
    return bool(_token_set(na) & _token_set(nb))


def _embedding_cosine_similarity(a: Any, b: Any) -> float:
    """Cosine similarity between two embedding vectors, clamped to [0, 1].

    Returns 0.0 if either side is empty / non-finite. Negative similarities are
    clamped because we use this as a soft positive gate.
    """
    av = np.asarray(a, dtype=np.float32).reshape(-1)
    bv = np.asarray(b, dtype=np.float32).reshape(-1)
    if av.size == 0 or bv.size != av.size:
        return 0.0
    if not (np.all(np.isfinite(av)) and np.all(np.isfinite(bv))):
        return 0.0
    na = float(np.linalg.norm(av))
    nb = float(np.linalg.norm(bv))
    if na <= 0 or nb <= 0:
        return 0.0
    return float(np.clip(np.dot(av, bv) / (na * nb), 0.0, 1.0))


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.int32, np.int64)):
        return int(value)
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# Per-label merge threshold defaults (meters).
# Small/movable objects use a tight gate, large furniture uses a looser one.
# ---------------------------------------------------------------------------
DEFAULT_LABEL_MERGE_THRESHOLDS_M: dict[str, float] = {
    "cup": 0.35, "mug": 0.35, "bottle": 0.40, "phone": 0.30, "spoon": 0.25,
    "fork": 0.25, "knife": 0.25, "remote": 0.30, "key": 0.20, "pen": 0.20,
    "book": 0.40, "laptop": 0.50, "chair": 0.80, "stool": 0.60,
    "table": 1.20, "desk": 1.20, "sofa": 1.50, "couch": 1.50, "bed": 1.80,
    "fridge": 1.00, "refrigerator": 1.00, "oven": 0.80, "microwave": 0.60,
    "sink": 0.60, "toilet": 0.60, "door": 0.80, "window": 0.80,
    "tv": 0.80, "monitor": 0.50, "person": 0.80,
}


def label_merge_threshold(label: str, default: float, overrides: dict[str, float] | None = None) -> float:
    """Return the merge threshold for a given label, falling back to ``default``."""
    nl = _norm_label(label)
    table = {**DEFAULT_LABEL_MERGE_THRESHOLDS_M, **(overrides or {})}
    if nl in table:
        return table[nl]
    for key, val in table.items():
        if key in nl or nl in key:
            return val
    return default


@dataclass
class ObservationRecord:
    timestamp: float
    frame_index: int
    track_id: str | None
    p_world: list[float]
    p_cam: list[float] | None = None
    confidence: float = 1.0
    pose_confidence: float = 1.0
    keyframe_path: str | None = None
    bbox_xyxy: tuple[float, float, float, float] | None = None
    bbox_area: float = 0.0  # used for keyframe selection

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self.__dict__)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ObservationRecord":
        kwargs = dict(data)
        kwargs.setdefault("pose_confidence", 1.0)
        kwargs.setdefault("bbox_area", 0.0)
        return cls(**kwargs)


@dataclass
class EntityNode:
    node_id: str
    semantic_label: str
    attributes: dict[str, Any] = field(default_factory=dict)
    reference_frame: str = "Frame_0_World_Origin"
    observations: list[ObservationRecord] = field(default_factory=list)
    keyframes: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    track_ids: list[str] = field(default_factory=list)
    place_id: str | None = None
    created_at: float | None = None
    updated_at: float | None = None
    # NOTE: ``spatial_egocentric_dynamic`` is *intentionally not* a field on
    # the node. Egocentric state is computed per-query by
    # :func:`directme.retrieval.egocentric.render_egocentric` and lives on the
    # per-query :class:`RetrievedItem`. Keeping it off the node guarantees
    # that saving the graph after a query never persists a stale camera-frame
    # snapshot. See architecture invariant 4 in docs/architecture.md.

    @property
    def p_world(self) -> np.ndarray:
        return _as_point(self.spatial_absolute["p_world"])

    @property
    def spatial_absolute(self) -> dict[str, Any]:
        if "_spatial_absolute" not in self.attributes:
            latest = self.observations[-1].p_world if self.observations else [0.0, 0.0, 0.0]
            self.attributes["_spatial_absolute"] = {
                "reference_frame": self.reference_frame,
                "p_world": latest,
                "observation_count": len(self.observations),
                "last_seen_timestamp": self.updated_at,
            }
        return self.attributes["_spatial_absolute"]

    def update_absolute_point(self, p_world: np.ndarray, alpha: float) -> None:
        current = _as_point(self.spatial_absolute["p_world"])
        updated = alpha * p_world + (1.0 - alpha) * current
        self.spatial_absolute["p_world"] = updated.tolist()

    def fuse_color_histogram(self, hist: list[float], alpha: float) -> None:
        """Running EMA of the HSV histogram for robust color identity."""
        if not hist:
            return
        existing = self.attributes.get("color_hsv_histogram")
        if existing is None or len(existing) != len(hist):
            self.attributes["color_hsv_histogram"] = list(hist)
            return
        e = np.asarray(existing, dtype=np.float32)
        n = np.asarray(hist, dtype=np.float32)
        merged = alpha * n + (1.0 - alpha) * e
        s = float(merged.sum())
        if s > 0:
            merged /= s
        self.attributes["color_hsv_histogram"] = merged.tolist()

    def fuse_semantic_embedding(self, embedding: list[float] | np.ndarray, alpha: float) -> None:
        """Running EMA of an L2-normalized semantic embedding (e.g. CLIP).

        Inspired by ConceptGraphs' multi-view CLIP fusion. The embedding is
        renormalized after each update so cosine similarity stays meaningful.
        """
        emb = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if emb.size == 0 or not np.all(np.isfinite(emb)):
            return
        existing = self.attributes.get("semantic_embedding")
        if existing is None or len(existing) != emb.size:
            n = np.linalg.norm(emb)
            self.attributes["semantic_embedding"] = (emb / n if n > 0 else emb).tolist()
            return
        e = np.asarray(existing, dtype=np.float32)
        merged = alpha * emb + (1.0 - alpha) * e
        n = np.linalg.norm(merged)
        if n > 0:
            merged /= n
        self.attributes["semantic_embedding"] = merged.tolist()

    def _get_keyframe_selector(self, budget: int) -> "KeyframeSelector":
        """Return the lazily-instantiated diversity selector.

        The selector is in-memory only (not persisted): on graph load we
        seed it with the already-saved keyframe paths via ``adopt_existing``
        so newly arriving observations can compete against — but not
        immediately clobber — the existing selection.
        """
        from directme.mapping.keyframe_selector import KeyframeSelector

        sel = getattr(self, "_keyframe_selector", None)
        if sel is None or sel.budget != budget:
            sel = KeyframeSelector(budget=budget)
            for kp in self.keyframes:
                sel.adopt_existing(kp)
            self._keyframe_selector = sel  # type: ignore[attr-defined]
        return sel

    def add_observation(
        self,
        obs: ObservationRecord,
        attributes: dict[str, Any] | None = None,
        max_observations: int = 128,
        keyframes_per_node: int = 4,
        dynamic_alpha: float = 0.70,
        static_alpha: float = 0.20,
        motion_overwrite_threshold_m: float = 0.50,
    ) -> None:
        attributes = attributes or {}
        for k, v in attributes.items():
            if k == "color":
                v_norm = normalize_color_name(str(v))
                if v_norm:
                    if not self.attributes.get("color"):
                        self.attributes["color"] = v_norm
            elif k == "color_hsv_histogram":
                continue  # handled via fuse_color_histogram below
            elif k == "semantic_embedding":
                continue  # handled via fuse_semantic_embedding below
            else:
                if k not in self.attributes or self.attributes[k] in (None, "", []):
                    self.attributes[k] = v

        if "color_hsv_histogram" in attributes:
            hist_alpha = dynamic_alpha if self.attributes.get("is_movable", False) else static_alpha
            self.fuse_color_histogram(attributes["color_hsv_histogram"], alpha=hist_alpha)

        if "semantic_embedding" in attributes:
            emb_alpha = dynamic_alpha if self.attributes.get("is_movable", False) else static_alpha
            self.fuse_semantic_embedding(attributes["semantic_embedding"], alpha=emb_alpha)

        if obs.track_id and obs.track_id not in self.track_ids:
            self.track_ids.append(obs.track_id)

        # Diversity-aware keyframe selection (v0.4): if the observation
        # carries a semantic embedding, the selector keeps the K candidates
        # that are mutually most dissimilar in cosine space (greedy
        # farthest-point sampling). Falls back to v0.3 bbox-area greedy
        # when no embedding is available, so the toy backend keeps working.
        if obs.keyframe_path:
            selector = self._get_keyframe_selector(keyframes_per_node)
            obs_embedding = attributes.get("semantic_embedding") if attributes else None
            selector.add(obs.keyframe_path, obs.bbox_area, obs_embedding)
            self.keyframes = selector.selected

        self.observations.append(obs)
        if len(self.observations) > max_observations:
            self.observations = self.observations[-max_observations:]

        # Effective alpha: scaled by pose confidence.
        is_movable = bool(self.attributes.get("is_movable", False))
        base_alpha = dynamic_alpha if is_movable else static_alpha
        eff_alpha = base_alpha * float(obs.pose_confidence)

        # Motion-aware overwrite for movable objects: if observation jumps far
        # from the current world anchor, overwrite rather than EMA-blend.
        new_pw = _as_point(obs.p_world)
        cur_pw = _as_point(self.spatial_absolute["p_world"])
        jump = float(np.linalg.norm(new_pw - cur_pw))
        if is_movable and jump > motion_overwrite_threshold_m:
            self.spatial_absolute["p_world"] = new_pw.tolist()
        else:
            self.update_absolute_point(new_pw, alpha=eff_alpha)

        self.updated_at = obs.timestamp
        self.spatial_absolute["observation_count"] = len(self.observations)
        self.spatial_absolute["last_seen_timestamp"] = obs.timestamp
        self.attributes["count_contribution"] = int(self.attributes.get("count_contribution", 1))

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "semantic_label": self.semantic_label,
            "aliases": self.aliases,
            "place_id": self.place_id,
            "attributes": {
                k: _jsonable(v)
                for k, v in self.attributes.items()
                if k != "_spatial_absolute"
            },
            "spatial_absolute": _jsonable(self.spatial_absolute),
            # spatial_egocentric_dynamic is *not* persisted: it is recomputed
            # per query against the user's current pose. See render_egocentric.
            "observations": [obs.to_dict() for obs in self.observations],
            "keyframes": self.keyframes,
            "track_ids": self.track_ids,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EntityNode":
        attrs = dict(data.get("attributes", {}))
        attrs["_spatial_absolute"] = data.get("spatial_absolute", {})
        # Older graph files (v0.2.0 and earlier) may still carry a stale
        # ``spatial_egocentric_dynamic`` field. We silently drop it here —
        # the field is purely query-time state and is regenerated by
        # render_egocentric on the next retrieval.
        return cls(
            node_id=data["node_id"],
            semantic_label=data["semantic_label"],
            aliases=list(data.get("aliases", [])),
            attributes=attrs,
            place_id=data.get("place_id"),
            observations=[ObservationRecord.from_dict(o) for o in data.get("observations", [])],
            keyframes=list(data.get("keyframes", [])),
            track_ids=list(data.get("track_ids", [])),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )


@dataclass
class SceneGraph:
    reference_frame: str = "Frame_0_World_Origin"
    merge_threshold_m: float = 0.5
    max_observations_per_node: int = 128
    keyframes_per_node: int = 4
    dynamic_update_alpha: float = 0.70
    static_update_alpha: float = 0.20
    motion_overwrite_threshold_m: float = 0.50
    color_histogram_min_similarity: float = 0.55
    semantic_embedding_min_similarity: float = 0.30
    track_match_max_gap_frames: int = 5
    track_match_max_jump_m: float = 2.0
    label_merge_thresholds_m: dict[str, float] = field(default_factory=dict)
    nodes: dict[str, EntityNode] = field(default_factory=dict)
    edges: list[dict[str, Any]] = field(default_factory=list)
    place_nodes: dict[str, dict[str, Any]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    _next_id: int = 1

    def _new_node_id(self) -> str:
        node_id = f"entity_{self._next_id:03d}"
        self._next_id += 1
        return node_id

    def _label_threshold(self, label: str) -> float:
        return label_merge_threshold(
            label, default=self.merge_threshold_m, overrides=self.label_merge_thresholds_m
        )

    def _find_match(
        self,
        label: str,
        p_world: np.ndarray,
        attributes: dict[str, Any] | None = None,
        track_id: str | None = None,
        timestamp: float | None = None,
        frame_index: int | None = None,
    ) -> tuple[EntityNode | None, float]:
        attributes = attributes or {}
        color = normalize_color_name(str(attributes.get("color"))) if attributes.get("color") else None
        new_hist = attributes.get("color_hsv_histogram")
        new_emb = attributes.get("semantic_embedding")
        threshold = self._label_threshold(label)

        # Strong association: stable tracker identity, but not an unconditional
        # bypass of world-space gating. Some lightweight trackers can carry a
        # stale ID across a cut / occlusion, which would collapse two physical
        # instances into one node. We therefore accept far same-track matches
        # only when the object is explicitly movable and the observation is
        # temporally continuous with the node's last observation.
        if track_id:
            for node in self.nodes.values():
                if track_id not in node.track_ids or not _semantic_compatible(label, node.semantic_label):
                    continue
                dist = float(np.linalg.norm(node.p_world - p_world))
                if dist <= threshold:
                    return node, dist

                last_obs = node.observations[-1] if node.observations else None
                frame_gap = math.inf
                if last_obs is not None and frame_index is not None:
                    frame_gap = abs(int(frame_index) - int(last_obs.frame_index))
                is_movable = bool(attributes.get("is_movable") or node.attributes.get("is_movable"))
                max_continuous_jump = self.track_match_max_jump_m * max(1.0, float(frame_gap))
                if (
                    is_movable
                    and frame_gap <= self.track_match_max_gap_frames
                    and dist <= max_continuous_jump
                ):
                    return node, dist

        best: EntityNode | None = None
        best_score = -math.inf
        best_dist = math.inf
        for node in self.nodes.values():
            if not _semantic_compatible(label, node.semantic_label):
                continue
            node_color = normalize_color_name(str(node.attributes.get("color"))) if node.attributes.get("color") else None
            if color and node_color and color != node_color:
                continue

            # Histogram gate: only reject if both sides have a histogram and similarity is low.
            existing_hist = node.attributes.get("color_hsv_histogram")
            if new_hist is not None and existing_hist is not None:
                hist_sim = histogram_cosine_similarity(new_hist, existing_hist)
                if hist_sim < self.color_histogram_min_similarity:
                    continue
            else:
                hist_sim = 0.5

            # Semantic-embedding gate (e.g. CLIP). Optional; skipped if either side is missing.
            existing_emb = node.attributes.get("semantic_embedding")
            if new_emb is not None and existing_emb is not None:
                emb_sim = _embedding_cosine_similarity(new_emb, existing_emb)
                if emb_sim < self.semantic_embedding_min_similarity:
                    continue
            else:
                emb_sim = 0.5

            dist = float(np.linalg.norm(node.p_world - p_world))
            if dist > threshold:
                continue

            # Composite score: closer + similar color histogram + similar semantic embedding.
            score = -dist + 0.20 * hist_sim + 0.20 * emb_sim
            if score > best_score:
                best, best_score, best_dist = node, score, dist

        return (best, best_dist) if best is not None else (None, math.inf)

    def upsert_object(
        self,
        label: str,
        p_world: Any,
        timestamp: float,
        frame_index: int,
        p_cam: Any | None = None,
        track_id: str | None = None,
        attributes: dict[str, Any] | None = None,
        confidence: float = 1.0,
        pose_confidence: float = 1.0,
        keyframe_path: str | None = None,
        bbox_xyxy: tuple[float, float, float, float] | None = None,
    ) -> tuple[EntityNode, str]:
        attributes = dict(attributes or {})
        if "color" in attributes:
            attributes["color"] = normalize_color_name(str(attributes["color"]))
        attributes.setdefault("count_contribution", 1)

        pw = _as_point(p_world)
        pc = _as_point(p_cam).tolist() if p_cam is not None else None
        match, _dist = self._find_match(
            label=label,
            p_world=pw,
            attributes=attributes,
            track_id=track_id,
            timestamp=float(timestamp),
            frame_index=int(frame_index),
        )

        bbox_area = 0.0
        if bbox_xyxy is not None:
            bbox_area = max(0.0, bbox_xyxy[2] - bbox_xyxy[0]) * max(0.0, bbox_xyxy[3] - bbox_xyxy[1])

        obs = ObservationRecord(
            timestamp=float(timestamp),
            frame_index=int(frame_index),
            track_id=track_id,
            p_world=pw.tolist(),
            p_cam=pc,
            confidence=float(confidence),
            pose_confidence=float(pose_confidence),
            keyframe_path=keyframe_path,
            bbox_xyxy=bbox_xyxy,
            bbox_area=float(bbox_area),
        )

        if match is None:
            node_id = self._new_node_id()
            attrs = dict(attributes)
            attrs["_spatial_absolute"] = {
                "reference_frame": self.reference_frame,
                "p_world": pw.tolist(),
                "observation_count": 1,
                "last_seen_timestamp": float(timestamp),
            }
            node = EntityNode(
                node_id=node_id,
                semantic_label=label,
                attributes=attrs,
                reference_frame=self.reference_frame,
                observations=[obs],
                keyframes=[keyframe_path] if keyframe_path else [],
                aliases=[],
                track_ids=[track_id] if track_id else [],
                created_at=float(timestamp),
                updated_at=float(timestamp),
            )
            self.nodes[node_id] = node
            return node, "spawn"

        match.add_observation(
            obs,
            attributes=attributes,
            max_observations=self.max_observations_per_node,
            keyframes_per_node=self.keyframes_per_node,
            dynamic_alpha=self.dynamic_update_alpha,
            static_alpha=self.static_update_alpha,
            motion_overwrite_threshold_m=self.motion_overwrite_threshold_m,
        )
        return match, "merge"

    def build_edges(self, k: int = 5, max_distance_m: float = 2.0) -> None:
        self.edges = []
        nodes = list(self.nodes.values())
        for src in nodes:
            distances: list[tuple[float, EntityNode]] = []
            for dst in nodes:
                if src.node_id == dst.node_id:
                    continue
                d = float(np.linalg.norm(src.p_world - dst.p_world))
                if d <= max_distance_m:
                    distances.append((d, dst))
            for d, dst in sorted(distances, key=lambda x: x[0])[:k]:
                self.edges.append(
                    {
                        "source": src.node_id,
                        "target": dst.node_id,
                        "relation": "near",
                        "distance_m": round(d, 3),
                        "reference_frame": self.reference_frame,
                    }
                )
        # Add object→place edges for nodes that have been assigned a place_id.
        for node in nodes:
            if node.place_id and node.place_id in self.place_nodes:
                self.edges.append(
                    {
                        "source": node.node_id,
                        "target": node.place_id,
                        "relation": "in_place",
                        "reference_frame": self.reference_frame,
                    }
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "directme.scene_graph.v2",
            "reference_frame": self.reference_frame,
            "metadata": self.metadata,
            "place_nodes": self.place_nodes,
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": self.edges,
            "settings": {
                "merge_threshold_m": self.merge_threshold_m,
                "max_observations_per_node": self.max_observations_per_node,
                "keyframes_per_node": self.keyframes_per_node,
                "dynamic_update_alpha": self.dynamic_update_alpha,
                "static_update_alpha": self.static_update_alpha,
                "motion_overwrite_threshold_m": self.motion_overwrite_threshold_m,
                "color_histogram_min_similarity": self.color_histogram_min_similarity,
                "semantic_embedding_min_similarity": self.semantic_embedding_min_similarity,
                "track_match_max_gap_frames": self.track_match_max_gap_frames,
                "track_match_max_jump_m": self.track_match_max_jump_m,
                "label_merge_thresholds_m": self.label_merge_thresholds_m,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SceneGraph":
        settings = dict(data.get("settings", {}))
        allowed = {
            "merge_threshold_m",
            "max_observations_per_node",
            "keyframes_per_node",
            "dynamic_update_alpha",
            "static_update_alpha",
            "motion_overwrite_threshold_m",
            "color_histogram_min_similarity",
            "semantic_embedding_min_similarity",
            "track_match_max_gap_frames",
            "track_match_max_jump_m",
            "label_merge_thresholds_m",
        }
        graph = cls(
            reference_frame=data.get("reference_frame", "Frame_0_World_Origin"),
            **{k: v for k, v in settings.items() if k in allowed},
        )
        graph.metadata = data.get("metadata", {})
        graph.place_nodes = dict(data.get("place_nodes", {}))
        for node_data in data.get("nodes", []):
            node = EntityNode.from_dict(node_data)
            graph.nodes[node.node_id] = node
        graph.edges = list(data.get("edges", []))
        max_id = 0
        for node_id in graph.nodes:
            try:
                max_id = max(max_id, int(node_id.split("_")[-1]))
            except ValueError:
                pass
        graph._next_id = max_id + 1
        return graph

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load_json(cls, path: str | Path) -> "SceneGraph":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)
