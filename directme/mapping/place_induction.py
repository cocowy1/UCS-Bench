"""Place node induction.

Cluster object nodes by world-frame position to discover semantic regions
(``kitchen``, ``living_room``, ``bedroom``, etc.) and attach each entity to a
``place_id``. Uses a simple greedy radius-based grouping which is robust and
deterministic for small to medium graphs without needing scikit-learn.

For very large graphs you can swap the implementation with HDBSCAN or
DBSCAN by replacing :func:`cluster_nodes`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from directme.mapping.scene_graph import EntityNode, SceneGraph


@dataclass
class Place:
    place_id: str
    label: str
    centroid: list[float]
    member_node_ids: list[str]


def _greedy_radius_clusters(points: np.ndarray, radius_m: float) -> list[list[int]]:
    """Greedy clustering: each new point joins the first cluster whose centroid
    is within ``radius_m``, otherwise it starts a new cluster.
    """
    clusters: list[list[int]] = []
    centroids: list[np.ndarray] = []
    for i, p in enumerate(points):
        joined = False
        for ci, c in enumerate(centroids):
            if np.linalg.norm(p - c) <= radius_m:
                clusters[ci].append(i)
                centroids[ci] = (centroids[ci] * (len(clusters[ci]) - 1) + p) / len(clusters[ci])
                joined = True
                break
        if not joined:
            clusters.append([i])
            centroids.append(p.copy())
    return clusters


def _label_by_majority(nodes: list[EntityNode]) -> str:
    """Pick a place label by majority semantic vote among constituent objects."""
    counts: Counter[str] = Counter()
    for n in nodes:
        scene = n.attributes.get("scene_tag")
        if scene:
            counts[str(scene).lower()] += 2
        counts[n.semantic_label.lower()] += 1
    if not counts:
        return "place"
    return counts.most_common(1)[0][0]


def induce_places(
    graph: SceneGraph,
    radius_m: float = 3.0,
    min_members: int = 1,
) -> list[Place]:
    """Run place induction over the current scene graph.

    Mutates ``graph.place_nodes`` and writes ``place_id`` on each member node.
    Returns the list of induced places for inspection.

    Strategy:
      1. Pre-partition nodes by their ``attributes['scene_tag']`` (if present).
         This guarantees that two nodes labeled by the scene classifier as
         belonging to different rooms are never merged into a single place,
         even if their world-frame centroids happen to be within ``radius_m``.
      2. Within each scene-tag group, run greedy radius clustering to split
         large rooms (e.g. ``"hallway"``) into multiple sub-places.
      3. Singleton clusters are kept by default (``min_members=1``) so that
         a single object in a small room still surfaces as a queryable place.
    """
    nodes = list(graph.nodes.values())
    if not nodes:
        graph.place_nodes = {}
        return []

    # Group node indices by normalized scene_tag. Nodes without a tag share
    # the synthetic ``"__untagged__"`` partition, which is geometry-only.
    groups: dict[str, list[int]] = {}
    for i, n in enumerate(nodes):
        tag = n.attributes.get("scene_tag") or ""
        key = str(tag).strip().lower().replace(" ", "_").replace("-", "_") or "__untagged__"
        groups.setdefault(key, []).append(i)

    induced: list[Place] = []
    new_place_nodes: dict[str, dict] = {}
    place_counter = 0
    for tag_key, idxs in groups.items():
        sub_points = np.stack([nodes[i].p_world for i in idxs], axis=0)
        local_clusters = _greedy_radius_clusters(sub_points, radius_m=radius_m)
        for local in local_clusters:
            global_idxs = [idxs[k] for k in local]
            if len(global_idxs) < min_members:
                for i in global_idxs:
                    nodes[i].place_id = None
                continue
            member_nodes = [nodes[i] for i in global_idxs]
            centroid = np.stack(
                [nodes[i].p_world for i in global_idxs], axis=0
            ).mean(axis=0)
            label = _label_by_majority(member_nodes)
            place_id = f"place_{place_counter:03d}"
            place_counter += 1
            for n in member_nodes:
                n.place_id = place_id
            place = Place(
                place_id=place_id,
                label=label,
                centroid=centroid.tolist(),
                member_node_ids=[n.node_id for n in member_nodes],
            )
            new_place_nodes[place_id] = {
                "place_id": place_id,
                "label": label,
                "centroid": place.centroid,
                "member_node_ids": place.member_node_ids,
            }
            induced.append(place)

    graph.place_nodes = new_place_nodes
    return induced
