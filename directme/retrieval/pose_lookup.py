"""Helpers for query-time current-pose resolution.

The online DirectMe path needs ``T_world_from_current_camera`` at the query
moment. In production this usually comes from the live pose backend. For offline
evaluation / CLI use, the mapping engine also writes an ``ego_pose_timeline``
into ``SceneGraph.metadata``. This module resolves the nearest recorded pose so
callers do not silently fall back to the world origin when a timestamped pose is
available.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from directme.geometry.poses import SE3
from directme.mapping.scene_graph import SceneGraph


def _record_pose(record: dict[str, Any]) -> SE3 | None:
    matrix = record.get("T_world_from_camera") or record.get("pose")
    if matrix is None:
        return None
    try:
        return SE3.from_list(matrix)
    except (TypeError, ValueError):
        return None


def pose_from_graph_timeline(
    graph: SceneGraph,
    timestamp: float | None = None,
    *,
    max_time_delta_s: float | None = None,
) -> SE3:
    """Return the latest / nearest ego pose stored in ``graph.metadata``.

    Args:
        graph: Scene graph whose metadata may contain ``ego_pose_timeline``.
        timestamp: If ``None``, return the latest valid pose. Otherwise return
            the pose whose timestamp is closest to the requested query time.
        max_time_delta_s: Optional guardrail. If set and the closest pose is
            farther than this many seconds from ``timestamp``, return identity
            instead of a stale pose.

    Returns:
        A valid :class:`SE3`. Falls back to identity only when no valid timeline
        pose is present or the optional freshness guard rejects it.
    """
    timeline = graph.metadata.get("ego_pose_timeline") or []
    valid: list[tuple[float, SE3]] = []
    for rec in timeline:
        if not isinstance(rec, dict):
            continue
        pose = _record_pose(rec)
        if pose is None:
            continue
        try:
            ts = float(rec.get("timestamp", 0.0))
        except (TypeError, ValueError):
            ts = 0.0
        valid.append((ts, pose))

    if not valid:
        return SE3.identity()

    if timestamp is None:
        return max(valid, key=lambda x: x[0])[1]

    target = float(timestamp)
    ts, pose = min(valid, key=lambda x: abs(x[0] - target))
    if max_time_delta_s is not None and abs(ts - target) > max_time_delta_s:
        return SE3.identity()
    return pose


def pose_record_from_se3(
    pose: SE3,
    *,
    timestamp: float,
    chunk_id: int | None = None,
    frame_index: int | None = None,
    scene_tag: str | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable timeline record from an SE3 pose."""
    return {
        "chunk_id": chunk_id,
        "frame_index": frame_index,
        "timestamp": float(timestamp),
        "T_world_from_camera": pose.to_list(),
        "translation": np.asarray(pose.translation, dtype=float).reshape(3).tolist(),
        "scene_tag": scene_tag,
    }
