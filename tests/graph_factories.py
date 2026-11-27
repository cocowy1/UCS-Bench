"""Shared factories for building synthetic scene graphs (pytest-local import)."""

from __future__ import annotations

import numpy as np

from directme.mapping.scene_graph import EntityNode, ObservationRecord, SceneGraph



def make_obs(
    timestamp: float,
    frame_index: int,
    p_world=(0.0, 0.0, 0.0),
    track_id="track_000001",
    confidence=0.8,
    pose_confidence=1.0,
    label="Cup",
    mask_score=None,
    keyframe_path=None,
):
    return ObservationRecord(
        timestamp=timestamp,
        frame_index=frame_index,
        track_id=track_id,
        p_world=list(p_world),
        confidence=confidence,
        pose_confidence=pose_confidence,
        label=label,
        mask_score=mask_score,
        keyframe_path=keyframe_path,
    )


def make_node(
    node_id="entity_001",
    label="Cup",
    observations=None,
    color="red",
    is_movable=False,
    first_seen=None,
):
    obs = observations or []
    node = EntityNode(
        node_id=node_id,
        semantic_label=label,
        attributes={"color": color},
        observations=obs,
        keyframes=[o.keyframe_path for o in obs if o.keyframe_path][:4],
        track_ids=sorted({o.track_id for o in obs if o.track_id}),
        created_at=obs[0].timestamp if obs else 0.0,
        updated_at=obs[-1].timestamp if obs else 0.0,
        first_seen_timestamp=first_seen if first_seen is not None else (obs[0].timestamp if obs else 0.0),
    )
    if is_movable:
        node.attributes["is_movable"] = True
    node.attributes["_spatial_absolute"] = {
        "reference_frame": "Frame_0_World_Origin",
        "p_world": list(np.mean([o.p_world for o in obs], axis=0)) if obs else [0.0, 0.0, 0.0],
        "observation_count": len(obs),
        "last_seen_timestamp": obs[-1].timestamp if obs else 0.0,
    }
    return node


def make_graph(nodes: list[EntityNode]) -> SceneGraph:
    g = SceneGraph()
    for n in nodes:
        g.nodes[n.node_id] = n
    return g


