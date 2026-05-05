"""Egocentric world→camera projection, discrete relation labels, and reachability.

A node's *egocentric* state is recomputed at query time from its world anchor and
the user's current pose:

    P_cam_current = inverse(T_world_from_camera_current) · P_world_node

We then derive three pieces of information from ``P_cam_current``:

1. **Discrete egocentric relation** (``EgoRelation``): one of
   ``front``, ``behind``, ``left``, ``right``, ``front_left``, ``front_right``,
   ``behind_left``, ``behind_right``. This is what the scene graph exposes as
   the relation type "object X is located <relation> me".

2. **Reachability**: ``True`` iff the Euclidean distance from the user is
   within ``reachable_radius_m`` (default 5.0 m), per project policy.

3. **Natural-language phrasing** in zh / en for prompt assembly.

Camera convention (consistent with the rest of DirectMe):

* x = right, y = down, z = forward.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from directme.geometry.poses import SE3
from directme.mapping.scene_graph import EntityNode


# ---------------------------------------------------------------------------
# Discrete relation labels
# ---------------------------------------------------------------------------

# Public constant set of relation strings. Persisted in scene-graph edges
# emitted at query time, and used by RuleBasedAnswerGenerator and prompt builders.
EGO_RELATION_LABELS: tuple[str, ...] = (
    "front",
    "behind",
    "left",
    "right",
    "front_left",
    "front_right",
    "behind_left",
    "behind_right",
)

DEFAULT_REACHABLE_RADIUS_M: float = 5.0
DEFAULT_LATERAL_TOLERANCE_RATIO: float = 0.20  # |x| < 0.20 * max(|z|, 1m) → "centered"
DEFAULT_DEPTH_NEUTRAL_TOLERANCE_M: float = 0.30  # |z| below this can be pure left/right


@dataclass(frozen=True)
class EgoRelation:
    """A user-relative spatial relation, computed at query time."""

    relation: str           # one of EGO_RELATION_LABELS
    p_cam: tuple[float, float, float]
    distance_m: float       # Euclidean distance ||p_cam||_2
    reachable: bool         # distance_m <= reachable_radius_m
    natural_language: str   # localized phrase, e.g. "在您的右前方约 0.4 米处（伸手可及）"


def world_to_camera_point(p_world: Any, current_world_from_camera: SE3) -> np.ndarray:
    return current_world_from_camera.inverse().transform_points(
        np.asarray(p_world, dtype=float).reshape(3)
    )


def classify_egocentric_relation(
    p_cam: Any,
    lateral_tolerance_ratio: float = DEFAULT_LATERAL_TOLERANCE_RATIO,
    depth_neutral_tolerance_m: float = DEFAULT_DEPTH_NEUTRAL_TOLERANCE_M,
) -> str:
    """Map a 3D camera-frame point to one of EGO_RELATION_LABELS.

    A point is "centered" laterally when ``|x| <= lateral_tolerance_ratio *
    max(|z|, 1.0)``. The 1-meter floor prevents tiny |z| values from collapsing
    the cone to zero width.
    """
    pc = np.asarray(p_cam, dtype=float).reshape(3)
    x, _y, z = float(pc[0]), float(pc[1]), float(pc[2])

    threshold = lateral_tolerance_ratio * max(abs(z), 1.0)
    if abs(x) <= threshold:
        return "front" if z >= 0 else "behind"  # centered cone

    lateral = "left" if x < 0 else "right"
    # If the object is roughly abreast of the camera wearer, do not force a
    # front/back component. This makes the public left/right labels reachable
    # and better matches natural egocentric language for side objects.
    if abs(z) <= depth_neutral_tolerance_m:
        return lateral

    forward_back = "front" if z > 0 else "behind"
    return f"{forward_back}_{lateral}"


# Phrasing tables, keyed by EGO_RELATION_LABELS.
_RELATION_ZH = {
    "front": "正前方",
    "behind": "正后方",
    "left": "左侧",
    "right": "右侧",
    "front_left": "左前方",
    "front_right": "右前方",
    "behind_left": "左后方",
    "behind_right": "右后方",
}

_RELATION_EN = {
    "front": "directly in front of you",
    "behind": "directly behind you",
    "left": "to your left",
    "right": "to your right",
    "front_left": "to your front-left",
    "front_right": "to your front-right",
    "behind_left": "to your back-left",
    "behind_right": "to your back-right",
}


def natural_language_location(
    relation: str, distance_m: float, reachable: bool, language: str = "zh"
) -> str:
    if language.lower().startswith("zh"):
        rel = _RELATION_ZH.get(relation, relation)
        reach = "伸手可及" if reachable else "不可及"
        if relation in ("front", "behind"):
            return f"在您的{rel}约 {distance_m:.1f} 米处（{reach}）"
        return f"在您的{rel}约 {distance_m:.1f} 米处（{reach}）"
    rel = _RELATION_EN.get(relation, relation)
    reach = "within reach" if reachable else "out of reach"
    return f"{distance_m:.1f} m {rel} ({reach})"


def compute_ego_relation(
    p_cam: Any,
    reachable_radius_m: float = DEFAULT_REACHABLE_RADIUS_M,
    lateral_tolerance_ratio: float = DEFAULT_LATERAL_TOLERANCE_RATIO,
    depth_neutral_tolerance_m: float = DEFAULT_DEPTH_NEUTRAL_TOLERANCE_M,
    language: str = "zh",
) -> EgoRelation:
    pc = np.asarray(p_cam, dtype=float).reshape(3)
    rel = classify_egocentric_relation(
        pc,
        lateral_tolerance_ratio=lateral_tolerance_ratio,
        depth_neutral_tolerance_m=depth_neutral_tolerance_m,
    )
    distance = float(math.sqrt(float(pc[0]) ** 2 + float(pc[1]) ** 2 + float(pc[2]) ** 2))
    reachable = distance <= reachable_radius_m
    text = natural_language_location(rel, distance, reachable, language=language)
    return EgoRelation(
        relation=rel,
        p_cam=(float(pc[0]), float(pc[1]), float(pc[2])),
        distance_m=distance,
        reachable=reachable,
        natural_language=text,
    )


def render_egocentric(
    node: EntityNode,
    current_world_from_camera: SE3,
    language: str = "zh",
    reachable_radius_m: float = DEFAULT_REACHABLE_RADIUS_M,
    lateral_tolerance_ratio: float = DEFAULT_LATERAL_TOLERANCE_RATIO,
    depth_neutral_tolerance_m: float = DEFAULT_DEPTH_NEUTRAL_TOLERANCE_M,
) -> dict[str, Any]:
    """Render a node's current ego-relative state.

    Pure function: does NOT mutate ``node``. The returned dict is meant to live
    on the per-query :class:`~directme.retrieval.retriever.RetrievedItem` only,
    never on the persistent scene graph. This keeps online QA a strictly read-
    only operation against the graph and prevents query-time state from leaking
    into long-term storage if the graph is saved after a query.
    """
    p_cam = world_to_camera_point(node.spatial_absolute["p_world"], current_world_from_camera)
    ego = compute_ego_relation(
        p_cam,
        reachable_radius_m=reachable_radius_m,
        lateral_tolerance_ratio=lateral_tolerance_ratio,
        depth_neutral_tolerance_m=depth_neutral_tolerance_m,
        language=language,
    )
    return {
        "reference_frame": "Current_Camera",
        "p_cam": list(ego.p_cam),
        "relation": ego.relation,           # one of EGO_RELATION_LABELS
        "distance_m": round(ego.distance_m, 4),
        "reachable": ego.reachable,
        "natural_language": ego.natural_language,
    }
