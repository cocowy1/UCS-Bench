"""Per-node reliability scoring for soft, noise-aware retrieval.

The scene graph is built by an automatic perception / tracking pipeline
(open-vocab detector + SAM2 masks + IoU tracking + depth-unprojection), so its
nodes are *evidence*, not ground truth.  This module computes a composite
reliability score in [0, 1] per :class:`~directme.mapping.scene_graph.EntityNode`
from the signals that are actually available on the node:

* detector confidence (``ObservationRecord.confidence``)          — all graphs
* SAM2 mask quality (``ObservationRecord.mask_score``)            — v0.6+ graphs
* track stability (fraction of observations sharing one track id)
* temporal persistence (saturating observation count)
* world-space position consistency (static objects only)
* semantic consistency (per-observation label agreement)          — v0.6+ graphs

The score is deliberately used for **ranking / down-weighting**, never as a
hard filter: a noisy observation can still be the only evidence a question
has, and the MLLM is given the raw keyframes to verify against.

All functions are pure (no graph mutation) and respect an optional
``as_of_timestamp`` so reliability reflects only what had been observed at
the question time (causal semantics).
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np

from directme.mapping.scene_graph import EntityNode, _norm_label


def _observations_as_of(node: EntityNode, as_of_timestamp: float | None):
    obs = node.observations
    if as_of_timestamp is None:
        return list(obs)
    return [o for o in obs if float(o.timestamp) <= float(as_of_timestamp)]


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


def _detection_confidence(obs) -> float | None:
    scores = [float(o.confidence) for o in obs if o.confidence is not None]
    if not scores:
        return None
    return float(np.clip(np.mean(scores), 0.0, 1.0))


def _mask_quality(obs) -> float | None:
    scores = [float(o.mask_score) for o in obs if getattr(o, "mask_score", None) is not None]
    if not scores:
        return None
    return float(np.clip(np.mean(scores), 0.0, 1.0))


def _track_stability(obs) -> float | None:
    """Fraction of observations carrying the modal track id.

    1.0 → a single persistent tracker identity (best case); low values mean
    the node's evidence is stitched together from many short tracker ids
    (fragmentation / re-detection), which is exactly the failure mode of
    long-sequence IoU tracking.
    """
    tids = [o.track_id for o in obs if o.track_id]
    if not tids:
        return None
    modal = Counter(tids).most_common(1)[0][1]
    return modal / len(tids)


def _temporal_persistence(obs, saturate_at: int = 8) -> float | None:
    if not obs:
        return None
    return min(1.0, len(obs) / float(saturate_at))


def _position_consistency(node: EntityNode, obs, scale_m: float = 0.25) -> float | None:
    """World-space stability of the node anchor over the retained observations.

    Only meaningful for static objects; movable objects legitimately move, so
    we return ``None`` (signal unavailable) when ``is_movable`` is set.  The
    score is ``1 / (1 + MAD / scale_m)`` on the median absolute deviation of
    the observed world positions, so a perfectly static node scores 1.0 and a
    node wandering by ~``scale_m`` metres scores ~0.5.
    """
    if bool(node.attributes.get("is_movable", False)):
        return None
    pts = np.asarray([o.p_world for o in obs if o.p_world is not None], dtype=float)
    if pts.ndim != 2 or len(pts) < 3:
        return None
    if not np.all(np.isfinite(pts)):
        return None
    # Median Euclidean deviation from the median point (a robust spread of the
    # node's world positions; per-axis medians would be diluted by static axes).
    med = np.median(pts, axis=0)
    mad = float(np.median(np.linalg.norm(pts - med, axis=1)))
    return 1.0 / (1.0 + mad / max(scale_m, 1e-6))


def _semantic_consistency(node: EntityNode, obs) -> float | None:
    """Agreement of per-observation detector labels (v0.6+ graphs).

    Fraction of retained observations whose normalized label equals the modal
    label; disagreement indicates detector flicker or a mis-associated merge.
    """
    labels = [_norm_label(str(o.label)) for o in obs if getattr(o, "label", None)]
    labels = [l for l in labels if l]
    if not labels:
        return None
    modal = Counter(labels).most_common(1)[0][1]
    return modal / len(labels)


# Subscore name → weight in the composite.  Weights sum to 1.0 over the
# subscores that are actually available; unavailable signals are skipped and
# the remaining weights renormalized.
_SUBSCORE_WEIGHTS: dict[str, float] = {
    "detection_confidence": 0.25,
    "track_stability": 0.20,
    "temporal_persistence": 0.15,
    "position_consistency": 0.20,
    "semantic_consistency": 0.10,
    "mask_quality": 0.10,
}


def node_reliability(
    node: EntityNode,
    as_of_timestamp: float | None = None,
) -> dict[str, Any]:
    """Compute a composite reliability score for one node.

    Returns a dict with the composite ``reliability`` in [0, 1] plus the
    individual subscores (``None`` where the signal is unavailable, e.g. on
    graphs built before per-observation labels / mask scores were stored).
    With no signals at all the score degrades to the neutral prior 0.5.
    """
    obs = _observations_as_of(node, as_of_timestamp)
    if not obs:
        return {
            "reliability": 0.0,
            "n_observations": 0,
            "detection_confidence": None,
            "track_stability": None,
            "temporal_persistence": None,
            "position_consistency": None,
            "semantic_consistency": None,
            "mask_quality": None,
        }

    subscores = {
        "detection_confidence": _detection_confidence(obs),
        "track_stability": _track_stability(obs),
        "temporal_persistence": _temporal_persistence(obs),
        "position_consistency": _position_consistency(node, obs),
        "semantic_consistency": _semantic_consistency(node, obs),
        "mask_quality": _mask_quality(obs),
    }

    total_w = 0.0
    acc = 0.0
    for name, value in subscores.items():
        if value is None or not math.isfinite(value):
            continue
        w = _SUBSCORE_WEIGHTS[name]
        acc += w * float(np.clip(value, 0.0, 1.0))
        total_w += w

    reliability = acc / total_w if total_w > 0 else 0.5
    out = dict(subscores)
    out["reliability"] = float(np.clip(reliability, 0.0, 1.0))
    out["n_observations"] = len(obs)
    return out


def position_as_of(
    node: EntityNode,
    as_of_timestamp: float | None = None,
    recency_tau_s: float | None = 30.0,
) -> np.ndarray | None:
    """Recompute the node's world anchor using only observations ≤ ``t_q``.

    * movable objects → position of the latest causal observation (motion
      overwrite semantics, matching :meth:`EntityNode.add_observation`);
    * static objects → confidence-weighted mean of the causal observations
      with an exponential **recency decay** (``exp(-(t_q - t_i)/tau)``). The
      decay matters: a plain historical mean lets early, pose-drifted chunk
      observations pollute the anchor, whereas the legacy full-history EMA
      implicitly favoured recent sightings. With ``recency_tau_s=None`` the
      decay is disabled (uniform causal mean).

    Returns ``None`` when the node has no observation at/before ``t_q``
    (the caller should then treat the node as not-yet-existing).
    """
    obs = _observations_as_of(node, as_of_timestamp)
    if not obs:
        return None
    kept = [(o, p) for o, p in ((o, o.p_world) for o in obs) if p is not None]
    pts = np.asarray([p for _o, p in kept], dtype=float)
    if pts.ndim != 2 or len(pts) == 0:
        return None
    if bool(node.attributes.get("is_movable", False)):
        return pts[-1].reshape(3)
    weights = np.asarray(
        [
            max(1e-6, float(o.confidence if o.confidence is not None else 1.0))
            * max(1e-6, float(o.pose_confidence if o.pose_confidence is not None else 1.0))
            for o, _p in kept
        ],
        dtype=float,
    )
    if recency_tau_s is not None and as_of_timestamp is not None:
        ages = np.asarray(
            [max(0.0, float(as_of_timestamp) - float(o.timestamp)) for o, _p in kept],
            dtype=float,
        )
        weights = weights * np.exp(-ages / max(float(recency_tau_s), 1e-6))
    if len(weights) != len(pts) or weights.sum() <= 0:
        return pts[-1].reshape(3)
    return (weights[:, None] * pts).sum(axis=0) / weights.sum()


def keyframes_as_of(
    node: EntityNode,
    as_of_timestamp: float | None = None,
    limit: int | None = None,
) -> list[str]:
    """Causal keyframe candidates: unique paths from observations ≤ ``t_q``.

    The node's persisted ``keyframes`` list is diversity-selected over the
    node's *full* history and carries no timestamps, so under strict causal
    semantics it can leak frames captured after the question time.  This
    helper rebuilds the candidate list from the observation records instead.
    """
    seen: set[str] = set()
    out: list[str] = []
    for o in _observations_as_of(node, as_of_timestamp):
        p = o.keyframe_path
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    if limit is not None:
        out = out[:limit]
    return out
