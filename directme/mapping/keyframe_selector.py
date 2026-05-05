"""Per-node keyframe selection.

Given a stream of candidate (path, bbox_area, embedding) tuples — typically
one per object observation — pick a small budget of keyframes that
**represent the object across viewpoints**, not just the closest views.

Two strategies are supported, automatically chosen per call:

1. **Diversity sampling on CLIP / DINO embeddings.** When an embedding is
   available, the selector keeps the K candidates that are mutually most
   dissimilar in cosine space (greedy farthest-point sampling, seeded by
   the highest-bbox candidate). This prevents the v0.3 failure mode where
   four near-identical consecutive frames win the budget because they all
   happen to have the largest bbox.

2. **Bbox-area greedy fallback.** Used when no embedding is supplied.
   Identical to the v0.3 behaviour and kept for backwards compatibility
   with toy / no-embedding pipelines.

Persistence note
----------------
Only the *selected* keyframe paths are persisted (on
``EntityNode.keyframes``). The candidate pool — including the cached
embeddings — is in-memory only. After a graph load the selector starts
fresh; this is correct because new observations after load will be
compared against new observations only, and the already-selected paths
remain pinned through subsequent diversity passes (they are admitted as
preexisting candidates with embedding ``None``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


@dataclass
class _Candidate:
    """Internal record for a single candidate keyframe."""

    path: str
    bbox_area: float
    embedding: np.ndarray | None  # 1-D float32, optional


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance in [0, 2]; returns ``2.0`` (max) if either vector is
    degenerate so degenerate vectors are favoured for selection."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-8 or nb <= 1e-8:
        return 2.0
    return 1.0 - float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))


def _farthest_point_sample(
    candidates: list[_Candidate], k: int
) -> list[_Candidate]:
    """Greedy farthest-point sampling on cosine distance over embeddings.

    Seeds with the candidate of largest bbox_area (most likely to be
    visually informative on its own), then at each step adds the candidate
    whose minimum distance to the already-selected set is largest.

    Candidates without an embedding are kept in the pool but never used
    as distance probes; they only enter the result if there are fewer than
    ``k`` candidates with embeddings.
    """
    if k <= 0 or not candidates:
        return []
    if len(candidates) <= k:
        return list(candidates)

    embedded = [c for c in candidates if c.embedding is not None]
    bare = [c for c in candidates if c.embedding is None]

    if not embedded:
        # All candidates lack embeddings → fall back to bbox area.
        return sorted(bare, key=lambda c: -c.bbox_area)[:k]

    # Seed with the largest-bbox embedded candidate.
    seed_idx = max(range(len(embedded)), key=lambda i: embedded[i].bbox_area)
    selected: list[_Candidate] = [embedded[seed_idx]]
    remaining: list[_Candidate] = [c for i, c in enumerate(embedded) if i != seed_idx]

    # Precompute per-remaining "min distance to selected so far".
    min_dists: list[float] = [
        _cosine_distance(c.embedding, selected[0].embedding)  # type: ignore[arg-type]
        for c in remaining
    ]

    while remaining and len(selected) < k:
        best_i = max(range(len(remaining)), key=lambda i: min_dists[i])
        chosen = remaining.pop(best_i)
        min_dists.pop(best_i)
        selected.append(chosen)
        # Update min distances against the newly added selection.
        for i, c in enumerate(remaining):
            d = _cosine_distance(c.embedding, chosen.embedding)  # type: ignore[arg-type]
            if d < min_dists[i]:
                min_dists[i] = d

    # Fill any leftover budget with bare (non-embedded) candidates by bbox area.
    if len(selected) < k and bare:
        selected.extend(sorted(bare, key=lambda c: -c.bbox_area)[: k - len(selected)])

    return selected


@dataclass
class KeyframeSelector:
    """Maintains a bounded set of representative keyframes for a single node.

    The selector is deliberately stateful: candidates accumulate across many
    observations, and the selected K is recomputed on every ``add`` call so
    later, more representative views can displace earlier near-duplicates.
    """

    budget: int = 4
    pool_cap: int = 16  # hard upper bound on the candidate pool to keep memory bounded
    _candidates: list[_Candidate] = field(default_factory=list)
    _seen_paths: set[str] = field(default_factory=set)

    def add(
        self,
        path: str,
        bbox_area: float,
        embedding: np.ndarray | list[float] | None = None,
    ) -> None:
        """Add a candidate observation. No-op if ``path`` was already added.

        Re-runs the diversity selection internally so :attr:`selected` is
        always up to date.
        """
        if not path or path in self._seen_paths:
            return
        emb_arr: np.ndarray | None = None
        if embedding is not None:
            emb_arr = np.asarray(embedding, dtype=np.float32).ravel()
            if emb_arr.size == 0:
                emb_arr = None

        self._candidates.append(
            _Candidate(path=path, bbox_area=float(bbox_area), embedding=emb_arr)
        )
        self._seen_paths.add(path)

        if len(self._candidates) > self.pool_cap:
            # Evict the least-informative candidate. "Least informative" =
            # smallest bbox area among those without an embedding, else
            # smallest bbox area overall.
            no_emb = [c for c in self._candidates if c.embedding is None]
            victim = min(
                no_emb if no_emb else self._candidates,
                key=lambda c: c.bbox_area,
            )
            self._candidates.remove(victim)
            self._seen_paths.discard(victim.path)

    def adopt_existing(self, path: str) -> None:
        """Pin a previously-saved keyframe path as a no-embedding candidate.

        Used right after ``SceneGraph.load_json`` to make the selector
        reflect what the graph already remembers, so newly arriving
        candidates compete against the existing selection rather than
        clobbering it on the next ``add``.
        """
        if path and path not in self._seen_paths:
            self._candidates.append(
                _Candidate(path=path, bbox_area=0.0, embedding=None)
            )
            self._seen_paths.add(path)

    @property
    def selected(self) -> list[str]:
        """Current set of selected keyframe paths, recomputed on demand."""
        chosen = _farthest_point_sample(self._candidates, self.budget)
        return [c.path for c in chosen]

    def __len__(self) -> int:
        return len(self._candidates)


def select_keyframes(
    candidates: Iterable[tuple[str, float, np.ndarray | list[float] | None]],
    budget: int = 4,
) -> list[str]:
    """One-shot helper: take an iterable of (path, bbox_area, embedding) and
    return up to ``budget`` paths after diversity sampling.

    Useful when you have all candidates up front and don't need stateful
    accumulation.
    """
    sel = KeyframeSelector(budget=budget, pool_cap=10**9)
    for path, area, emb in candidates:
        sel.add(path, area, emb)
    return sel.selected
