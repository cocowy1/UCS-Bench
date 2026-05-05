"""Tests for the diversity-aware keyframe selector."""

from __future__ import annotations

import numpy as np

from directme.mapping.keyframe_selector import (
    KeyframeSelector,
    select_keyframes,
)


def _emb(seed: int, dim: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def test_selector_falls_back_to_bbox_area_when_no_embeddings():
    """v0.3-compatible behaviour: when no candidates carry an embedding,
    we sort by bbox_area greedily — this matches the legacy logic."""
    sel = KeyframeSelector(budget=3)
    sel.add("a.jpg", bbox_area=100.0)
    sel.add("b.jpg", bbox_area=300.0)
    sel.add("c.jpg", bbox_area=200.0)
    sel.add("d.jpg", bbox_area=50.0)
    assert sel.selected[:3] == ["b.jpg", "c.jpg", "a.jpg"]


def test_selector_picks_diverse_views_when_embeddings_are_present():
    """If we have 4 nearly-identical candidates and 1 different one, the
    selector must include the different one in its top-3 — this is the
    failure mode the v0.3 bbox-area greedy had."""
    near_duplicates = _emb(0)
    odd_one_out = -near_duplicates  # cosine distance ≈ 2 (max)

    sel = KeyframeSelector(budget=3)
    sel.add("near_a.jpg", bbox_area=10.0, embedding=near_duplicates + 1e-3 * _emb(1))
    sel.add("near_b.jpg", bbox_area=11.0, embedding=near_duplicates + 1e-3 * _emb(2))
    sel.add("near_c.jpg", bbox_area=12.0, embedding=near_duplicates + 1e-3 * _emb(3))
    sel.add("near_d.jpg", bbox_area=13.0, embedding=near_duplicates + 1e-3 * _emb(4))
    sel.add("odd.jpg", bbox_area=5.0, embedding=odd_one_out)

    selected = sel.selected
    assert "odd.jpg" in selected, (
        f"Diversity selector should keep the visually different frame; got {selected}"
    )
    assert len(selected) == 3


def test_selector_is_idempotent_on_duplicate_paths():
    sel = KeyframeSelector(budget=4)
    sel.add("x.jpg", 100.0, embedding=_emb(0))
    sel.add("x.jpg", 999.0, embedding=_emb(99))  # duplicate path → ignored
    assert len(sel) == 1
    assert sel.selected == ["x.jpg"]


def test_selector_pool_cap_evicts_least_informative():
    sel = KeyframeSelector(budget=4, pool_cap=5)
    for i in range(20):
        sel.add(f"f_{i}.jpg", bbox_area=float(i), embedding=_emb(i))
    # pool_cap caps the pool, not the budget.
    assert len(sel) <= 5
    assert len(sel.selected) <= 4


def test_adopt_existing_seeds_pool_for_post_load_continuation():
    """After SceneGraph.load_json, EntityNode.keyframes is populated but the
    in-memory selector is empty. ``adopt_existing`` must register those
    paths so subsequent ``add`` calls treat them as competing candidates."""
    sel = KeyframeSelector(budget=2)
    sel.adopt_existing("preserved.jpg")
    assert sel.selected == ["preserved.jpg"]
    sel.add("new.jpg", bbox_area=100.0, embedding=_emb(0))
    selected = sel.selected
    assert "preserved.jpg" in selected
    assert "new.jpg" in selected


def test_select_keyframes_one_shot_helper():
    cands = [
        ("a.jpg", 10.0, _emb(0)),
        ("b.jpg", 12.0, _emb(0) + 1e-3 * _emb(1)),  # near a.jpg
        ("c.jpg", 8.0, -_emb(0)),                    # opposite of a.jpg
    ]
    selected = select_keyframes(cands, budget=2)
    assert len(selected) == 2
    # The diverse pair should include 'a.jpg' (largest bbox_area in its
    # cluster) and 'c.jpg' (most distant in cosine space).
    assert "c.jpg" in selected
