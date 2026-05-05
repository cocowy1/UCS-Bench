"""Tests for ConceptGraphs-style semantic embedding multi-view fusion."""

import numpy as np

from directme.mapping.scene_graph import SceneGraph


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def test_embedding_fuses_via_normalized_ema():
    graph = SceneGraph(merge_threshold_m=2.0)
    e1 = _unit([1.0, 0.0, 0.0, 0.0]).tolist()
    e2 = _unit([0.96, 0.28, 0.0, 0.0]).tolist()  # very close to e1

    graph.upsert_object(
        "cup", [0.0, 0.0, 0.0], 0, 0,
        attributes={"semantic_embedding": e1},
    )
    graph.upsert_object(
        "cup", [0.05, 0.0, 0.0], 1, 1,
        attributes={"semantic_embedding": e2},
    )

    # Single merged node, embedding stays unit-norm.
    assert len(graph.nodes) == 1
    node = next(iter(graph.nodes.values()))
    fused = np.asarray(node.attributes["semantic_embedding"], dtype=np.float32)
    assert abs(np.linalg.norm(fused) - 1.0) < 1e-5


def test_embedding_gate_prevents_wrong_merge_at_close_distance():
    """Two same-label same-color objects close in space but visually
    different (embeddings near-orthogonal) should NOT merge."""
    graph = SceneGraph(
        merge_threshold_m=1.0,
        semantic_embedding_min_similarity=0.30,
    )
    e_dog = _unit([1.0, 0.0, 0.0, 0.0]).tolist()
    e_cat = _unit([0.0, 1.0, 0.0, 0.0]).tolist()

    graph.upsert_object(
        "animal", [0.0, 0.0, 0.0], 0, 0,
        attributes={"semantic_embedding": e_dog},
    )
    graph.upsert_object(
        "animal", [0.3, 0.0, 0.0], 1, 1,
        attributes={"semantic_embedding": e_cat},
    )
    # Embeddings cosine = 0.0 < gate 0.3 → must spawn a second node.
    assert len(graph.nodes) == 2


def test_missing_embedding_does_not_block_merge():
    """If either side has no embedding, the gate is skipped (backward compat)."""
    graph = SceneGraph(merge_threshold_m=1.0)
    e = _unit([1.0, 0.0, 0.0]).tolist()

    graph.upsert_object(
        "cup", [0.0, 0.0, 0.0], 0, 0,
        attributes={"color": "red", "semantic_embedding": e},
    )
    graph.upsert_object(  # no embedding on this observation
        "cup", [0.1, 0.0, 0.0], 1, 1,
        attributes={"color": "red"},
    )
    assert len(graph.nodes) == 1


def test_embedding_renormalized_after_long_random_walk():
    graph = SceneGraph(merge_threshold_m=5.0)
    base = _unit([1.0, 0.0, 0.0, 0.0]).tolist()
    graph.upsert_object("cup", [0.0, 0.0, 0.0], 0, 0,
                        attributes={"semantic_embedding": base})

    rng = np.random.default_rng(0)
    for i in range(1, 20):
        noisy = _unit(rng.normal(0, 0.05, size=4) + np.array(base)).tolist()
        graph.upsert_object("cup", [0.0, 0.0, 0.0], i, i,
                            attributes={"semantic_embedding": noisy})

    assert len(graph.nodes) == 1
    fused = np.asarray(next(iter(graph.nodes.values())).attributes["semantic_embedding"])
    assert abs(np.linalg.norm(fused) - 1.0) < 1e-4
