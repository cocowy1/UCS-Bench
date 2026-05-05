"""Tests for SqliteSceneGraphStore round-trip."""

from directme.mapping.scene_graph import SceneGraph
from directme.storage.sqlite_store import SqliteSceneGraphStore


def test_sqlite_store_round_trip(tmp_path):
    db_path = tmp_path / "scene_graph.sqlite"
    store = SqliteSceneGraphStore(db_path)

    graph = SceneGraph(merge_threshold_m=0.5)
    graph.upsert_object("cup", [1.0, 0.0, 1.0], 0, 0, attributes={"color": "red"})
    graph.upsert_object("cup", [3.0, 0.0, 1.0], 1, 1, attributes={"color": "red"})
    graph.build_edges(k=2, max_distance_m=5.0)

    store.save(graph)
    loaded = store.load()

    assert set(loaded.nodes.keys()) == set(graph.nodes.keys())
    for nid in graph.nodes:
        a = graph.nodes[nid]
        b = loaded.nodes[nid]
        assert a.semantic_label == b.semantic_label
        assert a.attributes.get("color") == b.attributes.get("color")
    assert len(loaded.edges) == len(graph.edges)
    store.close()


def test_sqlite_store_incremental_update(tmp_path):
    db_path = tmp_path / "scene_graph.sqlite"
    store = SqliteSceneGraphStore(db_path)
    graph = SceneGraph()
    graph.upsert_object("chair", [0, 0, 0], 0, 0)
    store.save(graph)

    graph.upsert_object("table", [5, 0, 0], 1, 1)
    store.mark_dirty(["entity_002"])
    store.save(graph)

    loaded = store.load()
    labels = sorted(n.semantic_label for n in loaded.nodes.values())
    assert labels == ["chair", "table"]
    store.close()


def test_sqlite_store_persists_new_nodes_without_explicit_mark_dirty(tmp_path):
    """Regression test for v0.2.x SQLite bug: after the first save, the
    store would silently skip newly-spawned nodes if mark_dirty() was not
    called. v0.3 auto-detects new node_ids by diffing against the set of
    nodes already persisted, so even forgetful callers stay consistent.
    """
    db_path = tmp_path / "scene_graph.sqlite"
    store = SqliteSceneGraphStore(db_path)

    graph = SceneGraph()
    graph.upsert_object("cup", [0, 0, 1], 0, 0, attributes={"color": "red"})
    store.save(graph)
    assert len(store.load().nodes) == 1

    # Add a new node, do NOT call mark_dirty, save again.
    graph.upsert_object("cup", [3, 0, 1], 1, 1, attributes={"color": "red"})
    store.save(graph)
    loaded = store.load()
    assert len(loaded.nodes) == 2, (
        "SqliteSceneGraphStore must auto-detect new node_ids even when "
        "the caller forgot to call mark_dirty()."
    )

    # Deletion should also propagate.
    nid_to_drop = next(iter(graph.nodes))
    del graph.nodes[nid_to_drop]
    store.save(graph)
    assert len(store.load().nodes) == 1
    store.close()
