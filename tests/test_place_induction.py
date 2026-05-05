"""Tests for place node induction."""

from directme.mapping.place_induction import induce_places
from directme.mapping.scene_graph import SceneGraph


def _populate(graph: SceneGraph) -> None:
    # Cluster A around (0, 0, 0): a sofa and two cushions.
    graph.upsert_object("sofa", [0.0, 0.0, 0.0], 0, 0, attributes={"scene_tag": "living room"})
    graph.upsert_object("cushion", [0.5, 0.0, 0.5], 0, 0, attributes={"scene_tag": "living room"})
    graph.upsert_object("cushion", [-0.5, 0.0, 0.3], 0, 0, attributes={"scene_tag": "living room"})

    # Cluster B around (10, 0, 0): fridge + sink + oven.
    graph.upsert_object("fridge", [10.0, 0.0, 0.0], 0, 0, attributes={"scene_tag": "kitchen"})
    graph.upsert_object("sink", [10.5, 0.0, 0.5], 0, 0, attributes={"scene_tag": "kitchen"})
    graph.upsert_object("oven", [11.0, 0.0, 0.0], 0, 0, attributes={"scene_tag": "kitchen"})


def test_induce_places_creates_two_clusters():
    graph = SceneGraph()
    _populate(graph)
    places = induce_places(graph, radius_m=3.0, min_members=2)
    assert len(places) == 2
    labels = {p.label for p in places}
    assert "living room" in labels and "kitchen" in labels


def test_induce_places_assigns_place_id_to_members():
    graph = SceneGraph()
    _populate(graph)
    induce_places(graph, radius_m=3.0, min_members=2)
    place_ids = {n.place_id for n in graph.nodes.values() if n.place_id}
    assert len(place_ids) == 2
    # All nodes near origin share one place_id; all nodes near (10, 0, 0) share another.
    near_origin = [n for n in graph.nodes.values() if abs(n.p_world[0]) < 5]
    near_kitchen = [n for n in graph.nodes.values() if n.p_world[0] >= 5]
    assert len({n.place_id for n in near_origin}) == 1
    assert len({n.place_id for n in near_kitchen}) == 1
    assert near_origin[0].place_id != near_kitchen[0].place_id


def test_induce_places_drops_lonely_nodes():
    graph = SceneGraph()
    graph.upsert_object("door", [50.0, 0.0, 50.0], 0, 0)
    places = induce_places(graph, radius_m=3.0, min_members=2)
    assert places == []
    assert next(iter(graph.nodes.values())).place_id is None
