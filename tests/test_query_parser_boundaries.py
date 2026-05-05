from directme.retrieval.query_parser import parse_query
from directme.geometry.poses import SE3
from directme.mapping.scene_graph import SceneGraph
from directme.retrieval.retriever import GraphRetriever


def test_english_aliases_use_word_boundaries():
    intent = parse_query("Where is the cupboard?")
    assert "cup" not in intent.labels
    assert "cabinet" in intent.labels


def test_retriever_label_matching_does_not_match_cupboard_as_cup():
    graph = SceneGraph()
    graph.upsert_object("cupboard", [0, 0, 1], 0, 0)
    ctx = GraphRetriever(graph).retrieve("where is the cup?", SE3.identity(), language="en")
    assert ctx.count == 0
    assert ctx.items == []
