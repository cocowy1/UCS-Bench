from directme.config import DirectMeConfig
from directme.mapping.offline_engine import OfflineMappingEngine
from directme.perception.toy import build_living_room_kitchen_demo


def test_toy_offline_engine_builds_two_cups(tmp_path):
    frames, backend = build_living_room_kitchen_demo(tmp_path / "keyframes")
    cfg = DirectMeConfig()
    cfg.run_dir = str(tmp_path)
    engine = OfflineMappingEngine(backend=backend, config=cfg)
    events = engine.process_frames(frames, chunk_size=2)

    assert len(events) == 2
    assert engine.graph is not None
    assert len(engine.graph.nodes) == 2
    assert (tmp_path / "scene_graph.json").exists()
