from __future__ import annotations

from pathlib import Path

from directme.config import DirectMeConfig
from directme.geometry.poses import SE3
from directme.mapping.offline_engine import OfflineMappingEngine
from directme.perception.toy import build_living_room_kitchen_demo
from directme.qa.generator import RuleBasedAnswerGenerator
from directme.retrieval.retriever import GraphRetriever


def main() -> None:
    out = Path("runs/toy")
    config = DirectMeConfig()
    config.run_dir = str(out)
    frames, backend = build_living_room_kitchen_demo(out / "keyframes")

    engine = OfflineMappingEngine(backend=backend, config=config)
    events = engine.process_frames(frames, chunk_size=2)
    graph = engine.graph
    assert graph is not None

    current_pose = SE3.from_translation([7.0, 0.0, 0.0])
    question = "我身边有几个红杯子？在哪？"
    context = GraphRetriever(graph).retrieve(question, current_pose, language="zh")
    answer = RuleBasedAnswerGenerator().answer(context)

    print(f"events={len(events)} nodes={len(graph.nodes)}")
    print(answer)


if __name__ == "__main__":
    main()
