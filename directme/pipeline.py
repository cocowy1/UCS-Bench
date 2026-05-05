from __future__ import annotations

from dataclasses import dataclass

from directme.config import DirectMeConfig
from directme.geometry.poses import SE3
from directme.mapping.offline_engine import OfflineMappingEngine
from directme.mapping.scene_graph import SceneGraph
from directme.perception.base import PerceptionBackend, VideoFrame
from directme.qa.generator import AnswerGenerator, RuleBasedAnswerGenerator
from directme.retrieval.retriever import GraphRetriever, RetrievedContext


@dataclass
class DirectMe:
    """High-level DirectMe orchestrator."""

    config: DirectMeConfig
    graph: SceneGraph
    generator: AnswerGenerator

    @classmethod
    def with_empty_graph(
        cls,
        config: DirectMeConfig | None = None,
        generator: AnswerGenerator | None = None,
    ) -> "DirectMe":
        config = config or DirectMeConfig()
        graph = SceneGraph(
            reference_frame=config.world.reference_frame,
            merge_threshold_m=config.mapping.merge_threshold_m,
        )
        return cls(config=config, graph=graph, generator=generator or RuleBasedAnswerGenerator())

    def build_memory(self, frames: list[VideoFrame], backend: PerceptionBackend) -> SceneGraph:
        engine = OfflineMappingEngine(backend=backend, config=self.config, graph=self.graph)
        engine.process_frames(frames)
        self.graph = engine.graph or self.graph
        return self.graph

    def _retriever(self) -> GraphRetriever:
        return GraphRetriever(
            self.graph,
            reachable_radius_m=self.config.retrieval.reachable_radius_m,
            lateral_tolerance_ratio=self.config.retrieval.lateral_tolerance_ratio,
        )

    def retrieve(self, question: str, current_pose: SE3, language: str | None = None) -> RetrievedContext:
        return self._retriever().retrieve(
            question=question,
            current_pose=current_pose,
            top_k=self.config.retrieval.top_k,
            language=language or self.config.retrieval.language,
        )

    def answer(self, question: str, current_pose: SE3, language: str | None = None) -> str:
        context = self.retrieve(question, current_pose, language=language)
        return self.generator.answer(context)
