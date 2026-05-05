from __future__ import annotations

from pathlib import Path

from directme.mapping.scene_graph import SceneGraph


class JsonSceneGraphStore:
    """Atomic JSON persistence for the scene graph."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def save(self, graph: SceneGraph) -> None:
        graph.save_json(self.path)

    def load(self) -> SceneGraph:
        return SceneGraph.load_json(self.path)

    def exists(self) -> bool:
        return self.path.exists()
