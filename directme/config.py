from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class StreamConfig:
    fps: float = 1.0
    chunk_seconds: float = 10.0
    chunk_size_frames: int = 10


@dataclass
class WorldConfig:
    reference_frame: str = "Frame_0_World_Origin"


@dataclass
class MappingConfig:
    merge_threshold_m: float = 0.5
    max_observations_per_node: int = 128
    keyframes_per_node: int = 4
    nearest_edge_k: int = 5
    nearest_edge_max_distance_m: float = 2.0
    dynamic_update_alpha: float = 0.70
    static_update_alpha: float = 0.20
    motion_overwrite_threshold_m: float = 0.50
    color_histogram_min_similarity: float = 0.55
    semantic_embedding_min_similarity: float = 0.30
    track_match_max_gap_frames: int = 5
    track_match_max_jump_m: float = 2.0
    max_per_frame_jump_m: float = 5.0
    place_radius_m: float = 3.0
    place_min_members: int = 1

    # Drift telemetry warning thresholds. These values mirror the defaults
    # defined on :class:`directme.mapping.pose_propagation.ChunkPosePropagator`.
    # When the cumulative world-frame translation exceeds ``drift_warning_translation_m``
    # a warning is emitted in the graph metadata. Similarly, when the ratio
    # of rejected chunks to total chunks exceeds ``drift_warning_rejection_ratio``
    # or the cumulative rotational drift exceeds ``drift_warning_rotation_deg``
    # (in degrees) additional warnings are surfaced. Exposing these as
    # configuration fields lets users tune drift sensitivity based on their
    # deployment (e.g. longer videos may warrant tighter thresholds).
    drift_warning_translation_m: float = 100.0
    drift_warning_rejection_ratio: float = 0.10
    drift_warning_rotation_deg: float = 90.0


@dataclass
class RetrievalConfig:
    top_k: int = 8
    language: str = "zh"
    include_keyframes: bool = True
    reachable_radius_m: float = 5.0          # depth-based reachability threshold
    lateral_tolerance_ratio: float = 0.20    # cone width for "front"/"behind" centering


@dataclass
class StorageConfig:
    backend: str = "json"   # one of: "json", "sqlite"
    sqlite_filename: str = "scene_graph.sqlite"
    json_filename: str = "scene_graph.json"


@dataclass
class DirectMeConfig:
    stream: StreamConfig = field(default_factory=StreamConfig)
    world: WorldConfig = field(default_factory=WorldConfig)
    mapping: MappingConfig = field(default_factory=MappingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    run_dir: str = "runs/default"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DirectMeConfig":
        project = data.get("project", {})
        return cls(
            stream=StreamConfig(**data.get("stream", {})),
            world=WorldConfig(**data.get("world", {})),
            mapping=MappingConfig(**data.get("mapping", {})),
            retrieval=RetrievalConfig(**data.get("retrieval", {})),
            storage=StorageConfig(**data.get("storage", {})),
            run_dir=project.get("run_dir", data.get("run_dir", "runs/default")),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DirectMeConfig":
        with Path(path).open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stream": self.stream.__dict__,
            "world": self.world.__dict__,
            "mapping": self.mapping.__dict__,
            "retrieval": self.retrieval.__dict__,
            "storage": self.storage.__dict__,
            "run_dir": self.run_dir,
        }
