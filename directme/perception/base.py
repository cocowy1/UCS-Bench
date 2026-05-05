from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from directme.geometry.poses import SE3


@dataclass
class VideoFrame:
    index: int
    timestamp: float
    image_path: str | None = None
    image: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ObjectObservation:
    label: str
    track_id: str | None = None
    score: float = 1.0
    bbox_xyxy: tuple[float, float, float, float] | None = None
    mask: Any | None = None
    p_cam: tuple[float, float, float] | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    keyframe_path: str | None = None


@dataclass
class FramePerception:
    frame: VideoFrame
    local_pose: SE3
    intrinsics: np.ndarray | None = None
    depth: np.ndarray | None = None
    objects: list[ObjectObservation] = field(default_factory=list)
    scene_tag: str | None = None


@dataclass
class ChunkPerception:
    chunk_id: int
    frames: list[FramePerception]


class PerceptionBackend(ABC):
    """Adapter interface for depth, pose, detection, segmentation, and tracking.

    Implementations may call SCAL3R, Depth Anything 3, YOLO-World, SAM 2, MASA,
    or any alternative stack. The mapping engine only requires FramePerception.
    """

    @abstractmethod
    def process_chunk(self, frames: list[VideoFrame], chunk_id: int) -> ChunkPerception:
        raise NotImplementedError
