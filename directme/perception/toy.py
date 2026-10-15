"""Deterministic perception backend for tests and demos.

The script is keyed by frame index. Each frame returns a local chunk pose and a
list of already-localized camera-frame object observations. v2 adds HSV
histograms to verify the upgraded scene graph fusion path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from directme.geometry.poses import SE3
from directme.perception.base import (
    ChunkPerception,
    FramePerception,
    ObjectObservation,
    PerceptionBackend,
    VideoFrame,
)


@dataclass
class ScriptedFrame:
    local_pose: SE3
    objects: list[ObjectObservation]
    scene_tag: str | None = None


class ToyPerceptionBackend(PerceptionBackend):
    def __init__(self, script: dict[int, ScriptedFrame]):
        self.script = script

    def process_chunk(self, frames: list[VideoFrame], chunk_id: int) -> ChunkPerception:
        outputs: list[FramePerception] = []
        for frame in frames:
            spec = self.script.get(frame.index, ScriptedFrame(local_pose=SE3.identity(), objects=[]))
            metadata = {**frame.metadata, "pose_confidence": 1.0}
            outputs.append(
                FramePerception(
                    frame=VideoFrame(
                        index=frame.index,
                        timestamp=frame.timestamp,
                        image_path=frame.image_path,
                        image=None,
                        metadata=metadata,
                    ),
                    local_pose=spec.local_pose,
                    intrinsics=None,
                    depth=None,
                    objects=spec.objects,
                    scene_tag=spec.scene_tag,
                )
            )
        return ChunkPerception(chunk_id=chunk_id, frames=outputs)


def _red_cup_histogram(jitter: float = 0.0) -> list[float]:
    """A normalized histogram peaked at the red hue bin, with small noise."""
    hist = [0.0] * 12
    hist[0] = 0.85 + jitter
    hist[1] = 0.10 - 0.5 * jitter
    hist[11] = 0.05 - 0.5 * jitter
    s = sum(max(0.0, v) for v in hist) or 1.0
    return [max(0.0, v) / s for v in hist]


def build_living_room_kitchen_demo(out_dir: str | Path | None = None) -> tuple[list[VideoFrame], ToyPerceptionBackend]:
    """The running example used in docs and tests.

    World frame starts at the first frame. The wearer moves from a living room
    toward a kitchen. Two red cups are physically distinct and should become two
    graph nodes because their world positions are far apart.
    """
    out = Path(out_dir or "runs/toy/keyframes")
    out.mkdir(parents=True, exist_ok=True)

    keyframes = {}
    for idx, name in {
        0: "living_room_red_cup.txt",
        1: "living_room_walk.txt",
        2: "kitchen_entry.txt",
        3: "kitchen_red_cup.txt",
    }.items():
        path = out / name
        path.write_text(f"synthetic keyframe placeholder {idx}: {name}\n", encoding="utf-8")
        keyframes[idx] = str(path)

    frames = [
        VideoFrame(index=0, timestamp=0.0, image_path=keyframes[0]),
        VideoFrame(index=1, timestamp=10.0, image_path=keyframes[1]),
        VideoFrame(index=2, timestamp=11.0, image_path=keyframes[2]),
        VideoFrame(index=3, timestamp=20.0, image_path=keyframes[3]),
    ]

    script: dict[int, ScriptedFrame] = {
        0: ScriptedFrame(
            local_pose=SE3.identity(),
            scene_tag="living room",
            objects=[
                ObjectObservation(
                    label="cup",
                    track_id="track_living_cup",
                    p_cam=(2.0, 0.0, 3.0),
                    bbox_xyxy=(100.0, 100.0, 220.0, 260.0),
                    attributes={
                        "color": "red",
                        "color_hsv_histogram": _red_cup_histogram(0.0),
                        "is_movable": True,
                        "scene_tag": "living room",
                    },
                    keyframe_path=keyframes[0],
                )
            ],
        ),
        1: ScriptedFrame(
            local_pose=SE3.from_translation([3.0, 0.0, 0.0]),
            scene_tag="living room",
            objects=[],
        ),
        2: ScriptedFrame(
            local_pose=SE3.identity(),
            scene_tag="kitchen",
            objects=[],
        ),
        3: ScriptedFrame(
            local_pose=SE3.from_translation([4.0, 0.0, 0.0]),
            scene_tag="kitchen",
            objects=[
                ObjectObservation(
                    label="cup",
                    track_id="track_kitchen_cup",
                    p_cam=(0.30, 0.0, 0.40),
                    bbox_xyxy=(310.0, 200.0, 410.0, 320.0),
                    attributes={
                        "color": "red",
                        "color_hsv_histogram": _red_cup_histogram(0.01),
                        "is_movable": True,
                        "scene_tag": "kitchen",
                    },
                    keyframe_path=keyframes[3],
                )
            ],
        ),
    }

    return frames, ToyPerceptionBackend(script)
