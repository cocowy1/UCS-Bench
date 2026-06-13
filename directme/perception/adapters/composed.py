"""Composed real perception backend: DA3 + YOLO-World + SAM 2 + tracking.

Implements :class:`PerceptionBackend` end-to-end on real images. Heavy deps are
imported lazily through the individual adapters, so importing this module is
cheap when the adapters are not actually instantiated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from directme.geometry.poses import SE3
from directme.perception.adapters.depth_anything3 import DepthAnything3Adapter
from directme.perception.adapters.open_vocab_tracking import (
    Detection,
    OpenVocabularyTrackingAdapter,
)
from directme.perception.adapters.scene_classifier import (
    SceneClassifier,
    RuleBasedSceneClassifier,
    create_scene_classifier,
)
from directme.perception.base import (
    ChunkPerception,
    FramePerception,
    ObjectObservation,
    PerceptionBackend,
    VideoFrame,
)
from directme.perception.color_attributes import dominant_hsv_color, hsv_histogram_from_image_mask


def _load_image_rgb(path: str) -> np.ndarray:
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "ComposedRealBackend requires `opencv-python`. "
            "Install with `pip install directme[video]`."
        ) from exc
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


@dataclass
class ComposedRealBackend(PerceptionBackend):
    """A real :class:`PerceptionBackend` that runs DA3 + open-vocab tracking.

    Args:
        depth: a :class:`DepthAnything3Adapter` instance.
        tracker: an :class:`OpenVocabularyTrackingAdapter` instance configured
            with a YOLO-World detector and (optionally) a SAM 2 mask refiner.
        min_pose_confidence: chunks below this mean DA3 confidence are kept
            but flagged with ``low_confidence=True`` in metadata so the
            mapping engine can downweight them.
    """

    depth: DepthAnything3Adapter
    tracker: OpenVocabularyTrackingAdapter
    min_pose_confidence: float = 0.30
    color_hist_bins: int = 12

    # Scene classifier used to infer coarse room/landmark tags. Default to
    # rule-based classifier. Users may set this attribute to a different
    # implementation (e.g. QwenSceneClassifier) prior to processing. This
    # attribute must be defined before methods, hence declared here.
    scene_classifier: SceneClassifier = field(
        default_factory=RuleBasedSceneClassifier
    )

    def process_chunk(self, frames: list[VideoFrame], chunk_id: int) -> ChunkPerception:
        if not frames:
            return ChunkPerception(chunk_id=chunk_id, frames=[])

        image_paths = [f.image_path for f in frames if f.image_path]
        if len(image_paths) != len(frames):
            raise ValueError("ComposedRealBackend requires every VideoFrame to have image_path set")

        da3_outputs = self.depth.infer(image_paths)

        outputs: list[FramePerception] = []
        for frame, da3 in zip(frames, da3_outputs):
            image = _load_image_rgb(frame.image_path)  # type: ignore[arg-type]
            detections = self.tracker.step(frame_index=frame.index, image=image)

            objects: list[ObjectObservation] = []
            for det in detections:
                attrs: dict[str, Any] = {}
                if det.mask is not None:
                    attrs["color_hsv_histogram"] = hsv_histogram_from_image_mask(
                        image, det.mask, bins=self.color_hist_bins
                    )
                    attrs["color"] = dominant_hsv_color(image, det.mask)
                objects.append(
                    ObjectObservation(
                        label=det.label,
                        track_id=det.track_id,
                        score=det.score,
                        bbox_xyxy=det.bbox_xyxy,
                        mask=det.mask,
                        attributes=attrs,
                        keyframe_path=frame.image_path,
                    )
                )

            # Infer a coarse scene tag from detected object labels. If the
            # classifier supports using the full image (e.g. QwenSceneClassifier),
            # pass the image as well. If an error occurs (e.g. missing heavy
            # dependencies), fall back to a rule-based classifier. Note: even if
            # there are no detections, we still pass an empty label list to the
            # classifier; some implementations may use the image alone.
            try:
                scene_tag = self.scene_classifier(image, [det.label for det in detections])
            except Exception:
                fallback = RuleBasedSceneClassifier()
                scene_tag = fallback(image, [det.label for det in detections])

            metadata = {
                "pose_confidence": da3.pose_confidence,
                "low_confidence": da3.pose_confidence < self.min_pose_confidence,
            }
            fp = FramePerception(
                frame=VideoFrame(
                    index=frame.index,
                    timestamp=frame.timestamp,
                    image_path=frame.image_path,
                    image=None,  # not retained to save memory
                    metadata={**frame.metadata, **metadata},
                ),
                local_pose=da3.pose_local,
                intrinsics=da3.intrinsics,
                depth=da3.depth,
                objects=objects,
                scene_tag=scene_tag,
            )
            outputs.append(fp)

        return ChunkPerception(chunk_id=chunk_id, frames=outputs)





@dataclass
class UnifiedPerceptionBackend(PerceptionBackend):
    """Unified perception backend: DA3/SCAL3R depth-pose + YOLO-World tracking + optional scene tag."""

    depth_backend: str
    depth: Any
    tracker: OpenVocabularyTrackingAdapter
    min_pose_confidence: float = 0.30
    color_hist_bins: int = 12
    enable_scene_tag: bool = True
    scene_classifier: SceneClassifier = field(default_factory=RuleBasedSceneClassifier)

    def process_chunk(self, frames: list[VideoFrame], chunk_id: int) -> ChunkPerception:
        if not frames:
            return ChunkPerception(chunk_id=chunk_id, frames=[])

        image_paths = [f.image_path for f in frames if f.image_path]
        if len(image_paths) != len(frames):
            raise ValueError("UnifiedPerceptionBackend requires every VideoFrame to have image_path set")

        if self.depth_backend == "da3":
            depth_outputs = self.depth.infer(image_paths)
        elif self.depth_backend == "scal3r":
            if hasattr(self.depth, "infer_frames"):
                depth_outputs = self.depth.infer_frames(frames, chunk_id=chunk_id)
            elif hasattr(self.depth, "infer"):
                depth_outputs = self.depth.infer(frames)
            elif hasattr(self.depth, "process_chunk"):
                depth_chunk = self.depth.process_chunk(frames, chunk_id=chunk_id)
                depth_outputs = depth_chunk.frames
            else:
                raise TypeError(
                    "SCAL3R depth adapter must provide infer_frames(), infer(), or process_chunk()"
                )

        else:
            raise ValueError(f"Unknown depth_backend={self.depth_backend}; use da3 or scal3r")

        outputs: list[FramePerception] = []

        for frame, depth_out in zip(frames, depth_outputs):
            image = _load_image_rgb(frame.image_path)  # type: ignore[arg-type]
            detections = self.tracker.step(frame_index=frame.index, image=image)

            objects: list[ObjectObservation] = []
            for det in detections:
                attrs: dict[str, Any] = {}

                if det.mask is not None:
                    attrs["color_hsv_histogram"] = hsv_histogram_from_image_mask(
                        image, det.mask, bins=self.color_hist_bins
                    )
                    attrs["color"] = dominant_hsv_color(image, det.mask)

                objects.append(
                    ObjectObservation(
                        label=det.label,
                        track_id=det.track_id,
                        score=det.score,
                        bbox_xyxy=det.bbox_xyxy,
                        mask=det.mask,
                        attributes=attrs,
                        keyframe_path=frame.image_path,
                    )
                )

            if self.enable_scene_tag:
                try:
                    scene_tag = self.scene_classifier(
                        image,
                        [det.label for det in detections],
                    )
                except Exception:
                    scene_tag = RuleBasedSceneClassifier()(
                        image,
                        [det.label for det in detections],
                    )
            else:
                scene_tag = None

            # DA3Output: depth / intrinsics / pose_local / pose_confidence
            # SCAL3R FramePerception: depth / intrinsics / local_pose
            if self.depth_backend == "da3":
                depth = depth_out.depth
                intrinsics = depth_out.intrinsics
                local_pose = depth_out.pose_local
                pose_conf = float(getattr(depth_out, "pose_confidence", 1.0))
            else:
                depth = depth_out.depth
                intrinsics = depth_out.intrinsics
                local_pose = (
                    depth_out.local_pose
                    if hasattr(depth_out, "local_pose")
                    else depth_out.pose_local
                )
                pose_conf = float(getattr(depth_out, "pose_confidence", 1.0))

            metadata = {
                "depth_backend": self.depth_backend,
                "pose_confidence": pose_conf,
                "low_confidence": pose_conf < self.min_pose_confidence,
            }

            outputs.append(
                FramePerception(
                    frame=VideoFrame(
                        index=frame.index,
                        timestamp=frame.timestamp,
                        image_path=frame.image_path,
                        image=None,
                        metadata={**frame.metadata, **metadata},
                    ),
                    local_pose=local_pose,
                    intrinsics=intrinsics,
                    depth=depth,
                    objects=objects,
                    scene_tag=scene_tag,
                )
            )

        return ChunkPerception(chunk_id=chunk_id, frames=outputs)


def build_unified_perception_backend(
    *,
    depth_backend: str,
    tracker: OpenVocabularyTrackingAdapter,
    device: str = "cuda",
    da3_model_id: str = "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
    da3_process_res: int = 504,
    scal3r_config: str | None = None,
    scal3r_checkpoint: str | None = None,
    scal3r_work_dir: str | None = None,
    precomputed_scal3r_root: str | None = None,
    enable_scene_tag: bool = True,
    scene_classifier: SceneClassifier | None = None,
) -> UnifiedPerceptionBackend:
    depth_backend = depth_backend.lower()

    if depth_backend == "da3":
        from directme.perception.adapters.depth_anything3 import DepthAnything3Adapter

        depth = DepthAnything3Adapter(
            model_id=da3_model_id,
            device=device,
            process_res=da3_process_res,
        )

    elif depth_backend == "scal3r":
        from pathlib import Path
        from directme.perception.adapters.scal3r import (
            Scal3RDepthPoseAdapter,
            Scal3RRunner,
        )

        if not scal3r_config or not scal3r_checkpoint or not scal3r_work_dir:
            raise ValueError(
                "SCAL3R requires scal3r_config, scal3r_checkpoint, and scal3r_work_dir"
            )

        depth = Scal3RDepthPoseAdapter(
            runner=Scal3RRunner(
                config=scal3r_config,
                checkpoint=scal3r_checkpoint,
                device=device,
                save_dpt=1,
                save_xyz=0,
            ),
            precomputed_root=precomputed_scal3r_root or None,
            work_dir=Path(scal3r_work_dir),
            keep_work_dir=True,
        )


    else:
        raise ValueError(f"Unknown depth_backend={depth_backend}; use da3 or scal3r")

    return UnifiedPerceptionBackend(
        depth_backend=depth_backend,
        depth=depth,
        tracker=tracker,
        enable_scene_tag=enable_scene_tag,
        scene_classifier=scene_classifier or RuleBasedSceneClassifier(),
    )
