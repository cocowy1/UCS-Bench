"""Offline asynchronous mapping engine.

Phase-1 of DirectMe: consume a stream of frames in fixed-time chunks and
incrementally maintain the scene graph memory.

Robustness upgrades:
  * Skips chunks rejected by :class:`ChunkPosePropagator` (NaN / bad rotation /
    excessive jump) without polluting the world frame.
  * Honors per-frame ``pose_confidence`` from the perception backend.
  * Periodically runs place induction to keep object↔place edges fresh.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Iterable
from typing import Any

import numpy as np

from directme.config import DirectMeConfig
from directme.geometry.unprojection import unproject_bbox_center, unproject_mask_centroid
from directme.mapping.pose_propagation import ChunkPosePropagator, PosePropagationResult
from directme.retrieval.pose_lookup import pose_record_from_se3
from directme.mapping.place_induction import induce_places
from directme.mapping.scene_graph import SceneGraph
from directme.perception.base import (
    ChunkPerception,
    FramePerception,
    ObjectObservation,
    PerceptionBackend,
    VideoFrame,
)
from directme.storage.json_store import JsonSceneGraphStore


@dataclass
class MappingEvent:
    chunk_id: int
    frame_index: int
    node_id: str
    action: str
    label: str
    p_world: list[float]


@dataclass
class ChunkReport:
    chunk_id: int
    accepted: bool
    rejection_reason: str | None
    n_events: int


@dataclass
class OfflineMappingEngine:
    backend: PerceptionBackend
    config: DirectMeConfig = field(default_factory=DirectMeConfig)
    graph: SceneGraph | None = None
    store: Any | None = None  # any object exposing .save(graph)
    place_induction_every_n_chunks: int = 1
    chunk_reports: list[ChunkReport] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.graph is None:
            self.graph = SceneGraph(
                reference_frame=self.config.world.reference_frame,
                merge_threshold_m=self.config.mapping.merge_threshold_m,
                max_observations_per_node=self.config.mapping.max_observations_per_node,
                keyframes_per_node=self.config.mapping.keyframes_per_node,
                dynamic_update_alpha=self.config.mapping.dynamic_update_alpha,
                static_update_alpha=self.config.mapping.static_update_alpha,
                motion_overwrite_threshold_m=self.config.mapping.motion_overwrite_threshold_m,
                color_histogram_min_similarity=self.config.mapping.color_histogram_min_similarity,
                semantic_embedding_min_similarity=self.config.mapping.semantic_embedding_min_similarity,
                track_match_max_gap_frames=self.config.mapping.track_match_max_gap_frames,
                track_match_max_jump_m=self.config.mapping.track_match_max_jump_m,
            )
        # Construct the pose propagator with user-defined drift thresholds.  The
        # defaults on ChunkPosePropagator mirror those in MappingConfig,
        # so passing them explicitly makes the behaviour transparent and
        # configurable from YAML.
        self.pose_propagator = ChunkPosePropagator(
            max_per_frame_jump_m=self.config.mapping.max_per_frame_jump_m,
            drift_warning_translation_m=self.config.mapping.drift_warning_translation_m,
            drift_warning_rejection_ratio=self.config.mapping.drift_warning_rejection_ratio,
            drift_warning_rotation_deg=self.config.mapping.drift_warning_rotation_deg,
        )
        if self.store is None:
            run_dir = Path(self.config.run_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
            backend = self.config.storage.backend.lower()
            if backend == "json":
                self.store = JsonSceneGraphStore(run_dir / self.config.storage.json_filename)
            elif backend == "sqlite":
                # Lazy import keeps json-only deployments free of sqlite_store's
                # numpy import path side effects (and prevents accidental coupling).
                from directme.storage.sqlite_store import SqliteSceneGraphStore
                self.store = SqliteSceneGraphStore(run_dir / self.config.storage.sqlite_filename)
            else:
                raise ValueError(
                    f"Unknown storage.backend {backend!r}; expected 'json' or 'sqlite'."
                )
        self._chunk_counter = 0

    def _chunk_frames(self, frames: Iterable[VideoFrame], chunk_size: int) -> Iterable[list[VideoFrame]]:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        chunk: list[VideoFrame] = []
        for frame in frames:
            chunk.append(frame)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

    def process_frames(self, frames: Iterable[VideoFrame], chunk_size: int | None = None) -> list[MappingEvent]:
        """Process a frame iterable in fixed-size offline-incremental chunks.

        ``frames`` may be a generator produced by video decoding; the method no
        longer requires materializing the whole video before perception starts.
        A ``chunk_size`` of 60 means the backend is called once per 60 sampled
        frames, with a shorter final chunk if needed.
        """
        chunk_size = chunk_size or self.config.stream.chunk_size_frames
        events: list[MappingEvent] = []
        for chunk_id, chunk in enumerate(self._chunk_frames(frames, chunk_size)):
            events.extend(self.process_chunk(chunk, chunk_id=chunk_id))
        finalizer = getattr(self.backend, "finalize", None)
        if callable(finalizer):
            finalizer()
        return events

    def process_chunk(self, frames: list[VideoFrame], chunk_id: int) -> list[MappingEvent]:
        assert self.graph is not None
        assert self.store is not None

        perception = self.backend.process_chunk(frames=frames, chunk_id=chunk_id)
        events, report = self._fuse_chunk(perception)
        self.chunk_reports.append(report)

        # Record the per-chunk ego pose for trajectory memory. This is the
        # last accepted T_world_from_camera within this chunk; combined with
        # the chunk's median timestamp and its dominant scene_tag, it forms a
        # cheap "ego_pose_timeline" + "place_visit_timeline" that downstream
        # trajectory queries can read without re-running perception.
        if report.accepted and perception.frames:
            self._update_ego_timeline(perception)

        if report.accepted:
            self._chunk_counter += 1
            if self._chunk_counter % self.place_induction_every_n_chunks == 0:
                induce_places(
                    self.graph,
                    radius_m=self.config.mapping.place_radius_m,
                    min_members=self.config.mapping.place_min_members,
                )
                self.graph.build_edges(
                    k=self.config.mapping.nearest_edge_k,
                    max_distance_m=self.config.mapping.nearest_edge_max_distance_m,
                )

        # Mark every node touched by this chunk as dirty so the SQLite store
        # picks them up incrementally. (The store also auto-detects newly
        # created and deleted node_ids defensively, but explicit mark_dirty()
        # is what flags *in-place updates* — e.g. EMA position refinement of
        # an already-known node — for re-persistence.)
        if hasattr(self.store, "mark_dirty"):
            touched_ids = {ev.node_id for ev in events}
            if touched_ids:
                self.store.mark_dirty(touched_ids)

        # v0.4: write pose-drift telemetry into graph.metadata so downstream
        # consumers (trajectory evaluator, viz, user) can flag suspicious
        # portions of the trajectory. We do NOT correct drift here — that
        # would require loop closure and is intentionally out of scope.
        self.graph.metadata["drift_telemetry"] = self.pose_propagator.drift_telemetry()

        self.store.save(self.graph)
        return events

    def _update_ego_timeline(self, perception: ChunkPerception) -> None:
        """Append per-frame (timestamp, T_world_from_camera, scene_tag) entries."""
        assert self.graph is not None
        world_poses = getattr(self, "_last_accepted_world_poses", [])
        if not world_poses:
            world_poses = [self.pose_propagator.current_world_end] * len(perception.frames)

        ego_pose_timeline = self.graph.metadata.setdefault("ego_pose_timeline", [])
        place_visit_timeline = self.graph.metadata.setdefault("place_visit_timeline", [])

        for fp, pose in zip(perception.frames, world_poses):
            scene_tag = fp.scene_tag
            ego_pose_timeline.append(
                pose_record_from_se3(
                    pose,
                    timestamp=fp.frame.timestamp,
                    chunk_id=perception.chunk_id,
                    frame_index=fp.frame.index,
                    scene_tag=scene_tag,
                )
            )
            # Compress visits: only append when the scene_tag changes.
            if scene_tag is not None:
                if not place_visit_timeline or place_visit_timeline[-1].get("scene_tag") != scene_tag:
                    place_visit_timeline.append(
                        {
                            "chunk_id": perception.chunk_id,
                            "frame_index": fp.frame.index,
                            "timestamp": fp.frame.timestamp,
                            "scene_tag": scene_tag,
                        }
                    )

    def _fuse_chunk(self, perception: ChunkPerception) -> tuple[list[MappingEvent], ChunkReport]:
        assert self.graph is not None

        local_poses = [fp.local_pose for fp in perception.frames]
        pose_result: PosePropagationResult = self.pose_propagator.propagate(
            perception.chunk_id, local_poses
        )
        if not pose_result.accepted:
            self._last_accepted_world_poses = []
            return [], ChunkReport(
                chunk_id=perception.chunk_id,
                accepted=False,
                rejection_reason=pose_result.rejection_reason,
                n_events=0,
            )

        self._last_accepted_world_poses = list(pose_result.world_poses)
        events: list[MappingEvent] = []
        for frame_perception, world_pose in zip(perception.frames, pose_result.world_poses):
            pose_conf = float(frame_perception.frame.metadata.get("pose_confidence", 1.0))
            for obj in frame_perception.objects:
                p_cam = self._object_camera_point(obj, frame_perception)
                if p_cam is None:
                    continue
                if not np.all(np.isfinite(p_cam)):
                    continue
                p_world = world_pose.transform_points(p_cam)
                node, action = self.graph.upsert_object(
                    label=obj.label,
                    p_world=p_world,
                    p_cam=p_cam,
                    timestamp=frame_perception.frame.timestamp,
                    frame_index=frame_perception.frame.index,
                    track_id=obj.track_id,
                    attributes=obj.attributes,
                    confidence=obj.score,
                    pose_confidence=pose_conf,
                    keyframe_path=obj.keyframe_path or frame_perception.frame.image_path,
                    bbox_xyxy=obj.bbox_xyxy,
                )
                events.append(
                    MappingEvent(
                        chunk_id=perception.chunk_id,
                        frame_index=frame_perception.frame.index,
                        node_id=node.node_id,
                        action=action,
                        label=obj.label,
                        p_world=np.asarray(p_world, dtype=float).tolist(),
                    )
                )
        return events, ChunkReport(
            chunk_id=perception.chunk_id,
            accepted=True,
            rejection_reason=None,
            n_events=len(events),
        )

    @staticmethod
    def _object_camera_point(obj: ObjectObservation, fp: FramePerception) -> np.ndarray | None:
        if obj.p_cam is not None:
            return np.asarray(obj.p_cam, dtype=float).reshape(3)
        if fp.depth is not None and fp.intrinsics is not None:
            if obj.mask is not None:
                try:
                    return unproject_mask_centroid(obj.mask, fp.depth, fp.intrinsics)
                except ValueError:
                    pass
            if obj.bbox_xyxy is not None:
                try:
                    return unproject_bbox_center(obj.bbox_xyxy, fp.depth, fp.intrinsics)
                except ValueError:
                    return None
        return None
