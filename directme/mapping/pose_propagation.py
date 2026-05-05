"""Chunk-level pose propagation with failure detection.

Beyond the basic alignment formula, this module:
  * validates each local pose for NaNs / Infs / non-orthogonal rotations,
  * checks for implausibly large per-frame translation jumps,
  * skips bad chunks gracefully (returns the previous anchor unchanged) so the
    world frame is never corrupted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from directme.geometry.poses import SE3, propagate_chunk_local_poses


@dataclass
class PosePropagationResult:
    chunk_id: int
    world_poses: list[SE3]
    previous_anchor: SE3
    new_anchor: SE3
    accepted: bool
    rejection_reason: str | None = None


def is_valid_se3(pose: SE3, *, rotation_tol: float = 1e-3) -> bool:
    """Validate finiteness and rotation orthogonality."""
    m = pose.matrix
    if not np.all(np.isfinite(m)):
        return False
    R = m[:3, :3]
    err = np.linalg.norm(R @ R.T - np.eye(3))
    if err > rotation_tol:
        return False
    det = np.linalg.det(R)
    return 0.5 < det < 1.5


def max_translation_jump(local_poses: Sequence[SE3]) -> float:
    if len(local_poses) < 2:
        return 0.0
    translations = np.stack([p.translation for p in local_poses], axis=0)
    diffs = np.linalg.norm(np.diff(translations, axis=0), axis=1)
    return float(np.max(diffs)) if diffs.size else 0.0


class ChunkPosePropagator:
    """Maintains the world-frame camera pose across asynchronous chunks.

    v0.4 additions
    --------------
    The propagator records lightweight per-chunk **drift telemetry** so the
    rest of the pipeline can surface a "your trajectory may be unreliable
    after chunk N" warning to the user. The signals tracked are:

    * ``cumulative_translation_m`` — total Manhattan-style path length the
      camera has covered in the world frame (sum of per-chunk
      ``previous → new_anchor`` translations).
    * ``rejected_chunks`` — list of ``(chunk_id, reason)`` tuples for chunks
      whose local poses we refused to integrate.
    * ``per_chunk_jump_m`` — the largest per-frame translation jump within
      each chunk.

    These do **not** correct drift — that would require a full loop-closure
    / pose-graph step which is intentionally out of scope. They give the
    downstream evaluator and the user enough information to flag suspicious
    portions of the trajectory.
    """

    def __init__(
        self,
        initial_world_pose: SE3 | None = None,
        max_per_frame_jump_m: float = 5.0,
        drift_warning_translation_m: float = 100.0,
        drift_warning_rejection_ratio: float = 0.10,
        drift_warning_rotation_deg: float = 90.0,
    ):
        """Initialize a new pose propagator.

        Args:
            initial_world_pose: starting world-frame pose; defaults to identity.
            max_per_frame_jump_m: per-chunk per-frame translation jump threshold in meters.
            drift_warning_translation_m: cumulative translation threshold beyond which drift warnings
                are emitted. See :meth:`drift_telemetry`.
            drift_warning_rejection_ratio: ratio of rejected chunks beyond which drift warnings
                are emitted.
            drift_warning_rotation_deg: cumulative rotational drift threshold (in degrees).

        In v0.5 the propagator additionally tracks rotational drift. Large rotations between
        successive chunk anchors may indicate pose estimation instabilities (e.g. large yaw
        flips when the user turns around). When the cumulative rotation exceeds
        ``drift_warning_rotation_deg`` degrees a warning is emitted in the telemetry. This
        complements the existing translation-based drift signals and gives users better
        diagnostics of pose reliability over long sequences.
        """
        self.current_world_end = initial_world_pose or SE3.identity()
        self.max_per_frame_jump_m = max_per_frame_jump_m
        self.drift_warning_translation_m = drift_warning_translation_m
        self.drift_warning_rejection_ratio = drift_warning_rejection_ratio
        self.drift_warning_rotation_deg = drift_warning_rotation_deg

        # Telemetry state.
        self.cumulative_translation_m: float = 0.0
        self.cumulative_rotation_deg: float = 0.0
        self.rejected_chunks: list[tuple[int, str]] = []
        self.per_chunk_jump_m: list[tuple[int, float]] = []
        self.per_chunk_rotation_deg: list[tuple[int, float]] = []
        self.n_chunks_seen: int = 0

    def reset(self, pose: SE3 | None = None) -> None:
        self.current_world_end = pose or SE3.identity()
        self.cumulative_translation_m = 0.0
        self.cumulative_rotation_deg = 0.0
        self.rejected_chunks.clear()
        self.per_chunk_jump_m.clear()
        self.per_chunk_rotation_deg.clear()
        self.n_chunks_seen = 0

    def drift_telemetry(self) -> dict:
        """Return a JSON-serializable snapshot of drift indicators.

        The ``warnings`` list contains human-readable strings the engine can
        push into ``graph.metadata["drift_warnings"]`` for the trajectory
        evaluator and the user.
        """
        warnings: list[str] = []
        # Translation-based drift warning
        if self.cumulative_translation_m >= self.drift_warning_translation_m:
            warnings.append(
                f"cumulative_translation={self.cumulative_translation_m:.1f}m "
                f"exceeds threshold {self.drift_warning_translation_m:.1f}m; "
                f"world-frame drift is likely without loop closure."
            )
        # Rotation-based drift warning
        if self.cumulative_rotation_deg >= self.drift_warning_rotation_deg:
            warnings.append(
                f"cumulative_rotation={self.cumulative_rotation_deg:.1f}° exceeds "
                f"threshold {self.drift_warning_rotation_deg:.1f}°; orientation drift may be accumulating."
            )
        # Rejection ratio warning
        if self.n_chunks_seen > 0:
            ratio = len(self.rejected_chunks) / self.n_chunks_seen
            if ratio >= self.drift_warning_rejection_ratio:
                warnings.append(
                    f"{len(self.rejected_chunks)}/{self.n_chunks_seen} "
                    f"chunks were rejected ({ratio:.1%}); pose backend is "
                    f"unstable on this stream."
                )
        return {
            "cumulative_translation_m": float(self.cumulative_translation_m),
            "cumulative_rotation_deg": float(self.cumulative_rotation_deg),
            "n_chunks_seen": int(self.n_chunks_seen),
            "n_chunks_rejected": len(self.rejected_chunks),
            "rejected_chunks": list(self.rejected_chunks),
            "per_chunk_jump_m": list(self.per_chunk_jump_m),
            "per_chunk_rotation_deg": list(self.per_chunk_rotation_deg),
            "warnings": warnings,
        }

    def propagate(self, chunk_id: int, local_poses: list[SE3]) -> PosePropagationResult:
        previous = self.current_world_end.copy()
        self.n_chunks_seen += 1

        if not local_poses:
            self.rejected_chunks.append((chunk_id, "empty_chunk"))
            return PosePropagationResult(
                chunk_id=chunk_id,
                world_poses=[],
                previous_anchor=previous,
                new_anchor=previous,
                accepted=False,
                rejection_reason="empty_chunk",
            )

        for i, p in enumerate(local_poses):
            if not is_valid_se3(p):
                reason = f"invalid_se3_at_index_{i}"
                self.rejected_chunks.append((chunk_id, reason))
                return PosePropagationResult(
                    chunk_id=chunk_id,
                    world_poses=[],
                    previous_anchor=previous,
                    new_anchor=previous,
                    accepted=False,
                    rejection_reason=reason,
                )

        jump = max_translation_jump(local_poses)
        self.per_chunk_jump_m.append((chunk_id, float(jump)))
        if jump > self.max_per_frame_jump_m:
            reason = f"translation_jump_{jump:.2f}m"
            self.rejected_chunks.append((chunk_id, reason))
            return PosePropagationResult(
                chunk_id=chunk_id,
                world_poses=[],
                previous_anchor=previous,
                new_anchor=previous,
                accepted=False,
                rejection_reason=reason,
            )

        world_poses = propagate_chunk_local_poses(previous, local_poses)
        if not world_poses:
            self.rejected_chunks.append((chunk_id, "empty_world_poses"))
            return PosePropagationResult(
                chunk_id=chunk_id,
                world_poses=[],
                previous_anchor=previous,
                new_anchor=previous,
                accepted=False,
                rejection_reason="empty_world_poses",
            )

        self.current_world_end = world_poses[-1].copy()
        # Telemetry: how far did the world-frame anchor move in this chunk?
        delta = float(np.linalg.norm(self.current_world_end.translation - previous.translation))
        self.cumulative_translation_m += delta
        # Telemetry: how much did the world-frame orientation rotate in this chunk?
        try:
            R_prev = previous.matrix[:3, :3]
            R_curr = self.current_world_end.matrix[:3, :3]
            # Compute relative rotation: R_rel = R_prev^T * R_curr
            R_rel = R_prev.T @ R_curr
            # Rotation angle from trace formula. Clamp value for numerical stability.
            trace = float(np.trace(R_rel))
            cosine = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
            angle_rad = float(np.arccos(cosine))
            angle_deg = float(np.degrees(angle_rad))
        except Exception:
            angle_deg = 0.0
        self.cumulative_rotation_deg += abs(angle_deg)
        self.per_chunk_rotation_deg.append((chunk_id, angle_deg))

        return PosePropagationResult(
            chunk_id=chunk_id,
            world_poses=world_poses,
            previous_anchor=previous,
            new_anchor=self.current_world_end.copy(),
            accepted=True,
        )
