"""Tests for the pose-drift telemetry recorded in v0.4."""

from __future__ import annotations

import numpy as np

from directme.geometry.poses import SE3
from directme.mapping.pose_propagation import ChunkPosePropagator


def _identity_chunk(n: int) -> list[SE3]:
    return [SE3.identity() for _ in range(n)]


def _shifted_chunk(n: int, shift_per_frame: float) -> list[SE3]:
    """A clean linear translation: each frame is `shift_per_frame` further along x."""
    poses = []
    for i in range(n):
        T = np.eye(4)
        T[0, 3] = i * shift_per_frame
        poses.append(SE3(T))
    return poses


def test_drift_telemetry_starts_clean():
    pp = ChunkPosePropagator()
    tele = pp.drift_telemetry()
    assert tele["cumulative_translation_m"] == 0.0
    assert tele["n_chunks_seen"] == 0
    assert tele["n_chunks_rejected"] == 0
    assert tele["warnings"] == []


def test_cumulative_translation_accumulates_across_chunks():
    pp = ChunkPosePropagator()
    # Each chunk moves the camera +1 m along x in the chunk-local frame.
    pp.propagate(0, _shifted_chunk(3, 1.0))  # +2 m
    pp.propagate(1, _shifted_chunk(3, 1.0))  # +2 m more
    tele = pp.drift_telemetry()
    assert tele["n_chunks_seen"] == 2
    assert tele["n_chunks_rejected"] == 0
    assert abs(tele["cumulative_translation_m"] - 4.0) < 1e-6


def test_rejected_chunks_are_recorded():
    pp = ChunkPosePropagator(max_per_frame_jump_m=1.0)
    # Jump of 100 m per frame: must be rejected.
    pp.propagate(0, _shifted_chunk(3, 100.0))
    tele = pp.drift_telemetry()
    assert tele["n_chunks_rejected"] == 1
    assert tele["rejected_chunks"][0][0] == 0
    assert "translation_jump" in tele["rejected_chunks"][0][1]


def test_drift_warning_fires_above_translation_threshold():
    pp = ChunkPosePropagator(
        max_per_frame_jump_m=50.0,             # don't reject the synthetic input
        drift_warning_translation_m=5.0,
    )
    pp.propagate(0, _shifted_chunk(3, 5.0))    # ~10 m of accumulated translation
    tele = pp.drift_telemetry()
    assert tele["cumulative_translation_m"] > 5.0
    assert any(
        "cumulative_translation" in w for w in tele["warnings"]
    ), f"expected cumulative-translation warning, got {tele['warnings']}"


def test_drift_warning_fires_above_rejection_ratio():
    pp = ChunkPosePropagator(
        max_per_frame_jump_m=0.5,
        drift_warning_rejection_ratio=0.5,
    )
    # 1 good chunk + 1 bad chunk = 50 % rejection rate.
    pp.propagate(0, _identity_chunk(3))
    pp.propagate(1, _shifted_chunk(3, 100.0))
    tele = pp.drift_telemetry()
    assert any("rejected" in w for w in tele["warnings"])
