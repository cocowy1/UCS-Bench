"""Tests for ChunkPosePropagator failure handling."""

import numpy as np

from directme.geometry.poses import SE3
from directme.mapping.pose_propagation import (
    ChunkPosePropagator,
    is_valid_se3,
    max_translation_jump,
)


def _bad_rotation_se3():
    bad = np.eye(4)
    bad[:3, :3] = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=float)  # not orthogonal
    return SE3(bad)


def _nan_se3():
    bad = np.eye(4)
    bad[0, 3] = np.nan
    return SE3(bad)


def test_is_valid_se3_rejects_bad_rotation_and_nan():
    assert is_valid_se3(SE3.identity())
    assert not is_valid_se3(_bad_rotation_se3())
    assert not is_valid_se3(_nan_se3())


def test_max_translation_jump():
    poses = [
        SE3.from_translation([0, 0, 0]),
        SE3.from_translation([0.5, 0, 0]),
        SE3.from_translation([0.6, 0, 0]),
    ]
    jump = max_translation_jump(poses)
    assert abs(jump - 0.5) < 1e-9


def test_chunk_propagator_rejects_invalid_chunk_without_corrupting_anchor():
    prop = ChunkPosePropagator()
    good = [SE3.identity(), SE3.from_translation([1, 0, 0])]
    res1 = prop.propagate(0, good)
    assert res1.accepted
    anchor_before = prop.current_world_end.translation.copy()

    bad = [SE3.identity(), _nan_se3()]
    res2 = prop.propagate(1, bad)
    assert not res2.accepted
    assert res2.rejection_reason and "invalid_se3" in res2.rejection_reason
    np.testing.assert_array_equal(prop.current_world_end.translation, anchor_before)


def test_chunk_propagator_rejects_implausible_jump():
    prop = ChunkPosePropagator(max_per_frame_jump_m=2.0)
    too_far = [SE3.identity(), SE3.from_translation([10.0, 0, 0])]
    res = prop.propagate(0, too_far)
    assert not res.accepted
    assert "translation_jump" in (res.rejection_reason or "")
