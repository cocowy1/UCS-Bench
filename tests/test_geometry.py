import numpy as np

from directme.geometry.poses import SE3, propagate_chunk_local_poses
from directme.geometry.unprojection import backproject_pixel


def test_se3_inverse_roundtrip():
    t = SE3.from_translation([1, 2, 3])
    p = np.array([0.5, 0.0, 1.0])
    out = t.inverse().transform_points(t.transform_points(p))
    assert np.allclose(out, p)


def test_chunk_pose_propagation_with_non_identity_local_start():
    prev = SE3.from_translation([10, 0, 0])
    local_start = SE3.from_translation([2, 0, 0])
    local_end = SE3.from_translation([5, 0, 0])
    world = propagate_chunk_local_poses(prev, [local_start, local_end])
    assert np.allclose(world[0].translation, [10, 0, 0])
    assert np.allclose(world[1].translation, [13, 0, 0])


def test_backproject_pixel_center():
    k = np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1]], dtype=float)
    p = backproject_pixel(50, 50, 2.0, k)
    assert np.allclose(p, [0, 0, 2])
