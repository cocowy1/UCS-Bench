"""Tests for DA3 w2c extrinsics → DirectMe T_world_from_camera conversion."""

import numpy as np

from directme.geometry.poses import SE3
from directme.perception.adapters.depth_anything3 import w2c_3x4_to_T_world_from_camera


def test_identity_w2c_maps_to_identity_T():
    w2c = np.eye(4)[:3, :4]
    se3 = w2c_3x4_to_T_world_from_camera(w2c)
    np.testing.assert_allclose(se3.matrix, np.eye(4), atol=1e-8)


def test_w2c_inverse_gives_camera_in_world():
    # A camera placed at world position (3, 1, 2) looking with identity rotation
    # has w2c = [I | -t].
    t = np.array([3.0, 1.0, 2.0])
    R = np.eye(3)
    w2c_3x4 = np.concatenate([R, (-R @ t)[:, None]], axis=1)
    se3 = w2c_3x4_to_T_world_from_camera(w2c_3x4)
    # Camera origin in camera frame is (0, 0, 0). Mapped through T_world_from_camera
    # it should land at world position t.
    np.testing.assert_allclose(se3.translation, t, atol=1e-8)
    # Round-trip a non-zero camera-frame point.
    p_cam = np.array([0.5, 0.0, 1.0])
    p_world = se3.transform_points(p_cam)
    np.testing.assert_allclose(p_world, t + p_cam, atol=1e-8)
