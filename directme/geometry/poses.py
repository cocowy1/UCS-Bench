from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def _as_se3_matrix(matrix: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    if arr.shape == (16,):
        arr = arr.reshape(4, 4)
    if arr.shape != (4, 4):
        raise ValueError(f"SE3 matrix must have shape (4, 4), got {arr.shape}.")
    if not np.allclose(arr[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-6):
        raise ValueError("SE3 matrix last row must be [0, 0, 0, 1].")
    return arr


@dataclass
class SE3:
    """Small SE(3) helper with explicit homogeneous 4x4 matrices.

    Convention:
        T_world_from_cam transforms camera-frame homogeneous points into the world frame.
    """

    matrix: np.ndarray

    def __post_init__(self) -> None:
        self.matrix = _as_se3_matrix(self.matrix)

    @classmethod
    def identity(cls) -> "SE3":
        return cls(np.eye(4, dtype=float))

    @classmethod
    def from_list(cls, values: Sequence[Sequence[float]] | Sequence[float]) -> "SE3":
        return cls(np.asarray(values, dtype=float))

    @classmethod
    def from_rotation_translation(
        cls, rotation: Sequence[Sequence[float]] | np.ndarray, translation: Sequence[float] | np.ndarray
    ) -> "SE3":
        r = np.asarray(rotation, dtype=float)
        t = np.asarray(translation, dtype=float).reshape(3)
        if r.shape != (3, 3):
            raise ValueError("rotation must have shape (3, 3)")
        mat = np.eye(4, dtype=float)
        mat[:3, :3] = r
        mat[:3, 3] = t
        return cls(mat)

    @classmethod
    def from_translation(cls, translation: Sequence[float] | np.ndarray) -> "SE3":
        return cls.from_rotation_translation(np.eye(3), translation)

    @property
    def rotation(self) -> np.ndarray:
        return self.matrix[:3, :3]

    @property
    def translation(self) -> np.ndarray:
        return self.matrix[:3, 3]

    def inverse(self) -> "SE3":
        r = self.rotation
        t = self.translation
        inv = np.eye(4, dtype=float)
        inv[:3, :3] = r.T
        inv[:3, 3] = -r.T @ t
        return SE3(inv)

    def compose(self, other: "SE3") -> "SE3":
        return SE3(self.matrix @ other.matrix)

    def __matmul__(self, other: "SE3") -> "SE3":
        if not isinstance(other, SE3):
            raise TypeError("SE3 can only compose with SE3 via @. Use transform_points for points.")
        return self.compose(other)

    def transform_points(self, points_xyz: Sequence[float] | np.ndarray) -> np.ndarray:
        pts = np.asarray(points_xyz, dtype=float)
        squeeze = False
        if pts.shape == (3,):
            pts = pts[None, :]
            squeeze = True
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("points_xyz must have shape (3,) or (N, 3)")
        ones = np.ones((pts.shape[0], 1), dtype=float)
        hom = np.concatenate([pts, ones], axis=1)
        out = (self.matrix @ hom.T).T[:, :3]
        return out[0] if squeeze else out

    def to_list(self) -> list[list[float]]:
        return self.matrix.tolist()

    def copy(self) -> "SE3":
        return SE3(self.matrix.copy())


def propagate_chunk_local_poses(previous_world_end: SE3, local_poses: list[SE3]) -> list[SE3]:
    """Propagate local chunk poses into the absolute world frame.

    Robust alignment formula:
        T_world(t) = T_prev_world_end · inverse(T_local(start)) · T_local(t)

    If the first local pose is identity, this reduces to the commonly shown:
        T_world(t) = T_prev_world_end · T_local(t)
    """
    if not local_poses:
        return []
    local_start_inv = local_poses[0].inverse()
    world_from_local = previous_world_end.compose(local_start_inv)
    return [world_from_local.compose(local_pose) for local_pose in local_poses]
