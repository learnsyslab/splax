"""Shared camera utilities for scripts, tests, and benchmarks."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import RigidTransform


def look_at(eye: np.ndarray, target: np.ndarray, up: tuple | np.ndarray = (0, 1, 0)) -> np.ndarray:
    """World-to-camera OpenCV matrix looking from ``eye`` to ``target``.

    ``up`` picks the world up axis and must not be parallel to the view direction.
    """
    assert np.linalg.norm(target - eye) > 0, "eye and target must differ"
    z = (target - eye) / np.linalg.norm(target - eye)
    x = np.cross(z, up)  # x = right, y = down, so image up (-y) aligns with the world up axis
    x = x / np.linalg.norm(x)
    c2w = np.eye(4)
    c2w[:3, :3] = np.column_stack([x, np.cross(z, x), z])
    c2w[:3, 3] = eye
    return RigidTransform.from_matrix(c2w).inv().as_matrix().astype(np.float32)


def nerf_camera(frame: dict) -> np.ndarray:
    """Convert a NeRF blender camera pose to a world-to-camera view matrix."""
    c2w = np.array(frame["transform_matrix"], np.float64) @ np.diag([1.0, -1.0, -1.0, 1.0])
    return RigidTransform.from_matrix(c2w).inv().as_matrix().astype(np.float32)
