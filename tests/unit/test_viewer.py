"""Viewer construction and pose updates, headless and CPU-only because viser is web-based."""

from __future__ import annotations

import socket

import numpy as np
import pytest
from utils import scene_params

from splax.viewer import Viewer


def test_viewer_roundtrip():
    """Add (jax and numpy inputs), update, and remove splats on a live server."""
    *splats, _ = scene_params(50)

    with socket.socket() as s:  # Get a free port
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    viewer = Viewer(host="127.0.0.1", port=port)
    try:
        viewer.add_splats("splat_numpy", *(np.asarray(s) for s in splats))
        viewer.add_splats("splat_jax", *splats, position=(1.0, 2.0, 3.0))

        pos, wxyz = np.array([0.5, -0.5, 1.0]), np.array([0.0, 0.0, 0.0, 1.0])
        viewer.update_pose("splat_jax", pos, wxyz)
        handle = viewer._handles["splat_jax"]
        np.testing.assert_allclose(handle.position, pos)
        np.testing.assert_allclose(handle.wxyz, wxyz)

        with pytest.raises(KeyError, match="missing_splat"):
            viewer.update_pose("missing_splat", pos, wxyz)

        viewer.remove("splat_numpy")
        assert "splat_numpy" not in viewer._handles
    finally:
        viewer.close()
