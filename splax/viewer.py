"""Web-based splat viewer built on `viser <https://viser.studio>`_.

``Viewer`` wraps a ``viser.ViserServer`` and exposes splats as named rigid objects. ``add_splats``
uploads the gaussians of one object once (converted to the covariance form viser's browser-side
rasterizer consumes), and ``update_pose`` moves objects afterwards without re-uploading.

Viser is an optional dependency that can be installed with ``pip install splax[viewer]``.
"""

from __future__ import annotations

from typing import Literal

try:
    import viser
except ImportError as e:
    raise ImportError(
        "splax.viewer requires viser. Install it with `pip install splax[viewer]`."
    ) from e

import jax
import numpy as np
from scipy.spatial.transform import Rotation as R


def covariances(
    scales: jax.Array | np.ndarray, quats: jax.Array | np.ndarray
) -> np.ndarray:
    """Covariance matrices of gaussians from render-space scales and quats.

    Args:
        scales: (N, 3) positive per-axis scales.
        quats: (N, 4) wxyz unit quaternions.

    Returns:
        (N, 3, 3) float32 covariances ``rot @ diag(scales**2) @ rot.T``.
    """
    scales = np.asarray(scales, np.float32)
    rot = R.from_quat(np.asarray(quats, np.float32), scalar_first=True).as_matrix()
    return np.einsum("nij,nj,nkj->nik", rot, scales**2, rot).astype(np.float32)


class Viewer:
    """Splat viewer serving a web client, one scene node per rigid object."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8080,
        up_direction: Literal["+x", "+y", "+z", "-x", "-y", "-z"] = "+z",
    ):
        """Start the viser server and configure the scene.

        Args:
            host: Host address to bind the web server to.
            port: Port of the web server.
            up_direction: World up direction, e.g. ``"+z"`` (MuJoCo) or ``"+y"``.
        """
        self._server = viser.ViserServer(host=host, port=port)
        self._server.scene.set_up_direction(up_direction)
        self._handles: dict[str, viser.GaussianSplatHandle] = {}

    @property
    def server(self) -> viser.ViserServer:
        """The underlying viser server, for features beyond splats (gui, meshes, ...)."""
        return self._server

    def add_splats(
        self,
        name: str,
        means: jax.Array | np.ndarray,
        scales: jax.Array | np.ndarray,
        quats: jax.Array | np.ndarray,
        colors: jax.Array | np.ndarray,
        opacities: jax.Array | np.ndarray,
        *,
        position: jax.Array | np.ndarray | tuple[float, float, float] = (0.0, 0.0, 0.0),
        wxyz: jax.Array | np.ndarray | tuple[float, float, float, float] = (
            1.0,
            0.0,
            0.0,
            0.0,
        ),
    ) -> None:
        """Upload one rigid object's gaussians to the viewer under ``name``.

        Args:
            name: Name of the object.
            means: (N, 3) float32 centers in the object frame.
            scales: (N, 3) positive per-axis scales.
            quats: (N, 4) wxyz unit quaternions.
            colors: (N, 3) float32 RGB values.
            opacities: (N, 1) float32 opacity values.
            position: Initial world position of the object.
            wxyz: Initial world orientation of the object as a wxyz quaternion.
        """
        self._handles[name] = self._server.scene.add_gaussian_splats(
            f"/{name}",
            centers=np.asarray(means, np.float32),
            covariances=covariances(scales, quats),
            rgbs=np.asarray(colors, np.float32),
            opacities=np.asarray(opacities, np.float32).reshape(-1, 1),
            position=np.asarray(position, np.float32),
            wxyz=np.asarray(wxyz, np.float32),
        )

    def update_pose(
        self,
        name: str,
        position: jax.Array | np.ndarray,
        wxyz: jax.Array | np.ndarray,
    ) -> None:
        """Set the world pose of the object ``name``.

        Args:
            name: Name of the object.
            position: World position of the object.
            wxyz: World orientation of the object as a wxyz quaternion.
        """
        if (handle := self._handles.get(name)) is None:
            raise KeyError(f"No splats named {name!r}, add them with add_splats first")
        handle.position = np.asarray(position, np.float32)
        handle.wxyz = np.asarray(wxyz, np.float32)

    def remove(self, name: str) -> None:
        """Remove the object ``name`` from the viewer."""
        self._handles.pop(name).remove()

    def close(self) -> None:
        """Stop the web server."""
        self._server.stop()
