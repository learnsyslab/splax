"""Parity of ``splax.project`` against ``gsplat.fully_fused_projection`` (forward).

gsplat is a different CUDA kernel from splax's Warp port, so the two cannot agree bit-for-bit the
way a faithful port would. They DO share the projection math (EWA covariance, the same 0.3 px eps2d
dilation, the same pinhole intrinsics), so for every gaussian visible in both the projected
quantities match to a tight numeric tolerance:

  - xys (means2d): pixel coordinates, close to sub-pixel.
  - depths: camera-space z, essentially identical.
  - conics: inverse projected-2D-covariance (a, b, c), close.

Integer tile counts (radii / n_tiles_hit) are NOT compared. gsplat returns a per-axis pixel radius
under a different visibility/tiling convention than splax's scalar 3-sigma radius, so only the
visibility they induce is cross-checked (the gaussians each backend keeps agree on the overwhelming
majority). See tests/_gsplat.py for the full list of convention conversions.

Every test here is marked ``gsplat`` and skipped when gsplat is not installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from utils import VIEWMAT, camera, scene

import splax

if TYPE_CHECKING:
    from pathlib import Path
    from types import ModuleType


def _assert_parity(
    splax_out: tuple[jax.Array, ...], ref: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
):
    """Assert the projected quantities of a splax and a gsplat call agree where both keep them."""
    xys_s, depths_s, radii_s, conics_s = (np.asarray(x) for x in splax_out[:4])
    radii_g, xys_g, depths_g, conics_g = ref

    vis_s = radii_s.ravel() > 0
    vis_g = (radii_g > 0).all(-1)

    # The two visibility sets agree on the vast majority of gaussians (the few that
    # differ sit exactly on a cull boundary). Cross-check on the intersection.
    mask = vis_s & vis_g
    assert mask.sum() > 0
    agree = np.mean(vis_s == vis_g)
    assert agree > 0.98, f"visibility agreement only {agree:.3%}"

    np.testing.assert_allclose(xys_s[mask], xys_g[mask], atol=2e-3)
    np.testing.assert_allclose(depths_s.ravel()[mask], depths_g.ravel()[mask], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(conics_s[mask], conics_g[mask], atol=2e-3, rtol=1e-3)


@pytest.mark.gsplat
def test_project_vs_gsplat(gsplat_shim: ModuleType):
    means, scales, quats, *_ = scene(10_000, seed=1, dense=True)
    opacities = jnp.full((means.shape[0],), 0.99)
    kw = camera(256, 256)
    splax_out = splax.project(means, scales, quats, VIEWMAT, opacities=opacities, **kw)
    _assert_parity(splax_out, gsplat_shim.project(means, scales, quats, VIEWMAT, **kw))


@pytest.mark.gsplat
def test_project_vs_gsplat_jit(gsplat_shim: ModuleType):
    means, scales, quats, *_ = scene(10_000, seed=2, dense=True)
    opacities = jnp.full((means.shape[0],), 0.99)
    kw = camera(256, 256)
    project = jax.jit(lambda m, s, q, v: splax.project(m, s, q, v, opacities=opacities, **kw))
    splax_out = project(means, scales, quats, VIEWMAT)
    _assert_parity(splax_out, gsplat_shim.project(means, scales, quats, VIEWMAT, **kw))


@pytest.mark.gsplat
def test_project_vs_gsplat_lego(gsplat_shim: ModuleType, lego_meta: dict, lego_ply: Path):
    means, scales, quats, _colors, _opacities = splax.io.load_ply(lego_ply)
    means, scales, quats = means[:50_000], scales[:50_000], quats[:50_000]
    H = W = 800
    focal = float(0.5 * W / np.tan(0.5 * lego_meta["camera_angle_x"]))
    viewmat = jnp.asarray(splax.utils.nerf_camera(lego_meta["frames"][0]["transform_matrix"]))
    kw = {"img_shape": (H, W), "f": (focal, focal), "c": (W // 2, H // 2)}
    opacities = jnp.full((means.shape[0],), 0.99)
    splax_out = splax.project(means, scales, quats, viewmat, opacities=opacities, **kw)
    _assert_parity(splax_out, gsplat_shim.project(means, scales, quats, viewmat, **kw))
