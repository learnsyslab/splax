"""Test the projection against the gsplat reference, on random scenes and on lego."""

from __future__ import annotations

from functools import partial
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


# gsplat returns a per-axis pixel radius under a different visibility convention than splax's
# scalar 3-sigma radius, so the tile counts are not compared, only the visibility they induce.
def _assert_parity(
    splax_out: tuple[jax.Array, ...], ref: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
):
    """Assert the projected quantities of a splax and a gsplat call agree where both keep them."""
    xys_s, depths_s, radii_s, conics_s = splax_out[:4]
    radii_g, xys_g, depths_g, conics_g = ref

    vis_s = radii_s.ravel() > 0
    vis_g = (radii_g > 0).all(-1)

    # The two visibility sets agree on the vast majority of gaussians (the few that differ sit
    # exactly on a cull boundary). Cross-check on the intersection.
    mask = vis_s & vis_g
    assert mask.sum() > 0
    assert (agree := np.mean(vis_s == vis_g)) > 0.98, f"visibility agreement only {agree:.3%}"
    np.testing.assert_allclose(xys_s[mask], xys_g[mask], atol=2e-3)
    np.testing.assert_allclose(depths_s.ravel()[mask], depths_g.ravel()[mask], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(conics_s[mask], conics_g[mask], atol=2e-3, rtol=1e-3)


@pytest.mark.gsplat
def test_project_vs_gsplat(gsplat_shim: ModuleType):
    """Match a projected random splat against the gsplat projection of the same scene."""
    means, scales, quats, *_ = scene(10_000, seed=1, dense=True)
    opacities = jnp.full((means.shape[0],), 0.99)
    kw = camera(256, 256)
    project = jax.jit(partial(splax.project, opacities=opacities, **kw))
    splax_out = project(means, scales, quats, VIEWMAT)
    gplat_out = gsplat_shim.project(means, scales, quats, VIEWMAT, **kw)
    _assert_parity(splax_out, gplat_out)


@pytest.mark.gsplat
def test_project_vs_gsplat_lego(gsplat_shim: ModuleType, lego_meta: dict, lego_ply: Path):
    """Match a projected lego scene against the gsplat projection of the same scene."""
    means, log_scales, quats, sh_colors, logit_opacities = splax.io.load_ply(lego_ply)
    scales, _, _ = splax.io.apply_activations(log_scales, sh_colors, logit_opacities)
    means, scales, quats = means[:50_000], scales[:50_000], quats[:50_000]
    H = W = 800
    focal = float(0.5 * W / np.tan(0.5 * lego_meta["camera_angle_x"]))
    viewmat = jnp.asarray(splax.utils.nerf_camera(lego_meta["frames"][0]["transform_matrix"]))
    kw = {"img_shape": (H, W), "f": (focal, focal), "c": (W // 2, H // 2)}
    opacities = jnp.full((means.shape[0],), 0.99)
    project = jax.jit(partial(splax.project, opacities=opacities, **kw))
    splax_out = project(means, scales, quats, viewmat)
    gplat_out = gsplat_shim.project(means, scales, quats, viewmat, **kw)
    _assert_parity(splax_out, gplat_out)
