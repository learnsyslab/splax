"""Test PLY export round-trip."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import pytest
from utils import coeff_scene_params

import splax
from splax.io import load_ply

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("degree", [0, 3])
def test_write_ply_is_load_ply_inverse(degree: int, tmp_path: Path):
    """Random splats through write_ply then load_ply reproduce the parameters exactly."""
    means, log_scales, quats, sh_colors, logit_opacities = coeff_scene_params(5000, degree=degree)[
        :5
    ]
    out = tmp_path / "rand.ply"
    splax.io.write_ply(out, means, log_scales, quats, sh_colors, logit_opacities)

    lm, ls, lq, lc, lo = (np.asarray(x) for x in load_ply(out))

    np.testing.assert_array_equal(lm, means)
    np.testing.assert_array_equal(ls, log_scales)
    np.testing.assert_array_equal(lq, quats)
    np.testing.assert_array_equal(lc, sh_colors)
    np.testing.assert_array_equal(lo, logit_opacities)


@pytest.mark.parametrize("degree", [0, 3])
def test_repeated_ply_cycles_are_stable(degree: int, tmp_path: Path):
    """A second load and write cycle writes the identical bytes."""
    splats = coeff_scene_params(5000, seed=1, degree=degree)[:5]
    first, second = tmp_path / "first.ply", tmp_path / "second.ply"
    splax.io.write_ply(first, *splats)
    splax.io.write_ply(second, *load_ply(first))
    assert second.read_bytes() == first.read_bytes()


def test_prune_drops_strays_faint_and_blobs():
    """Prune removes the far, the faint and the oversized and keeps everything else."""
    n = 200
    rng = np.random.default_rng(0)
    means = jnp.asarray(rng.normal(size=(n, 3)), jnp.float32)
    log_scales = jnp.full((n, 3), -3.0, jnp.float32)
    quats = jnp.tile(jnp.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
    sh_colors = jnp.zeros((n, 1, 3), jnp.float32)
    logit_opacities = jnp.full((n,), 2.0, jnp.float32)
    means = means.at[0].set([500.0, 0.0, 0.0])  # stray
    logit_opacities = logit_opacities.at[1].set(-8.0)  # faint
    log_scales = log_scales.at[2].set([2.0, -3.0, -3.0])  # blob
    kept = splax.io.prune(means, log_scales, quats, sh_colors, logit_opacities)
    assert all(k.shape[0] == n - 3 for k in kept)
    p_means, p_log_scales, _, _, p_logit_opacities = kept
    assert np.linalg.norm(np.asarray(p_means), axis=1).max() < 100
    assert np.asarray(p_logit_opacities).min() == 2.0
    assert np.asarray(p_log_scales).max() == -3.0
