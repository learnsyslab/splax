"""Test PLY export round-trip."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from utils import scene_params

import splax
from splax.io import load_ply

if TYPE_CHECKING:
    from pathlib import Path


def test_write_ply_is_load_ply_inverse(tmp_path: Path):
    """Random splats through write_ply then load_ply reproduce the parameters exactly."""
    means, log_scales, quats, sh_colors, logit_opacities = scene_params(5000)[:5]
    out = tmp_path / "rand.ply"
    splax.io.write_ply(out, means, log_scales, quats, sh_colors, logit_opacities)

    lm, ls, lq, lc, lo = (np.asarray(x) for x in load_ply(out))

    np.testing.assert_array_equal(lm, means)
    np.testing.assert_array_equal(ls, log_scales)
    np.testing.assert_array_equal(lq, quats)
    np.testing.assert_array_equal(lc, sh_colors)
    np.testing.assert_array_equal(lo, logit_opacities)


def test_repeated_ply_cycles_are_stable(tmp_path: Path):
    """A second load and write cycle writes the identical bytes."""
    splats = scene_params(5000, seed=1)[:5]
    first, second = tmp_path / "first.ply", tmp_path / "second.ply"
    splax.io.write_ply(first, *splats)
    splax.io.write_ply(second, *load_ply(first))
    assert second.read_bytes() == first.read_bytes()
