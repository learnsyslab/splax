"""PLY export round-trip.

``splax.io.write_ply`` must be the exact inverse of ``splax.io.load_ply``. Random render-space
splats written and reloaded must reproduce the inputs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from utils import scene

import splax
from splax.io import load_ply

if TYPE_CHECKING:
    from pathlib import Path


def test_write_ply_is_load_ply_inverse(tmp_path: Path):
    """Random splats through write_ply then load_ply reproduce the render-space inputs."""
    means, scales, quats, colors, opac, _bg = scene(5000)
    out = tmp_path / "rand.ply"
    splax.io.write_ply(out, means, scales, quats, colors, opac)

    lm, ls, lq, lc, lo = (np.asarray(x) for x in load_ply(out))

    np.testing.assert_allclose(lm, means, rtol=0, atol=1e-6)
    np.testing.assert_allclose(ls, scales, rtol=1e-5, atol=1e-6)
    # quats are normalized on both sides, compare up to sign is unnecessary since
    # write_ply preserves the stored raw quat direction and load re-normalizes.
    np.testing.assert_allclose(lq, quats, rtol=0, atol=1e-6)
    np.testing.assert_allclose(lc, colors, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(lo, opac, rtol=1e-4, atol=1e-4)
