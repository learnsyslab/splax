"""Test the packed sort key of the rasterizer against the wide key on the lego scene."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from utils import VIEWMAT, camera, psnr, rasterize_both_keymodes

import splax

if TYPE_CHECKING:
    from pathlib import Path


def test_rasterize_packed_vs_64bit_lego(lego_ply: Path):
    """Match the packed 32-bit sort key against the 64-bit key on the real lego scene."""
    means, log_scales, quats, sh_colors, logit_opacities = splax.io.load_ply(lego_ply)
    scales, colors, opacities = splax.io.apply_activations(log_scales, sh_colors, logit_opacities)
    H, W = 720, 1280
    project = jax.jit(partial(splax.project, opacities=opacities, **camera(H, W)))
    xys, depths, radii, conics, _, cum = project(means, scales, quats, VIEWMAT.at[2, 3].set(6.0))

    args = (colors, opacities, jnp.ones(3), xys, depths, radii, conics, cum)
    packed, wide = rasterize_both_keymodes(args, H, W)
    deviation, quality = np.abs(packed - wide).max(), psnr(packed, wide)
    assert deviation < 0.05, f"packed vs 64-bit max abs diff {deviation:.2e}"
    assert quality > 65, f"packed vs 64-bit PSNR only {quality:.1f} dB"
