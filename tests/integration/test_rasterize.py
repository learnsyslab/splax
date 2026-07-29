"""Rasterization stage against the gsplat reference and the real lego scene.

gsplat exposes no standalone blend entry point, so the reference goes through its full
``rasterization`` and the comparison feeds ``splax.rasterize`` from ``splax.project``. The
projection itself is pinned against ``gsplat.fully_fused_projection`` in the projection tests, so
what these bound is the blend. The two blends use a different sort, tiling, and accumulation order
and cannot agree bit-for-bit, so the difference is bounded perceptually with the max abs difference
and the PSNR.

The depth channel has no counterpart in the shim, so ``rasterize_depth`` is compared on its image
only, while the depth map itself is pinned by the ``D == pvz · A`` identity in the unit tests.

The packed-key check runs on the real lego scene, where the tight key emission is exercised at full
resolution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import pytest
from utils import VIEWMAT, camera, rasterize_both_keymodes, scene

import splax

if TYPE_CHECKING:
    from pathlib import Path
    from types import ModuleType


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    """Return the PSNR of two images in dB, capped where they are identical."""
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse == 0 else -10 * np.log10(mse)


@pytest.mark.integration
@pytest.mark.gsplat
@pytest.mark.parametrize("n,H,W", [(20_000, 256, 256), (100_000, 512, 512)])
def test_rasterize_vs_gsplat(n: int, H: int, W: int, gsplat_shim: ModuleType):
    """Bound the blended image against gsplat's rasterization.

    splax's tight tiling drops per-gaussian tails already below 1/255 and gsplat's classic
    rasterizer keeps a slightly different set, so a handful of pixels move by a few 1/255 while the
    bulk agrees to well under that. Measured ~100 dB and a max abs difference ~0.003 across sizes.
    """
    means, scales, quats, colors, opacities, background = scene(n, seed=n, dense=True)
    xys, depths, radii, conics, _nth, cum = splax.project(
        means, scales, quats, VIEWMAT, opacities=opacities, **camera(H, W)
    )
    a = np.asarray(
        splax.rasterize(
            colors, opacities, background, xys, depths, radii, conics, cum, img_shape=(H, W)
        )
    )
    b = gsplat_shim.render(
        means,
        scales,
        quats,
        colors,
        opacities,
        viewmat=VIEWMAT,
        background=background,
        **camera(H, W),
    )
    assert a.shape == b.shape
    assert _psnr(a, b) > 60.0, f"splax vs gsplat blend PSNR only {_psnr(a, b):.1f} dB"
    assert np.abs(a - b).max() < 0.03, f"max abs diff {np.abs(a - b).max():.3f}"


@pytest.mark.integration
@pytest.mark.gsplat
@pytest.mark.parametrize("n,H,W", [(20_000, 256, 256), (100_000, 512, 512)])
def test_rasterize_depth_vs_gsplat(n: int, H: int, W: int, gsplat_shim: ModuleType):
    """Bound the image of the depth blend against gsplat's rasterization.

    The depth map has no reference in the shim, so only the colour channel is compared here. The
    depth accumulator must leave that channel on the same perceptual bound as the plain blend.
    """
    means, scales, quats, colors, opacities, background = scene(n, seed=n, dense=True)
    xys, depths, radii, conics, _nth, cum = splax.project(
        means, scales, quats, VIEWMAT, opacities=opacities, **camera(H, W)
    )
    img, depth = splax.rasterize_depth(
        colors, opacities, background, xys, depths, radii, conics, cum, img_shape=(H, W)
    )
    a = np.asarray(img)
    b = gsplat_shim.render(
        means,
        scales,
        quats,
        colors,
        opacities,
        viewmat=VIEWMAT,
        background=background,
        **camera(H, W),
    )
    assert _psnr(a, b) > 60.0, f"splax vs gsplat depth blend PSNR only {_psnr(a, b):.1f} dB"
    assert np.abs(a - b).max() < 0.03, f"max abs diff {np.abs(a - b).max():.3f}"
    # the expected depth stays inside the projected depth range wherever a gaussian contributes
    depth = np.asarray(depth)
    covered = depth > 0.0
    assert covered.any()
    assert depth[covered].max() <= float(np.asarray(depths).max())


@pytest.mark.integration
def test_rasterize_packed_vs_64bit_lego(lego_ply: Path):
    """Match the packed 32-bit sort key against the 64-bit key on the real lego scene."""
    means, scales, quats, colors, opacities = splax.io.load_ply(lego_ply)
    H, W = 720, 1280
    xys, depths, radii, conics, _nth, cum = splax.project(
        means, scales, quats, VIEWMAT.at[2, 3].set(6.0), opacities=opacities, **camera(H, W)
    )
    args = (colors, opacities, jnp.ones(3), xys, depths, radii, conics, cum)
    packed, wide = rasterize_both_keymodes(args, H, W)
    d = np.abs(packed - wide)
    assert d.max() < 0.05, f"packed vs 64-bit max abs diff {d.max():.2e}"
    assert _psnr(packed, wide) > 65, f"packed vs 64-bit PSNR only {_psnr(packed, wide):.1f} dB"
