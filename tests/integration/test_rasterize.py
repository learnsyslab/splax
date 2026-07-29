"""Rasterization stage against the gsplat reference and the real lego scene.

gsplat exposes no standalone blend entry point, so the reference goes through its full
``rasterization`` and the comparison feeds ``splax.rasterize`` from ``splax.project``. The
projection itself is pinned against ``gsplat.fully_fused_projection`` in the projection tests, so
what these bound is the blend. The two blends use a different sort, tiling, and accumulation order
and cannot agree bit-for-bit, so the difference is bounded perceptually with the max abs difference
and the PSNR.

The accumulated alpha ``sum_i w_i`` is gsplat's ``render_alphas``, bounded the same way. It is a
coverage in [0, 1] on both sides, so its bounds are absolute rather than relative to a range.

``rasterize_depth`` renders ``sum_i w_i z_i / sum_i w_i``, which gsplat renders under
``render_mode="ED"``, so both its image and its depth map are bounded against the shim. The depth is
an unbounded positive quantity in camera units rather than a [0, 1] image, so it is bounded relative
to its own range.

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
    img, _alpha = splax.rasterize(
        colors, opacities, background, xys, depths, radii, conics, cum, img_shape=(H, W)
    )
    a = np.asarray(img)
    b, _b_alpha = gsplat_shim.render(
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


@pytest.mark.gsplat
@pytest.mark.parametrize("n,H,W", [(20_000, 256, 256), (100_000, 512, 512)])
def test_rasterize_alpha_vs_gsplat(n: int, H: int, W: int, gsplat_shim: ModuleType):
    """Bound the accumulated alpha against gsplat's render alphas.

    The alpha is the same visibility accumulation the colours ride on, so the tail gaussians the two
    tilings order differently move it on the same handful of pixels and by the same margin as the
    image. It is a coverage in [0, 1] on both sides, so the bounds are absolute.
    """
    means, scales, quats, colors, opacities, background = scene(n, seed=n, dense=True)
    xys, depths, radii, conics, _nth, cum = splax.project(
        means, scales, quats, VIEWMAT, opacities=opacities, **camera(H, W)
    )
    _img, alpha = splax.rasterize(
        colors, opacities, background, xys, depths, radii, conics, cum, img_shape=(H, W)
    )
    a = np.asarray(alpha)
    _b, b_alpha = gsplat_shim.render(
        means,
        scales,
        quats,
        colors,
        opacities,
        viewmat=VIEWMAT,
        background=background,
        **camera(H, W),
    )
    assert a.shape == b_alpha.shape
    assert a.min() >= 0.0 and a.max() <= 1.0
    assert _psnr(a, b_alpha) > 60.0, f"splax vs gsplat alpha PSNR only {_psnr(a, b_alpha):.1f} dB"
    assert np.abs(a - b_alpha).max() < 0.03, f"max abs diff {np.abs(a - b_alpha).max():.3f}"


@pytest.mark.gsplat
@pytest.mark.parametrize("n,H,W", [(20_000, 256, 256), (100_000, 512, 512)])
def test_rasterize_depth_vs_gsplat(n: int, H: int, W: int, gsplat_shim: ModuleType):
    """Bound the image and the expected depth map against gsplat's rasterization.

    The depth accumulator runs beside the colour blend and must leave the image on the same
    perceptual bound as the plain blend.
    """
    means, scales, quats, colors, opacities, background = scene(n, seed=n, dense=True)
    xys, depths, radii, conics, _nth, cum = splax.project(
        means, scales, quats, VIEWMAT, opacities=opacities, **camera(H, W)
    )
    img, _alpha = splax.rasterize_depth(
        colors, opacities, background, xys, depths, radii, conics, cum, img_shape=(H, W)
    )
    ref, _ref_alpha = gsplat_shim.render_depth(
        means,
        scales,
        quats,
        colors,
        opacities,
        viewmat=VIEWMAT,
        background=background,
        **camera(H, W),
    )
    a, a_depth = np.asarray(img)[..., :3], np.asarray(img)[..., 3]
    b, b_depth = ref[..., :3], ref[..., 3]
    assert a.shape == b.shape
    assert a_depth.shape == b_depth.shape
    assert _psnr(a, b) > 60.0, f"splax vs gsplat depth blend PSNR only {_psnr(a, b):.1f} dB"
    assert np.abs(a - b).max() < 0.03, f"max abs diff {np.abs(a - b).max():.3f}"
    # Both renders leave the same pixels uncovered and read depth 0 there.
    np.testing.assert_array_equal(a_depth == 0.0, b_depth == 0.0)
    # The depth carries camera units, so both bounds are taken against its own range. The tail
    # gaussians that the two tilings order differently move a handful of pixels by ~2e-3 of that
    # range while the 99.9th percentile stays three orders below, measured 101 dB and 112 dB
    # normalized PSNR across the two sizes.
    scale = b_depth.max()
    rel = np.abs(a_depth - b_depth).max() / scale
    depth_psnr = _psnr(a_depth / scale, b_depth / scale)
    assert rel < 0.01, f"relative max depth diff {rel:.2e}"
    assert depth_psnr > 80.0, f"splax vs gsplat depth PSNR only {depth_psnr:.1f} dB"


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
