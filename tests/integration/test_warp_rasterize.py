"""Perceptual parity of the full splax render against gsplat's ``rasterization``.

gsplat cannot be matched bit-for-bit (different sort, blend, and tiling), so the difference is
bounded perceptually (max abs diff + PSNR) rather than element-wise-exactly. These are marked
``gsplat`` and skipped when gsplat is not installed. The packed-key check runs on the real lego
scene, where the tight key emission is exercised at full resolution.

Convention conversions for the gsplat reference are documented in tests/_gsplat.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import pytest
from _warp_rasterize import RastKW, RenderKW, rasterize_both_keymodes, render_scene

import splax

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from types import ModuleType


@pytest.mark.integration
@pytest.mark.gsplat
@pytest.mark.parametrize("n,H,W", [(20_000, 256, 256), (100_000, 512, 512)])
def test_render_vs_gsplat(n: int, H: int, W: int, gsplat_shim: ModuleType):
    """splax.render vs gsplat.rasterization on the same scene.

    Different kernels (sort order, blend, opacity-aware tiling), so we bound the
    image difference perceptually. splax's tight tiling drops per-gaussian tails
    already below 1/255, and gsplat's classic rasterizer keeps a slightly different
    set, so a handful of pixels move by a few 1/255. The bulk agree to well under
    that. Empirically PSNR is comfortably above 30 dB across sizes.
    """
    (splats, kw) = render_scene(n, H, W, seed=n)
    a = np.asarray(splax.render(*splats, **kw)[0])
    b = gsplat_shim.render(*splats, **kw)
    assert a.shape == b.shape
    mse = float(np.mean((a - b) ** 2))
    psnr = -10 * np.log10(mse) if mse > 0 else float("inf")
    # Measured ~100 dB / max abs ~0.003 across these sizes, bounded with margin.
    assert psnr > 60.0, f"splax vs gsplat render PSNR only {psnr:.1f} dB"
    assert np.abs(a - b).max() < 0.03, f"max abs diff {np.abs(a - b).max():.3f}"


@pytest.mark.integration
def test_packed_vs_64bit_lego(lego_ply: Path):
    """Packed vs 64-bit on the real lego scene (tight key emission)."""
    means, scales, quats, colors, opac = splax.io.load_ply(lego_ply)
    H, W = 720, 1280
    viewmat = jnp.asarray(
        np.array([[1, 0, 0, 0.2], [0, 1, 0, -0.1], [0, 0, 1, 6.0], [0, 0, 0, 1]], np.float32)
    )
    xys, depths, radii, conics, _nth, cum = splax.project(
        means,
        scales,
        quats,
        viewmat,
        opacities=opac,
        img_shape=(H, W),
        f=(float(H), float(H)),
        c=(W // 2, H // 2),
        glob_scale=1.0,
        clip_thresh=0.01,
    )
    kw: RastKW = {"img_shape": (H, W)}
    args = (colors, opac, jnp.ones(3), xys, depths, radii, conics, cum)
    packed, wide = rasterize_both_keymodes(args, kw)
    d = np.abs(packed - wide)
    mse = float(np.mean((packed - wide) ** 2))
    psnr = 99.0 if mse == 0 else -10 * np.log10(mse)
    assert d.max() < 0.05, f"packed vs 64-bit max abs diff {d.max():.2e}"
    assert psnr > 65, f"packed vs 64-bit PSNR only {psnr:.1f} dB"


@pytest.mark.integration
@pytest.mark.gsplat
def test_render_lego_vs_gsplat(
    gsplat_shim: ModuleType, lego_meta: dict, lego_view: Callable[[str], np.ndarray], lego_ply: Path
):
    """Splax vs gsplat full render on the real lego scene, from a dataset pose.

    A realistic-scene perceptual check to complement the random-scene parity and
    the GT-PSNR regression gate (tests/integration/test_lego_regression.py). Different kernels,
    so bounded by PSNR rather than exactly.
    """
    means, scales, quats, colors, opac = splax.io.load_ply(lego_ply)
    frame = lego_meta["frames"][0]
    gt = lego_view(frame["file_path"])
    H, W = gt.shape[:2]
    ff = 0.5 * W / np.tan(0.5 * lego_meta["camera_angle_x"])
    kw: RenderKW = {
        "viewmat": jnp.asarray(splax.utils.nerf_camera(frame["transform_matrix"])),
        "background": jnp.ones(3),
        "img_shape": (H, W),
        "f": (float(ff), float(ff)),
        "c": (W // 2, H // 2),
        "glob_scale": 1.0,
        "clip_thresh": 0.01,
    }
    a = np.asarray(splax.render(means, scales, quats, colors, opac, **kw)[0])
    b = gsplat_shim.render(means, scales, quats, colors, opac, **kw)
    mse = float(np.mean((a - b) ** 2))
    psnr = -10 * np.log10(mse) if mse > 0 else float("inf")
    # Measured ~82 dB on this pose, bounded well below with margin for scene detail.
    assert psnr > 45.0, f"splax vs gsplat lego render PSNR only {psnr:.1f} dB"
