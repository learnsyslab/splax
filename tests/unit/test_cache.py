"""Warp backend machinery invariants behind the rasterizer.

The backend keeps persistent grow-only scratch buffers per device and picks between a packed 32-bit
and a 64-bit radix sort key. Both are invisible from the outside, yet a stale sort buffer or an
un-zeroed tile_bins prefix silently corrupts a render, and a divergence between the projected tile
count and the emitted keys corrupts the per-gaussian sort buffer offsets. The tests here pin the
scratch reuse and release, the agreement and fallback of the packed key, and the exact tile
emission.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import warp as wp
from utils import VIEWMAT, camera, projected, rasterize_both_keymodes, scene

import splax
import splax._cache as _cache
from splax._rasterize._sort._kernels import map_intersects_64bit

# region scratch invariants


def test_scratch_reuse_across_sizes():
    """Persistent grow-only scratch must not leak state between renders.

    Render several scenes of *different* intersection counts back to back (shrinking
    then growing), each time comparing against a freshly cleared reference render of
    the same scene. A stale sort buffer or an un-zeroed tile_bins prefix would make
    the second render of a smaller scene disagree with its clean-slate render.
    """
    configs = [(200_000, 300, 400), (10_000, 256, 256), (100_000, 512, 512), (10_000, 256, 256)]
    for n, H, W in configs:
        args = projected(n, H, W, seed=n)
        # render while scratch is warm from previous (differently sized) iterations
        warm = np.asarray(splax.rasterize(*args, img_shape=(H, W))[0])
        # clean reference: drop all cached scratch, render the identical scene again
        splax.clear_cache()
        cold = np.asarray(splax.rasterize(*args, img_shape=(H, W))[0])
        assert np.array_equal(warm, cold), (
            f"scratch reuse changed output at n={n} {H}x{W}: max|d|={np.abs(warm - cold).max():.2e}"
        )


def test_scratch_released_on_signature_change():
    """Test that switching to a smaller workload releases a bigger sort scratch."""
    dev = wp.get_device()
    splax.clear_cache()
    splax.rasterize(*projected(500_000, 1080, 1920, seed=1), img_shape=(1080, 1920))
    big = wp.get_mempool_used_mem_current(dev)
    splax.rasterize(*projected(5_000, 128, 128, seed=2), img_shape=(128, 128))
    small = wp.get_mempool_used_mem_current(dev)
    assert small < big * 0.5, f"scratch not released on signature change: {small} vs {big}"


# region packed 32-bit sort key


def test_packed_key_matches_64bit():
    """The packed 32-bit key agrees with the 64-bit key to a high perceptual bound.

    Depth is linearly quantized into depth_bits (>=16) buckets over the per-frame
    range, so the blend order changes only for near-coincident (same-bucket)
    gaussians, giving >65 dB PSNR, <0.05 max abs diff vs the 64-bit path (measured
    floor ~80-140 dB across configs).
    """
    packed, wide = rasterize_both_keymodes(projected(100_000, 512, 512, seed=7), 512, 512)
    d = np.abs(packed - wide)
    mse = float(np.mean((packed - wide) ** 2))
    psnr = 99.0 if mse == 0 else -10 * np.log10(mse)
    assert d.max() < 0.05, f"packed vs 64-bit max abs diff {d.max():.2e}"
    assert psnr > 65, f"packed vs 64-bit PSNR only {psnr:.1f} dB"


def test_packed_key_falls_back_when_bits_dont_fit():
    """When image+tile bits leave <16 for depth, the 64-bit key is used (fallback).

    Observed via the scratch key buffer dtype: packed gives int32, fallback gives int64.
    B=8 at 1080p gives image(3)+tile(13)=16 bits, so depth_bits=15 <16 and it falls back.
    B=1 at 1080p gives 13 bits, so depth_bits=18 and it packs.
    """
    dev = str(wp.get_device("cuda:0"))
    H, W = 1080, 1920
    means, scales, quats, colors, opac, _bg = scene(4000, seed=1, dense=True)
    kw = {"background": jnp.zeros(3), **camera(H, W)}
    splats = (means, scales, quats, colors, opac)

    # B=1 packs into int32 scratch
    splax.clear_cache()
    img, _ = splax.render(*splats, viewmat=VIEWMAT, **kw)
    img.block_until_ready()
    assert _cache._scratch_cache[dev]["isect_dtype"] == wp.int32

    # B=8: image(3)+tile(13)=16 > 15, falls back to int64 scratch
    splax.clear_cache()
    views = jnp.stack([VIEWMAT.at[2, 3].set(5.0 + 0.1 * i) for i in range(8)])
    jax.jit(jax.vmap(lambda vm: splax.render(*splats, viewmat=vm, **kw)[0]))(
        views
    ).block_until_ready()
    assert _cache._scratch_cache[dev]["isect_dtype"] == wp.int64
    splax.clear_cache()


# region tight tile intersection

# Opacity-aware tight tile intersection. Projection counts tiles via the ellipse
# walk and rasterize walks the identical ellipse to emit keys. The count and the
# emit must agree exactly or the per-gaussian sort buffer offsets corrupt.


@pytest.mark.parametrize("n,H,W", [(20_000, 256, 256), (100_000, 512, 512)])
def test_tile_emission_matches_count(n: int, H: int, W: int):
    """The AccuTile key-emission writes EXACTLY n_tiles_hit keys per gaussian.

    Launch the emission kernel into a sentinel-filled buffer and verify that every
    slot in [cum[i-1], cum[i]) is written by gaussian i and no slot is left stale or
    overwritten, i.e. the emitted count agrees bit-for-bit with the projection's
    AccuTile tile count. A divergence between the count (projection) and the walk
    (emission) would show up as sentinels remaining or a wrong gaussian id.
    """
    means, scales, quats, _colors, opac, _bg = scene(n, seed=n, dense=True)
    xys, depths, radii, conics, nth, cum = splax.project(
        means, scales, quats, VIEWMAT, opacities=opac, **camera(H, W)
    )
    nth_np = np.asarray(nth).astype(np.int64)
    cum_np = np.asarray(cum).astype(np.int64)
    total = int(cum_np[-1])
    assert total > 0
    # structural: cum is the inclusive scan of n_tiles_hit
    np.testing.assert_array_equal(cum_np, np.cumsum(nth_np))

    bw = 16
    tbx = (W + bw - 1) // bw
    tby = (H + bw - 1) // bw
    n_tiles = tbx * tby
    tile_n_bits = (n_tiles - 1).bit_length()  # bits to index [0, n_tiles), see sort_and_bin

    dev = "cuda:0"
    xys_w = wp.array(np.asarray(xys), dtype=wp.vec2, device=dev)
    depths_int = wp.array(np.asarray(depths).view(np.int32), dtype=wp.int32, device=dev)
    radii_w = wp.array(np.asarray(radii).astype(np.int32), dtype=wp.int32, device=dev)
    conics_w = wp.array(np.asarray(conics), dtype=wp.vec3, device=dev)
    opac_w = wp.array(np.asarray(opac).astype(np.float32), dtype=wp.float32, device=dev)
    cum_w = wp.array(cum_np.astype(np.int32), dtype=wp.int32, device=dev)

    SENT = np.int64(-999)
    isect = wp.array(np.full(total, SENT, np.int64), dtype=wp.int64, device=dev)
    gids = wp.array(np.full(total, -1, np.int32), dtype=wp.int32, device=dev)

    wp.launch(
        map_intersects_64bit,
        dim=n,
        inputs=[xys_w, depths_int, radii_w, conics_w, opac_w, cum_w, n, n, tile_n_bits, tbx, tby],
        outputs=[isect, gids],
        device=dev,
    )
    wp.synchronize()

    isect_np = isect.numpy()
    gids_np = gids.numpy()
    # no stale slot: every key slot was written (count == emit, no gaps/overflow)
    assert not (isect_np == SENT).any(), f"{(isect_np == SENT).sum()} unwritten slots"
    assert (gids_np >= 0).all()
    # each gaussian owns exactly n_tiles_hit[i] contiguous slots at cum[i-1]
    starts = np.concatenate([[0], cum_np[:-1]])
    for i in np.nonzero(nth_np > 0)[0][:2000]:
        s, e = int(starts[i]), int(cum_np[i])
        assert (gids_np[s:e] == i).all(), f"gaussian {i} slot ownership mismatch"
