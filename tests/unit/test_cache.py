"""Test the scratch buffers, sort keys and tile intersection behind the rasterizer."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import warp as wp
from utils import VIEWMAT, camera, projected, psnr, rasterize_both_keymodes, scene, scene_params

import splax
import splax._cache as _cache
from splax._rasterize._sort._kernels import map_intersects_64bit

# region scratch invariants


def test_scratch_reuse_across_sizes():
    """Test that reused scratch does not leak state between renders of different sizes."""
    configs = [(2_000, 32, 32), (100, 64, 64), (1000, 128, 128), (100, 256, 256)]
    rasterize = jax.jit(splax.rasterize, static_argnames=("img_shape",))
    for n, H, W in configs:
        args = projected(n, H, W, seed=n)
        # render while scratch is warm from previous (differently sized) iterations
        warm = rasterize(*args, img_shape=(H, W))[0]
        # clean reference: drop all cached scratch, render the identical scene again
        splax.clear_cache()
        cold = rasterize(*args, img_shape=(H, W))[0]
        assert np.array_equal(warm, cold), "scratch reuse changed output"


def test_scratch_released_on_signature_change():
    """Test that switching to a smaller workload releases the scratch of the bigger one."""
    dev = str(wp.get_device())
    splax.clear_cache()
    rasterize = jax.jit(splax.rasterize, static_argnames=("img_shape",))
    jax.block_until_ready(rasterize(*projected(5_000, 256, 256, seed=1), img_shape=(256, 256)))
    big = sum(a.capacity for a in _cache._scratch_cache[dev].values() if isinstance(a, wp.array))
    jax.block_until_ready(rasterize(*projected(1_000, 128, 128, seed=2), img_shape=(128, 128)))
    small = sum(a.capacity for a in _cache._scratch_cache[dev].values() if isinstance(a, wp.array))
    assert small < big / 2, "scratch has not released after signature change"


# region packed 32-bit sort key


def test_packed_key_matches_64bit():
    """Compare renders from the packed 32-bit sort key to renders from the 64-bit key."""
    packed, wide = rasterize_both_keymodes(projected(1_000, 32, 32, seed=1), 32, 32)
    deviation, quality = np.abs(packed - wide).max(), psnr(packed, wide)
    assert deviation < 0.05, f"packed vs 64-bit max abs diff {deviation:.2e}"
    assert quality > 65, f"packed vs 64-bit PSNR only {quality:.1f} dB"


def test_packed_key_falls_back_when_bits_dont_fit():
    """Test that the wide sort key takes over when too few bits remain for depth."""
    dev = str(wp.get_device("cuda:0"))
    H, W = 1080, 1920
    splats = scene_params(4000, seed=1, dense=True)[:5]
    kw = {"background": jnp.zeros(3), **camera(H, W)}

    # B=1 packs into int32 scratch
    splax.clear_cache()
    render = jax.jit(splax.render, static_argnames=("f", "c", "img_shape"))
    img, _ = render(*splats, viewmat=VIEWMAT, **kw)
    img.block_until_ready()
    assert _cache._scratch_cache[dev]["isect_dtype"] == wp.int32

    # B=8: image(3)+tile(13)=16 > 15, falls back to int64 scratch
    splax.clear_cache()
    views = jnp.stack([VIEWMAT.at[2, 3].set(5.0 + 0.1 * i) for i in range(8)])
    jax.block_until_ready(jax.jit(jax.vmap(partial(render, *splats, **kw)))(viewmat=views))
    assert _cache._scratch_cache[dev]["isect_dtype"] == wp.int64
    splax.clear_cache()


# region tight tile intersection

# Projection counts the tiles an ellipse covers and rasterization walks the same ellipse to emit
# the keys. Any disagreement corrupts the per-gaussian offsets into the sort buffer.


@pytest.mark.parametrize("n,H,W", [(2_000, 256, 256), (10_000, 512, 512)])
def test_tile_emission_matches_count(n: int, H: int, W: int):
    """Test that key emission writes exactly as many keys per gaussian as projection counted."""
    means, scales, quats, _, opac, _ = scene(n, seed=n, dense=True)
    project = jax.jit(splax.project, static_argnames=("f", "c", "img_shape"))
    xys, depths, radii, conics, nth, cum = project(
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
    # no stale slot: every key slot was written
    assert not (isect_np == SENT).any(), f"{(isect_np == SENT).sum()} unwritten slots"
    assert (gids_np >= 0).all()
    # each gaussian owns exactly n_tiles_hit[i] contiguous slots at cum[i-1]
    starts = np.concatenate([[0], cum_np[:-1]])
    for i in np.nonzero(nth_np > 0)[0][:2000]:
        s, e = int(starts[i]), int(cum_np[i])
        assert (gids_np[s:e] == i).all(), f"gaussian {i} slot ownership mismatch"
