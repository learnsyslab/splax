"""splax-internal rasterization invariants that need no external reference.

Covered here: persistent scratch reuse and release, the packed 32-bit vs 64-bit sort key and its
fallback, the SNUGBOX/AccuTile tight tile emission matching its count, and jit self-consistency.
"""

from __future__ import annotations

from typing import TypedDict

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import warp as wp
from _warp_rasterize import RastKW, rasterize_both_keymodes, render_scene

import splax
import splax._cache as _cache
from splax._rasterize._sort._kernels import map_intersects_64bit


class _ProjKW(TypedDict):
    img_shape: tuple[int, int]
    f: tuple[float, float]
    c: tuple[float, float]
    glob_scale: float
    clip_thresh: float


class _Scene(TypedDict):
    colors: jax.Array
    opacities: jax.Array
    background: jax.Array
    xys: jax.Array
    depths: jax.Array
    radii: jax.Array
    conics: jax.Array
    cum_tiles_hit: jax.Array
    img_shape: tuple[int, int]


class _RenderKWNoView(TypedDict):
    background: jax.Array
    img_shape: tuple[int, int]
    f: tuple[float, float]
    c: tuple[float, float]
    glob_scale: float
    clip_thresh: float


def _random_scene(n: int, H: int, W: int, seed: int = 0) -> _Scene:
    """Random scene projected with splax into rasterize inputs."""
    key = jax.random.key(seed)
    k = jax.random.split(key, 6)
    means = jax.random.normal(k[0], (n, 3))
    scales = jax.random.uniform(k[1], (n, 3), minval=0.005, maxval=0.05)
    quats = jax.random.normal(k[2], (n, 4))
    quats = quats / jnp.linalg.norm(quats, axis=-1, keepdims=True)
    colors = jax.random.uniform(k[3], (n, 3))
    opacities = jax.random.uniform(k[4], (n,))
    background = jax.random.uniform(k[5], (3,))
    viewmat = jnp.array([[1, 0, 0, 0.2], [0, 1, 0, -0.1], [0, 0, 1, 5], [0, 0, 0, 1]], jnp.float32)
    proj_args: _ProjKW = {
        "img_shape": (H, W),
        "f": (float(H), float(H)),
        "c": (W // 2, H // 2),
        "glob_scale": 1.0,
        "clip_thresh": 0.01,
    }
    xys, depths, radii, conics, _nth, cum = splax.project(
        means, scales, quats, viewmat, opacities=opacities, **proj_args
    )
    return {
        "colors": colors,
        "opacities": opacities,
        "background": background,
        "xys": xys,
        "depths": depths,
        "radii": radii,
        "conics": conics,
        "cum_tiles_hit": cum,
        "img_shape": (H, W),
    }


def _rast_args(scene: _Scene) -> tuple[jax.Array, ...]:
    return (
        scene["colors"],
        scene["opacities"],
        scene["background"],
        scene["xys"],
        scene["depths"],
        scene["radii"],
        scene["conics"],
        scene["cum_tiles_hit"],
    )


def _splax_rast(scene: _Scene) -> np.ndarray:
    kw: RastKW = {"img_shape": scene["img_shape"]}
    return np.asarray(splax.rasterize(*_rast_args(scene), **kw))


@pytest.mark.unit
def test_render_under_jit_matches_eager():
    """splax.render under jit is byte-identical to eager (splax-only, no gsplat)."""
    (splats, kw) = render_scene(50_000, 256, 256, seed=99)
    eager = np.asarray(splax.render(*splats, **kw)[0])
    jitted = np.asarray(jax.jit(lambda *x: splax.render(*x, **kw)[0])(*splats))
    assert np.array_equal(eager, jitted)


@pytest.mark.unit
def test_principal_point_defaults_to_center():
    """Omitting c is byte-identical to passing the image center explicitly."""
    (splats, kw) = render_scene(10_000, 256, 256, seed=42)
    explicit = np.asarray(splax.render(*splats, **kw)[0])
    kw_no_c = dict(kw)
    del kw_no_c["c"]
    defaulted = np.asarray(splax.render(*splats, **kw_no_c)[0])
    assert np.array_equal(explicit, defaulted)


# region scratch invariants


@pytest.mark.unit
def test_scratch_reuse_across_sizes():
    """Persistent grow-only scratch must not leak state between renders.

    Render several scenes of *different* intersection counts back to back (shrinking
    then growing), each time comparing against a freshly cleared reference render of
    the same scene. A stale sort buffer or an un-zeroed tile_bins prefix would make
    the second render of a smaller scene disagree with its clean-slate render.
    """
    configs = [(200_000, 300, 400), (10_000, 256, 256), (100_000, 512, 512), (10_000, 256, 256)]
    for n, H, W in configs:
        scene = _random_scene(n, H, W, seed=n)
        # render while scratch is warm from previous (differently sized) iterations
        warm = _splax_rast(scene)
        # clean reference: drop all cached scratch, render the identical scene again
        splax.clear_cache()
        cold = _splax_rast(scene)
        assert np.array_equal(warm, cold), (
            f"scratch reuse changed output at n={n} {H}x{W}: max|d|={np.abs(warm - cold).max():.2e}"
        )


@pytest.mark.unit
def test_scratch_dropped_on_signature_change():
    """Test that switching to a smaller workload releases a bigger sort scratch."""
    dev = wp.get_device()
    splax.clear_cache()
    _splax_rast(_random_scene(500_000, 1080, 1920, seed=1))
    big = wp.get_mempool_used_mem_current(dev)
    _splax_rast(_random_scene(5_000, 128, 128, seed=2))
    small = wp.get_mempool_used_mem_current(dev)
    assert small < big * 0.5, f"scratch not released on signature change: {small} vs {big}"


# region packed 32-bit sort key


@pytest.mark.unit
def test_packed_vs_64bit_random():
    """The packed 32-bit key agrees with the 64-bit key to a high perceptual bound.

    Depth is linearly quantized into depth_bits (>=16) buckets over the per-frame
    range, so the blend order changes only for near-coincident (same-bucket)
    gaussians, giving >65 dB PSNR, <0.05 max abs diff vs the 64-bit path (measured
    floor ~80-140 dB across configs).
    """
    scene = _random_scene(100_000, 512, 512, seed=7)
    kw: RastKW = {"img_shape": scene["img_shape"]}
    packed, wide = rasterize_both_keymodes(_rast_args(scene), kw)
    d = np.abs(packed - wide)
    mse = float(np.mean((packed - wide) ** 2))
    psnr = 99.0 if mse == 0 else -10 * np.log10(mse)
    assert d.max() < 0.05, f"packed vs 64-bit max abs diff {d.max():.2e}"
    assert psnr > 65, f"packed vs 64-bit PSNR only {psnr:.1f} dB"


@pytest.mark.unit
def test_packed_fallback_triggers_when_bits_dont_fit():
    """When image+tile bits leave <16 for depth, the 64-bit key is used (fallback).

    Observed via the scratch key buffer dtype: packed gives int32, fallback gives int64.
    B=8 at 1080p gives image(3)+tile(13)=16 bits, so depth_bits=15 <16 and it falls back.
    B=1 at 1080p gives 13 bits, so depth_bits=18 and it packs.
    """
    dev = str(wp.get_device("cuda:0"))
    m, s, q, c, o = (
        jax.random.normal(jax.random.key(1), (4000, 3)),
        jax.random.uniform(jax.random.key(2), (4000, 3), minval=0.01, maxval=0.05),
        _norm_quats(jax.random.normal(jax.random.key(3), (4000, 4))),
        jax.random.uniform(jax.random.key(4), (4000, 3)),
        jax.random.uniform(jax.random.key(5), (4000,)),
    )
    H, W = 1080, 1920
    kw: _RenderKWNoView = {
        "background": jnp.zeros(3),
        "img_shape": (H, W),
        "f": (float(H), float(H)),
        "c": (W // 2, H // 2),
        "glob_scale": 1.0,
        "clip_thresh": 0.01,
    }

    # B=1 packs into int32 scratch
    splax.clear_cache()
    img, _ = splax.render(m, s, q, c, o, viewmat=_id_viewmat(), **kw)
    img.block_until_ready()
    assert _cache._scratch_cache[dev]["isect_dtype"] == wp.int32

    # B=8: image(3)+tile(13)=16 > 15, falls back to int64 scratch
    splax.clear_cache()
    views = jnp.stack([_id_viewmat(dz=5.0 + 0.1 * i) for i in range(8)])
    jax.jit(jax.vmap(lambda vm: splax.render(m, s, q, c, o, viewmat=vm, **kw)[0]))(
        views
    ).block_until_ready()
    assert _cache._scratch_cache[dev]["isect_dtype"] == wp.int64
    splax.clear_cache()


def _norm_quats(q: jax.Array) -> jax.Array:
    return q / jnp.linalg.norm(q, axis=-1, keepdims=True)


def _id_viewmat(dz: float = 5.0) -> jax.Array:
    return jnp.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, dz], [0, 0, 0, 1]], jnp.float32)


# region tight tile intersection

# Opacity-aware tight tile intersection. Projection counts tiles via the ellipse
# walk and rasterize walks the identical ellipse to emit keys. The count and the
# emit must agree exactly or the per-gaussian sort buffer offsets corrupt.


def _project_tight(
    n: int, H: int, W: int, seed: int
) -> tuple[
    jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, tuple[int, int]
]:
    """splax.project WITH opacities gives SNUGBOX radii + AccuTile n_tiles_hit."""
    key = jax.random.key(seed)
    k = jax.random.split(key, 5)
    means = jax.random.normal(k[0], (n, 3))
    scales = jax.random.uniform(k[1], (n, 3), minval=0.005, maxval=0.05)
    quats = jax.random.normal(k[2], (n, 4))
    quats = quats / jnp.linalg.norm(quats, axis=-1, keepdims=True)
    opacities = jax.random.uniform(k[3], (n,))
    viewmat = jnp.array([[1, 0, 0, 0.2], [0, 1, 0, -0.1], [0, 0, 1, 5], [0, 0, 0, 1]], jnp.float32)
    pk: _ProjKW = {
        "img_shape": (H, W),
        "f": (float(H), float(H)),
        "c": (W // 2, H // 2),
        "glob_scale": 1.0,
        "clip_thresh": 0.01,
    }
    xys, depths, radii, conics, nth, cum = splax.project(
        means, scales, quats, viewmat, opacities=opacities, **pk
    )
    return xys, depths, radii, conics, opacities, nth, cum, (H, W)


@pytest.mark.unit
@pytest.mark.parametrize("n,H,W", [(20_000, 256, 256), (100_000, 512, 512)])
def test_snugbox_emit_matches_count(n: int, H: int, W: int):
    """The AccuTile key-emission writes EXACTLY n_tiles_hit keys per gaussian.

    Launch the emission kernel into a sentinel-filled buffer and verify that every
    slot in [cum[i-1], cum[i]) is written by gaussian i and no slot is left stale or
    overwritten, i.e. the emitted count agrees bit-for-bit with the projection's
    AccuTile tile count. A divergence between the count (projection) and the walk
    (emission) would show up as sentinels remaining or a wrong gaussian id.
    """
    xys, depths, radii, conics, opac, nth, cum, (H, W) = _project_tight(n, H, W, seed=n)
    nth_np = np.asarray(nth).ravel().astype(np.int64)
    cum_np = np.asarray(cum).ravel().astype(np.int64)
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
    depths_int = wp.array(np.asarray(depths).ravel().view(np.int32), dtype=wp.int32, device=dev)
    radii_w = wp.array(np.asarray(radii).ravel().astype(np.int32), dtype=wp.int32, device=dev)
    conics_w = wp.array(np.asarray(conics), dtype=wp.vec3, device=dev)
    opac_w = wp.array(np.asarray(opac).ravel().astype(np.float32), dtype=wp.float32, device=dev)
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
