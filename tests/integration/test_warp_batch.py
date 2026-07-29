"""Native batching: jax.vmap over the pure-Warp splax pipeline.

splax.project / splax.rasterize / splax.render carry vmap_method="expand_dims" and
launch a single grid over the whole batch (gsplat-style, no host loop). These tests
assert that a vmapped call equals ``jnp.stack`` of the per-element unbatched calls.

Batching must NOT change blend order within an image. The global sort packs the
image id above the tile bits, so each image's (tile, depth) ordering is identical to
the unbatched sort, and the front-to-back accumulation is bit-identical. We therefore
require **bit-exact** equality (array_equal), not just an allclose tolerance.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import warp as wp
from _warp_batch import (
    CAMERA,
    KW,
    VIEWS,
    N,
    faithful_64bit_keys,  # noqa: F401 (autouse fixture)
    rand_scene,
    render,
    viewmat,
)

import splax
import splax._cache as _cache
import splax._rasterize._sort._sort as _sort


@pytest.mark.integration
def test_render_vmap_over_viewmats():
    """vmap(render) over B=3 viewmats == jnp.stack of 3 unbatched renders."""
    m, s, q, c, o = rand_scene(N, seed=2)
    ref = jnp.stack([render(m, s, q, c, o, VIEWS[i]) for i in range(3)])
    out = jax.vmap(lambda vm: render(m, s, q, c, o, vm))(VIEWS)
    assert out.shape == ref.shape
    np.testing.assert_array_equal(np.asarray(out), np.asarray(ref))


@pytest.mark.integration
def test_render_vmap_over_splats():
    """Vmap over batched splat params (shared viewmat) == stacked unbatched."""
    scenes = [rand_scene(N, seed=s) for s in (3, 4, 5)]
    mb, sb, qb, cb, ob = (jnp.stack([sc[i] for sc in scenes]) for i in range(5))
    vm = viewmat(0.1)
    ref = jnp.stack([render(*scenes[i], vm) for i in range(3)])
    out = jax.vmap(lambda m, s, q, c, o: render(m, s, q, c, o, vm))(mb, sb, qb, cb, ob)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(ref))


@pytest.mark.integration
def test_render_vmap_mixed_both_batched():
    """Vmap over both splat params AND viewmats (mixed dims in one call)."""
    scenes = [rand_scene(N, seed=s) for s in (6, 7, 8)]
    mb, sb, qb, cb, ob = (jnp.stack([sc[i] for sc in scenes]) for i in range(5))
    ref = jnp.stack([render(*scenes[i], VIEWS[i]) for i in range(3)])
    out = jax.vmap(render)(mb, sb, qb, cb, ob, VIEWS)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(ref))


@pytest.mark.integration
def test_render_nested_vmap_grid():
    """Nested vmap over an A splats x B viewmats grid == the A x B double loop."""
    scenes = [rand_scene(N, seed=s) for s in (3, 4, 5)]
    mb, sb, qb, cb, ob = (jnp.stack([sc[i] for sc in scenes]) for i in range(5))
    B = VIEWS.shape[0]
    ref = jnp.stack(
        [jnp.stack([render(*scenes[a], VIEWS[b]) for b in range(B)]) for a in range(len(scenes))]
    )
    grid = jax.vmap(lambda m, s, q, c, o: jax.vmap(lambda vm: render(m, s, q, c, o, vm))(VIEWS))
    out = grid(mb, sb, qb, cb, ob)
    assert out.shape == ref.shape
    np.testing.assert_array_equal(np.asarray(out), np.asarray(ref))


@pytest.mark.integration
def test_render_nested_shared_splat_transforms():
    """Nested vmap, shared splat, viewmat on both axes, transforms on the outer axis only."""
    m, s, q, c, o = rand_scene(N, seed=6)
    slices = ((0, N // 2), (N // 2, N))
    n_worlds, n_cams = 2, 3
    vms = jnp.stack(
        [
            jnp.stack([viewmat(0.1 * w + 0.03 * cam) for cam in range(n_cams)])
            for w in range(n_worlds)
        ]
    )
    tfs = jnp.stack(
        [jnp.broadcast_to(jnp.eye(4).at[0, 3].set(0.02 * w), (2, 4, 4)) for w in range(n_worlds)]
    )

    def one(vm: jax.Array, tf: jax.Array) -> jax.Array:
        tf_kw = {"gaussian_transforms": tf, "gaussian_slices": slices}
        return splax.render(m, s, q, c, o, viewmat=vm, **CAMERA, **tf_kw)[0]

    ref = jnp.stack(
        [jnp.stack([one(vms[w, cam], tfs[w]) for cam in range(n_cams)]) for w in range(n_worlds)]
    )
    out = jax.vmap(lambda vmw, tfw: jax.vmap(lambda vm: one(vm, tfw))(vmw))(vms, tfs)
    assert out.shape == ref.shape
    np.testing.assert_array_equal(np.asarray(out), np.asarray(ref))


@pytest.mark.integration
def test_render_nested_vmap_grad():
    """Nested vmap over jax.grad of the differentiable render == the double loop of gradients."""
    m, s, q, c, o = rand_scene(N, seed=12)
    means = jnp.stack([m, m * 1.01])

    def loss(mm: jax.Array, vm: jax.Array) -> jax.Array:
        return render(mm, s, q, c, o, vm).sum()

    ref = jnp.stack(
        [
            jnp.stack([jax.grad(loss)(means[a], VIEWS[b]) for b in range(VIEWS.shape[0])])
            for a in range(means.shape[0])
        ]
    )
    out = jax.vmap(lambda mm: jax.vmap(lambda vm: jax.grad(loss)(mm, vm))(VIEWS))(means)
    assert out.shape == ref.shape
    # gradient magnitudes reach ~1e3 and batched reductions reorder, so allclose not bit-exact
    np.testing.assert_allclose(np.asarray(out), np.asarray(ref), rtol=1e-4, atol=1e-2)


@pytest.mark.integration
def test_render_jit_vmap():
    """jit(vmap(render)) matches unbatched."""
    m, s, q, c, o = rand_scene(N, seed=9)
    ref = jnp.stack([render(m, s, q, c, o, VIEWS[i]) for i in range(3)])
    fn = jax.jit(jax.vmap(lambda vm: render(m, s, q, c, o, vm)))
    np.testing.assert_array_equal(np.asarray(fn(VIEWS)), np.asarray(ref))


@pytest.mark.integration
def test_render_vmap_b1_equals_unbatched():
    """B=1 vmap is identical to the plain unbatched render."""
    m, s, q, c, o = rand_scene(N, seed=10)
    vm = viewmat(0.15)
    unb = render(m, s, q, c, o, vm)
    b1 = jax.vmap(lambda v: render(m, s, q, c, o, v))(vm[None])
    assert b1.shape == (1,) + unb.shape
    np.testing.assert_array_equal(np.asarray(b1[0]), np.asarray(unb))


@pytest.mark.integration
def test_render_vmap_larger_batch_and_res():
    """B=8 at a larger resolution: image-id/tile-id key packing stays correct."""
    m, s, q, c, o = rand_scene(12_000, seed=11)
    B, hh, ww = 8, 512, 512
    kw: KW = {**CAMERA, "img_shape": (hh, ww), "f": (float(hh), float(hh)), "c": (ww // 2, hh // 2)}
    views = jnp.stack([viewmat(0.1 * i) for i in range(B)])
    ref = jnp.stack([splax.render(m, s, q, c, o, viewmat=views[i], **kw)[0] for i in range(B)])
    out = jax.jit(jax.vmap(lambda vm: splax.render(m, s, q, c, o, viewmat=vm, **kw)[0]))(views)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(ref))


@pytest.mark.integration
def test_render_vmap_packed_matches_stack(monkeypatch: pytest.MonkeyPatch):
    """Match packed batched render output against stacked unbatched output."""
    monkeypatch.setattr(_sort, "_use_32bit_keys", lambda depth_bits: depth_bits >= 16)
    m, s, q, c, o = rand_scene(12_000, seed=11)
    B, hh, ww = 8, 512, 512
    kw: KW = {**CAMERA, "img_shape": (hh, ww), "f": (float(hh), float(hh)), "c": (ww // 2, hh // 2)}
    views = jnp.stack([viewmat(0.1 * i) for i in range(B)])
    ref = jnp.stack(
        [splax.render(m, s, q, c, o, viewmat=views[i], **kw)[0] for i in range(B)]
    )  # B=1 renders each pack with depth_bits=21
    splax.clear_cache()
    out = np.asarray(
        jax.jit(jax.vmap(lambda vm: splax.render(m, s, q, c, o, viewmat=vm, **kw)[0]))(views)
    )  # B=8 render packs with depth_bits=18 (image 3 + tile 10)
    assert _cache._scratch_cache[str(wp.get_device("cuda:0"))]["isect_dtype"] == wp.int32
    d = np.abs(out - np.asarray(ref))
    mse = float(np.mean((out - np.asarray(ref)) ** 2))
    psnr = 99.0 if mse == 0 else -10 * np.log10(mse)
    assert d.max() < 0.05, f"packed batched vs stacked max abs diff {d.max():.2e}"
    assert psnr > 65, f"packed batched vs stacked PSNR only {psnr:.1f} dB"
    splax.clear_cache()
