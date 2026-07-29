"""Unit tests for the projection stage.

``splax.project`` is walked along one matrix of aspects. A plain call, the vmap that has to match
the loop over its single calls, mixed batched and shared operands, the gradient, and the gradient
under vmap. Parity against the gsplat reference lives in ``tests/integration/test_project.py``.

The projection is differentiable with respect to the means, scales, quaternions and the viewmat.
Opacities only drive the integer tile counts and carry no gradient, so the gradient tests keep them
shared and differentiate the four remaining operands.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from utils import VIEWMAT, VIEWS, assert_finite_difference, camera, poses, scene

import splax

B = len(VIEWS)


def _assert_matches_loop(batched: tuple[jax.Array, ...], refs: list[tuple[jax.Array, ...]]):
    """Assert each per-image slice of a batched projection is bit-exact against its single call.

    ``cum_tiles_hit`` is excluded because it scans the whole batch and has no per-image counterpart.
    """
    for i, ref in enumerate(refs):
        for batched_out, ref_out in zip(batched[:5], ref[:5]):
            np.testing.assert_array_equal(np.asarray(batched_out[i]), np.asarray(ref_out))


def _weights(n: int, seed: int) -> tuple[jax.Array, ...]:
    """Draw the loss weights of the xys, depths and conics of ``n`` gaussians.

    The weights are signed, which keeps the loss close to zero so that a float32 loss still
    resolves a finite-difference step instead of drowning it in cancellation.
    """
    keys = jax.random.split(jax.random.key(seed), 3)
    shapes = [(n, 2), (n,), (n, 3)]
    return tuple(jax.random.uniform(k, s, minval=-1.0) for k, s in zip(keys, shapes))


def _loss(
    mean3ds: jax.Array,
    scales: jax.Array,
    quats: jax.Array,
    viewmat: jax.Array,
    *,
    opacities: jax.Array,
    weights: tuple[jax.Array, ...],
    img_shape: tuple[int, int],
) -> jax.Array:
    """Reduce a projection to the weighted scalar the gradient tests differentiate.

    Screen-space centers, depths and conics all enter, so every differentiable operand receives an
    O(1) gradient. Scales and quaternions reach the loss through the conics alone.
    """
    xys, depths, _radii, conics, _nth, _cum = splax.project(
        mean3ds, scales, quats, viewmat, opacities=opacities, **camera(*img_shape)
    )
    w_xys, w_depths, w_conics = weights
    return jnp.sum(w_xys * xys) + jnp.sum(w_depths * depths) + jnp.sum(w_conics * conics)


# region regular invocation


def test_project():
    """Project a scene and check the outputs against the pinhole model they encode."""
    n, H, W = 3000, 128, 128
    means, scales, quats, _colors, opacities, _bg = scene(n, seed=1, dense=True)
    kw = camera(H, W)
    xys, depths, radii, conics, n_tiles_hit, cum_tiles_hit = splax.project(
        means, scales, quats, VIEWMAT, opacities=opacities, **kw
    )
    assert xys.shape == (n, 2)
    assert depths.shape == radii.shape == n_tiles_hit.shape == cum_tiles_hit.shape == (n,)
    assert conics.shape == (n, 3)

    live = np.asarray(radii) > 0
    assert live.mean() > 0.5, f"only {live.mean():.1%} of a centered scene survived culling"
    # The reference runs in float64 numpy, so the GPU's tf32 matmul rounding stays out of it.
    view = np.asarray(VIEWMAT, np.float64)
    cam = np.asarray(means, np.float64) @ view[:3, :3].T + view[:3, 3]
    xys_ref = np.array(kw["f"]) * cam[:, :2] / cam[:, 2:] + np.array(kw["c"])
    np.testing.assert_allclose(np.asarray(xys)[live], xys_ref[live], atol=1e-3)
    np.testing.assert_allclose(np.asarray(depths)[live], cam[live, 2], rtol=1e-6)

    # The conic is the inverse of the dilated 2d covariance, so it is positive definite wherever a
    # gaussian survives.
    a, b, c = (np.asarray(conics)[live, i] for i in range(3))
    assert np.all(a > 0) and np.all(c > 0) and np.all(a * c - b * b > 0)
    # Culled gaussians are written as exact zeros, and cum_tiles_hit is the inclusive scan.
    assert not np.asarray(xys)[~live].any()
    assert not np.asarray(depths)[~live].any()
    assert not np.asarray(conics)[~live].any()
    assert not np.asarray(n_tiles_hit)[~live].any()
    nth = np.asarray(n_tiles_hit).astype(np.int64)
    np.testing.assert_array_equal(np.asarray(cum_tiles_hit).astype(np.int64), np.cumsum(nth))


def test_project_principal_point_default():
    """Leaving the principal point out puts it at the image center.

    The image is deliberately not square, so a principal point that swaps its axes lands the
    projection somewhere else.
    """
    n, H, W = 3000, 128, 192
    means, scales, quats, _colors, opacities, _bg = scene(n, seed=2, dense=True)
    kw = camera(H, W)
    del kw["c"]
    xys, _depths, radii, *_ = splax.project(
        means, scales, quats, VIEWMAT, opacities=opacities, **kw
    )
    view = np.asarray(VIEWMAT, np.float64)
    cam = np.asarray(means, np.float64) @ view[:3, :3].T + view[:3, 3]
    xys_ref = np.array(kw["f"]) * cam[:, :2] / cam[:, 2:] + np.array([W / 2, H / 2])
    live = np.asarray(radii) > 0
    assert live.any()
    np.testing.assert_allclose(np.asarray(xys)[live], xys_ref[live], atol=1e-3)


def test_opacity_compensation():
    """Rebuild the compensation factor from the conic and bound it to [0, 1].

    The factor is the square root of the determinant ratio of the undilated and the dilated 2D
    covariance, and a culled gaussian carries no dilation to compensate, so it reads exactly 1.
    """
    n, H, W = 3000, 128, 128
    means, scales, quats, _colors, opacities, _bg = scene(n, seed=1)
    _xys, _depths, radii, conics, _nth, _cum = splax.project(
        means, scales, quats, VIEWMAT, opacities=opacities, **camera(H, W)
    )
    rho = np.asarray(splax.opacity_compensation(conics, radii))
    conics = np.asarray(conics)
    radii = np.asarray(radii)
    eps = 0.3
    # Reference: rebuild the dilated 2D covariance from the conic, which is its inverse, strip the
    # eps dilation, take the det ratio directly.
    a, b, c = conics[:, 0], conics[:, 1], conics[:, 2]
    live = radii > 0
    det_conic = a * c - b * b
    det_d = np.where(live, 1.0 / np.where(det_conic == 0, 1.0, det_conic), 1.0)
    cxx, cyy, cxy = c * det_d, a * det_d, -b * det_d
    det_o = (cxx - eps) * (cyy - eps) - cxy * cxy
    ref = np.sqrt(np.clip(np.where(live, det_o / det_d, 1.0), 0.0, 1.0))
    assert np.allclose(rho[live], ref[live], atol=1e-5), (
        f"max deviation {np.abs(rho[live] - ref[live]).max():.2e}"
    )
    assert np.all(rho >= 0.0) and np.all(rho <= 1.0), "the compensation leaves [0, 1]"
    assert np.allclose(rho[~live], 1.0), "culled gaussians must not be compensated"
    assert rho[live].min() < 0.98, "no thin gaussian is compensated at all"


# region batching


def test_project_vmap_matches_loop():
    """Match a vmap over a batch of scenes and cameras against the loop over its single calls.

    ``splax.project`` carries vmap_method="expand_dims" and launches a single grid over the whole
    batch. Every output except ``cum_tiles_hit`` is per-gaussian and batch-invariant, so it matches
    the stacked unbatched projections exactly. ``cum_tiles_hit`` is intentionally a *global*
    inclusive prefix sum across the whole batch, the gsplat single-sort layout in which every
    image's intersections lie contiguously, so it equals the global cumsum of the flattened
    ``n_tiles_hit`` rather than the per-image cumsum.
    """
    n, H, W = 8_000, 128, 128
    draws = [scene(n, seed=seed, dense=True) for seed in range(B)]
    means, scales, quats = (jnp.stack([draw[i] for draw in draws]) for i in range(3))
    opacities = draws[0][4]
    project = partial(splax.project, opacities=opacities, **camera(H, W))

    batched = jax.vmap(project)(means, scales, quats, VIEWS)
    refs = [project(means[i], scales[i], quats[i], VIEWS[i]) for i in range(B)]
    _assert_matches_loop(batched, refs)

    nth = np.asarray(batched[4]).reshape(-1).astype(np.int64)
    cum = np.asarray(batched[5]).reshape(-1).astype(np.int64)
    np.testing.assert_array_equal(cum, np.cumsum(nth))


def test_project_broadcast():
    """Mixed batched and shared operands match the loop that spells the batch out by hand."""
    n, H, W = 4000, 128, 128
    means, scales, quats, _colors, opacities, _bg = scene(n, seed=3, dense=True)
    batched_means = means + 0.05 * jax.random.normal(jax.random.key(4), (B, n, 3))
    batched_opacities = opacities * jnp.linspace(0.2, 1.0, B)[:, None]
    project = partial(splax.project, **camera(H, W))

    # Shared gaussians, one camera per image, the multi-view regime.
    batched = jax.vmap(partial(project, opacities=opacities), in_axes=(None, None, None, 0))(
        means, scales, quats, VIEWS
    )
    refs = [project(means, scales, quats, viewmat, opacities=opacities) for viewmat in VIEWS]
    _assert_matches_loop(batched, refs)

    # Per-image means, shared scales, quaternions and camera.
    batched = jax.vmap(partial(project, opacities=opacities), in_axes=(0, None, None, None))(
        batched_means, scales, quats, VIEWMAT
    )
    refs = [project(m, scales, quats, VIEWMAT, opacities=opacities) for m in batched_means]
    _assert_matches_loop(batched, refs)

    # Per-image opacities over a fully shared scene. Opacities drive the tile counts, so only the
    # integer outputs move.
    batched = jax.vmap(project, in_axes=(None, None, None, None))(
        means, scales, quats, VIEWMAT, opacities=batched_opacities
    )
    refs = [project(means, scales, quats, VIEWMAT, opacities=o) for o in batched_opacities]
    _assert_matches_loop(batched, refs)


# region gradient


def test_project_grad():
    """Validate the gradient of every differentiable operand with directional finite differences.

    Stepping along the gradient of one operand at a time maximizes the directional-derivative
    signal against float32 noise and attributes a mismatch to that operand.
    """
    n, H, W = 400, 96, 96
    means, scales, quats, _colors, opacities, _bg = scene(n, seed=5)
    loss = partial(_loss, opacities=opacities, weights=_weights(n, seed=6), img_shape=(H, W))
    args = (means, scales, quats, VIEWMAT)
    names = ["means", "scales", "quats", "viewmat"]

    grads = jax.grad(loss, argnums=(0, 1, 2, 3))(*args)
    for name, g in zip(names, grads):
        assert np.all(np.isfinite(np.asarray(g))), f"{name} gradient has non-finite entries"
        assert np.linalg.norm(np.asarray(g)) > 0, f"{name} gradient is all zero"
    assert np.allclose(np.asarray(grads[3])[3], 0.0), "the viewmat bottom row is constant"

    for i, name in enumerate(names):

        def perturbed(operand: jax.Array, i: int = i) -> jax.Array:
            return loss(*args[:i], operand, *args[i + 1 :])

        assert_finite_difference(
            perturbed, (args[i],), (grads[i],), eps=1e-3, rtol=5e-3, name=f"{name} "
        )


def test_project_quat_scale():
    """Scaling the quaternions leaves the projection unchanged and rescales its gradient.

    The rotation is built from the normalized quaternion, so the outputs depend on the quaternion
    direction alone while the gradient carries the ``1 / |q|`` factor of that normalization. A
    power-of-two scale keeps both relations bit-exact in float32.
    """
    n, H, W = 400, 96, 96
    means, scales, quats, _colors, opacities, _bg = scene(n, seed=15)
    loss = partial(_loss, opacities=opacities, weights=_weights(n, seed=16), img_shape=(H, W))
    grad = jax.grad(loss, argnums=2)

    assert float(loss(means, scales, 2.0 * quats, VIEWMAT)) == float(
        loss(means, scales, quats, VIEWMAT)
    )
    scaled = np.asarray(grad(means, scales, 2.0 * quats, VIEWMAT))
    np.testing.assert_array_equal(2.0 * scaled, np.asarray(grad(means, scales, quats, VIEWMAT)))


def test_project_grad_vmap_matches_loop():
    """Match vmap(grad) over a batch of scenes and cameras against the loop over single gradients.

    The per-gaussian gradients are written by the thread owning the gaussian, so they stay
    bit-exact. The viewmat gradient accumulates every gaussian of an image with atomics, whose
    order follows the launch geometry, so it is compared numerically.
    """
    n, H, W = 800, 96, 96
    draws = [scene(n, seed=seed) for seed in range(B)]
    means, scales, quats = (jnp.stack([draw[i] for draw in draws]) for i in range(3))
    viewmats = poses(B, seed=8)
    loss = partial(_loss, opacities=draws[0][4], weights=_weights(n, seed=9), img_shape=(H, W))

    grad = jax.grad(loss, argnums=(0, 1, 2, 3))
    batched = jax.vmap(grad)(means, scales, quats, viewmats)
    refs = [grad(means[i], scales[i], quats[i], viewmats[i]) for i in range(B)]
    for i, name in enumerate(["means", "scales", "quats"]):
        ref = np.stack([np.asarray(r[i]) for r in refs])
        np.testing.assert_array_equal(np.asarray(batched[i]), ref, err_msg=name)
    ref_viewmat = np.stack([np.asarray(r[3]) for r in refs])
    np.testing.assert_allclose(np.asarray(batched[3]), ref_viewmat, rtol=1e-4, atol=1e-6)


def test_project_grad_broadcast():
    """The gradient of a shared operand is the sum over the batch axis of the per-image gradients.

    Batching the camera alone makes the projected geometry per-image while the gaussians stay
    shared. The vjp of a broadcast is a sum, so the gradient of the summed batch loss with respect
    to the shared gaussians equals the summed per-image gradients.
    """
    n, H, W = 800, 96, 96
    means, scales, quats, _colors, opacities, _bg = scene(n, seed=11)
    viewmats = poses(B, seed=12)
    loss = partial(_loss, opacities=opacities, weights=_weights(n, seed=13), img_shape=(H, W))

    def batch_loss(m: jax.Array, s: jax.Array, q: jax.Array) -> jax.Array:
        return jnp.sum(jax.vmap(loss, in_axes=(None, None, None, 0))(m, s, q, viewmats))

    grad = jax.grad(loss, argnums=(0, 1, 2))
    batched = jax.grad(batch_loss, argnums=(0, 1, 2))(means, scales, quats)
    per_image = [grad(means, scales, quats, viewmat) for viewmat in viewmats]
    for i, name in enumerate(["means", "scales", "quats"]):
        summed = np.sum([np.asarray(g[i]) for g in per_image], axis=0)
        np.testing.assert_allclose(
            np.asarray(batched[i]), summed, rtol=1e-4, atol=1e-6, err_msg=name
        )
