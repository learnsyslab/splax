"""Test the projection stage across plain calls, batching and gradients."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from utils import VIEWMAT, VIEWS, assert_finite_difference, camera, poses, scene

import splax

if TYPE_CHECKING:
    from collections.abc import Callable

B = len(VIEWS)


def _target(*outputs: jax.Array, seed: int) -> tuple[jax.Array, ...]:
    """Offset a projection into the target the gradient losses are fitted against."""
    keys = jax.random.split(jax.random.key(seed), len(outputs))
    perturb = (jax.random.uniform(k, o.shape, minval=-1.0) for o, k in zip(outputs, keys))
    return tuple(o + 0.01 * jnp.std(o) * p for o, p in zip(outputs, perturb))


def _loss(
    project: Callable[..., tuple[jax.Array, ...]],
    target_xys: jax.Array,
    target_depths: jax.Array,
    target_conics: jax.Array,
    mean3ds: jax.Array,
    scales: jax.Array,
    quats: jax.Array,
    viewmat: jax.Array,
) -> jax.Array:
    """Sum the squared residual between a projection and the target it is fitted against."""
    xys, depths, _, conics, _, _ = project(mean3ds, scales, quats, viewmat)
    return (
        jnp.sum((xys - target_xys) ** 2)
        + jnp.sum((depths - target_depths) ** 2)
        + jnp.sum((conics - target_conics) ** 2)
    )


# region regular invocation


def test_project():
    """Project a scene and check the outputs against the pinhole model they encode."""
    n, H, W = 3000, 128, 192
    means, scales, quats, _, opacities, _ = scene(n, seed=1, dense=True)
    # Distinct focals and an off-center principal point, so no axis or fallback can stand in for
    # another.
    kw = camera(H, W) | {"f": (150.0, 110.0), "c": (100.0, 70.0)}
    project = jax.jit(splax.project, static_argnames=("img_shape", "f", "c"))
    xys, depths, radii, conics, n_tiles_hit, cum_tiles_hit = project(
        means, scales, quats, VIEWMAT, opacities=opacities, **kw
    )
    assert xys.shape == (n, 2)
    assert depths.shape == radii.shape == n_tiles_hit.shape == cum_tiles_hit.shape == (n,)
    assert conics.shape == (n, 3)

    live = radii > 0
    assert live.mean() > 0.5, f"only {live.mean():.1%} of a centered scene survived culling"
    assert not live.all(), "a few gaussians must be culled to test the culled outputs"
    # The reference runs in float64 numpy to have a high-precision reference.
    view = np.asarray(VIEWMAT, np.float64)
    cam = np.asarray(means, np.float64) @ view[:3, :3].T + view[:3, 3]
    xys_ref = np.array(kw["f"]) * cam[:, :2] / cam[:, 2:] + np.array(kw["c"])
    np.testing.assert_allclose(xys[live], xys_ref[live], atol=1e-3)
    np.testing.assert_allclose(depths[live], cam[live, 2], rtol=1e-6)

    # The conic is the inverse of the dilated 2d covariance, so it is positive definite wherever a
    # gaussian survives.
    a, b, c = (conics[live, i] for i in range(3))
    assert np.all(a > 0) and np.all(c > 0) and np.all(a * c - b * b > 0)
    # Culled gaussians are written as exact zeros, and cum_tiles_hit is the inclusive scan.
    assert not xys[~live].any()
    assert not depths[~live].any()
    assert not conics[~live].any()
    assert not n_tiles_hit[~live].any()
    nth = np.asarray(n_tiles_hit, np.int64)
    np.testing.assert_array_equal(np.asarray(cum_tiles_hit, np.int64), np.cumsum(nth))


def test_opacity_compensation_tracks_gaussian_size():
    """Test that growing the gaussians past the dilation leaves less and less to compensate."""
    means, scales, quats, _, opacities, _ = scene(3000, seed=1)
    project = jax.jit(
        partial(
            splax.project, means, scales, quats, VIEWMAT, opacities=opacities, **camera(128, 128)
        ),
        static_argnames="glob_scale",
    )
    sweep = [project(glob_scale=glob_scale) for glob_scale in (0.25, 0.5, 1.0, 2.0, 4.0)]
    rho = np.stack([splax.opacity_compensation(out[3], out[2]) for out in sweep])

    assert (np.diff(rho, axis=0) > 0).all(), "a larger gaussian was compensated more, not less"
    assert rho.min() > 0.0 and rho.max() < 1.0
    assert rho[0].max() < 0.6, "the smallest gaussians are barely compensated"
    assert rho[-1].min() > 0.9, (
        "the largest gaussians are compensated despite dwarfing the dilation"
    )


def test_opacity_compensation_depends_on_size_relative_to_the_dilation():
    """Test that shrinking a gaussian and widening the dilation to match cancel each other."""
    means, scales, quats, _, opacities, _ = scene(3000, seed=1)
    project = jax.jit(partial(splax.project, opacities=opacities, **camera(128, 128)))
    _, _, radii, conics, _, _ = project(means, scales, quats, VIEWMAT)
    # A conic divided by s describes a gaussian s times wider, which a dilation s times wider meets.
    s = 4.0
    wide_conics = splax.opacity_compensation(conics / s, radii, s * 0.4)
    regular_conics = splax.opacity_compensation(conics, radii, 0.4)
    np.testing.assert_allclose(wide_conics, regular_conics, atol=1e-6)


def test_opacity_compensation_spares_culled_gaussians():
    """Test that a gaussian the projection culled is handed back uncompensated."""
    means, scales, quats, _, opacities, _ = scene(3000, seed=1)
    project = jax.jit(partial(splax.project, opacities=opacities, **camera(128, 128)))
    _, _, radii, conics, _, _ = project(means, scales, quats, VIEWMAT)
    rho = splax.opacity_compensation(conics, jnp.zeros_like(radii))
    np.testing.assert_array_equal(rho, np.ones(len(rho), np.float32))


def test_project_jit():
    """Test that project jits with the camera arguments declared static."""
    means, scales, quats, _, opacities, _ = scene(3000, seed=1, dense=True)
    project = jax.jit(partial(splax.project, opacities=opacities, **camera(128, 128)))
    jax.block_until_ready(project(means, scales, quats, VIEWMAT))


# region batching


def test_project_vmap_matches_loop():
    """Test that a vmap over a batch of scenes and cameras matches the loop over single calls."""
    n, H, W = 8_000, 128, 128
    draws = [scene(n, seed=seed, dense=True) for seed in range(B)]
    means, scales, quats = (jnp.stack([draw[i] for draw in draws]) for i in range(3))
    opacities = draws[0][4]
    project = jax.jit(partial(splax.project, opacities=opacities, **camera(H, W)))

    xys, depths, radii, conics, n_tiles_hit, cum_tiles_hit = jax.jit(jax.vmap(project))(
        means, scales, quats, VIEWS
    )
    refs = [project(means[i], scales[i], quats[i], VIEWS[i]) for i in range(B)]
    # cum_tiles_hit scans the whole batch and has no per-image counterpart to stack.
    r_xys, r_depths, r_radii, r_conics, r_n_tiles_hit, _ = (jnp.stack(outs) for outs in zip(*refs))
    np.testing.assert_array_equal(xys, r_xys)
    np.testing.assert_array_equal(depths, r_depths)
    np.testing.assert_array_equal(radii, r_radii)
    np.testing.assert_array_equal(conics, r_conics)
    np.testing.assert_array_equal(n_tiles_hit, r_n_tiles_hit)

    # cum_tiles_hit is a global inclusive prefix sum across the whole batch rather than a
    # per-image cumsum, so compare it against the cumsum of the flattened n_tiles_hit.
    nth = np.asarray(n_tiles_hit, np.int64).reshape(-1)
    cum = np.asarray(cum_tiles_hit, np.int64).reshape(-1)
    np.testing.assert_array_equal(cum, np.cumsum(nth))


def test_project_broadcast_shared_scene():
    """Test that one camera per image over shared gaussians matches the loop over the cameras."""
    n, H, W = 4000, 128, 128
    means, scales, quats, _, opacities, _ = scene(n, seed=3, dense=True)
    project = jax.jit(partial(splax.project, opacities=opacities, **camera(H, W)))

    vmap_project = jax.jit(jax.vmap(project, in_axes=(None, None, None, 0)))
    xys, depths, radii, conics, n_tiles_hit, _ = vmap_project(means, scales, quats, VIEWS)
    refs = [project(means, scales, quats, viewmat) for viewmat in VIEWS]
    r_xys, r_depths, r_radii, r_conics, r_n_tiles_hit, _ = (jnp.stack(outs) for outs in zip(*refs))
    np.testing.assert_array_equal(xys, r_xys)
    np.testing.assert_array_equal(depths, r_depths)
    np.testing.assert_array_equal(radii, r_radii)
    np.testing.assert_array_equal(conics, r_conics)
    np.testing.assert_array_equal(n_tiles_hit, r_n_tiles_hit)


def test_project_broadcast_per_image_means():
    """Test that per-image means over a shared camera match the loop over the means."""
    n, H, W = 4000, 128, 128
    means, scales, quats, _, opacities, _ = scene(n, seed=3, dense=True)
    batched_means = means + 0.05 * jax.random.normal(jax.random.key(4), (B, n, 3))
    project = jax.jit(partial(splax.project, opacities=opacities, **camera(H, W)))

    vmap_project = jax.jit(jax.vmap(project, in_axes=(0, None, None, None)))
    xys, depths, radii, conics, n_tiles_hit, _ = vmap_project(batched_means, scales, quats, VIEWMAT)
    refs = [project(m, scales, quats, VIEWMAT) for m in batched_means]
    r_xys, r_depths, r_radii, r_conics, r_n_tiles_hit, _ = (jnp.stack(outs) for outs in zip(*refs))
    np.testing.assert_array_equal(xys, r_xys)
    np.testing.assert_array_equal(depths, r_depths)
    np.testing.assert_array_equal(radii, r_radii)
    np.testing.assert_array_equal(conics, r_conics)
    np.testing.assert_array_equal(n_tiles_hit, r_n_tiles_hit)


def test_project_broadcast_per_image_opacities():
    """Test that per-image opacities over a shared scene match the loop over the opacities."""
    n, H, W = 4000, 128, 128
    means, scales, quats, _, opacities, _ = scene(n, seed=3, dense=True)
    batched_opacities = opacities * jnp.linspace(0.2, 1.0, B)[:, None]
    project = jax.jit(partial(splax.project, **camera(H, W)))

    # in_axes covers the positional operands, while the mapped opacities arrive as a keyword.
    vmap_project = jax.jit(jax.vmap(project, in_axes=(None, None, None, None)))
    xys, depths, radii, conics, n_tiles_hit, _ = vmap_project(
        means, scales, quats, VIEWMAT, opacities=batched_opacities
    )
    refs = [project(means, scales, quats, VIEWMAT, opacities=o) for o in batched_opacities]
    r_xys, r_depths, r_radii, r_conics, r_n_tiles_hit, _ = (jnp.stack(outs) for outs in zip(*refs))
    np.testing.assert_array_equal(xys, r_xys)
    np.testing.assert_array_equal(depths, r_depths)
    np.testing.assert_array_equal(radii, r_radii)
    np.testing.assert_array_equal(conics, r_conics)
    np.testing.assert_array_equal(n_tiles_hit, r_n_tiles_hit)


# region gradient


def test_project_grad():
    """Test the gradient of every differentiable operand with directional finite differences."""
    n, H, W = 400, 96, 96
    means, scales, quats, _, opacities, _ = scene(n, seed=5)
    args = (means, scales, quats, VIEWMAT)
    project = jax.jit(partial(splax.project, opacities=opacities, **camera(H, W)))
    xys, depths, _, conics, _, _ = project(*args)
    loss = partial(_loss, project, *_target(xys, depths, conics, seed=6))

    grads = jax.jit(jax.grad(loss, argnums=(0, 1, 2, 3)))(*args)
    assert all(np.all(np.isfinite(g)) for g in grads), "a gradient has non-finite entries"
    assert all(np.linalg.norm(g) > 0 for g in grads), "a gradient is all zero"
    np.testing.assert_array_equal(grads[3][3], jnp.zeros(4), err_msg="bottom row is not constant")

    assert_finite_difference(loss, args, grads, eps=1e-3, rtol=5e-4)


def test_project_quat_scale():
    """Test that scaling the quaternions leaves the projection unchanged and scales its gradient."""
    n, H, W = 400, 96, 96
    means, scales, quats, _, opacities, _ = scene(n, seed=15)
    args = (means, scales, quats, VIEWMAT)
    project = jax.jit(partial(splax.project, opacities=opacities, **camera(H, W)))
    xys, depths, _, conics, _, _ = project(*args)
    loss = partial(_loss, project, *_target(xys, depths, conics, seed=16))
    grad = jax.jit(jax.grad(loss, argnums=2))

    assert loss(means, scales, 2.0 * quats, VIEWMAT) == loss(*args)
    scaled = grad(means, scales, 2.0 * quats, VIEWMAT)
    np.testing.assert_array_equal(2.0 * scaled, grad(*args))


def test_project_grad_vmap_matches_loop():
    """Test that vmap(grad) over a batch matches the loop over the single gradients."""
    n, H, W = 800, 96, 96
    draws = [scene(n, seed=seed) for seed in range(B)]
    means, scales, quats = (jnp.stack([draw[i] for draw in draws]) for i in range(3))
    viewmats = poses(B, seed=8)
    project = jax.jit(partial(splax.project, opacities=draws[0][4], **camera(H, W)))
    # One target for the whole batch, taken from the first image and shared by every gradient.
    xys, depths, _, conics, _, _ = project(means[0], scales[0], quats[0], viewmats[0])
    loss = partial(_loss, project, *_target(xys, depths, conics, seed=9))

    grad = jax.jit(jax.grad(loss, argnums=(0, 1, 2, 3)))
    g_means, g_scales, g_quats, g_viewmat = jax.jit(jax.vmap(grad))(means, scales, quats, viewmats)
    refs = [grad(means[i], scales[i], quats[i], viewmats[i]) for i in range(B)]
    r_means, r_scales, r_quats, r_viewmat = (jnp.stack(gs) for gs in zip(*refs))
    # Per-gaussian gradients are written by the thread owning the gaussian, so they stay exact.
    np.testing.assert_array_equal(g_means, r_means)
    np.testing.assert_array_equal(g_scales, r_scales)
    np.testing.assert_array_equal(g_quats, r_quats)
    # The viewmat gradient accumulates over an image's gaussians with atomics, whose order
    # follows the launch geometry, so it is compared numerically instead of exactly.
    np.testing.assert_allclose(g_viewmat, r_viewmat, rtol=1e-4, atol=1e-6)


def test_project_grad_broadcast():
    """Test that the gradient of a shared operand equals the sum of the per-image gradients."""
    n, H, W = 800, 96, 96
    means, scales, quats, _, opacities, _ = scene(n, seed=11)
    viewmats = poses(B, seed=12)
    project = jax.jit(partial(splax.project, opacities=opacities, **camera(H, W)))
    xys, depths, _, conics, _, _ = project(means, scales, quats, viewmats[0])
    loss = partial(_loss, project, *_target(xys, depths, conics, seed=13))

    def batch_loss(m: jax.Array, s: jax.Array, q: jax.Array) -> jax.Array:
        return jnp.sum(jax.vmap(loss, in_axes=(None, None, None, 0))(m, s, q, viewmats))

    grad = jax.jit(jax.grad(loss, argnums=(0, 1, 2)))
    g_means, g_scales, g_quats = jax.jit(jax.grad(batch_loss, argnums=(0, 1, 2)))(
        means, scales, quats
    )
    per_image = [grad(means, scales, quats, viewmat) for viewmat in viewmats]
    s_means, s_scales, s_quats = (jnp.stack(gs).sum(axis=0) for gs in zip(*per_image))
    np.testing.assert_allclose(g_means, s_means, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(g_scales, s_scales, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(g_quats, s_quats, rtol=1e-4, atol=1e-6)
