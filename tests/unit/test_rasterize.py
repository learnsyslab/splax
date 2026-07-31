"""Test the images and alpha maps produced by rasterize and rasterize_depth."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from utils import VIEWMAT, VIEWS, assert_finite_difference, camera, projected, scene

import splax

if TYPE_CHECKING:
    from collections.abc import Callable

B = len(VIEWS)
# rasterize and rasterize_depth differ only in the channel count of the image they return
RASTERIZERS = [pytest.param(splax.rasterize, 3), pytest.param(splax.rasterize_depth, 4)]


@partial(jax.jit, static_argnames=("H", "W"))
def _project_geoms(
    means: jax.Array,
    scales: jax.Array,
    quats: jax.Array,
    opacities: jax.Array,
    viewmat: jax.Array,
    *,
    H: int,
    W: int,
) -> tuple[jax.Array, ...]:
    """Project a splat from ``viewmat``, dropping the tile count rasterize does not take."""
    xys, depths, radii, conics, _, cum = splax.project(
        means, scales, quats, viewmat, opacities=opacities, **camera(H, W)
    )
    return xys, depths, radii, conics, cum


# region regular invocation


def test_rasterize():
    """Show the background through exactly the coverage the alpha map reports as missing."""
    n, H, W = 4000, 96, 96
    colors, opacities, background, *geoms = projected(n, H, W, seed=1)
    rasterize = jax.jit(partial(splax.rasterize, img_shape=(H, W)))

    img, alpha = rasterize(colors, opacities, background, *geoms)
    unlit, unlit_alpha = rasterize(jnp.zeros((n, 3)), opacities, background, *geoms)
    colored, _ = rasterize(colors, opacities, jnp.zeros(3), *geoms)
    assert img.shape == (H, W, 3)
    assert 0.1 < alpha.mean() < 0.9, "the splat covers the image too evenly to see the background"
    # Colors light the splat without moving its coverage.
    np.testing.assert_array_equal(alpha, unlit_alpha)
    transmitted = background * (1.0 - alpha)[..., None]
    np.testing.assert_allclose(unlit, transmitted, rtol=1e-5, atol=1e-6)
    # Colors and background reach the pixel independently of one another.
    np.testing.assert_allclose(img, colored + unlit, rtol=1e-5, atol=1e-6)


def test_rasterize_alpha():
    """Read the accumulated alpha of a splat that saturates in the centre and misses the corners."""
    n, H, W = 600, 96, 96
    means, scales, quats, _, _, _ = scene(n, seed=8)
    means, opacities = means * 0.15, jnp.full((n,), 0.9)  # a tight, opaque cluster
    geoms = _project_geoms(means, scales, quats, opacities, VIEWMAT, H=H, W=W)
    rasterize = jax.jit(partial(splax.rasterize, img_shape=(H, W)))

    img, alpha = rasterize(jnp.ones((n, 3)), opacities, jnp.zeros(3), *geoms)
    assert alpha.shape == (H, W)
    assert (alpha >= 0.0).all() and (alpha <= 1.0).all(), "the accumulated alpha leaves [0, 1]"
    assert alpha[0, 0] == 0.0, "the image corner is uncovered"
    assert alpha.max() > 0.99, "the cluster does not saturate anywhere"
    # White gaussians over a black background composite to the coverage itself.
    coverage = jnp.broadcast_to(alpha[..., None], (H, W, 3))
    np.testing.assert_allclose(img, coverage, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("rasterize", [splax.rasterize, splax.rasterize_depth])
def test_rasterize_antialiased_dims_the_coverage(
    rasterize: Callable[..., tuple[jax.Array, jax.Array]],
):
    """Scale the coverage down everywhere by the Mip-Splatting compensation."""
    n, H, W = 4000, 96, 96
    colors, opacities, background, *geoms = projected(n, H, W, seed=9)
    rasterize = jax.jit(partial(rasterize, img_shape=(H, W)), static_argnames="antialiased")

    _, alpha = rasterize(colors, opacities, background, *geoms)
    _, compensated = rasterize(colors, opacities, background, *geoms, antialiased=True)
    # The compensation is bounded by 1, so it can only ever take coverage away.
    assert (compensated <= alpha + 1e-6).all(), "the compensation raised the coverage"
    assert compensated.max() < alpha.max(), "the compensation left the coverage untouched"


def test_rasterize_depth():
    """Read back a single gaussian's own camera-space depth on every pixel it covers."""
    H = W = 64
    means = jnp.array([[0.1, -0.05, 0.0]])
    scales = jnp.full((1, 3), 0.12)
    quats = jnp.array([[1.0, 0.0, 0.0, 0.0]])
    opacities = jnp.array([0.9])
    viewmat = jnp.eye(4).at[2, 3].set(4.0)
    camera_z = (viewmat[:3, :3] @ means[0] + viewmat[:3, 3])[2]
    geoms = _project_geoms(means, scales, quats, opacities, viewmat, H=H, W=W)
    rasterize_depth = jax.jit(partial(splax.rasterize_depth, img_shape=(H, W)))

    img, alpha = rasterize_depth(jnp.full((1, 3), 0.5), opacities, jnp.zeros(3), *geoms)
    depth = img[..., 3]
    covered = alpha > 0.0

    assert img.shape == (H, W, 4)
    deviation = jnp.abs(depth[covered] - camera_z).max()
    assert deviation < 1e-4, f"max deviation {deviation:.2e}"
    # the depth carries camera units and is nowhere near the [0, 1] the alpha lives in
    assert depth[covered].min() > 1.0, "the depth is not a metric camera distance"
    assert alpha[covered].min() < 0.05 < 0.5 < alpha[covered].max(), "the coverage barely varies"
    assert not covered[0, 0] and depth[0, 0] == 0.0, "the image corner is background"
    assert covered.mean() > 0.01, "the gaussian barely contributes"


def test_rasterize_depth_image_byte_identical():
    """Return the plain rasterize image and alpha unchanged from the depth path."""
    n, H, W = 4000, 128, 128
    args = projected(n, H, W, seed=1)
    plain, plain_alpha = jax.jit(partial(splax.rasterize, img_shape=(H, W)))(*args)
    img, alpha = jax.jit(partial(splax.rasterize_depth, img_shape=(H, W)))(*args)
    np.testing.assert_array_equal(plain, img[..., :3])
    np.testing.assert_array_equal(plain_alpha, alpha)


@pytest.mark.parametrize("rasterize", [splax.rasterize, splax.rasterize_depth])
def test_rasterize_jit(rasterize: Callable[..., tuple[jax.Array, jax.Array]]):
    """Test that the blend traces and runs with the image shape declared static."""
    H = W = 128
    args = projected(4000, H, W, seed=1)
    jax.block_until_ready(jax.jit(rasterize, static_argnames="img_shape")(*args, img_shape=(H, W)))


# region batching


@pytest.mark.usefixtures("faithful_64bit_keys")
@pytest.mark.parametrize("rasterize,channels", RASTERIZERS)
def test_rasterize_vmap_matches_loop(
    rasterize: Callable[..., tuple[jax.Array, jax.Array]], channels: int
):
    """Match the batched rasterize of B views against the loop over the unbatched calls."""
    n, H, W = 4000, 96, 96
    means, scales, quats, colors, opacities, background = scene(n, seed=1, dense=True)
    project = partial(_project_geoms, means, scales, quats, opacities, H=H, W=W)
    batched = jax.vmap(project)(VIEWS)
    rasterize = jax.jit(partial(rasterize, colors, opacities, background, img_shape=(H, W)))

    img, alpha = jax.jit(jax.vmap(rasterize))(*batched)
    refs = [rasterize(*project(viewmat)) for viewmat in VIEWS]
    r_img, r_alpha = (jnp.stack(outs) for outs in zip(*refs))
    assert img.shape == (B, H, W, channels) and alpha.shape == (B, H, W)
    np.testing.assert_array_equal(img, r_img)
    np.testing.assert_array_equal(alpha, r_alpha)


@pytest.mark.usefixtures("faithful_64bit_keys")
@pytest.mark.parametrize("rasterize,channels", RASTERIZERS)
def test_rasterize_broadcast(rasterize: Callable[..., tuple[jax.Array, jax.Array]], channels: int):
    """Share colors, opacities, and background across a batched geometry."""
    n, H, W = 4000, 96, 96
    means, scales, quats, colors, opacities, background = scene(n, seed=3, dense=True)
    batched = jax.vmap(partial(_project_geoms, means, scales, quats, opacities, H=H, W=W))(VIEWS)
    rasterize = jax.jit(partial(rasterize, img_shape=(H, W)))

    broadcast = jax.jit(jax.vmap(rasterize, in_axes=(None, None, None, 0, 0, 0, 0, 0)))
    shared_img, shared_alpha = broadcast(colors, opacities, background, *batched)
    tiled_img, tiled_alpha = jax.jit(jax.vmap(rasterize))(
        jnp.broadcast_to(colors, (B, n, 3)),
        jnp.broadcast_to(opacities, (B, n)),
        jnp.broadcast_to(background, (B, 3)),
        *batched,
    )
    assert shared_img.shape == (B, H, W, channels)
    np.testing.assert_array_equal(shared_img, tiled_img)
    np.testing.assert_array_equal(shared_alpha, tiled_alpha)


@pytest.mark.parametrize("rasterize,channels", RASTERIZERS)
def test_rasterize_broadcast_geometry(
    rasterize: Callable[..., tuple[jax.Array, jax.Array]], channels: int
):
    """Share one projection across a batch of appearances."""
    n, H, W = 4000, 96, 96
    means, scales, quats, colors, opacities, background = scene(n, seed=3, dense=True)
    geoms = _project_geoms(means, scales, quats, opacities, VIEWMAT, H=H, W=W)
    batch_colors = colors * jnp.linspace(0.4, 1.0, B)[:, None, None]
    rasterize = jax.jit(partial(rasterize, img_shape=(H, W)))

    broadcast = jax.jit(jax.vmap(rasterize, in_axes=(0, None, None, None, None, None, None, None)))
    img, alpha = broadcast(batch_colors, opacities, background, *geoms)
    refs = [rasterize(c, opacities, background, *geoms) for c in batch_colors]
    r_img, r_alpha = (jnp.stack(outs) for outs in zip(*refs))
    assert img.shape == (B, H, W, channels) and alpha.shape == (B, H, W)
    assert img.std(axis=0).max() > 0.0, "the batch renders the same image B times"
    assert (alpha == alpha[0]).all(), "the appearance changed the coverage"
    np.testing.assert_array_equal(img, r_img)
    np.testing.assert_array_equal(alpha, r_alpha)


# region gradient

# The finite-difference losses draw signed weights to surface local changes


def test_rasterize_grad():
    """Check the rasterize gradients against a central-difference directional derivative."""
    n, H, W = 400, 80, 80
    colors, opacities, background, xys, depths, radii, conics, cum = projected(
        n, H, W, seed=7, dense=False
    )
    rasterize = jax.jit(partial(splax.rasterize, img_shape=(H, W)))
    w = jax.random.uniform(jax.random.key(5), (H, W, 3), minval=-1.0)

    def loss(c: jax.Array, o: jax.Array, xy: jax.Array, cn: jax.Array) -> jax.Array:
        img, _ = rasterize(c, o, background, xy, depths, radii, cn, cum)
        return jnp.mean(w * img)

    args = (colors, opacities, xys, conics)
    grads = jax.jit(jax.grad(loss, argnums=(0, 1, 2, 3)))(*args)
    assert all(jnp.linalg.norm(g) > 0.0 for g in grads), "the rasterize drops a gradient path"
    assert_finite_difference(loss, args, grads, eps=1e-3)


def test_rasterize_alpha_grad():
    """Check the accumulated alpha gradients against a central-difference directional derivative."""
    n, H, W = 400, 80, 80
    colors, opacities, background, xys, depths, radii, conics, cum = projected(
        n, H, W, seed=7, dense=False
    )
    rasterize = jax.jit(partial(splax.rasterize, img_shape=(H, W)))
    w = jax.random.uniform(jax.random.key(11), (H, W), minval=-1.0)

    def loss(c: jax.Array, o: jax.Array, xy: jax.Array, cn: jax.Array) -> jax.Array:
        _, alpha = rasterize(c, o, background, xy, depths, radii, cn, cum)
        return jnp.mean(w * alpha)

    args = (colors, opacities, xys, conics)
    grads = jax.jit(jax.grad(loss, argnums=(0, 1, 2, 3)))(*args)
    assert jnp.linalg.norm(grads[0]) == 0.0, "the alpha does not depend on the colors"
    assert all(jnp.linalg.norm(g) > 0.0 for g in grads[1:]), "the alpha drops a path"
    assert_finite_difference(loss, args, grads, eps=1e-3, name="alpha ")


@pytest.mark.parametrize("mode", ["depth_only", "mixed"])
def test_rasterize_depth_grad(mode: str):
    """Check the rasterize_depth gradients against a central-difference directional derivative."""
    n, H, W = 400, 80, 80
    colors, opacities, background, xys, depths, radii, conics, cum = projected(
        n, H, W, seed=7, dense=False
    )
    rasterize_depth = jax.jit(partial(splax.rasterize_depth, img_shape=(H, W)))
    wd = jax.random.uniform(jax.random.key(9), (H, W), minval=-1.0)
    wc = jax.random.uniform(jax.random.key(10), (H, W, 3), minval=-1.0)

    def loss(c: jax.Array, o: jax.Array, xy: jax.Array, cn: jax.Array, d: jax.Array) -> jax.Array:
        img, _ = rasterize_depth(c, o, background, xy, d, radii, cn, cum)
        depth_loss = jnp.mean(wd * img[..., 3])
        return depth_loss if mode == "depth_only" else jnp.mean(wc * img[..., :3]) + depth_loss

    args = (colors, opacities, xys, conics, depths)
    grads = jax.jit(jax.grad(loss, argnums=(0, 1, 2, 3, 4)))(*args)
    if mode == "depth_only":
        assert jnp.linalg.norm(grads[0]) == 0.0, "depth does not depend on the colors"
    assert jnp.linalg.norm(grads[2]) > 0.0, "the geometry gradient drives the regularizer"
    assert jnp.linalg.norm(grads[4]) > 0.0, "depths carry a cotangent"
    assert_finite_difference(loss, args, grads, eps=1e-3, name=f"{mode} ")


def test_rasterize_depth_grad_empty_pixels():
    """Keep the depth gradient finite on an image a single gaussian barely covers."""
    H = W = 64
    means = jnp.array([[0.1, -0.05, 0.0]])
    scales = jnp.full((1, 3), 0.12)
    quats = jnp.array([[1.0, 0.0, 0.0, 0.0]])
    opacities = jnp.array([0.9])
    viewmat = jnp.eye(4).at[2, 3].set(4.0)
    xys, depths, radii, conics, cum = _project_geoms(
        means, scales, quats, opacities, viewmat, H=H, W=W
    )
    grey = jnp.full((1, 3), 0.5)
    rasterize_depth = jax.jit(partial(splax.rasterize_depth, grey, img_shape=(H, W)))

    img, _ = rasterize_depth(opacities, jnp.zeros(3), xys, depths, radii, conics, cum)
    empty = jnp.mean(img[..., 3] == 0.0)
    assert 0.5 < empty < 1.0, f"only {1 - empty:.0%} of the image is empty"

    def depth_sum(o: jax.Array, xy: jax.Array, cn: jax.Array, d: jax.Array) -> jax.Array:
        depth_img, _ = rasterize_depth(o, jnp.zeros(3), xy, d, radii, cn, cum)
        return jnp.sum(depth_img[..., 3])

    grads = jax.jit(jax.grad(depth_sum, argnums=(0, 1, 2, 3)))(opacities, xys, conics, depths)
    assert all(jnp.all(jnp.isfinite(g)) for g in grads), "empty pixels leak a NaN or inf gradient"
    assert jnp.linalg.norm(grads[3]) > 0.0, "depths carry a cotangent"


def test_rasterize_grad_vmap_matches_loop():
    """Match the batch-native geometry gradients against the loop over the unbatched gradients."""
    n, H, W = 2000, 96, 96
    means, scales, quats, colors, opacities, background = scene(n, seed=2)
    project = partial(_project_geoms, means, scales, quats, opacities, H=H, W=W)
    batched = jax.vmap(project)(VIEWS)
    rasterize = jax.jit(partial(splax.rasterize, colors, opacities, background, img_shape=(H, W)))
    w = jax.random.uniform(jax.random.key(5), (B, H, W, 3))

    def loss(
        xy: jax.Array, d: jax.Array, r: jax.Array, cn: jax.Array, cm: jax.Array, weight: jax.Array
    ) -> jax.Array:
        img, _ = rasterize(xy, d, r, cn, cm)
        return jnp.sum(weight * img)

    grad = jax.jit(jax.grad(loss, argnums=(0, 3)))
    g_xys, g_conics = jax.jit(jax.vmap(grad))(*batched, w)
    refs = [grad(*project(viewmat), weight) for viewmat, weight in zip(VIEWS, w)]
    r_xys, r_conics = (jnp.stack(gs) for gs in zip(*refs))
    np.testing.assert_allclose(g_xys, r_xys, rtol=2e-3, atol=1e-4)
    np.testing.assert_allclose(g_conics, r_conics, rtol=2e-3, atol=1e-4)


def test_rasterize_depth_grad_vmap_matches_loop():
    """Match the batch-native rasterize_depth gradients against the loop over the unbatched ones."""
    n, H, W = 2000, 96, 96
    means, scales, quats, colors, opacities, background = scene(n, seed=2)
    project = partial(_project_geoms, means, scales, quats, opacities, H=H, W=W)
    batched = jax.vmap(project)(VIEWS)
    rasterize_depth = jax.jit(
        partial(splax.rasterize_depth, colors, opacities, background, img_shape=(H, W))
    )
    wd = jax.random.uniform(jax.random.key(6), (B, H, W))

    def loss(
        xy: jax.Array, d: jax.Array, r: jax.Array, cn: jax.Array, cm: jax.Array, weight: jax.Array
    ) -> jax.Array:
        img, _ = rasterize_depth(xy, d, r, cn, cm)
        return jnp.sum(weight * img[..., 3])

    grad = jax.jit(jax.grad(loss, argnums=(0, 1, 3)))
    g_xys, g_depths, g_conics = jax.jit(jax.vmap(grad))(*batched, wd)
    refs = [grad(*project(viewmat), weight) for viewmat, weight in zip(VIEWS, wd)]
    r_xys, r_depths, r_conics = (jnp.stack(gs) for gs in zip(*refs))
    np.testing.assert_allclose(g_xys, r_xys, rtol=2e-3, atol=1e-4)
    np.testing.assert_allclose(g_depths, r_depths, rtol=2e-3, atol=1e-4)
    np.testing.assert_allclose(g_conics, r_conics, rtol=2e-3, atol=1e-4)


@pytest.mark.parametrize("rasterize,channels", RASTERIZERS)
def test_rasterize_grad_broadcast(
    rasterize: Callable[..., tuple[jax.Array, jax.Array]], channels: int
):
    """Sum the per-image gradients into an operand shared across the batch."""
    n, H, W = 2000, 96, 96
    means, scales, quats, colors, opacities, background = scene(n, seed=4)
    project = partial(_project_geoms, means, scales, quats, opacities, H=H, W=W)
    batched = jax.vmap(project)(VIEWS)
    rasterize = jax.jit(partial(rasterize, img_shape=(H, W)))
    w = jax.random.uniform(jax.random.key(7), (B, H, W, channels))

    def loss(c: jax.Array, o: jax.Array) -> jax.Array:
        imgs, _ = jax.vmap(rasterize, in_axes=(None, None, None, 0, 0, 0, 0, 0))(
            c, o, background, *batched
        )
        return jnp.sum(w * imgs)

    def view_loss(c: jax.Array, o: jax.Array, viewmat: jax.Array, weight: jax.Array) -> jax.Array:
        img, _ = rasterize(c, o, background, *project(viewmat))
        return jnp.sum(weight * img)

    g_colors, g_opacities = jax.jit(jax.grad(loss, argnums=(0, 1)))(colors, opacities)
    view_grad = jax.jit(jax.grad(view_loss, argnums=(0, 1)))
    per_view = [view_grad(colors, opacities, viewmat, weight) for viewmat, weight in zip(VIEWS, w)]
    s_colors, s_opacities = (jnp.stack(gs).sum(axis=0) for gs in zip(*per_view))
    np.testing.assert_allclose(g_colors, s_colors, rtol=2e-3, atol=1e-4)
    np.testing.assert_allclose(g_opacities, s_opacities, rtol=2e-3, atol=1e-4)


def test_rasterize_alpha_grad_broadcast():
    """Sum the per-image alpha gradients into an operand shared across the batch."""
    n, H, W = 2000, 96, 96
    means, scales, quats, colors, opacities, background = scene(n, seed=4)
    project = partial(_project_geoms, means, scales, quats, opacities, H=H, W=W)
    batched = jax.vmap(project)(VIEWS)
    rasterize = jax.jit(partial(splax.rasterize, img_shape=(H, W)))
    w = jax.random.uniform(jax.random.key(12), (B, H, W))

    def loss(o: jax.Array) -> jax.Array:
        _, alphas = jax.vmap(rasterize, in_axes=(None, None, None, 0, 0, 0, 0, 0))(
            colors, o, background, *batched
        )
        return jnp.sum(w * alphas)

    def view_loss(o: jax.Array, viewmat: jax.Array, weight: jax.Array) -> jax.Array:
        _, alpha = rasterize(colors, o, background, *project(viewmat))
        return jnp.sum(weight * alpha)

    g_opacities = jax.jit(jax.grad(loss))(opacities)
    view_grad = jax.jit(jax.grad(view_loss))
    per_view = [view_grad(opacities, viewmat, weight) for viewmat, weight in zip(VIEWS, w)]
    assert jnp.linalg.norm(g_opacities) > 0.0, "the alpha drops the opacity gradient"
    np.testing.assert_allclose(g_opacities, jnp.stack(per_view).sum(axis=0), rtol=2e-3, atol=1e-4)


def test_rasterize_grad_broadcast_geometry():
    """Split the gradients of a batch of appearances over the projection they share."""
    n, H, W = 2000, 96, 96
    means, scales, quats, colors, opacities, background = scene(n, seed=4)
    xys, depths, radii, conics, cum = _project_geoms(
        means, scales, quats, opacities, VIEWMAT, H=H, W=W
    )
    batch_colors = colors * jnp.linspace(0.4, 1.0, B)[:, None, None]
    rasterize = jax.jit(partial(splax.rasterize, img_shape=(H, W)))
    w = jax.random.uniform(jax.random.key(13), (B, H, W, 3))
    wa = jax.random.uniform(jax.random.key(14), (B, H, W))

    def loss(c: jax.Array, o: jax.Array, xy: jax.Array, cn: jax.Array) -> jax.Array:
        imgs, alphas = jax.vmap(rasterize, in_axes=(0, None, None, None, None, None, None, None))(
            c, o, background, xy, depths, radii, cn, cum
        )
        return jnp.sum(w * imgs) + jnp.sum(wa * alphas)

    def view_loss(
        c: jax.Array, o: jax.Array, xy: jax.Array, cn: jax.Array, wi: jax.Array, wai: jax.Array
    ) -> jax.Array:
        img, alpha = rasterize(c, o, background, xy, depths, radii, cn, cum)
        return jnp.sum(wi * img) + jnp.sum(wai * alpha)

    args = (batch_colors, opacities, xys, conics)
    grads = jax.jit(jax.grad(loss, argnums=(0, 1, 2, 3)))(*args)
    g_colors, g_opacities, g_xys, g_conics = grads
    view_grad = jax.jit(jax.grad(view_loss, argnums=(0, 1, 2, 3)))
    per_view = [
        view_grad(c, opacities, xys, conics, wi, wai) for c, wi, wai in zip(batch_colors, w, wa)
    ]
    r_colors, r_opacities, r_xys, r_conics = (jnp.stack(gs) for gs in zip(*per_view))
    assert all(jnp.linalg.norm(g) > 0.0 for g in grads), "the rasterize drops a gradient path"
    # Each appearance keeps its own gradient, the shared projection collects the sum over the batch.
    np.testing.assert_allclose(g_colors, r_colors, rtol=2e-3, atol=1e-4)
    np.testing.assert_allclose(g_opacities, r_opacities.sum(axis=0), rtol=2e-3, atol=1e-4)
    np.testing.assert_allclose(g_xys, r_xys.sum(axis=0), rtol=2e-3, atol=1e-4)
    np.testing.assert_allclose(g_conics, r_conics.sum(axis=0), rtol=2e-3, atol=1e-4)


def test_rasterize_depth_grad_broadcast_geometry():
    """Split the depth gradients of a batch of appearances over the projection they share."""
    n, H, W = 2000, 96, 96
    means, scales, quats, colors, opacities, background = scene(n, seed=4)
    xys, depths, radii, conics, cum = _project_geoms(
        means, scales, quats, opacities, VIEWMAT, H=H, W=W
    )
    batch_colors = colors * jnp.linspace(0.4, 1.0, B)[:, None, None]
    rasterize_depth = jax.jit(partial(splax.rasterize_depth, img_shape=(H, W)))
    w = jax.random.uniform(jax.random.key(15), (B, H, W, 4))

    def loss(c: jax.Array, o: jax.Array, xy: jax.Array, cn: jax.Array, d: jax.Array) -> jax.Array:
        imgs, _ = jax.vmap(rasterize_depth, in_axes=(0, None, None, None, None, None, None, None))(
            c, o, background, xy, d, radii, cn, cum
        )
        return jnp.sum(w * imgs)

    def view_loss(
        c: jax.Array, o: jax.Array, xy: jax.Array, cn: jax.Array, d: jax.Array, wi: jax.Array
    ) -> jax.Array:
        img, _ = rasterize_depth(c, o, background, xy, d, radii, cn, cum)
        return jnp.sum(wi * img)

    args = (batch_colors, opacities, xys, conics, depths)
    grads = jax.jit(jax.grad(loss, argnums=(0, 1, 2, 3, 4)))(*args)
    g_colors, g_opacities, g_xys, g_conics, g_depths = grads
    view_grad = jax.jit(jax.grad(view_loss, argnums=(0, 1, 2, 3, 4)))
    per_view = [view_grad(c, opacities, xys, conics, depths, wi) for c, wi in zip(batch_colors, w)]
    r_colors, r_opacities, r_xys, r_conics, r_depths = (jnp.stack(gs) for gs in zip(*per_view))
    assert all(jnp.linalg.norm(g) > 0.0 for g in grads), "the rasterize_depth_depth drops a path"
    np.testing.assert_allclose(g_colors, r_colors, rtol=2e-3, atol=1e-4)
    np.testing.assert_allclose(g_opacities, r_opacities.sum(axis=0), rtol=2e-3, atol=1e-4)
    np.testing.assert_allclose(g_xys, r_xys.sum(axis=0), rtol=2e-3, atol=1e-4)
    np.testing.assert_allclose(g_conics, r_conics.sum(axis=0), rtol=2e-3, atol=1e-4)
    np.testing.assert_allclose(g_depths, r_depths.sum(axis=0), rtol=2e-3, atol=1e-4)
