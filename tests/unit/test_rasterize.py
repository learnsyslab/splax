"""Rasterization stage tests.

``splax.rasterize`` blends the projected gaussians into an image. ``splax.rasterize_depth``
additionally accumulates the expected depth map ``D(p) = Σ wᵢ dᵢ`` from the same visibility weights
``wᵢ`` as the colour blend, which COLMAP sparse-point depth regularization optimizes. Both are
covered along the same matrix, i.e. the plain call, batching under ``jax.vmap``, broadcasting of the
operands that stay shared across the batch, and gradients in the plain and the batched setting.

Two properties pin the depth channel down. For a single gaussian the expected depth is its
camera-space depth times the accumulated alpha, so ``D == pvz · A`` with ``A = Σ wᵢ`` the
unit-colour render over a black background, and pixels the splat does not cover carry depth 0. The
depth accumulator is separate from the colour blend, so the image the depth path returns is
bit-for-bit the plain ``rasterize`` image.

The gradients are checked against a central-difference directional derivative at the 8e-2 relative
bound the splat finite-difference tests use. The hard 1/255 cull and the early-termination cutoff
that the difference steps cross are the intrinsic residual. A depth-only loss leaves an exactly zero
colour gradient, since the depth map does not depend on the colours, and nonzero geometry gradients.

The geometry defines the batch. ``xys``, ``depths``, ``radii``, ``conics``, and ``cum_tiles_hit``
carry one entry per image, while ``colors``, ``opacities``, and ``background`` may stay shared and
are then indexed modulo the per-image gaussian count. Gradients follow the same split, so a shared
operand collects the sum of its per-image gradients.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from utils import VIEWS, camera, projected, scene

import splax

B = len(VIEWS)


def _geometry(
    viewmat: jax.Array,
    means: jax.Array,
    scales: jax.Array,
    quats: jax.Array,
    opacities: jax.Array,
    H: int,
    W: int,
) -> tuple[jax.Array, ...]:
    """Project a splat from ``viewmat`` into the geometry arguments the blend consumes."""
    xys, depths, radii, conics, _nth, cum = splax.project(
        means, scales, quats, viewmat, opacities=opacities, **camera(H, W)
    )
    return xys, depths, radii, conics, cum


# region regular invocation


@pytest.mark.unit
def test_rasterize():
    """Decompose a blended image into its colour blend and its background composite.

    The image is ``Σ wᵢ cᵢ + T · background``, so rendering over a black background and rendering
    unlit gaussians over the background add back up to the full image.
    """
    n, H, W = 4000, 96, 96
    colors, opacities, background, *geometry = projected(n, H, W, seed=1)
    black = jnp.zeros(3)
    unlit = jnp.zeros((n, 3))
    img = splax.rasterize(colors, opacities, background, *geometry, img_shape=(H, W))
    blend = splax.rasterize(colors, opacities, black, *geometry, img_shape=(H, W))
    composite = splax.rasterize(unlit, opacities, background, *geometry, img_shape=(H, W))
    assert img.shape == (H, W, 3)
    assert float(blend.max()) > 0.1, "the splat barely covers the image"
    np.testing.assert_allclose(np.asarray(img), np.asarray(blend + composite), rtol=1e-5, atol=1e-6)


@pytest.mark.unit
def test_rasterize_depth():
    """Match a single gaussian's expected depth against its accumulated alpha.

    A single gaussian contributes one depth to every pixel it covers, so the expected depth
    collapses to that camera-space depth times the accumulated alpha, which the unit-colour render
    over a black background measures directly.
    """
    H = W = 64
    means = jnp.array([[0.1, -0.05, 0.0]])
    scales = jnp.full((1, 3), 0.12)
    quats = jnp.array([[1.0, 0.0, 0.0, 0.0]])
    opacities = jnp.array([0.9])
    viewmat = jnp.eye(4).at[2, 3].set(4.0)
    pvz = float((viewmat[:3, :3] @ means[0] + viewmat[:3, 3])[2])
    geometry = _geometry(viewmat, means, scales, quats, opacities, H, W)
    black = jnp.zeros(3)

    alpha = splax.rasterize(jnp.ones((1, 3)), opacities, black, *geometry, img_shape=(H, W))
    grey = jnp.full((1, 3), 0.5)
    _img, depth = splax.rasterize_depth(grey, opacities, black, *geometry, img_shape=(H, W))
    A = np.asarray(alpha)[..., 0]
    depth = np.asarray(depth)

    assert depth.shape == (H, W)
    assert np.allclose(depth, pvz * A, atol=1e-4), f"max dev {np.abs(depth - pvz * A).max():.2e}"
    assert depth[0, 0] == 0.0 and A[0, 0] == 0.0, "the image corner is background"
    assert depth.max() > 0.5 * pvz, "the gaussian barely contributes"


@pytest.mark.unit
def test_rasterize_depth_image_byte_identical():
    """Return bit-for-bit the plain rasterize image from the depth path.

    The expected depth accumulates in its own kernel and accumulator, which must not perturb the
    colour blend.
    """
    n, H, W = 4000, 128, 128
    args = projected(n, H, W, seed=1)
    plain = splax.rasterize(*args, img_shape=(H, W))
    img, _depth = splax.rasterize_depth(*args, img_shape=(H, W))
    np.testing.assert_array_equal(np.asarray(plain), np.asarray(img))


# region batching


@pytest.mark.unit
@pytest.mark.usefixtures("faithful_64bit_keys")
def test_rasterize_vmap_matches_loop():
    """Match the batched blend of B views against the loop over the unbatched blends.

    Batching packs the image id above the tile bits of the sort key, so each image keeps the
    blend order it has on its own and the images come out bit-identical.
    """
    n, H, W = 4000, 96, 96
    means, scales, quats, colors, opacities, background = scene(n, seed=1, dense=True)
    geometry = partial(
        _geometry, means=means, scales=scales, quats=quats, opacities=opacities, H=H, W=W
    )
    batched = jax.vmap(geometry)(VIEWS)

    out = jax.vmap(lambda *g: splax.rasterize(colors, opacities, background, *g, img_shape=(H, W)))(
        *batched
    )
    ref = jnp.stack(
        [
            splax.rasterize(colors, opacities, background, *geometry(VIEWS[i]), img_shape=(H, W))
            for i in range(B)
        ]
    )
    assert out.shape == (B, H, W, 3)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(ref))


@pytest.mark.unit
@pytest.mark.usefixtures("faithful_64bit_keys")
def test_rasterize_depth_vmap_matches_loop():
    """Match the batched depth blend of B views against the loop over the unbatched blends."""
    n, H, W = 4000, 96, 96
    means, scales, quats, colors, opacities, background = scene(n, seed=1, dense=True)
    geometry = partial(
        _geometry, means=means, scales=scales, quats=quats, opacities=opacities, H=H, W=W
    )
    batched = jax.vmap(geometry)(VIEWS)

    out_img, out_depth = jax.vmap(
        lambda *g: splax.rasterize_depth(colors, opacities, background, *g, img_shape=(H, W))
    )(*batched)
    ref = [
        splax.rasterize_depth(colors, opacities, background, *geometry(VIEWS[i]), img_shape=(H, W))
        for i in range(B)
    ]
    assert out_img.shape == (B, H, W, 3) and out_depth.shape == (B, H, W)
    np.testing.assert_array_equal(np.asarray(out_img), np.asarray(jnp.stack([r[0] for r in ref])))
    np.testing.assert_array_equal(np.asarray(out_depth), np.asarray(jnp.stack([r[1] for r in ref])))


@pytest.mark.unit
@pytest.mark.usefixtures("faithful_64bit_keys")
def test_rasterize_broadcast():
    """Share colors, opacities, and background across a batched geometry.

    The shared operands are indexed modulo the per-image gaussian count and a single background is
    selected for every image, which must reproduce the fully batched blend that carries one explicit
    copy per image.
    """
    n, H, W = 4000, 96, 96
    means, scales, quats, colors, opacities, background = scene(n, seed=3, dense=True)
    geometry = partial(
        _geometry, means=means, scales=scales, quats=quats, opacities=opacities, H=H, W=W
    )
    batched = jax.vmap(geometry)(VIEWS)

    shared = jax.vmap(
        lambda *g: splax.rasterize(colors, opacities, background, *g, img_shape=(H, W))
    )(*batched)
    tiled = jax.vmap(lambda c, o, bg, *g: splax.rasterize(c, o, bg, *g, img_shape=(H, W)))(
        jnp.broadcast_to(colors, (B, n, 3)),
        jnp.broadcast_to(opacities, (B, n)),
        jnp.broadcast_to(background, (B, 3)),
        *batched,
    )
    np.testing.assert_array_equal(np.asarray(shared), np.asarray(tiled))


@pytest.mark.unit
@pytest.mark.usefixtures("faithful_64bit_keys")
def test_rasterize_depth_broadcast():
    """Share colors, opacities, and background across a batched depth blend."""
    n, H, W = 4000, 96, 96
    means, scales, quats, colors, opacities, background = scene(n, seed=3, dense=True)
    geometry = partial(
        _geometry, means=means, scales=scales, quats=quats, opacities=opacities, H=H, W=W
    )
    batched = jax.vmap(geometry)(VIEWS)

    shared = jax.vmap(
        lambda *g: splax.rasterize_depth(colors, opacities, background, *g, img_shape=(H, W))
    )(*batched)
    tiled = jax.vmap(lambda c, o, bg, *g: splax.rasterize_depth(c, o, bg, *g, img_shape=(H, W)))(
        jnp.broadcast_to(colors, (B, n, 3)),
        jnp.broadcast_to(opacities, (B, n)),
        jnp.broadcast_to(background, (B, 3)),
        *batched,
    )
    for a, b in zip(shared, tiled):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


# region gradient


@pytest.mark.unit
def test_rasterize_grad():
    """Check the blend gradients against a central-difference directional derivative.

    Colors, opacities, xys, and conics are perturbed along their own gradient direction at once,
    which maximizes the directional-derivative signal against the float32 render noise. jit must
    leave the gradients untouched.
    """
    n, H, W = 400, 80, 80
    colors, opacities, background, xys, depths, radii, conics, cum = projected(
        n, H, W, seed=7, dense=False
    )
    w = jax.random.uniform(jax.random.key(5), (H, W, 3))

    def loss(c: jax.Array, o: jax.Array, xy: jax.Array, cn: jax.Array) -> jax.Array:
        img = splax.rasterize(c, o, background, xy, depths, radii, cn, cum, img_shape=(H, W))
        return jnp.mean(w * img)

    args = (colors, opacities, xys, conics)
    grad = jax.grad(loss, argnums=(0, 1, 2, 3))
    grads = grad(*args)
    assert all(float(jnp.linalg.norm(g)) > 0.0 for g in grads), "the blend drops a gradient path"
    for a, b in zip(grads, jax.jit(grad)(*args)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-5, atol=1e-6)

    dirs = [g / (jnp.linalg.norm(g) + 1e-12) for g in grads]
    analytic = sum(float(jnp.vdot(g, d)) for g, d in zip(grads, dirs))
    eps = 2e-3
    plus = [a + eps * d for a, d in zip(args, dirs)]
    minus = [a - eps * d for a, d in zip(args, dirs)]
    numeric = (float(loss(*plus)) - float(loss(*minus))) / (2 * eps)
    rel = abs(analytic - numeric) / (abs(numeric) + 1e-12)
    assert rel < 8e-2, f"FD mismatch: {analytic:.6e} vs {numeric:.6e} (rel {rel:.2e})"


@pytest.mark.unit
@pytest.mark.parametrize("mode", ["depth_only", "mixed"])
def test_rasterize_depth_grad(mode: str):
    """Check the depth blend gradients against a central-difference directional derivative.

    A depth-only loss isolates the depth cotangent chain, a mixed colour and depth loss runs it
    alongside the colour chain. Depth is independent of the colours, so the depth-only loss leaves
    an exactly zero colour gradient while the geometry gradients stay nonzero.
    """
    n, H, W = 400, 80, 80
    colors, opacities, background, xys, depths, radii, conics, cum = projected(
        n, H, W, seed=7, dense=False
    )
    wd = jax.random.uniform(jax.random.key(9), (H, W))
    wc = jax.random.uniform(jax.random.key(10), (H, W, 3))

    def loss(c: jax.Array, o: jax.Array, xy: jax.Array, cn: jax.Array, d: jax.Array) -> jax.Array:
        img, depth = splax.rasterize_depth(
            c, o, background, xy, d, radii, cn, cum, img_shape=(H, W)
        )
        dl = jnp.mean(wd * depth)
        return dl if mode == "depth_only" else jnp.mean(wc * img) + dl

    args = (colors, opacities, xys, conics, depths)
    grad = jax.grad(loss, argnums=(0, 1, 2, 3, 4))
    grads = grad(*args)
    if mode == "depth_only":
        assert float(jnp.linalg.norm(grads[0])) == 0.0, "depth does not depend on the colours"
    assert float(jnp.linalg.norm(grads[2])) > 0.0, "the geometry gradient drives the regularizer"
    assert float(jnp.linalg.norm(grads[4])) > 0.0, "depths carry a cotangent"
    for a, b in zip(grads, jax.jit(grad)(*args)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-5, atol=1e-6)

    dirs = [g / (jnp.linalg.norm(g) + 1e-12) for g in grads]
    analytic = sum(float(jnp.vdot(g, d)) for g, d in zip(grads, dirs))
    eps = 2e-3
    plus = [a + eps * d for a, d in zip(args, dirs)]
    minus = [a - eps * d for a, d in zip(args, dirs)]
    numeric = (float(loss(*plus)) - float(loss(*minus))) / (2 * eps)
    rel = abs(analytic - numeric) / (abs(numeric) + 1e-12)
    assert rel < 8e-2, f"{mode} FD mismatch: {analytic:.6e} vs {numeric:.6e} (rel {rel:.2e})"


@pytest.mark.unit
def test_rasterize_grad_vmap_matches_loop():
    """Match the batch-native geometry gradients against the loop over the unbatched gradients.

    The backward accumulates per gaussian with atomics, so the batched and the sequential reduction
    orders differ and the comparison is a tolerance rather than an equality.
    """
    n, H, W = 2000, 96, 96
    means, scales, quats, colors, opacities, background = scene(n, seed=2)
    geometry = partial(
        _geometry, means=means, scales=scales, quats=quats, opacities=opacities, H=H, W=W
    )
    batched = jax.vmap(geometry)(VIEWS)
    w = jax.random.uniform(jax.random.key(5), (B, H, W, 3))

    def loss(
        xy: jax.Array, d: jax.Array, r: jax.Array, cn: jax.Array, cm: jax.Array, weight: jax.Array
    ) -> jax.Array:
        img = splax.rasterize(colors, opacities, background, xy, d, r, cn, cm, img_shape=(H, W))
        return jnp.sum(weight * img)

    grad = jax.grad(loss, argnums=(0, 3))
    out = jax.vmap(grad)(*batched, w)
    for i in range(B):
        ref = grad(*geometry(VIEWS[i]), w[i])
        for a, b in zip(out, ref):
            np.testing.assert_allclose(np.asarray(a[i]), np.asarray(b), rtol=2e-3, atol=1e-4)


@pytest.mark.unit
def test_rasterize_depth_grad_vmap_matches_loop():
    """Match the batch-native depth blend gradients against the loop over the unbatched ones."""
    n, H, W = 2000, 96, 96
    means, scales, quats, colors, opacities, background = scene(n, seed=2)
    geometry = partial(
        _geometry, means=means, scales=scales, quats=quats, opacities=opacities, H=H, W=W
    )
    batched = jax.vmap(geometry)(VIEWS)
    wd = jax.random.uniform(jax.random.key(6), (B, H, W))

    def loss(
        xy: jax.Array, d: jax.Array, r: jax.Array, cn: jax.Array, cm: jax.Array, weight: jax.Array
    ) -> jax.Array:
        _img, depth = splax.rasterize_depth(
            colors, opacities, background, xy, d, r, cn, cm, img_shape=(H, W)
        )
        return jnp.sum(weight * depth)

    grad = jax.grad(loss, argnums=(0, 1, 3))
    out = jax.vmap(grad)(*batched, wd)
    for i in range(B):
        ref = grad(*geometry(VIEWS[i]), wd[i])
        for a, b in zip(out, ref):
            np.testing.assert_allclose(np.asarray(a[i]), np.asarray(b), rtol=2e-3, atol=1e-4)


@pytest.mark.unit
def test_rasterize_grad_broadcast():
    """Sum the per-image gradients into an operand shared across the batch.

    colors and opacities feed every image of the batch, so their gradient is the sum of the
    gradients the images produce on their own.
    """
    n, H, W = 2000, 96, 96
    means, scales, quats, colors, opacities, background = scene(n, seed=4)
    geometry = partial(
        _geometry, means=means, scales=scales, quats=quats, opacities=opacities, H=H, W=W
    )
    batched = jax.vmap(geometry)(VIEWS)
    w = jax.random.uniform(jax.random.key(7), (B, H, W, 3))

    def loss(c: jax.Array, o: jax.Array) -> jax.Array:
        imgs = jax.vmap(lambda *g: splax.rasterize(c, o, background, *g, img_shape=(H, W)))(
            *batched
        )
        return jnp.sum(w * imgs)

    def view_loss(c: jax.Array, o: jax.Array, i: int) -> jax.Array:
        img = splax.rasterize(c, o, background, *geometry(VIEWS[i]), img_shape=(H, W))
        return jnp.sum(w[i] * img)

    out = jax.grad(loss, argnums=(0, 1))(colors, opacities)
    per_view = [jax.grad(view_loss, argnums=(0, 1))(colors, opacities, i) for i in range(B)]
    for k, a in enumerate(out):
        ref = sum(p[k] for p in per_view)
        np.testing.assert_allclose(np.asarray(a), np.asarray(ref), rtol=2e-3, atol=1e-4)


@pytest.mark.unit
def test_rasterize_depth_grad_broadcast():
    """Sum the per-image depth blend gradients into an operand shared across the batch."""
    n, H, W = 2000, 96, 96
    means, scales, quats, colors, opacities, background = scene(n, seed=4)
    geometry = partial(
        _geometry, means=means, scales=scales, quats=quats, opacities=opacities, H=H, W=W
    )
    batched = jax.vmap(geometry)(VIEWS)
    w = jax.random.uniform(jax.random.key(7), (B, H, W, 3))
    wd = jax.random.uniform(jax.random.key(8), (B, H, W))

    def loss(c: jax.Array, o: jax.Array) -> jax.Array:
        imgs, depths = jax.vmap(
            lambda *g: splax.rasterize_depth(c, o, background, *g, img_shape=(H, W))
        )(*batched)
        return jnp.sum(w * imgs) + jnp.sum(wd * depths)

    def view_loss(c: jax.Array, o: jax.Array, i: int) -> jax.Array:
        img, depth = splax.rasterize_depth(c, o, background, *geometry(VIEWS[i]), img_shape=(H, W))
        return jnp.sum(w[i] * img) + jnp.sum(wd[i] * depth)

    out = jax.grad(loss, argnums=(0, 1))(colors, opacities)
    per_view = [jax.grad(view_loss, argnums=(0, 1))(colors, opacities, i) for i in range(B)]
    for k, a in enumerate(out):
        ref = sum(p[k] for p in per_view)
        np.testing.assert_allclose(np.asarray(a), np.asarray(ref), rtol=2e-3, atol=1e-4)
