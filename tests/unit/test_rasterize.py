"""Rasterization stage tests.

``splax.rasterize`` blends the projected gaussians into an image and the accumulated alpha
``A(p) = sum_i w_i``, the coverage the splat contributes to a pixel. ``splax.rasterize_depth``
additionally packs the expected depth map ``D(p) = sum_i w_i d_i / sum_i w_i`` into the fourth image
channel, built from the same visibility weights ``w_i`` as the colour blend, which COLMAP
sparse-point depth regularization optimizes. All three outputs are covered along the same matrix,
i.e. the plain call, batching under ``jax.vmap``, broadcasting of the operands that stay shared
across the batch, and gradients in the plain and the batched setting.

The alpha is bounded in [0, 1] by construction, reads exactly 0 where no gaussian contributes, and
saturates towards 1 under a dense opaque splat. It is what the image composites the background with,
so rendering unit colours over a black background reproduces it channel for channel.

Two properties pin the depth channel down. For a single gaussian the expected depth is metric, i.e.
it equals that gaussian's camera-space depth on every pixel it covers however thin the coverage, and
pixels the splat does not cover carry depth 0. The depth accumulator is separate from the colour
blend, so the RGB channels the depth path returns are the plain ``rasterize`` image.

The gradients are checked against a central-difference directional derivative at the 8e-2 relative
bound the splat finite-difference tests use. The hard 1/255 cull and the early-termination cutoff
that the difference steps cross are the intrinsic residual. A depth-only loss leaves an exactly zero
colour gradient, since the depth map does not depend on the colours, and nonzero geometry gradients.
The depth normalization divides by a quantity that vanishes on uncovered pixels, so those pixels are
additionally checked to leave the gradient finite.

Geometry and appearance batch independently. ``xys``, ``depths``, ``radii``, ``conics``, and
``cum_tiles_hit`` carry either one entry per image or a single entry shared across the batch, and so
do ``colors``, ``opacities``, and ``background``.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from utils import VIEWMAT, VIEWS, assert_finite_difference, camera, projected, scene

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


def test_rasterize():
    """Decompose a blended image into its colour blend and its background composite.

    The image is ``sum_i w_i c_i + T * background``, so rendering over a black background and
    rendering unlit gaussians over the background add back up to the full image.
    """
    n, H, W = 4000, 96, 96
    colors, opacities, background, *geometry = projected(n, H, W, seed=1)
    black = jnp.zeros(3)
    unlit = jnp.zeros((n, 3))
    img, _alpha = splax.rasterize(colors, opacities, background, *geometry, img_shape=(H, W))
    blend, _ = splax.rasterize(colors, opacities, black, *geometry, img_shape=(H, W))
    composite, _ = splax.rasterize(unlit, opacities, background, *geometry, img_shape=(H, W))
    assert img.shape == (H, W, 3)
    assert float(blend.max()) > 0.1, "the splat barely covers the image"
    np.testing.assert_allclose(np.asarray(img), np.asarray(blend + composite), rtol=1e-5, atol=1e-6)


def test_rasterize_alpha():
    """Read the accumulated alpha of a splat that saturates in the centre and misses the corners.

    The alpha is the coverage the blend accumulates, so it is bounded in [0, 1], reads exactly 0
    where no gaussian contributes, and saturates towards 1 under a dense opaque cluster. It is also
    the weight the image composites the background with, which unit colours over a black background
    reproduce channel for channel.
    """
    n, H, W = 600, 96, 96
    means, scales, quats, _colors, _opacities, _bg = scene(n, seed=8)
    means, opacities = means * 0.15, jnp.full((n,), 0.9)  # a tight, opaque cluster
    geometry = _geometry(VIEWMAT, means, scales, quats, opacities, H, W)

    img, alpha = splax.rasterize(
        jnp.ones((n, 3)), opacities, jnp.zeros(3), *geometry, img_shape=(H, W)
    )
    a = np.asarray(alpha)
    assert a.shape == (H, W)
    assert (a >= 0.0).all() and (a <= 1.0).all(), "the accumulated alpha leaves [0, 1]"
    assert a[0, 0] == 0.0, "the image corner is uncovered"
    assert a.max() > 0.99, "the cluster does not saturate anywhere"
    composite = np.repeat(a[..., None], 3, -1)
    np.testing.assert_allclose(np.asarray(img), composite, rtol=1e-5, atol=1e-5)


def test_rasterize_depth():
    """Read back a single gaussian's own camera-space depth on every pixel it covers.

    Normalizing by the accumulated alpha makes the depth metric, so a lone gaussian reports its
    centre depth independently of how thinly it covers a pixel. The accumulated alpha separates the
    covered pixels from the background.
    """
    H = W = 64
    means = jnp.array([[0.1, -0.05, 0.0]])
    scales = jnp.full((1, 3), 0.12)
    quats = jnp.array([[1.0, 0.0, 0.0, 0.0]])
    opacities = jnp.array([0.9])
    viewmat = jnp.eye(4).at[2, 3].set(4.0)
    pvz = float((viewmat[:3, :3] @ means[0] + viewmat[:3, 3])[2])
    geometry = _geometry(viewmat, means, scales, quats, opacities, H, W)

    grey = jnp.full((1, 3), 0.5)
    img, alpha = splax.rasterize_depth(grey, opacities, jnp.zeros(3), *geometry, img_shape=(H, W))
    A = np.asarray(alpha)
    depth = np.asarray(img)[..., 3]
    covered = A > 0.0

    assert img.shape == (H, W, 4)
    dev = np.abs(depth[covered] - pvz).max()
    assert dev < 1e-4, f"max dev {dev:.2e}"
    # the depth carries camera units and is nowhere near the [0, 1] the alpha lives in
    assert depth[covered].min() > 1.0, "the depth is not a metric camera distance"
    assert A[covered].min() < 0.05 < 0.5 < A[covered].max(), "the coverage barely varies"
    assert not covered[0, 0] and depth[0, 0] == 0.0, "the image corner is background"
    assert covered.mean() > 0.01, "the gaussian barely contributes"


def test_rasterize_depth_image_byte_identical():
    """Return the plain rasterize image and alpha unchanged from the depth path.

    The expected depth accumulates in its own kernel and accumulator, which must not perturb the
    colour blend.
    """
    n, H, W = 4000, 128, 128
    args = projected(n, H, W, seed=1)
    plain, plain_alpha = splax.rasterize(*args, img_shape=(H, W))
    img, alpha = splax.rasterize_depth(*args, img_shape=(H, W))
    np.testing.assert_array_equal(np.asarray(plain), np.asarray(img)[..., :3])
    np.testing.assert_array_equal(np.asarray(plain_alpha), np.asarray(alpha))


# region batching


@pytest.mark.usefixtures("faithful_64bit_keys")
def test_rasterize_vmap_matches_loop():
    """Match the batched blend of B views against the loop over the unbatched blends.

    Batching packs the image id above the tile bits of the sort key, so each image keeps the
    blend order it has on its own and the images come out the same.
    """
    n, H, W = 4000, 96, 96
    means, scales, quats, colors, opacities, background = scene(n, seed=1, dense=True)
    geometry = partial(
        _geometry, means=means, scales=scales, quats=quats, opacities=opacities, H=H, W=W
    )
    batched = jax.vmap(geometry)(VIEWS)

    out_img, out_alpha = jax.vmap(
        lambda *g: splax.rasterize(colors, opacities, background, *g, img_shape=(H, W))
    )(*batched)
    ref = [
        splax.rasterize(colors, opacities, background, *geometry(VIEWS[i]), img_shape=(H, W))
        for i in range(B)
    ]
    assert out_img.shape == (B, H, W, 3) and out_alpha.shape == (B, H, W)
    np.testing.assert_array_equal(np.asarray(out_img), np.asarray(jnp.stack([r[0] for r in ref])))
    np.testing.assert_array_equal(np.asarray(out_alpha), np.asarray(jnp.stack([r[1] for r in ref])))


@pytest.mark.usefixtures("faithful_64bit_keys")
def test_rasterize_depth_vmap_matches_loop():
    """Match the batched depth blend of B views against the loop over the unbatched blends."""
    n, H, W = 4000, 96, 96
    means, scales, quats, colors, opacities, background = scene(n, seed=1, dense=True)
    geometry = partial(
        _geometry, means=means, scales=scales, quats=quats, opacities=opacities, H=H, W=W
    )
    batched = jax.vmap(geometry)(VIEWS)

    out_img, out_alpha = jax.vmap(
        lambda *g: splax.rasterize_depth(colors, opacities, background, *g, img_shape=(H, W))
    )(*batched)
    ref = [
        splax.rasterize_depth(colors, opacities, background, *geometry(VIEWS[i]), img_shape=(H, W))
        for i in range(B)
    ]
    assert out_img.shape == (B, H, W, 4) and out_alpha.shape == (B, H, W)
    np.testing.assert_array_equal(np.asarray(out_img), np.asarray(jnp.stack([r[0] for r in ref])))
    np.testing.assert_array_equal(np.asarray(out_alpha), np.asarray(jnp.stack([r[1] for r in ref])))


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
    for a, b in zip(shared, tiled):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


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


def _appearances(colors: jax.Array, opacities: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Scale one splat's colors and opacities into the B appearances of a shared projection."""
    factors = 0.4 + 0.6 * jnp.arange(B) / (B - 1)
    return colors * factors[:, None, None], opacities * factors[:, None]


def test_rasterize_broadcast_geometry():
    """Share one projection across a batch of appearances.

    A single projection feeds B images that differ in their colours and opacities alone, so the sort
    runs once and every image blends the same gaussians in the same order. The tile emission stays
    on the shared opacities, which is what the shared cumulative tile counts describe.
    """
    n, H, W = 4000, 96, 96
    means, scales, quats, colors, opacities, background = scene(n, seed=3, dense=True)
    geometry = _geometry(VIEWMAT, means, scales, quats, opacities, H, W)
    bcolors, bopacities = _appearances(colors, opacities)

    def blend(c: jax.Array, o: jax.Array) -> tuple[jax.Array, jax.Array]:
        return splax.rasterize(
            c, o, background, *geometry, img_shape=(H, W), map_opacities=opacities
        )

    out_img, out_alpha = jax.vmap(blend)(bcolors, bopacities)
    ref = [blend(bcolors[i], bopacities[i]) for i in range(B)]
    assert out_img.shape == (B, H, W, 3) and out_alpha.shape == (B, H, W)
    assert float(out_alpha.std(axis=0).max()) > 0.0, "the batch renders the same alpha B times"
    np.testing.assert_array_equal(np.asarray(out_img), np.asarray(jnp.stack([r[0] for r in ref])))
    np.testing.assert_array_equal(np.asarray(out_alpha), np.asarray(jnp.stack([r[1] for r in ref])))


def test_rasterize_depth_broadcast_geometry():
    """Share one projection across a batch of appearances in the depth blend."""
    n, H, W = 4000, 96, 96
    means, scales, quats, colors, opacities, background = scene(n, seed=3, dense=True)
    geometry = _geometry(VIEWMAT, means, scales, quats, opacities, H, W)
    bcolors, bopacities = _appearances(colors, opacities)

    def blend(c: jax.Array, o: jax.Array) -> tuple[jax.Array, jax.Array]:
        return splax.rasterize_depth(
            c, o, background, *geometry, img_shape=(H, W), map_opacities=opacities
        )

    out_img, out_alpha = jax.vmap(blend)(bcolors, bopacities)
    ref = [blend(bcolors[i], bopacities[i]) for i in range(B)]
    assert out_img.shape == (B, H, W, 4) and out_alpha.shape == (B, H, W)
    depth_spread = float(out_img[..., 3].std(axis=0).max())
    assert depth_spread > 0.0, "the batch renders the same depth B times"
    np.testing.assert_array_equal(np.asarray(out_img), np.asarray(jnp.stack([r[0] for r in ref])))
    np.testing.assert_array_equal(np.asarray(out_alpha), np.asarray(jnp.stack([r[1] for r in ref])))


# region gradient


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
        img, _alpha = splax.rasterize(
            c, o, background, xy, depths, radii, cn, cum, img_shape=(H, W)
        )
        return jnp.mean(w * img)

    args = (colors, opacities, xys, conics)
    grad = jax.grad(loss, argnums=(0, 1, 2, 3))
    grads = grad(*args)
    assert all(float(jnp.linalg.norm(g)) > 0.0 for g in grads), "the blend drops a gradient path"
    for a, b in zip(grads, jax.jit(grad)(*args)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-5, atol=1e-6)

    assert_finite_difference(loss, args, grads)


def test_rasterize_alpha_grad():
    """Check the accumulated alpha gradients against a central-difference directional derivative.

    The alpha is built from the opacities and the geometry alone, so an alpha-only loss leaves an
    exactly zero colour gradient while the remaining paths stay nonzero.
    """
    n, H, W = 400, 80, 80
    colors, opacities, background, xys, depths, radii, conics, cum = projected(
        n, H, W, seed=7, dense=False
    )
    w = jax.random.uniform(jax.random.key(11), (H, W))

    def loss(c: jax.Array, o: jax.Array, xy: jax.Array, cn: jax.Array) -> jax.Array:
        _img, alpha = splax.rasterize(
            c, o, background, xy, depths, radii, cn, cum, img_shape=(H, W)
        )
        return jnp.mean(w * alpha)

    args = (colors, opacities, xys, conics)
    grad = jax.grad(loss, argnums=(0, 1, 2, 3))
    grads = grad(*args)
    assert float(jnp.linalg.norm(grads[0])) == 0.0, "the alpha does not depend on the colours"
    assert all(float(jnp.linalg.norm(g)) > 0.0 for g in grads[1:]), "the alpha drops a path"
    for a, b in zip(grads, jax.jit(grad)(*args)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-5, atol=1e-6)

    assert_finite_difference(loss, args, grads, name="alpha ")


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
        img, _alpha = splax.rasterize_depth(
            c, o, background, xy, d, radii, cn, cum, img_shape=(H, W)
        )
        dl = jnp.mean(wd * img[..., 3])
        return dl if mode == "depth_only" else jnp.mean(wc * img[..., :3]) + dl

    args = (colors, opacities, xys, conics, depths)
    grad = jax.grad(loss, argnums=(0, 1, 2, 3, 4))
    grads = grad(*args)
    if mode == "depth_only":
        assert float(jnp.linalg.norm(grads[0])) == 0.0, "depth does not depend on the colours"
    assert float(jnp.linalg.norm(grads[2])) > 0.0, "the geometry gradient drives the regularizer"
    assert float(jnp.linalg.norm(grads[4])) > 0.0, "depths carry a cotangent"
    for a, b in zip(grads, jax.jit(grad)(*args)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-5, atol=1e-6)

    assert_finite_difference(loss, args, grads, name=f"{mode} ")


def test_rasterize_depth_grad_empty_pixels():
    """Keep the depth gradient finite on an image a single gaussian barely covers.

    The normalization divides by the accumulated alpha, which is exactly zero wherever no gaussian
    contributes. Those pixels are masked out of the quotient on both the forward and the backward
    side, so summing the whole depth map leaves finite gradients.
    """
    H = W = 64
    means = jnp.array([[0.1, -0.05, 0.0]])
    scales = jnp.full((1, 3), 0.12)
    quats = jnp.array([[1.0, 0.0, 0.0, 0.0]])
    opacities = jnp.array([0.9])
    viewmat = jnp.eye(4).at[2, 3].set(4.0)
    xys, depths, radii, conics, cum = _geometry(viewmat, means, scales, quats, opacities, H, W)
    grey = jnp.full((1, 3), 0.5)

    def render(o: jax.Array, xy: jax.Array, cn: jax.Array, d: jax.Array) -> jax.Array:
        img, _alpha = splax.rasterize_depth(
            grey, o, jnp.zeros(3), xy, d, radii, cn, cum, img_shape=(H, W)
        )
        return img[..., 3]

    depth = render(opacities, xys, conics, depths)
    empty = float(jnp.mean(depth == 0.0))
    assert 0.5 < empty < 1.0, f"only {1 - empty:.0%} of the image is empty"
    grad = jax.grad(lambda *args: jnp.sum(render(*args)), argnums=(0, 1, 2, 3))
    grads = grad(opacities, xys, conics, depths)
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in grads), "empty pixels leak a NaN gradient"
    assert float(jnp.linalg.norm(grads[3])) > 0.0, "depths carry a cotangent"


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
        img, _alpha = splax.rasterize(
            colors, opacities, background, xy, d, r, cn, cm, img_shape=(H, W)
        )
        return jnp.sum(weight * img)

    grad = jax.grad(loss, argnums=(0, 3))
    out = jax.vmap(grad)(*batched, w)
    for i in range(B):
        ref = grad(*geometry(VIEWS[i]), w[i])
        for a, b in zip(out, ref):
            np.testing.assert_allclose(np.asarray(a[i]), np.asarray(b), rtol=2e-3, atol=1e-4)


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
        img, _alpha = splax.rasterize_depth(
            colors, opacities, background, xy, d, r, cn, cm, img_shape=(H, W)
        )
        return jnp.sum(weight * img[..., 3])

    grad = jax.grad(loss, argnums=(0, 1, 3))
    out = jax.vmap(grad)(*batched, wd)
    for i in range(B):
        ref = grad(*geometry(VIEWS[i]), wd[i])
        for a, b in zip(out, ref):
            np.testing.assert_allclose(np.asarray(a[i]), np.asarray(b), rtol=2e-3, atol=1e-4)


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
        imgs, _alphas = jax.vmap(
            lambda *g: splax.rasterize(c, o, background, *g, img_shape=(H, W))
        )(*batched)
        return jnp.sum(w * imgs)

    def view_loss(c: jax.Array, o: jax.Array, i: int) -> jax.Array:
        img, _alpha = splax.rasterize(c, o, background, *geometry(VIEWS[i]), img_shape=(H, W))
        return jnp.sum(w[i] * img)

    out = jax.grad(loss, argnums=(0, 1))(colors, opacities)
    per_view = [jax.grad(view_loss, argnums=(0, 1))(colors, opacities, i) for i in range(B)]
    for k, a in enumerate(out):
        ref = sum(p[k] for p in per_view)
        np.testing.assert_allclose(np.asarray(a), np.asarray(ref), rtol=2e-3, atol=1e-4)


def test_rasterize_grad_broadcast_geometry():
    """Split the gradients of a batch of appearances over the projection they share.

    Colours and opacities carry one entry per image and keep their own gradient, while the shared
    projection feeds every image and collects the sum of the per-image gradients. The loss weights
    the image and the accumulated alpha so both cotangent paths run.
    """
    n, H, W = 2000, 96, 96
    means, scales, quats, colors, opacities, background = scene(n, seed=4)
    xys, depths, radii, conics, cum = _geometry(VIEWMAT, means, scales, quats, opacities, H, W)
    bcolors, bopacities = _appearances(colors, opacities)
    w = jax.random.uniform(jax.random.key(13), (B, H, W, 3))
    wa = jax.random.uniform(jax.random.key(14), (B, H, W))

    def view_loss(c: jax.Array, o: jax.Array, xy: jax.Array, cn: jax.Array, i: int) -> jax.Array:
        img, alpha = splax.rasterize(
            c, o, background, xy, depths, radii, cn, cum, img_shape=(H, W), map_opacities=opacities
        )
        return jnp.sum(w[i] * img) + jnp.sum(wa[i] * alpha)

    def loss(c: jax.Array, o: jax.Array, xy: jax.Array, cn: jax.Array) -> jax.Array:
        imgs, alphas = jax.vmap(
            lambda ci, oi: splax.rasterize(
                ci,
                oi,
                background,
                xy,
                depths,
                radii,
                cn,
                cum,
                img_shape=(H, W),
                map_opacities=opacities,
            )
        )(c, o)
        return jnp.sum(w * imgs) + jnp.sum(wa * alphas)

    args = (bcolors, bopacities, xys, conics)
    out = jax.grad(loss, argnums=(0, 1, 2, 3))(*args)
    per_view = [
        jax.grad(view_loss, argnums=(0, 1, 2, 3))(bcolors[i], bopacities[i], xys, conics, i)
        for i in range(B)
    ]
    assert all(float(jnp.linalg.norm(g)) > 0.0 for g in out), "the blend drops a gradient path"
    for k in (0, 1):  # per-image appearance keeps its own gradient
        ref = jnp.stack([p[k] for p in per_view])
        np.testing.assert_allclose(np.asarray(out[k]), np.asarray(ref), rtol=2e-3, atol=1e-4)
    for k in (2, 3):  # the shared projection collects the sum over the batch
        ref = sum(p[k] for p in per_view)
        np.testing.assert_allclose(np.asarray(out[k]), np.asarray(ref), rtol=2e-3, atol=1e-4)


def test_rasterize_depth_grad_broadcast_geometry():
    """Split the depth blend gradients of a batch of appearances over their shared projection."""
    n, H, W = 2000, 96, 96
    means, scales, quats, colors, opacities, background = scene(n, seed=4)
    xys, depths, radii, conics, cum = _geometry(VIEWMAT, means, scales, quats, opacities, H, W)
    bcolors, bopacities = _appearances(colors, opacities)
    w = jax.random.uniform(jax.random.key(15), (B, H, W, 3))
    wd = jax.random.uniform(jax.random.key(16), (B, H, W))

    def view_loss(
        c: jax.Array, o: jax.Array, xy: jax.Array, cn: jax.Array, d: jax.Array, i: int
    ) -> jax.Array:
        img, _alpha = splax.rasterize_depth(
            c, o, background, xy, d, radii, cn, cum, img_shape=(H, W), map_opacities=opacities
        )
        return jnp.sum(w[i] * img[..., :3]) + jnp.sum(wd[i] * img[..., 3])

    def loss(c: jax.Array, o: jax.Array, xy: jax.Array, cn: jax.Array, d: jax.Array) -> jax.Array:
        imgs, _alphas = jax.vmap(
            lambda ci, oi: splax.rasterize_depth(
                ci, oi, background, xy, d, radii, cn, cum, img_shape=(H, W), map_opacities=opacities
            )
        )(c, o)
        return jnp.sum(w * imgs[..., :3]) + jnp.sum(wd * imgs[..., 3])

    args = (bcolors, bopacities, xys, conics, depths)
    out = jax.grad(loss, argnums=(0, 1, 2, 3, 4))(*args)
    per_view = [
        jax.grad(view_loss, argnums=(0, 1, 2, 3, 4))(
            bcolors[i], bopacities[i], xys, conics, depths, i
        )
        for i in range(B)
    ]
    assert all(float(jnp.linalg.norm(g)) > 0.0 for g in out), "the depth blend drops a path"
    for k in (0, 1):
        ref = jnp.stack([p[k] for p in per_view])
        np.testing.assert_allclose(np.asarray(out[k]), np.asarray(ref), rtol=2e-3, atol=1e-4)
    for k in (2, 3, 4):
        ref = sum(p[k] for p in per_view)
        np.testing.assert_allclose(np.asarray(out[k]), np.asarray(ref), rtol=2e-3, atol=1e-4)


def test_rasterize_alpha_grad_broadcast():
    """Sum the per-image alpha gradients into an operand shared across the batch."""
    n, H, W = 2000, 96, 96
    means, scales, quats, colors, opacities, background = scene(n, seed=4)
    geometry = partial(
        _geometry, means=means, scales=scales, quats=quats, opacities=opacities, H=H, W=W
    )
    batched = jax.vmap(geometry)(VIEWS)
    w = jax.random.uniform(jax.random.key(12), (B, H, W))

    def loss(o: jax.Array) -> jax.Array:
        _imgs, alphas = jax.vmap(
            lambda *g: splax.rasterize(colors, o, background, *g, img_shape=(H, W))
        )(*batched)
        return jnp.sum(w * alphas)

    def view_loss(o: jax.Array, i: int) -> jax.Array:
        _img, alpha = splax.rasterize(colors, o, background, *geometry(VIEWS[i]), img_shape=(H, W))
        return jnp.sum(w[i] * alpha)

    out = jax.grad(loss)(opacities)
    ref = sum(jax.grad(view_loss)(opacities, i) for i in range(B))
    assert float(jnp.linalg.norm(out)) > 0.0, "the alpha drops the opacity gradient"
    np.testing.assert_allclose(np.asarray(out), np.asarray(ref), rtol=2e-3, atol=1e-4)


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
        imgs, _alphas = jax.vmap(
            lambda *g: splax.rasterize_depth(c, o, background, *g, img_shape=(H, W))
        )(*batched)
        return jnp.sum(w * imgs[..., :3]) + jnp.sum(wd * imgs[..., 3])

    def view_loss(c: jax.Array, o: jax.Array, i: int) -> jax.Array:
        img, _alpha = splax.rasterize_depth(c, o, background, *geometry(VIEWS[i]), img_shape=(H, W))
        return jnp.sum(w[i] * img[..., :3]) + jnp.sum(wd[i] * img[..., 3])

    out = jax.grad(loss, argnums=(0, 1))(colors, opacities)
    per_view = [jax.grad(view_loss, argnums=(0, 1))(colors, opacities, i) for i in range(B)]
    for k, a in enumerate(out):
        ref = sum(p[k] for p in per_view)
        np.testing.assert_allclose(np.asarray(a), np.asarray(ref), rtol=2e-3, atol=1e-4)
