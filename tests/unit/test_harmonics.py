"""Test the spherical harmonics evaluation on its own and inside the render."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from utils import VIEWMAT, assert_finite_difference, camera, coeff_scene_params

import splax


def coeffs_and_directions(n: int, seed: int) -> tuple[jax.Array, jax.Array]:
    """Random degree-3 coefficients and directions that are not unit length."""
    k = jax.random.split(jax.random.key(seed), 2)
    return jax.random.normal(k[0], (n, 16, 3)), jax.random.normal(k[1], (n, 3)) * 3.0


def test_scale_invariance():
    """Directions are normalized internally, so their magnitude cannot change the color."""
    coeffs, directions = coeffs_and_directions(4096, seed=5)
    base = np.asarray(splax.spherical_harmonics(coeffs, directions))
    for scale in (1e-3, 12.5):
        scaled = np.asarray(splax.spherical_harmonics(coeffs, directions * scale))
        difference = np.abs(scaled - base).max()
        assert difference < 1e-5, f"scale {scale} changed the color by {difference:.2e}"


@pytest.mark.parametrize("count", [2, 3, 7, 15])
def test_partial_band_count_raises(count: int):
    """A coefficient count that is not a whole set of bands has no degree, so it raises."""
    coeffs, directions = coeffs_and_directions(64, seed=count)
    with pytest.raises(AssertionError, match="coefficients must be one of"):
        splax.spherical_harmonics(coeffs[:, :count], directions)


def test_single_coefficient_is_the_stored_color_map():
    """One coefficient per gaussian evaluates to exactly what ``splax.io.sh_to_rgb`` maps."""
    coeffs, directions = coeffs_and_directions(4096, seed=30)
    np.testing.assert_allclose(
        np.asarray(splax.spherical_harmonics(coeffs[:, :1], directions)),
        np.asarray(splax.io.sh_to_rgb(coeffs[:, 0])),
        atol=2e-7,
    )


def test_render_is_view_dependent():
    """The higher bands change the image, and change it differently from another camera."""
    n, H, W = 20_000, 128, 128
    means, log_scales, quats, sh_colors, logit_opacities, background = coeff_scene_params(n, seed=3)
    splats = (means, log_scales, quats, sh_colors, logit_opacities)
    kw = {"viewmat": VIEWMAT, "background": background, **camera(H, W)}
    base_splat = (means, log_scales, quats, sh_colors[:, :1], logit_opacities)
    base = np.asarray(splax.render(*base_splat, **kw)[0])
    full = np.asarray(splax.render(*splats, **kw)[0])
    assert np.abs(full - base).max() > 1e-2, "the higher bands must change the render"

    moved = {**kw, "viewmat": VIEWMAT.at[0, 3].set(1.5)}
    other = np.asarray(splax.render(*splats, **moved)[0])
    other_base = np.asarray(splax.render(*base_splat, **moved)[0])
    # both cameras see the same geometry, so the difference must come from the view-dependence
    assert np.abs((other - other_base) - (full - base)).max() > 1e-2


@pytest.mark.parametrize("in_axes", [(None, 0, 0), (0, None, 0)], ids=["coeffs", "directions"])
def test_spherical_harmonics_grad_broadcast(in_axes: tuple[int | None, ...]):
    """Sum the gradient of an operand shared across nested views over exactly those views."""
    n, scenes, views = 512, 2, 3
    shared, batched = in_axes.index(None), in_axes.index(0)
    args = list(coeffs_and_directions(n, seed=0))
    args[batched] = jax.random.normal(jax.random.key(1), (scenes, views, *args[batched].shape))
    weights = jax.random.normal(jax.random.key(2), (scenes, views, n, 3))

    def view_loss(coeffs: jax.Array, directions: jax.Array, weight: jax.Array) -> jax.Array:
        return jnp.sum(splax.spherical_harmonics(coeffs, directions) * weight)

    def scene_loss(coeffs: jax.Array, directions: jax.Array, weight: jax.Array) -> jax.Array:
        return jnp.sum(jax.vmap(view_loss, in_axes)(coeffs, directions, weight))

    def loss(coeffs: jax.Array, directions: jax.Array) -> jax.Array:
        return jnp.sum(jax.vmap(scene_loss, in_axes)(coeffs, directions, weights))

    grad = partial(jax.grad, argnums=shared)
    per_view_grad = jax.vmap(jax.vmap(grad(view_loss), in_axes), in_axes)(*args, weights)
    per_scene_grad = jax.vmap(grad(scene_loss), in_axes)(*args, weights)
    total_grad = jax.grad(loss, argnums=(0, 1))(*args)
    assert_finite_difference(loss, tuple(args), total_grad)
    np.testing.assert_allclose(per_view_grad.sum(1), per_scene_grad, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(per_scene_grad.sum(0), total_grad[shared], rtol=1e-5, atol=1e-5)
