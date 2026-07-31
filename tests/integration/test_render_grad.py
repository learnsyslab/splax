"""Test the render gradients against finite differences, the unbatched loop, and gsplat."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial.transform import RigidTransform
from utils import VIEWMAT, assert_finite_difference, camera, poses, scene_params

import splax

if TYPE_CHECKING:
    from pathlib import Path
    from types import ModuleType

B = 3
PARAMS = ("means", "log_scales", "quats", "sh_colors", "logit_opacities")
LEGO_HEIGHT = LEGO_WIDTH = 800
LEGO_OFFSET = jnp.array([0.01, -0.01, 0.01, 0.01, -0.01, 0.01])


def _render_loss(
    splats: tuple[jax.Array, ...], kw: dict, weight: jax.Array, viewmat: jax.Array
) -> jax.Array:
    """Render a splat from one camera pose and score it against a per-pixel weight."""
    img, _ = splax.render(*splats, viewmat=viewmat, **kw)
    return jnp.mean(weight * img)


# region gradient


def test_render_grad():
    """Produce finite, nonzero gradients for the five splat parameters and the viewmat."""
    n, H, W = 2000, 96, 96
    means, log_scales, quats, sh_colors, logit_opacities, background = scene_params(n, seed=1)
    kw = {"background": background, **camera(H, W)}
    weights = jax.random.uniform(jax.random.key(2), (H, W, 3))

    def loss(
        m: jax.Array, s: jax.Array, q: jax.Array, c: jax.Array, o: jax.Array, v: jax.Array
    ) -> jax.Array:
        img, _ = splax.render(m, s, q, c, o, viewmat=v, **kw)
        return jnp.mean(weights * img)

    args = (means, log_scales, quats, sh_colors, logit_opacities, VIEWMAT)
    grads = jax.jit(jax.grad(loss, argnums=(0, 1, 2, 3, 4, 5)))(*args)
    assert all(jnp.isfinite(g).all() for g in grads), "a gradient has non-finite entries"
    assert all(jnp.linalg.norm(g) > 0.0 for g in grads), "a gradient is all zero"


@pytest.mark.parametrize("antialiased", [False, True])
def test_render_grad_finite_difference(antialiased: bool):
    """Match the analytic directional derivative against central finite differences."""
    n, H, W = 400, 80, 80
    means, log_scales, quats, sh_colors, logit_opacities, background = scene_params(n, seed=7)
    kw = {"viewmat": VIEWMAT, "background": background, **camera(H, W)}
    weights = jax.random.uniform(jax.random.key(5), (H, W, 3))

    def loss(m: jax.Array, s: jax.Array, q: jax.Array, c: jax.Array, o: jax.Array) -> jax.Array:
        img, _ = splax.render(m, s, q, c, o, antialiased=antialiased, **kw)
        return jnp.mean(weights * img)

    args = (means, log_scales, quats, sh_colors, logit_opacities)
    grads = jax.jit(jax.grad(loss, argnums=(0, 1, 2, 3, 4)))(*args)
    assert_finite_difference(loss, args, grads)


def test_render_grad_viewmat():
    """Check the camera-pose gradient with directional finite differences."""
    n, H, W = 4000, 120, 120
    *splats, background = scene_params(n, seed=11)
    kw = {"background": background, **camera(H, W)}
    weights = jax.random.uniform(jax.random.key(4), (H, W, 3))
    loss = partial(_render_loss, splats, kw, weights)

    grad = jax.jit(jax.grad(loss))(VIEWMAT)
    assert jnp.isfinite(grad).all(), "viewmat grad has non-finite entries"
    np.testing.assert_array_equal(grad[3], jnp.zeros(4), err_msg="viewmat bottom row is constant")
    assert_finite_difference(loss, (VIEWMAT,), (grad,), eps=1e-3, name="viewmat ")


def test_render_grad_viewmat_pose_chain_rule():
    """Check the gradient through an se3 pose parameterization with finite differences."""
    n, H, W = 4000, 120, 120
    *splats, background = scene_params(n, seed=13)
    kw = {"background": background, **camera(H, W)}
    weights = jax.random.uniform(jax.random.key(8), (H, W, 3))
    xi0 = jnp.array([0.03, -0.02, 0.015, 0.04, -0.03, 0.02])
    render_loss = partial(_render_loss, splats, kw, weights)

    def loss(xi: jax.Array) -> jax.Array:
        generator = jnp.array(
            [
                [0.0, -xi[2], xi[1], xi[3]],
                [xi[2], 0.0, -xi[0], xi[4]],
                [-xi[1], xi[0], 0.0, xi[5]],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        return render_loss(jax.scipy.linalg.expm(generator) @ VIEWMAT)

    grad = jax.jit(jax.grad(loss))(xi0)
    assert jnp.isfinite(grad).all() and jnp.linalg.norm(grad) > 0
    assert_finite_difference(loss, (xi0,), (grad,), eps=1e-3, name="pose chain-rule ")


# region broadcasted gradient

# A gradient for a batched operand is per image, one for a shared operand sums over the batch.


def test_render_grad_vmap_matches_loop():
    """Match the vmapped gaussian gradients against the sequential gradients."""
    n, H, W = 500, 96, 96
    means, log_scales, quats, sh_colors, logit_opacities, background = scene_params(n, seed=2)
    kw = {"viewmat": VIEWMAT, "background": background, **camera(H, W)}
    batched_means = means + 0.02 * jax.random.normal(jax.random.key(1), (B, n, 3))

    def loss(m: jax.Array) -> jax.Array:
        img, _ = splax.render(m, log_scales, quats, sh_colors, logit_opacities, **kw)
        return jnp.sum(img)

    grad = jax.jit(jax.grad(loss))
    batched = jax.jit(jax.vmap(grad))(batched_means)
    sequential = jnp.stack([grad(m) for m in batched_means])
    # The backward accumulates with atomics, so even the sequential path jitters between runs.
    np.testing.assert_allclose(batched, sequential, rtol=2e-3, atol=1e-4)


def test_render_grad_vmap_matches_loop_multiview():
    """Match per-image gaussian gradients under batched poses against the sequential stack."""
    n, H, W = 800, 96, 96
    means, log_scales, quats, sh_colors, logit_opacities, background = scene_params(n, seed=2)
    kw = {"background": background, **camera(H, W)}
    viewmats = poses(B, 7)
    weights = jax.random.uniform(jax.random.key(3), (B, H, W, 3))
    batched_means = means + 0.02 * jax.random.normal(jax.random.key(4), (B, n, 3))

    def loss(m: jax.Array, viewmat: jax.Array, weight: jax.Array) -> jax.Array:
        img, _ = splax.render(
            m, log_scales, quats, sh_colors, logit_opacities, viewmat=viewmat, **kw
        )
        return jnp.mean(weight * img)

    grad = jax.jit(jax.grad(loss))
    batched = jax.jit(jax.vmap(grad))(batched_means, viewmats, weights)
    sequential = jnp.stack([grad(*a) for a in zip(batched_means, viewmats, weights)])
    assert jnp.isfinite(batched).all(), "the vmapped means gradient is not finite"
    np.testing.assert_allclose(batched, sequential, rtol=1e-4, atol=1e-6)


def test_render_grad_viewmat_vmap_matches_loop():
    """Match the vmapped per-pose camera gradients against the sequential gradients."""
    n, H, W = 800, 96, 96
    *splats, background = scene_params(n, seed=11)
    kw = {"background": background, **camera(H, W)}
    viewmats = poses(B, 12)
    weights = jax.random.uniform(jax.random.key(13), (B, H, W, 3))

    grad = jax.jit(jax.grad(partial(_render_loss, splats, kw), argnums=1))
    batched = jax.jit(jax.vmap(grad))(weights, viewmats)
    sequential = jnp.stack([grad(w, vm) for w, vm in zip(weights, viewmats)])
    assert jnp.isfinite(batched).all(), "the vmapped viewmat gradient is not finite"
    np.testing.assert_allclose(batched, sequential, rtol=1e-4, atol=1e-6)
    msg = "viewmat bottom row must be constant"
    np.testing.assert_array_equal(batched[:, 3, :], jnp.zeros((B, 4)), err_msg=msg)


def test_render_grad_viewmat_vmap_finite_difference():
    """Check one batched camera gradient with directional finite differences."""
    n, H, W = 3000, 110, 110
    *splats, background = scene_params(n, seed=21)
    kw = {"background": background, **camera(H, W)}
    viewmats = poses(B, 22)
    weights = jax.random.uniform(jax.random.key(23), (B, H, W, 3))
    loss = partial(_render_loss, splats, kw)

    grad = jax.jit(jax.vmap(jax.grad(loss, argnums=1)))(weights, viewmats)[0]
    assert jnp.isfinite(grad).all()
    assert_finite_difference(
        partial(loss, weights[0]), (viewmats[0],), (grad,), eps=1e-3, name="batched "
    )


@pytest.mark.parametrize("param", PARAMS)
def test_render_grad_broadcast_summed(param: str):
    """Match the summed per-image gradients of a shared parameter against the total-loss one."""
    n, H, W = 800, 96, 96
    means, log_scales, quats, sh_colors, logit_opacities, background = scene_params(n, seed=5)
    splat = (means, log_scales, quats, sh_colors, logit_opacities)
    index = PARAMS.index(param)
    kw = {"background": background, **camera(H, W)}
    viewmats = poses(B, 8)
    weights = jax.random.uniform(jax.random.key(6), (B, H, W, 3))

    def loss(p: jax.Array, viewmat: jax.Array, weight: jax.Array) -> jax.Array:
        args = splat[:index] + (p,) + splat[index + 1 :]
        img, _ = splax.render(*args, viewmat=viewmat, **kw)
        return jnp.mean(weight * img)

    def total_loss(p: jax.Array) -> jax.Array:
        return sum(loss(p, viewmat, weight) for viewmat, weight in zip(viewmats, weights))

    per_image = jax.jit(jax.vmap(jax.grad(loss), in_axes=(None, 0, 0)))(
        splat[index], viewmats, weights
    )
    total = jax.jit(jax.grad(total_loss))(splat[index])
    summed = jnp.sum(per_image, axis=0)
    assert jnp.isfinite(summed).all(), f"the summed {param} gradient is not finite"
    np.testing.assert_allclose(summed, total, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("param", ["means", "sh_colors", "logit_opacities"])
def test_render_grad_broadcast_geometry(param: str):
    """Match the vmapped gradients against the loop when only the loss weight is batched."""
    n, H, W = 600, 80, 80
    means, log_scales, quats, sh_colors, logit_opacities, background = scene_params(n, seed=17)
    splat = (means, log_scales, quats, sh_colors, logit_opacities)
    index = PARAMS.index(param)
    kw = {"viewmat": VIEWMAT, "background": background, **camera(H, W)}
    weights = jax.random.uniform(jax.random.key(18), (B, H, W, 3))

    def loss(p: jax.Array, weight: jax.Array) -> jax.Array:
        args = splat[:index] + (p,) + splat[index + 1 :]
        img, _ = splax.render(*args, **kw)
        return jnp.mean(weight * img)

    grad = jax.jit(jax.grad(loss))
    batched = jax.jit(jax.vmap(grad, in_axes=(None, 0)))(splat[index], weights)
    sequential = jnp.stack([grad(splat[index], weight) for weight in weights])
    assert jnp.isfinite(batched).all(), f"the vmapped {param} gradient is not finite"
    np.testing.assert_allclose(batched, sequential, rtol=1e-4, atol=1e-6)


# region gsplat equivalence


@pytest.mark.gsplat
@pytest.mark.parametrize("n,H,W", [(3000, 128, 128), (8000, 160, 160)])
def test_render_grad_vs_gsplat(n: int, H: int, W: int, gsplat_shim: ModuleType):
    """Match the gradients of a weighted squared-error loss against gsplat's torch autograd."""
    means, log_scales, quats, sh_colors, logit_opacities, background = scene_params(n, seed=n)
    splat = (means, log_scales, quats, sh_colors, logit_opacities)
    kw = {"viewmat": VIEWMAT, "background": background, **camera(H, W)}
    weights = jax.random.uniform(jax.random.key(123), (H, W, 3))

    def loss(m: jax.Array, s: jax.Array, q: jax.Array, c: jax.Array, o: jax.Array) -> jax.Array:
        img, _ = splax.render(m, s, q, c, o, **kw)
        return jnp.mean(weights * img**2)

    g_means, g_log_scales, g_quats, g_sh_colors, g_logit_opacities = jax.jit(
        jax.grad(loss, argnums=(0, 1, 2, 3, 4))
    )(*splat)
    scales, colors, opacities = splax.io.apply_activations(log_scales, sh_colors, logit_opacities)
    ref_means, ref_scales, ref_quats, ref_colors, ref_opacities = gsplat_shim.grad(
        means, scales, quats, colors, opacities, **kw, weight=np.asarray(weights)
    )
    # gsplat differentiates the activated arrays, so its gradients pull back through the activations
    _, pullback = jax.vjp(splax.io.apply_activations, log_scales, sh_colors, logit_opacities)
    ref_log_scales, ref_sh_colors, ref_logit_opacities = pullback(
        (jnp.asarray(ref_scales), jnp.asarray(ref_colors), jnp.asarray(ref_opacities))
    )

    # Accumulation ordering in the kernels leads to small differences in the gradients
    np.testing.assert_allclose(g_means, ref_means, rtol=1e-2, atol=1e-6)
    np.testing.assert_allclose(g_log_scales, ref_log_scales, rtol=1e-2, atol=1e-6)
    np.testing.assert_allclose(g_quats, ref_quats, rtol=1e-2, atol=1e-6)
    np.testing.assert_allclose(g_sh_colors, ref_sh_colors, rtol=1e-2, atol=1e-6)
    np.testing.assert_allclose(g_logit_opacities, ref_logit_opacities, rtol=1e-2, atol=1e-6)


@pytest.mark.gsplat
def test_render_grad_vs_gsplat_lego_viewmat(
    gsplat_shim: ModuleType, lego_meta: dict, lego_ply: Path
):
    """Match the lego camera gradient against gsplat's torch autograd."""
    splat = splax.io.load_ply(lego_ply)
    means, log_scales, quats, sh_colors, logit_opacities = splat
    # gsplat renders from the activated arrays the parameters map onto
    scales, colors, opacities = splax.io.apply_activations(log_scales, sh_colors, logit_opacities)
    gsplat_splat = (means, scales, quats, colors, opacities)
    viewmat = splax.utils.nerf_camera(lego_meta["frames"][0]["transform_matrix"])
    focal = float(0.5 * LEGO_WIDTH / np.tan(0.5 * lego_meta["camera_angle_x"]))
    kw = {
        "background": jnp.ones(3),
        "img_shape": (LEGO_HEIGHT, LEGO_WIDTH),
        "f": (focal, focal),
        "c": (LEGO_WIDTH / 2, LEGO_HEIGHT / 2),
    }
    offset = RigidTransform.from_exp_coords(LEGO_OFFSET).as_matrix()
    target = jax.jit(partial(splax.render, *splat, **kw))(viewmat=offset @ viewmat)[0]
    pose = RigidTransform.from_exp_coords(0.5 * LEGO_OFFSET).as_matrix() @ viewmat

    def loss(viewmat_in: jax.Array) -> jax.Array:
        img, _ = splax.render(*splat, viewmat=viewmat_in, **kw)
        return jnp.mean((img - target) ** 2)

    grad = jax.jit(jax.grad(loss))(pose)
    reference = gsplat_shim.viewmat_grad(
        *gsplat_splat, viewmat=pose, target=np.asarray(target), **kw
    )
    assert jnp.linalg.norm(grad) > 0.0, "the camera gradient is zero"
    np.testing.assert_allclose(grad, reference, rtol=1e-3, atol=1e-4)
