"""Gradient tests for ``splax.render``.

Three axes are covered. Central finite differences validate the gradients with respect to the five
splat parameters and the camera ``viewmat`` without any external reference. Batch-native gradients
under ``jax.vmap`` are matched against the per-sample sequential ``jax.grad`` loop. Parity against
the gsplat reference cross-checks the whole gradient field against a second CUDA implementation.

Gradient selection is purely ``jax.grad`` argnums. Differentiating with respect to a gaussian input
runs the gaussian-grad kernels, differentiating with respect to the ``viewmat`` runs the
camera-pose accumulator, and both together run the joint kernel.

The batched contract. The projection and rasterization backward FFIs are
``vmap_method="expand_dims"``, so ``jax.vmap(jax.grad(loss))`` runs a single batched backward
launch instead of falling back to a per-sample Python loop. A gradient with respect to a batched
operand is per image, while a gradient with respect to a broadcast operand is the SUM over the
batch axis, because the vjp of a broadcast is a sum. In the degenerate case the whole render is
shared and only the image cotangent is batched, so the backward indexes geometry from image 0
while scattering per-output-image gradients.

Tolerances. Float32 central differences land within 8e-2 relative of the analytic directional
derivative, the residual being intrinsic to splatting's hard 1/255 cull and early-termination
discontinuities, which an FD step crosses. The vmap and the sequential path differ only by float32
atomic-add ordering across the launch geometry, which stays well under 1e-3 relative. Against
gsplat every parameter agrees to ~1e-4 relative Frobenius, so a 2e-3 bound holds with margin.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial.transform import RigidTransform
from utils import VIEWMAT, assert_finite_difference, camera, poses, scene

import splax

if TYPE_CHECKING:
    from pathlib import Path
    from types import ModuleType

B = 3
PARAMS = ("means", "scales", "quats", "colors", "opacities")
LEGO_HEIGHT = LEGO_WIDTH = 800
LEGO_OFFSET = jnp.array([0.01, -0.01, 0.01, 0.01, -0.01, 0.01])


def _assert_grads_close(name: str, batched: jax.Array, sequential: jax.Array):
    """Match a batched gradient against the per-sample sequential stack."""
    batched, sequential = np.asarray(batched), np.asarray(sequential)
    assert batched.shape == sequential.shape, f"{name}: shape {batched.shape} vs {sequential.shape}"
    assert np.all(np.isfinite(batched)), f"{name}: non-finite vmap grad"
    rel = np.linalg.norm(batched - sequential) / (np.linalg.norm(sequential) + 1e-12)
    assert rel < 1e-3, f"{name}: vmap vs sequential rel error {rel:.2e}"
    assert np.allclose(batched, sequential, rtol=1e-4, atol=1e-6), (
        f"{name}: max|d| = {np.abs(batched - sequential).max():.2e}"
    )


# region gradient


def test_render_grad():
    """Produce finite, nonzero gradients for the five splat parameters and the viewmat."""
    n, H, W = 2000, 96, 96
    means, scales, quats, colors, opacities, background = scene(n, seed=1)
    kw = {"background": background, **camera(H, W)}
    weights = jax.random.uniform(jax.random.key(2), (H, W, 3))

    def loss(
        m: jax.Array, s: jax.Array, q: jax.Array, c: jax.Array, o: jax.Array, v: jax.Array
    ) -> jax.Array:
        img, _ = splax.render(m, s, q, c, o, viewmat=v, **kw)
        return jnp.mean(weights * img)

    args = (means, scales, quats, colors, opacities, VIEWMAT)
    grads = jax.grad(loss, argnums=(0, 1, 2, 3, 4, 5))(*args)
    for name, grad, arg in zip((*PARAMS, "viewmat"), grads, args):
        grad = np.asarray(grad)
        assert grad.shape == arg.shape, f"{name}: grad shape {grad.shape} vs {arg.shape}"
        assert np.all(np.isfinite(grad)), f"{name}: non-finite grad"
        assert np.linalg.norm(grad) > 0.0, f"{name}: zero grad"


@pytest.mark.parametrize("antialiased", [False, True])
def test_render_grad_finite_difference(antialiased: bool):
    """Match the analytic directional derivative against central finite differences.

    Hundreds of random parameters are exercised at once via a random unit direction per array.
    Central differences in float32 give ~1e-2 relative accuracy, so a loose relative bound is used.
    """
    n, H, W = 400, 80, 80
    means, scales, quats, colors, opacities, background = scene(n, seed=7)
    kw = {"viewmat": VIEWMAT, "background": background, **camera(H, W)}
    weights = jax.random.uniform(jax.random.key(5), (H, W, 3))

    def loss(m: jax.Array, s: jax.Array, q: jax.Array, c: jax.Array, o: jax.Array) -> jax.Array:
        # Linear, mean-reduced loss keeps the loss magnitude small (float32 render, so minimal FD
        # cancellation) while giving an O(1) gradient over all five parameter arrays at once
        # (~4800 perturbed entries).
        img, _ = splax.render(m, s, q, c, o, antialiased=antialiased, **kw)
        return jnp.mean(weights * img)

    args = (means, scales, quats, colors, opacities)
    grads = jax.grad(loss, argnums=(0, 1, 2, 3, 4))(*args)
    assert_finite_difference(loss, args, grads)


def test_render_grad_jit_matches_eager():
    """Match the jitted gradients against the eager gradients."""
    n, H, W = 2000, 128, 128
    means, scales, quats, colors, opacities, background = scene(n, seed=3)
    kw = {"viewmat": VIEWMAT, "background": background, **camera(H, W)}
    weights = jax.random.uniform(jax.random.key(123), (H, W, 3))

    def loss(m: jax.Array, s: jax.Array, q: jax.Array, c: jax.Array, o: jax.Array) -> jax.Array:
        img, _ = splax.render(m, s, q, c, o, **kw)
        return jnp.mean(weights * img**2)

    args = (means, scales, quats, colors, opacities)
    eager = jax.grad(loss, argnums=(0, 1, 2, 3, 4))(*args)
    jitted = jax.jit(jax.grad(loss, argnums=(0, 1, 2, 3, 4)))(*args)
    for a, b in zip(eager, jitted):
        assert np.allclose(np.asarray(a), np.asarray(b), rtol=1e-5, atol=1e-6)


def test_render_grad_viewmat():
    """Check the camera-pose gradient with directional finite differences."""
    n, H, W = 4000, 120, 120
    means, scales, quats, colors, opacities, background = scene(n, seed=11)
    kw = {"background": background, **camera(H, W)}
    weights = jax.random.uniform(jax.random.key(4), (H, W, 3))

    def loss(viewmat: jax.Array) -> jax.Array:
        img, _ = splax.render(means, scales, quats, colors, opacities, viewmat=viewmat, **kw)
        return jnp.mean(weights * img)

    grad = jax.grad(loss)(VIEWMAT)
    assert np.all(np.isfinite(np.asarray(grad))), "viewmat grad has non-finite entries"
    assert np.allclose(np.asarray(grad)[3], 0.0), "the viewmat bottom row is constant"
    # The bottom row is constant, so the step stays on the 12 differentiable entries.
    differentiable = grad.at[3].set(0.0)
    assert_finite_difference(loss, (VIEWMAT,), (differentiable,), eps=1e-3, name="viewmat ")


def test_render_grad_viewmat_pose_chain_rule():
    """Check the gradient through an se3 pose parameterization with finite differences."""
    n, H, W = 4000, 120, 120
    means, scales, quats, colors, opacities, background = scene(n, seed=13)
    kw = {"background": background, **camera(H, W)}
    weights = jax.random.uniform(jax.random.key(8), (H, W, 3))
    xi0 = jnp.array([0.03, -0.02, 0.015, 0.04, -0.03, 0.02])

    def loss(xi: jax.Array) -> jax.Array:
        generator = jnp.array(
            [
                [0.0, -xi[2], xi[1], xi[3]],
                [xi[2], 0.0, -xi[0], xi[4]],
                [-xi[1], xi[0], 0.0, xi[5]],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        viewmat = jax.scipy.linalg.expm(generator) @ VIEWMAT
        img, _ = splax.render(means, scales, quats, colors, opacities, viewmat=viewmat, **kw)
        return jnp.mean(weights * img)

    grad = jax.grad(loss)(xi0)
    assert np.all(np.isfinite(np.asarray(grad))) and np.linalg.norm(np.asarray(grad)) > 0
    assert_finite_difference(loss, (xi0,), (grad,), eps=1e-3, name="pose chain-rule ")


def test_render_grad_selection_consistency():
    """Match the joint-kernel gradients against the single-path kernel gradients."""
    n, H, W = 3000, 110, 110
    means, scales, quats, colors, opacities, background = scene(n, seed=5)
    kw = {"background": background, **camera(H, W)}
    weights = jax.random.uniform(jax.random.key(6), (H, W, 3))

    def loss(m: jax.Array, viewmat: jax.Array) -> jax.Array:
        img, _ = splax.render(m, scales, quats, colors, opacities, viewmat=viewmat, **kw)
        return jnp.mean(weights * img)

    # Gaussian grad: means-only (gaussian kernel) vs joint (both kernel).
    means_only = jax.grad(loss, argnums=0)(means, VIEWMAT)
    means_joint, viewmat_joint = jax.grad(loss, argnums=(0, 1))(means, VIEWMAT)
    assert np.allclose(np.asarray(means_only), np.asarray(means_joint), rtol=1e-5, atol=1e-6)

    # Camera grad: viewmat-only (view kernel) vs joint (both kernel).
    viewmat_only = np.asarray(jax.grad(loss, argnums=1)(means, VIEWMAT))
    assert np.allclose(viewmat_only, np.asarray(viewmat_joint), rtol=1e-4, atol=1e-6)


# region broadcasted gradient


def test_render_grad_vmap_matches_loop():
    """Match the vmapped gaussian gradients against the sequential gradients."""
    n, H, W = 500, 96, 96
    means, scales, quats, colors, opacities, background = scene(n, seed=2)
    kw = {"viewmat": VIEWMAT, "background": background, **camera(H, W)}
    batched_means = means + 0.02 * jax.random.normal(jax.random.key(1), (B, n, 3))

    def loss(m: jax.Array) -> jax.Array:
        img, _ = splax.render(m, scales, quats, colors, opacities, **kw)
        return jnp.sum(img)

    batched = np.asarray(jax.vmap(jax.grad(loss))(batched_means))
    sequential = np.stack([np.asarray(jax.grad(loss)(batched_means[i])) for i in range(B)])
    # The rasterize backward accumulates with atomics, so even the sequential path jitters run to
    # run. rtol=1e-4 flaked on this sum-reduced loss.
    assert np.allclose(batched, sequential, rtol=2e-3, atol=1e-4)


def test_render_grad_vmap_matches_loop_multiview():
    """Match per-image gaussian gradients under batched poses against the sequential stack."""
    n, H, W = 800, 96, 96
    means, scales, quats, colors, opacities, background = scene(n, seed=2)
    kw = {"background": background, **camera(H, W)}
    viewmats = poses(B, 7)
    weights = jax.random.uniform(jax.random.key(3), (B, H, W, 3))
    batched_means = means + 0.02 * jax.random.normal(jax.random.key(4), (B, n, 3))

    def loss(m: jax.Array, viewmat: jax.Array, i: jax.Array | int) -> jax.Array:
        img, _ = splax.render(m, scales, quats, colors, opacities, viewmat=viewmat, **kw)
        return jnp.mean(weights[i] * img)

    batched = jax.vmap(jax.grad(loss))(batched_means, viewmats, jnp.arange(B))
    sequential = jnp.stack([jax.grad(loss)(batched_means[i], viewmats[i], i) for i in range(B)])
    _assert_grads_close("batched means", batched, sequential)


def test_render_grad_viewmat_vmap_matches_loop():
    """Match the vmapped per-pose camera gradients against the sequential gradients."""
    n, H, W = 800, 96, 96
    means, scales, quats, colors, opacities, background = scene(n, seed=11)
    kw = {"background": background, **camera(H, W)}
    viewmats = poses(B, 12)
    weights = jax.random.uniform(jax.random.key(13), (B, H, W, 3))

    def loss(viewmat: jax.Array, i: jax.Array | int) -> jax.Array:
        img, _ = splax.render(means, scales, quats, colors, opacities, viewmat=viewmat, **kw)
        return jnp.mean(weights[i] * img)

    batched = jax.vmap(jax.grad(loss))(viewmats, jnp.arange(B))
    sequential = jnp.stack([jax.grad(loss)(viewmats[i], i) for i in range(B)])
    _assert_grads_close("batched viewmat", batched, sequential)
    assert np.allclose(np.asarray(batched)[:, 3, :], 0.0), "viewmat bottom row must be constant"


def test_render_grad_viewmat_vmap_finite_difference():
    """Check one batched camera gradient with directional finite differences.

    The batched camera grad is thereby validated against numerics and not only against the
    sequential kernel.
    """
    n, H, W = 3000, 110, 110
    means, scales, quats, colors, opacities, background = scene(n, seed=21)
    kw = {"background": background, **camera(H, W)}
    viewmats = poses(B, 22)
    weights = jax.random.uniform(jax.random.key(23), (B, H, W, 3))

    def loss(viewmat: jax.Array, i: jax.Array | int) -> jax.Array:
        img, _ = splax.render(means, scales, quats, colors, opacities, viewmat=viewmat, **kw)
        return jnp.mean(weights[i] * img)

    grad = jax.vmap(jax.grad(loss))(viewmats, jnp.arange(B))[0]
    assert np.all(np.isfinite(np.asarray(grad)))
    # The bottom row is constant, so the step stays on the 12 differentiable entries.
    differentiable = grad.at[3].set(0.0)
    assert_finite_difference(
        partial(loss, i=0), (viewmats[0],), (differentiable,), eps=1e-3, name="batched viewmat "
    )


@pytest.mark.parametrize("param", PARAMS)
def test_render_grad_broadcast_summed(param: str):
    """Match the summed per-image gradient of a broadcast parameter against the total-loss one.

    The multi-view regime batches the camera pose, which makes the whole projected geometry
    batched, while the gaussians stay shared.
    """
    n, H, W = 800, 96, 96
    means, scales, quats, colors, opacities, background = scene(n, seed=5)
    splat = (means, scales, quats, colors, opacities)
    index = PARAMS.index(param)
    kw = {"background": background, **camera(H, W)}
    viewmats = poses(B, 8)
    weights = jax.random.uniform(jax.random.key(6), (B, H, W, 3))

    def loss(p: jax.Array, viewmat: jax.Array, i: jax.Array | int) -> jax.Array:
        args = splat[:index] + (p,) + splat[index + 1 :]
        img, _ = splax.render(*args, viewmat=viewmat, **kw)
        return jnp.mean(weights[i] * img)

    per_image = jax.vmap(jax.grad(loss), in_axes=(None, 0, 0))(
        splat[index], viewmats, jnp.arange(B)
    )
    total = jax.grad(lambda p: sum(loss(p, viewmats[i], i) for i in range(B)))(splat[index])
    _assert_grads_close(f"{param} broadcast", jnp.sum(per_image, axis=0), total)


@pytest.mark.parametrize("param", ["means", "colors", "opacities"])
def test_render_grad_broadcast_geometry(param: str):
    """Match the vmapped gradients of a render shared across the batch against the loop.

    Nothing inside the render is batched, only the per-image loss weight, so the projected
    geometry is broadcast while the image cotangent is batched.
    """
    n, H, W = 600, 80, 80
    means, scales, quats, colors, opacities, background = scene(n, seed=17)
    splat = (means, scales, quats, colors, opacities)
    index = PARAMS.index(param)
    kw = {"viewmat": VIEWMAT, "background": background, **camera(H, W)}
    weights = jax.random.uniform(jax.random.key(18), (B, H, W, 3))

    def loss(p: jax.Array, i: jax.Array | int) -> jax.Array:
        args = splat[:index] + (p,) + splat[index + 1 :]
        img, _ = splax.render(*args, **kw)
        return jnp.mean(weights[i] * img)

    batched = jax.vmap(jax.grad(loss), in_axes=(None, 0))(splat[index], jnp.arange(B))
    sequential = jnp.stack([jax.grad(loss)(splat[index], i) for i in range(B)])
    _assert_grads_close(f"broadcast geometry {param}", batched, sequential)


# region gsplat equivalence


@pytest.mark.gsplat
@pytest.mark.parametrize("n,H,W", [(3000, 128, 128), (8000, 160, 160)])
@pytest.mark.parametrize("which", ["sum", "wmse"])
def test_render_grad_vs_gsplat(n: int, H: int, W: int, which: str, gsplat_shim: ModuleType):
    """Match the gradients of two scalar losses against gsplat's torch autograd."""
    means, scales, quats, colors, opacities, background = scene(n, seed=n)
    splat = (means, scales, quats, colors, opacities)
    kw = {"viewmat": VIEWMAT, "background": background, **camera(H, W)}
    weights = jax.random.uniform(jax.random.key(123), (H, W, 3))

    def loss(m: jax.Array, s: jax.Array, q: jax.Array, c: jax.Array, o: jax.Array) -> jax.Array:
        img, _ = splax.render(m, s, q, c, o, **kw)
        return jnp.sum(img) if which == "sum" else jnp.mean(weights * img**2)

    grads = jax.grad(loss, argnums=(0, 1, 2, 3, 4))(*splat)
    weight = None if which == "sum" else np.asarray(weights)
    reference = gsplat_shim.grad(*splat, **kw, weight=weight)

    for name, grad, ref in zip(PARAMS, grads, reference):
        grad, ref = np.asarray(grad), np.asarray(ref)
        # Relative Frobenius is the meaningful metric across the two kernels. The whole gradient
        # field agrees to ~1e-4 relative.
        rel = np.linalg.norm(grad - ref) / (np.linalg.norm(ref) + 1e-12)
        assert rel < 2e-3, f"{which}/{name} relative grad error {rel:.2e}"


@pytest.mark.gsplat
def test_render_grad_vs_gsplat_lego_viewmat(
    gsplat_shim: ModuleType, lego_meta: dict, lego_ply: Path
):
    """Match the lego camera gradients against gsplat's torch autograd.

    Both implementations differentiate a pixelwise MSE against a target rendered at a slightly
    offset pose. Single-view and vmap-batched gradients are covered.
    """
    means, scales, quats, colors, opacities = splax.io.load_ply(lego_ply)
    viewmat = splax.utils.nerf_camera(lego_meta["frames"][0]["transform_matrix"])
    focal = float(0.5 * LEGO_WIDTH / np.tan(0.5 * lego_meta["camera_angle_x"]))
    kw = {
        "background": jnp.ones(3),
        "img_shape": (LEGO_HEIGHT, LEGO_WIDTH),
        "f": (focal, focal),
        "c": (LEGO_WIDTH / 2, LEGO_HEIGHT / 2),
    }
    target_viewmat = RigidTransform.from_exp_coords(LEGO_OFFSET).as_matrix() @ viewmat
    # Three current poses at scaled offsets from the base pose.
    tangents = jnp.array([0.25, 0.5, 0.75])[:, None] * LEGO_OFFSET[None, :]
    viewmats = RigidTransform.from_exp_coords(tangents).as_matrix() @ viewmat

    # Render the target in the same framework to reduce noise from framework differences.
    target = splax.render(means, scales, quats, colors, opacities, viewmat=target_viewmat, **kw)[0]
    gsplat_target, _gsplat_alpha = gsplat_shim.render(
        means, scales, quats, colors, opacities, viewmat=target_viewmat, **kw
    )

    def loss(viewmat_in: jax.Array) -> jax.Array:
        img, _ = splax.render(means, scales, quats, colors, opacities, viewmat=viewmat_in, **kw)
        return jnp.mean((img - target) ** 2)

    single = np.asarray(jax.grad(loss)(viewmats[1]))
    reference = gsplat_shim.viewmat_grad(
        means, scales, quats, colors, opacities, viewmat=viewmats[1], target=gsplat_target, **kw
    )
    assert np.allclose(single, reference, rtol=1e-3, atol=1e-4)
    assert not np.allclose(single, np.zeros_like(single)), "gradient is zero"

    batched = np.asarray(jax.vmap(jax.grad(loss))(viewmats))
    batched_reference = np.asarray(
        [
            gsplat_shim.viewmat_grad(
                means,
                scales,
                quats,
                colors,
                opacities,
                viewmat=viewmats[view],
                target=gsplat_target,
                **kw,
            )
            for view in range(B)
        ]
    )
    assert np.allclose(batched, batched_reference, rtol=1e-3, atol=1e-4)
