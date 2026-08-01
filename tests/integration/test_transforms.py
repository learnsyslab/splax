"""Test the per-object rigid transforms, in the forward render and in the gradients."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial.transform import RigidTransform as TF
from scipy.spatial.transform import Rotation as R
from utils import VIEWMAT, camera, manual_move, psnr, scene_params

import splax


def test_identity_transforms_byte_identical():
    """Leave the render untouched when every active transform is the identity."""
    n = 4000
    means, log_scales, quats, sh_colors, logit_opac, bg = scene_params(n, seed=1)
    splats = (means, log_scales, quats, sh_colors, logit_opac)
    kw = {"viewmat": VIEWMAT, "background": bg, **camera(128, 128)}
    render = jax.jit(partial(splax.render, *splats, **kw))
    identity = jax.jit(
        partial(splax.render, *splats, **kw, gaussian_slices=((0, 1000), (2000, 3000)))
    )

    eye = jnp.broadcast_to(jnp.eye(4, dtype=jnp.float32), (2, 4, 4))
    np.testing.assert_array_equal(render()[0], identity(gaussian_transforms=eye)[0])


def test_render_matches_manual_transform():
    """Match a transformed render against the same move applied to the splat arrays."""
    n = 4000
    means, log_scales, quats, sh_colors, logit_opac, bg = scene_params(n, seed=3)
    kw = {"viewmat": VIEWMAT, "background": bg, **camera(128, 128)}
    rot = R.from_euler("xyz", [0.26, -0.17, 0.52])
    T = TF.from_components((0.3, -0.2, 0.1), rot).as_matrix().astype(np.float32)
    render = jax.jit(partial(splax.render, **kw))
    transformed = jax.jit(partial(splax.render, **kw, gaussian_slices=((0, 1000),)))

    moved = transformed(
        means, log_scales, quats, sh_colors, logit_opac, gaussian_transforms=jnp.asarray(T)[None]
    )[0]
    m2, q2 = manual_move(means, quats, T, 0, 1000)
    ref = render(m2, log_scales, q2, sh_colors, logit_opac)[0]
    quality = psnr(moved, ref)
    assert quality > 60, f"kernel vs manual transform PSNR only {quality:.1f} dB"
    # the transform must actually change the image
    plain = render(means, log_scales, quats, sh_colors, logit_opac)[0]
    assert jnp.abs(moved - plain).max() > 1e-2


def test_vmap_over_transforms_matches_sequential():
    """Match a vmap over a stack of transforms against the loop over the single renders."""
    n, B = 4000, 3
    means, log_scales, quats, sh_colors, logit_opac, bg = scene_params(n, seed=4)
    kw = {"viewmat": VIEWMAT, "background": bg, **camera(96, 96)}
    angles = np.array([[0.0, 0.0, 0.3 * i] for i in range(B)])
    trans = np.array([[0.05 * i, -0.03 * i, 0.0] for i in range(B)])
    Ts = TF.from_components(trans, R.from_euler("xyz", angles)).as_matrix().astype(np.float32)
    tfs = jnp.asarray(Ts)[:, None]  # (B, 1, 4, 4)

    render_tf = jax.jit(
        partial(
            splax.render,
            means,
            log_scales,
            quats,
            sh_colors,
            logit_opac,
            **kw,
            gaussian_slices=((500, 1500),),
        )
    )

    out = jax.jit(jax.vmap(render_tf))(gaussian_transforms=tfs)[0]
    seq = jnp.stack([render_tf(gaussian_transforms=tf)[0] for tf in tfs])
    np.testing.assert_array_equal(out, seq)
    # elements genuinely differ
    assert jnp.abs(out[0] - out[B - 1]).max() > 1e-2


def test_two_objects_move_independently():
    """Move two slices independently and match the manual reference."""
    n = 4000
    means, log_scales, quats, sh_colors, logit_opac, bg = scene_params(n, seed=5)
    kw = {"viewmat": VIEWMAT, "background": bg, **camera(128, 128)}
    rot_a = R.from_euler("xyz", [0.0, 0.0, 0.4])
    Ta = TF.from_components((0.2, 0.0, 0.0), rot_a).as_matrix().astype(np.float32)
    rot_b = R.from_euler("xyz", [0.3, 0.0, 0.0])
    Tb = TF.from_components((-0.1, 0.15, 0.0), rot_b).as_matrix().astype(np.float32)
    render = jax.jit(partial(splax.render, **kw))
    transformed = jax.jit(
        partial(
            splax.render,
            means,
            log_scales,
            quats,
            sh_colors,
            logit_opac,
            **kw,
            gaussian_slices=((0, 800), (2000, 2600)),
        )
    )

    both = transformed(gaussian_transforms=jnp.asarray(np.stack([Ta, Tb])))[0]
    m2, q2 = manual_move(means, quats, Ta, 0, 800)
    m2, q2 = manual_move(m2, q2, Tb, 2000, 2600)
    ref = render(m2, log_scales, q2, sh_colors, logit_opac)[0]
    quality = psnr(both, ref)
    assert quality > 60, f"two-object transform PSNR only {quality:.1f} dB"

    swapped = transformed(gaussian_transforms=jnp.asarray(np.stack([Tb, Ta])))[0]
    assert jnp.abs(both - swapped).max() > 1e-2


# region gradients

N = 2000
SLICES = ((0, 700), (1000, 1600))
K = len(SLICES)
IDS = np.full(N, -1, np.int32)
for k, (start, stop) in enumerate(SLICES):
    IDS[start:stop] = k
MOVED = IDS >= 0

ROTVECS = jnp.asarray([[0.15, -0.08, 0.3], [-0.2, 0.12, 0.05]], jnp.float32)
TRANS = jnp.asarray([[0.1, -0.05, 0.02], [-0.08, 0.03, 0.1]], jnp.float32)


def _loss_kernel(
    means: jax.Array, log_scales: jax.Array, quats: jax.Array, tfs: jax.Array, *, extras: tuple
) -> jax.Array:
    """Render a splat with the transforms applied by the kernel and score it on a target."""
    sh_colors, logit_opac, kw, target = extras
    img, _ = splax.render(
        means,
        log_scales,
        quats,
        sh_colors,
        logit_opac,
        **kw,
        gaussian_transforms=tfs,
        gaussian_slices=SLICES,
    )
    return jnp.mean((img - target) ** 2)


def _loss_reference(
    means: jax.Array, log_scales: jax.Array, quats: jax.Array, tfs: jax.Array, *, extras: tuple
) -> jax.Array:
    """Render the same transform applied to the splat arrays in jax and score it on a target."""
    sh_colors, logit_opac, kw, target = extras
    # assume_valid skips the orthonormalization, whose branchy gradient is NaN under jax.grad
    rot = R.from_matrix(tfs[:, :3, :3], assume_valid=True)[IDS]
    moved = rot.apply(means) + tfs[:, :3, 3][IDS]
    composed = rot * R.from_quat(quats, scalar_first=True)
    means_ref = jnp.where(MOVED[:, None], moved, means)
    quats_ref = jnp.where(MOVED[:, None], composed.as_quat(scalar_first=True), quats)
    img, _ = splax.render(means_ref, log_scales, quats_ref, sh_colors, logit_opac, **kw)
    return jnp.mean((img - target) ** 2)


def _setup(seed: int) -> tuple:
    """Draw a splat, a camera, and the target image the transform losses are scored on."""
    means, log_scales, quats, sh_colors, logit_opac, bg = scene_params(N, seed=seed)
    kw = {"viewmat": VIEWMAT, "background": bg, **camera(96, 96)}
    target = jax.random.uniform(jax.random.key(100 + seed), (96, 96, 3))
    return means, log_scales, quats, (sh_colors, logit_opac, kw, target)


def test_identity_transforms_match_plain_grads():
    """Match the gradients under identity transforms against the plain untransformed ones."""
    means, log_scales, quats, extras = _setup(seed=2)
    eye = jnp.broadcast_to(jnp.eye(4, dtype=jnp.float32), (K, 4, 4))
    id_means, id_log_scales, id_quats = jax.grad(_loss_kernel, argnums=(0, 1, 2))(
        means, log_scales, quats, eye, extras=extras
    )

    def loss_plain(means: jax.Array, log_scales: jax.Array, quats: jax.Array) -> jax.Array:
        sh_colors, logit_opac, kw, target = extras
        img, _ = splax.render(means, log_scales, quats, sh_colors, logit_opac, **kw)
        return jnp.mean((img - target) ** 2)

    p_means, p_log_scales, p_quats = jax.jit(jax.grad(loss_plain, argnums=(0, 1, 2)))(
        means, log_scales, quats
    )
    np.testing.assert_allclose(id_means, p_means, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(id_log_scales, p_log_scales, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(id_quats, p_quats, rtol=1e-5, atol=1e-8)


def test_gaussian_grads_match_jax_reference():
    """Match the kernel's gaussian gradients against the pure jax reference transform."""
    means, log_scales, quats, extras = _setup(seed=3)
    tfs = TF.from_components(TRANS, R.from_rotvec(ROTVECS)).as_matrix()
    kernel_grad = jax.jit(partial(jax.grad(_loss_kernel, argnums=(0, 1, 2)), extras=extras))
    reference_grad = jax.jit(partial(jax.grad(_loss_reference, argnums=(0, 1, 2)), extras=extras))
    k_means, k_log_scales, k_quats = kernel_grad(means, log_scales, quats, tfs)
    r_means, r_log_scales, r_quats = reference_grad(means, log_scales, quats, tfs)

    np.testing.assert_allclose(k_means, r_means, rtol=5e-3, atol=2e-6)
    np.testing.assert_allclose(k_log_scales, r_log_scales, rtol=5e-3, atol=2e-6)
    # quats in the tangent space of the unit sphere, see the module docstring
    k_radial = jnp.sum(k_quats * quats, axis=1, keepdims=True) * quats
    r_radial = jnp.sum(r_quats * quats, axis=1, keepdims=True) * quats
    np.testing.assert_allclose(k_quats - k_radial, r_quats - r_radial, atol=2e-6)


def test_pose_grads_match_jax_reference():
    """Match the transform gradients contracted to rotvec and translation coordinates."""
    means, log_scales, quats, extras = _setup(seed=4)

    def transforms(rotvecs: jax.Array, trans: jax.Array) -> jax.Array:
        # RigidTransform holds its rotation as a quaternion, and the round trip out to a matrix
        # does not carry the gradient back to the rotation vector, so the matrix is built directly.
        eye = jnp.broadcast_to(jnp.eye(4, dtype=jnp.float32), (K, 4, 4))
        rotation = R.from_rotvec(rotvecs).as_matrix()
        return eye.at[:, :3, :3].set(rotation).at[:, :3, 3].set(trans)

    def kernel_pose_loss(rotvecs: jax.Array, trans: jax.Array) -> jax.Array:
        return _loss_kernel(means, log_scales, quats, transforms(rotvecs, trans), extras=extras)

    def reference_pose_loss(rotvecs: jax.Array, trans: jax.Array) -> jax.Array:
        return _loss_reference(means, log_scales, quats, transforms(rotvecs, trans), extras=extras)

    k_rotvecs, k_trans = jax.jit(jax.grad(kernel_pose_loss, argnums=(0, 1)))(ROTVECS, TRANS)
    r_rotvecs, r_trans = jax.jit(jax.grad(reference_pose_loss, argnums=(0, 1)))(ROTVECS, TRANS)
    np.testing.assert_allclose(k_rotvecs, r_rotvecs, rtol=5e-3, atol=2e-6)
    np.testing.assert_allclose(k_trans, r_trans, rtol=5e-3, atol=2e-6)
    assert jnp.abs(k_rotvecs).max() > 0 and jnp.abs(k_trans).max() > 0


def test_transform_grad_bottom_row_zero():
    """Leave the constant bottom row of every transform without a gradient."""
    means, log_scales, quats, extras = _setup(seed=5)
    tfs = TF.from_components(TRANS, R.from_rotvec(ROTVECS)).as_matrix()
    kernel_grad = jax.jit(partial(jax.grad(_loss_kernel, argnums=3), extras=extras))
    grad = kernel_grad(means, log_scales, quats, tfs)
    np.testing.assert_array_equal(grad[:, 3, :], jnp.zeros((K, 4)))


def test_vmap_grad_over_transform_stack():
    """Match a vmapped transform gradient against the loop over the single gradients."""
    means, log_scales, quats, extras = _setup(seed=6)
    rots = R.from_rotvec(jnp.stack([ROTVECS, -ROTVECS]))
    tfs = TF.from_components(jnp.stack([ROTVECS, -ROTVECS]), rots).as_matrix()
    stack = jnp.stack([tfs[0], tfs[1]])
    grad = jax.jit(
        partial(jax.grad(_loss_kernel, argnums=3), means, log_scales, quats, extras=extras)
    )
    batched = jax.jit(jax.vmap(grad))(stack)
    sequential = jnp.stack([grad(tfs) for tfs in stack])
    np.testing.assert_allclose(batched, sequential, rtol=1e-5, atol=1e-8)


def test_vmap_grad_over_viewmats_with_transforms():
    """Match a vmapped camera gradient under active transforms against the unbatched loop."""
    means, log_scales, quats, extras = _setup(seed=7)
    sh_colors, logit_opac, kw, target = extras
    tfs = TF.from_components(TRANS, R.from_rotvec(ROTVECS)).as_matrix()
    viewmats = jnp.stack([kw["viewmat"], kw["viewmat"].at[0, 3].add(0.15)])

    def loss(viewmat: jax.Array) -> jax.Array:
        view_kw = kw | {"viewmat": viewmat}
        img, _ = splax.render(
            means,
            log_scales,
            quats,
            sh_colors,
            logit_opac,
            **view_kw,
            gaussian_transforms=tfs,
            gaussian_slices=SLICES,
        )
        return jnp.mean((img - target) ** 2)

    grad = jax.jit(jax.grad(loss))
    batched = jax.jit(jax.vmap(grad))(viewmats)
    sequential = jnp.stack([grad(viewmat) for viewmat in viewmats])
    np.testing.assert_allclose(batched, sequential, rtol=1e-5, atol=1e-8)
