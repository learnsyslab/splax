"""Differentiable rendering entry point.

``splax.training.render`` composes the projection and rasterization primitives with their custom
autodiff rules, so ``jax.grad`` flows through it with respect to the gaussian parameters and the
camera pose. It shares every Warp kernel with the inference path and differs only in the forward
rule keeping the blend residuals alive for the backward pass.

Batched gradients are batch-native. ``jax.vmap(jax.grad(render))`` runs a single batched backward
launch and matches per-sample sequential gradients. Inputs shared across the batch get their
gradients summed over the batch axis, while per-image inputs such as a batch of camera poses get
per-image gradients.

The module also carries the training toolkit on top of the differentiable renderer.
``render_params`` renders from the unconstrained log/logit parameterization, ``make_step`` builds a
jitted train step with the 3DGS loss terms, and the exposure and pose helpers provide the per-image
auxiliary corrections for real captures.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import dm_pix
import jax
import jax.numpy as jnp
import optax

from splax._project import opacity_compensation, project
from splax._rasterize import rasterize, rasterize_depth

if TYPE_CHECKING:
    from collections.abc import Callable


def render(
    means3d: jax.Array,
    scales: jax.Array,
    quats: jax.Array,
    colors: jax.Array,
    opacities: jax.Array,
    *,
    viewmat: jax.Array,
    background: jax.Array,
    img_shape: tuple[int, int],
    f: tuple[float, float],
    c: tuple[float, float] | None = None,
    glob_scale: float = 1.0,
    clip_thresh: float = 0.01,
    antialiased: bool = False,
    render_depth: bool = False,
) -> tuple[jax.Array, jax.Array | None]:
    """Render a splat with autodiff support.

    The image matches the forward result of ``splax.inference.render``. Gradient selection follows
    ``jax.grad`` and its argnums, and the backward pass launches only the kernels the requested
    gradients need, so camera pose optimization for example pays only for the camera gradient.

    Antialiased rendering enables the Mip-Splatting opacity compensation. The per-gaussian
    compensation factor is multiplied into the blend opacity and cancels the area inflation that
    thin gaussians get from the screen-space dilation. Its gradient chains back to the scales,
    quaternions, and means.

    Depth rendering additionally returns the alpha-blended expected depth map, used for sparse-point
    depth regularization. The depth channel is differentiable and routes its cotangent to the
    gaussian geometry and the camera pose. The image is identical to the plain path either way.

    The background is always a constant. For inference without autodiff use
    ``splax.inference.render``.

    Args:
        means3d: Gaussian centers, shape ``(N, 3)``.
        scales: Per-axis scales, shape ``(N, 3)``.
        quats: Rotations as wxyz quaternions, shape ``(N, 4)``.
        colors: Gaussian colors, shape ``(N, 3)``.
        opacities: Gaussian opacities, one entry per gaussian.
        viewmat: World-to-camera matrix, shape ``(4, 4)``.
        background: Constant background color, shape ``(3,)``.
        img_shape: Image size as ``(height, width)`` in pixels.
        f: Focal lengths ``(fx, fy)`` in pixels.
        c: Principal point ``(cx, cy)`` in pixels, defaulting to the image center.
        glob_scale: Global factor applied to all scales.
        clip_thresh: Near-plane clipping threshold.
        antialiased: Enable the Mip-Splatting opacity compensation.
        render_depth: Additionally render the expected depth map.

    Returns:
        Tuple of the rendered image and the depth map, where the depth map is None unless
        ``render_depth`` is True.
    """
    xys, depths, radii, conics, _num_tiles_hit, cum_tiles_hit = project(
        means3d,
        scales,
        quats,
        viewmat,
        opacities=opacities,
        img_shape=img_shape,
        f=f,
        c=c,
        glob_scale=glob_scale,
        clip_thresh=clip_thresh,
    )

    blend_opacities = opacities
    map_opacities = None
    if antialiased:
        rho = opacity_compensation(conics, radii)
        blend_opacities = opacities * rho.reshape(opacities.shape)
        map_opacities = opacities  # the tile intersection stays on the raw opacity

    if render_depth:
        return rasterize_depth(
            colors,
            blend_opacities,
            background,
            xys,
            depths,
            radii,
            conics,
            cum_tiles_hit,
            img_shape=img_shape,
            map_opacities=map_opacities,
        )
    img = rasterize(
        colors,
        blend_opacities,
        background,
        xys,
        depths,
        radii,
        conics,
        cum_tiles_hit,
        img_shape=img_shape,
        map_opacities=map_opacities,
    )
    return img, None


def render_params(
    p: dict[str, jax.Array],
    viewmat: jax.Array,
    H: int,
    W: int,
    intr: tuple[float, float, float, float],
    background: jax.Array | None = None,
    antialiased: bool = False,
    render_depth: bool = False,
) -> tuple[jax.Array, jax.Array | None]:
    """Render one view from parameterized splats."""
    fx, fy, cx, cy = intr
    means = p["means"]
    scales = jnp.exp(p["log_scales"])
    quats = p["quats"] / (jnp.linalg.norm(p["quats"], axis=-1, keepdims=True) + 1e-8)
    colors = jax.nn.sigmoid(p["colors_logit"])
    opac = jax.nn.sigmoid(p["opac_logit"])
    if background is None:
        background = jnp.ones(3)
    return render(
        means,
        scales,
        quats,
        colors,
        opac,
        viewmat=viewmat,
        background=background,
        img_shape=(H, W),
        f=(fx, fy),
        c=(cx, cy),
        glob_scale=1.0,
        clip_thresh=0.01,
        antialiased=antialiased,
        render_depth=render_depth,
    )


def _bilinear_sample(D: jax.Array, uv: jax.Array) -> jax.Array:
    """Bilinearly sample the (H, W) depth map at pixel coords ``uv`` (K, 2) = (x, y)."""
    H, W = D.shape
    x = jnp.clip(uv[:, 0] - 0.5, 0.0, W - 1.0)
    y = jnp.clip(uv[:, 1] - 0.5, 0.0, H - 1.0)
    x0 = jnp.floor(x).astype(jnp.int32)
    y0 = jnp.floor(y).astype(jnp.int32)
    x1 = jnp.minimum(x0 + 1, W - 1)
    y1 = jnp.minimum(y0 + 1, H - 1)
    wx = x - x0
    wy = y - y0
    d00 = D[y0, x0]
    d01 = D[y0, x1]
    d10 = D[y1, x0]
    d11 = D[y1, x1]
    top = d00 * (1.0 - wx) + d01 * wx
    bot = d10 * (1.0 - wx) + d11 * wx
    return top * (1.0 - wy) + bot * wy


# Per-image exposure correction

# Real captures drift in exposure / white-balance across frames. Without correction the splat
# absorbs that per-view color error as spurious view-dependent color. The affine fix learns one 3x4
# color transform per *training* image so the shared 3D color no longer has to explain per-image ISP
# variation.

# Held-out views have NO learned transform, as letting eval fit its own transform would let it cheat
# by regressing the render onto the GT. So eval always scores the RAW render vs GT


def init_exposure(ntr: int) -> jax.Array:
    """Per-training-image affine color transforms, identity-initialized."""
    eye = jnp.broadcast_to(jnp.eye(3, dtype=jnp.float32), (ntr, 3, 3))
    off = jnp.zeros((ntr, 3, 1), jnp.float32)
    return jnp.concatenate([eye, off], axis=2)


def apply_exposure(img: jax.Array, affine: jax.Array) -> jax.Array:
    """Apply one image's 3x4 affine color transform to an (H, W, 3) render."""
    M, b = affine[:, :3], affine[:, 3]
    return jnp.einsum("ij,hwj->hwi", M, img) + b


# Per-image pose refinement

# COLMAP poses of handheld video carry small residual errors (blur, rolling shutter, aliased loop
# closures). A per-training-view 6D delta (axis-angle w, translation t) is left-composed onto the
# w2c viewmat and optimized jointly with the splat. splax accumulates viewmat gradients, so this is
# a first-class parameter. Held-out views keep their raw COLMAP poses: eval stays honest.


def init_pose_deltas(ntr: int) -> jax.Array:
    """Per-training-image 6D pose deltas (axis-angle, translation), zero-initialized."""
    return jnp.zeros((ntr, 6), jnp.float32)


def apply_pose_delta(vm: jax.Array, delta: jax.Array) -> jax.Array:
    """Left-compose a small SE3 delta onto a w2c viewmat: R' = Rd R, t' = Rd t + td.

    Rodrigues with the smooth A = sin(t)/t, B = (1-cos(t))/t^2 parameterization so the
    zero-rotation init has well-defined gradients.
    """
    w, t = delta[:3], delta[3:]
    # Not scipy's Rotation.from_rotvec here: it returns NaN gradients at the zero-vector init.
    # The smooth A/B form below keeps jax.grad finite at theta = 0.
    theta2 = jnp.sum(w * w) + 1e-12
    theta = jnp.sqrt(theta2)
    A = jnp.sin(theta) / theta
    B = (1.0 - jnp.cos(theta)) / theta2
    K = jnp.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])
    Rd = jnp.eye(3) + A * K + B * (K @ K)
    out = jnp.eye(4, dtype=vm.dtype)
    out = out.at[:3, :3].set(Rd @ vm[:3, :3])
    out = out.at[:3, 3].set(Rd @ vm[:3, 3] + t)
    return out


def make_step(
    opt: optax.GradientTransformation,
    H: int,
    W: int,
    intr: tuple[float, float, float, float],
    ssim_lambda: float,
    opacity_reg: float,
    scale_reg: float,
    opacity_entropy: float = 0.0,
    flat_reg: float = 0.0,
    antialiased: bool = False,
    depth_loss: bool = False,
    depth_lambda: float = 1e-2,
    aux_tx: optax.GradientTransformation | None = None,
    exp_opt: bool = False,
    pose_opt: bool = False,
    pose_reg: float = 0.0,
    batch: int = 1,
) -> Callable:
    """Build a jitted train step.

    Loss terms:
        * ``bg`` is a per-step render-side background color
        * ``depth_loss`` adds a scale-normalized masked L1 between the rendered expected-depth
        channel and those points' camera-space depths, weighted by ``depth_lambda``
        * ``aux_tx`` optimizer for the per-image auxiliary tables (dict with any of ``exp`` /
        ``pose``, per ``exp_opt`` / ``pose_opt``). ``exp`` applies a view's 3x4 affine to the
        render before the photometric terms; ``pose`` left-composes a 6D delta onto the viewmat.
    """

    def per_view(
        p: dict[str, jax.Array],
        aux_p: dict[str, jax.Array] | None,
        gt: jax.Array,
        vm: jax.Array,
        bg: jax.Array,
        vi: jax.Array,
        pts_uv: jax.Array,
        pts_depth: jax.Array,
        pts_mask: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Photometric + depth terms for ONE view (vmapped over the batch axis)."""
        if pose_opt:
            assert aux_p is not None
            dlt = jax.lax.dynamic_index_in_dim(aux_p["pose"], vi, axis=0, keepdims=False)
            vm = apply_pose_delta(vm, dlt)
        if depth_loss:
            img, depth = render_params(
                p, vm, H, W, intr, background=bg, antialiased=antialiased, render_depth=True
            )
            assert depth is not None
            dpred = _bilinear_sample(depth, pts_uv)
            npts = jnp.sum(pts_mask) + 1e-8
            # per-view scale normalization: divide the L1 residual by the mean target
            # depth so the term is dimensionless / scale-invariant.
            scale = jnp.sum(pts_mask * pts_depth) / npts + 1e-8
            dl = jnp.sum(pts_mask * jnp.abs(dpred - pts_depth)) / npts / scale
        else:
            img, _ = render_params(p, vm, H, W, intr, background=bg, antialiased=antialiased)
            dl = jnp.array(0.0, jnp.float32)
        if exp_opt:
            assert aux_p is not None
            affine = jax.lax.dynamic_index_in_dim(aux_p["exp"], vi, axis=0, keepdims=False)
            img = apply_exposure(img, affine)
        l1 = jnp.mean(jnp.abs(img - gt))
        dssim = jnp.asarray(1.0 - dm_pix.ssim(img, gt))
        return l1, dssim, dl

    def loss_fn(
        p: dict[str, jax.Array],
        aux_p: dict[str, jax.Array] | None,
        gt: jax.Array,
        vm: jax.Array,
        bg: jax.Array,
        vi: jax.Array,
        pts_uv: jax.Array,
        pts_depth: jax.Array,
        pts_mask: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        l1s, dssims, dls = jax.vmap(per_view, in_axes=(None, None, 0, 0, 0, 0, 0, 0, 0))(
            p, aux_p, gt, vm, bg, vi, pts_uv, pts_depth, pts_mask
        )
        l1 = jnp.mean(l1s)  # batch-mean photometric (gsplat)
        loss = (1.0 - ssim_lambda) * l1 + ssim_lambda * jnp.mean(dssims)
        loss = loss + opacity_reg * jnp.mean(jax.nn.sigmoid(p["opac_logit"]))
        loss = loss + scale_reg * jnp.mean(jnp.exp(p["log_scales"]))
        if opacity_entropy > 0:
            # SuGaR-style binarization: drive opacities toward 0 or 1 so gaussians
            # act as opaque surface elements rather than semi-transparent fog.
            a = jax.nn.sigmoid(p["opac_logit"])
            ent = -(a * jnp.log(a + 1e-8) + (1.0 - a) * jnp.log(1.0 - a + 1e-8))
            loss = loss + opacity_entropy * jnp.mean(ent)
        if flat_reg > 0:
            # SuGaR-style flatness: shrink only the smallest axis so gaussians
            # become disks that can align with surfaces.
            loss = loss + flat_reg * jnp.mean(jnp.min(jnp.exp(p["log_scales"]), axis=-1))
        if depth_loss:
            loss = loss + depth_lambda * jnp.mean(dls)
        if pose_opt and pose_reg > 0:
            # L2 anchor on the pose deltas: keeps the train poses in the COLMAP gauge so the
            # fixed held-out poses stay consistent with the reconstructed world.
            assert aux_p is not None
            loss = loss + pose_reg * jnp.mean(aux_p["pose"] ** 2)
        return loss, l1

    if aux_tx is None:

        @jax.jit
        def step(
            p: dict[str, jax.Array],
            opt_state: optax.OptState,
            gt: jax.Array,
            vm: jax.Array,
            bg: jax.Array,
            pts_uv: jax.Array,
            pts_depth: jax.Array,
            pts_mask: jax.Array,
        ) -> tuple[dict[str, jax.Array], optax.OptState, jax.Array]:
            vi = jnp.zeros((batch,), jnp.int32)  # unused when aux_tx is None
            (loss, l1), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                p, None, gt, vm, bg, vi, pts_uv, pts_depth, pts_mask
            )
            updates, opt_state = opt.update(grads, opt_state, p)
            # apply_updates is typed as the broad optax ArrayTree; the params stay a dict.
            return (cast("dict[str, jax.Array]", optax.apply_updates(p, updates)), opt_state, l1)
    else:

        @jax.jit
        def step(
            p: dict[str, jax.Array],
            opt_state: optax.OptState,
            aux_p: dict[str, jax.Array],
            aux_state: optax.OptState,
            gt: jax.Array,
            vm: jax.Array,
            bg: jax.Array,
            vi: jax.Array,
            pts_uv: jax.Array,
            pts_depth: jax.Array,
            pts_mask: jax.Array,
        ) -> tuple[
            dict[str, jax.Array], optax.OptState, dict[str, jax.Array], optax.OptState, jax.Array
        ]:
            (loss, l1), (grads, aux_grads) = jax.value_and_grad(
                loss_fn, argnums=(0, 1), has_aux=True
            )(p, aux_p, gt, vm, bg, vi, pts_uv, pts_depth, pts_mask)
            updates, opt_state = opt.update(grads, opt_state, p)
            aux_updates, aux_state = aux_tx.update(aux_grads, aux_state, aux_p)
            # apply_updates is typed as the broad optax ArrayTree; the pytrees keep their types.
            return (
                cast("dict[str, jax.Array]", optax.apply_updates(p, updates)),
                opt_state,
                cast("dict[str, jax.Array]", optax.apply_updates(aux_p, aux_updates)),
                aux_state,
                l1,
            )

    return step
