"""Differentiable rendering entry point.

``splax.training.render`` composes the projection and rasterization primitives with their custom
autodiff rules, so ``jax.grad`` flows through it with respect to the gaussian parameters and the
camera pose. It shares every Warp kernel with the inference path and differs only in the forward
rule keeping the blend residuals alive for the backward pass.

Batched gradients are batch-native. ``jax.vmap(jax.grad(render))`` runs a single batched backward
launch and matches per-sample sequential gradients. Inputs shared across the batch get their
gradients summed over the batch axis, while per-image inputs such as a batch of camera poses get
per-image gradients.

``render_log`` renders from the unconstrained log/logit parameterization on top of it.

Note:
    For efficient inference without gradients, use ``splax.inference.render`` instead.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from splax._project import opacity_compensation, project
from splax._rasterize import rasterize, rasterize_depth


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

    The image matches the forward result of ``splax.inference.render``, but is compatible with
    ``jax.grad``. It is also aware of the arguments with respect to which gradients are requested,
    and will only compute the necessary intermediate values. Depth rendering additionally returns
    the differentiable alpha-blended expected depth map for sparse-point depth regularization.

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
    camera: dict = {"img_shape": img_shape, "f": f, "c": c}
    camera |= {"glob_scale": glob_scale, "clip_thresh": clip_thresh}
    xys, depths, radii, conics, _, cum_tiles_hit = project(
        means3d, scales, quats, viewmat, opacities=opacities, **camera
    )

    blend_opacities = opacities
    map_opacities = None
    if antialiased:
        rho = opacity_compensation(conics, radii)
        blend_opacities = opacities * rho.reshape(opacities.shape)
        map_opacities = opacities  # the tile intersection stays on the raw opacity

    inputs = (colors, blend_opacities, background, xys, depths, radii, conics, cum_tiles_hit)
    if render_depth:
        return rasterize_depth(*inputs, img_shape=img_shape, map_opacities=map_opacities)
    return rasterize(*inputs, img_shape=img_shape, map_opacities=map_opacities), None


def render_log(
    means3d: jax.Array,
    log_scales: jax.Array,
    quats: jax.Array,
    logit_colors: jax.Array,
    logit_opacities: jax.Array,
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
    """Render from the unconstrained log/logit parameterization used for training.

    Maps the log/logit arrays to the constrained splat arrays (``exp`` on the log scales, normalized
    quaternions, ``sigmoid`` on the color and opacity logits) and calls ``render`` with the same
    camera arguments. ``splax.mcmc`` operates in the same parameterization.

    Args:
        means3d: Gaussian centers, shape ``(N, 3)``.
        log_scales: Log of the per-axis scales, shape ``(N, 3)``.
        quats: Rotations as wxyz quaternions, not necessarily normalized, shape ``(N, 4)``.
        logit_colors: Color logits, shape ``(N, 3)``.
        logit_opacities: Opacity logits, one entry per gaussian.
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
    scales = jnp.exp(log_scales)
    quats = quats / (jnp.linalg.norm(quats, axis=-1, keepdims=True) + 1e-8)
    colors = jax.nn.sigmoid(logit_colors)
    opacities = jax.nn.sigmoid(logit_opacities)
    camera: dict = {"viewmat": viewmat, "background": background, "img_shape": img_shape}
    camera |= {"f": f, "c": c, "glob_scale": glob_scale, "clip_thresh": clip_thresh}
    camera |= {"antialiased": antialiased, "render_depth": render_depth}
    return render(means3d, scales, quats, colors, opacities, **camera)
