"""Differentiable rendering entry point.

``splax.training.render`` composes the projection and rasterization primitives with their custom
autodiff rules, so ``jax.grad`` flows through it with respect to the gaussian parameters and the
camera pose. It shares every Warp kernel with the inference path and differs only in the forward
rule keeping the blend residuals alive for the backward pass.

Batched gradients are batch-native. ``jax.vmap(jax.grad(render))`` runs a single batched backward
launch and matches per-sample sequential gradients. Inputs shared across the batch get their
gradients summed over the batch axis, while per-image inputs such as a batch of camera poses get
per-image gradients.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from splax._project import opacity_compensation, project
from splax._rasterize import rasterize, rasterize_depth

if TYPE_CHECKING:
    import jax


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
