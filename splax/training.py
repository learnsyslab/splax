"""Differentiable rendering entry point.

splax.training.render composes the jax.custom_vjp projection and rasterization
primitives, so jax.grad flows through it with respect to means3d, scales, quats,
colors, opacities, and the camera pose viewmat. It shares every Warp kernel with
the inference path. The only difference is that the forward rule keeps final_Ts
and final_idx alive as backward residuals.

Batched gradients are batch-native. jax.vmap(jax.grad(render)) runs a single
batched backward launch and matches per-sample sequential grads. Broadcast
inputs shared across the batch get their gradients summed over the batch axis,
per-image inputs such as a batch of camera poses get per-image gradients.
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

    Returns (image, depths) where depths is None unless render_depth=True. The
    image matches splax.inference.render's forward result.

    Gradient selection is pure jax.grad / argnums. The projection backward
    launches only the kernels the requested gradients need, so for example
    camera-pose optimization pays only for the camera gradient.

    antialiased=True enables the Mip-Splatting opacity compensation. The
    per-gaussian det-ratio factor from opacity_compensation is multiplied into
    the blend opacity, cancelling the area inflation thin gaussians get from the
    0.3 px dilation. Its gradient chains back to scales, quats, and means through
    the existing conic vjp. The tile intersection stays on the raw opacity so the
    sort offsets remain valid.

    render_depth=True also returns the alpha-blended expected-depth map, used for
    sparse-point depth regularization. The depth channel is differentiable and
    routes its cotangent to the gaussian geometry and camera pose. The image is
    identical to the plain path either way.

    Background is always a constant. For inference without autodiff use
    splax.inference.render.
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


__all__ = ["render"]
