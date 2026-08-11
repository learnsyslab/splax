"""Rendering entry point.

``splax.render`` composes the projection and rasterization primitives with their custom autodiff
rules, so ``jax.grad`` flows through it with respect to the gaussian parameters, the camera pose,
and the per-object rigid transforms.

Batched gradients are batch-native. ``jax.vmap(jax.grad(render))`` runs a single batched backward
launch and matches per-sample sequential gradients. Inputs shared across the batch get their
gradients summed over the batch axis, while per-image inputs such as a batch of camera poses get
per-image gradients.

``render`` takes the unconstrained parameters, i.e. log scales, degree-0 SH colors, and logit
opacities, and applies their activations before the projection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from splax._project import project, transform_ids
from splax._rasterize import rasterize, rasterize_depth
from splax.io import apply_activations

if TYPE_CHECKING:
    from collections.abc import Sequence

    import jax


def render(
    means3d: jax.Array,
    log_scales: jax.Array,
    quats: jax.Array,
    sh_colors: jax.Array,
    logit_opacities: jax.Array,
    *,
    viewmat: jax.Array,
    background: jax.Array,
    img_shape: tuple[int, int],
    f: tuple[float, float],
    c: tuple[float, float] | None = None,
    dist: tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0),
    glob_scale: float = 1.0,
    clip_thresh: float = 0.01,
    antialiased: bool = False,
    render_depth: bool = False,
    gaussian_transforms: jax.Array | None = None,
    gaussian_slices: Sequence[tuple[int, int]] | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Render a splat.

    The parameters are the ones a ``.ply`` stores and a trainer optimizes. Their activations, the
    ``exp`` on the scales, the SH map on the colors, and the ``sigmoid`` on the opacities, are
    applied here, and the quaternions are normalized inside the projection kernel.

    The render is differentiable with respect to the parameters, the camera pose, and the
    transforms. Depth rendering additionally packs the differentiable expected depth map into the
    image for sparse-point depth regularization.

    Slices of the gaussians can follow rigid transforms for composed dynamic scenes. The gaussians
    in slice k move by ``gaussian_transforms[k]``, while all others stay static.  Transforms are
    handled in the projection kernel without copying the splat. Gradients flow to the transforms as
    well, so object poses can be optimized directly.

    Args:
        means3d: Gaussian centers, shape ``(N, 3)``.
        log_scales: Log of the per-axis scales, shape ``(N, 3)``.
        quats: Rotations as wxyz quaternions, not necessarily normalized, shape ``(N, 4)``.
        sh_colors: Degree-0 SH color coefficients, shape ``(N, 3)``.
        logit_opacities: Opacity logits, shape ``(N,)``.
        viewmat: World-to-camera matrix, shape ``(4, 4)``.
        background: Constant background color, shape ``(3,)``.
        img_shape: Image size as ``(height, width)`` in pixels.
        f: Focal lengths ``(fx, fy)`` in pixels.
        c: Principal point ``(cx, cy)`` in pixels, defaulting to the image center.
        dist: Brown-Conrady coefficients ``(k1, k2, p1, p2, k3)``, zero for an ideal pinhole. The
            coefficients are static and carry no gradient.
        glob_scale: Global factor applied to all scales.
        clip_thresh: Near-plane clipping threshold.
        antialiased: Enable the Mip-Splatting opacity compensation.
        render_depth: Additionally render the expected depth map in camera-space z.
        gaussian_transforms: Rigid world-space transforms, shape ``(K, 4, 4)``.
        gaussian_slices: K matching, non-overlapping ``(start, stop)`` gaussian index ranges.

    Returns:
        Tuple of the rendered image and the ``(H, W)`` accumulated alpha, the coverage the gaussians
        contribute, which is 0 on pixels no gaussian covers. The image is ``(H, W, 3)`` RGB, or
        ``(H, W, 4)`` with the expected depth in the fourth channel if ``render_depth`` is True.
        Depth is the expected camera-space z along the optical axis, not a Euclidean range, and
        reads 0 on pixels no gaussian covers.
    """
    if (gaussian_transforms is None) != (gaussian_slices is None):
        raise ValueError("gaussian_transforms and gaussian_slices must be passed together")
    tf_ids = None
    if gaussian_transforms is not None and gaussian_slices is not None:
        if gaussian_transforms.shape[-3:] != (len(gaussian_slices), 4, 4):
            raise ValueError(
                f"gaussian_transforms shape {gaussian_transforms.shape} does not "
                f"match {len(gaussian_slices)} slices, expected (K, 4, 4)"
            )
        tf_ids = transform_ids(means3d.shape[0], gaussian_slices)

    scales, colors, opacities = apply_activations(log_scales, sh_colors, logit_opacities)
    camera: dict = {"img_shape": img_shape, "f": f, "c": c, "dist": dist}
    camera |= {"glob_scale": glob_scale, "clip_thresh": clip_thresh}
    xys, depths, radii, conics, _, cum_tiles_hit = project(
        means3d,
        scales,
        quats,
        viewmat,
        opacities=opacities,
        **camera,
        gaussian_transforms=gaussian_transforms,
        transform_ids=tf_ids,
    )

    inputs = (colors, opacities, background, xys, depths, radii, conics, cum_tiles_hit)
    if render_depth:
        return rasterize_depth(*inputs, img_shape=img_shape, antialiased=antialiased)
    return rasterize(*inputs, img_shape=img_shape, antialiased=antialiased)
