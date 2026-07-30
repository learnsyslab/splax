"""Differentiable rasterization stage.

``rasterize`` and ``rasterize_depth`` blend the projected gaussians into an image and the
accumulated alpha map. The depth variant packs an expected depth map in camera-space z into the
fourth image channel.

The stage takes activated arrays, i.e. RGB colors and opacities in ``[0, 1]``.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from splax._rasterize._kernels import (
    rasterize_bwd_depth_ffi,
    rasterize_bwd_ffi,
    rasterize_depth_ffi,
    rasterize_ffi,
)

# region public API


def rasterize(
    colors: jax.Array,
    opacities: jax.Array,
    background: jax.Array,
    xys: jax.Array,
    depths: jax.Array,
    radii: jax.Array,
    conics: jax.Array,
    cum_tiles_hit: jax.Array,
    *,
    img_shape: tuple[int, int],
    map_opacities: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Blend projected gaussians into an image and an accumulated alpha map.

    Differentiable with respect to colors, opacities, xys, and conics. background, depths, radii,
    and cum_tiles_hit are non-differentiable.

    Returns:
        The ``(H, W, 3)`` image and the ``(H, W)`` accumulated alpha, the coverage the gaussians
        coming out of the blend contribute, which is 0 on pixels no gaussian covers.
    """
    if any(isinstance(v, jax.Array) for v in img_shape):
        raise TypeError(
            "img_shape sizes the kernel launch and must be static. Under jax.jit either close "
            'over it or pass static_argnames="img_shape".'
        )
    n = colors.shape[0]
    H, W = img_shape
    if map_opacities is None:
        map_opacities = opacities
    out_img, final_Ts, _ = _rasterize(
        colors,
        opacities,
        map_opacities,
        background,
        xys,
        depths,
        radii,
        conics,
        cum_tiles_hit,
        n,
        H,
        W,
    )
    return out_img, 1.0 - final_Ts


def rasterize_depth(
    colors: jax.Array,
    opacities: jax.Array,
    background: jax.Array,
    xys: jax.Array,
    depths: jax.Array,
    radii: jax.Array,
    conics: jax.Array,
    cum_tiles_hit: jax.Array,
    *,
    img_shape: tuple[int, int],
    map_opacities: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Blend gaussians into an RGB and depth image (H, W, 4) and an alpha map (H, W).

    The depth channel is the expected depth, the visibility-weighted mean of the gaussians'
    camera-space z normalized by the accumulated alpha. It is measured along the optical axis rather
    than as a Euclidean range, and pixels that no gaussian covers read 0.

    Returns:
        The ``(H, W, 4)`` image, RGB in the first three channels and the expected depth in the
        fourth, and the ``(H, W)`` accumulated alpha, which is 0 on pixels no gaussian covers.
    """
    if any(isinstance(v, jax.Array) for v in img_shape):
        raise TypeError(
            "img_shape sizes the kernel launch and must be static. Under jax.jit either close "
            'over it or pass static_argnames="img_shape".'
        )
    n = colors.shape[0]
    H, W = img_shape
    if map_opacities is None:
        map_opacities = opacities
    out_img, out_depth, final_Ts, _ = _rasterize_depth(
        colors,
        opacities,
        map_opacities,
        background,
        xys,
        depths,
        radii,
        conics,
        cum_tiles_hit,
        n,
        H,
        W,
    )
    # The accumulated alpha vanishes exactly on uncovered pixels, so the quotient is masked to 0
    # there. Dividing by the masked denominator keeps the gradient of those pixels finite as well.
    covered = final_Ts < 1.0
    accum_alpha = jnp.where(covered, 1.0 - final_Ts, 1.0)
    depth = jnp.where(covered, out_depth / accum_alpha, 0.0)
    return jnp.concatenate([out_img, depth[..., None]], axis=-1), 1.0 - final_Ts


# region custom vjp


@partial(jax.custom_vjp, nondiff_argnums=(9, 10, 11))
def _rasterize(
    colors: jax.Array,
    opacities: jax.Array,
    map_opacities: jax.Array,
    background: jax.Array,
    xys: jax.Array,
    depths: jax.Array,
    radii: jax.Array,
    conics: jax.Array,
    cum_tiles_hit: jax.Array,
    n: int,
    H: int,
    W: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Custom vjp for the blend, returning (out_img, final_Ts, final_idx).

    final_Ts is the per-pixel final transmittance the public rasterize turns into the alpha map, and
    final_idx is the last contributing gaussian, a backward residual the public rasterize discards.
    JAX requires a rigid array signature for custom_vjps, so the None default of map_opacities is
    resolved in the public rasterize before this is called.
    """
    final_Ts, final_idx, out_img = rasterize_ffi(
        colors,
        opacities,
        map_opacities,
        background.reshape(1, 3),
        xys,
        depths,
        radii,
        conics,
        cum_tiles_hit,
        n,
        H,
        W,
        output_dims=(H, W),
    )
    return out_img, final_Ts, final_idx


def _rasterize_fwd(
    colors: jax.Array,
    opacities: jax.Array,
    map_opacities: jax.Array,
    background: jax.Array,
    xys: jax.Array,
    depths: jax.Array,
    radii: jax.Array,
    conics: jax.Array,
    cum_tiles_hit: jax.Array,
    n: int,
    H: int,
    W: int,
) -> tuple[tuple[jax.Array, jax.Array, jax.Array], tuple[jax.Array, ...]]:
    """Forward pass of _rasterize, reusing the primitive and keeping its residuals."""
    out = _rasterize(
        colors,
        opacities,
        map_opacities,
        background,
        xys,
        depths,
        radii,
        conics,
        cum_tiles_hit,
        n,
        H,
        W,
    )
    _, final_Ts, final_idx = out
    residuals = (
        colors,
        opacities,
        map_opacities,
        background,
        xys,
        depths,
        radii,
        conics,
        cum_tiles_hit,
        final_Ts,
        final_idx,
    )
    return out, residuals


def _rasterize_bwd(
    n: int, H: int, W: int, residuals: tuple[jax.Array, ...], cotangents: tuple[jax.Array, ...]
) -> tuple[jax.Array | None, ...]:
    """Backward pass of _rasterize."""
    (
        colors,
        opacities,
        map_opacities,
        background,
        xys,
        depths,
        radii,
        conics,
        cum_tiles_hit,
        final_Ts,
        final_idx,
    ) = residuals
    # The accumulated alpha is 1 - final_Ts, so final_Ts carries a cotangent alongside the image
    v_img, v_final_Ts, _ = cotangents
    v_colors, v_opacity, v_xy, v_conic = rasterize_bwd_ffi(
        colors,
        opacities,
        map_opacities,
        background.reshape(1, 3),
        xys,
        depths,
        radii,
        conics,
        cum_tiles_hit,
        final_Ts,
        final_idx,
        v_img,
        v_final_Ts,
        n,
        H,
        W,
        output_dims=n,
    )
    # Cotangents for (colors, opacities, map_opacities, background, xys, depths, radii, conics,
    # cum_tiles_hit). map_opacities feeds only the integer key emission, so it is non-diff like
    # background, depths, radii, and the cumsum.
    return v_colors, v_opacity, None, None, v_xy, None, None, v_conic, None


_rasterize.defvjp(_rasterize_fwd, _rasterize_bwd)


@partial(jax.custom_vjp, nondiff_argnums=(9, 10, 11))
def _rasterize_depth(
    colors: jax.Array,
    opacities: jax.Array,
    map_opacities: jax.Array,
    background: jax.Array,
    xys: jax.Array,
    depths: jax.Array,
    radii: jax.Array,
    conics: jax.Array,
    cum_tiles_hit: jax.Array,
    n: int,
    H: int,
    W: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Custom vjp for the depth blend, returning (out_img, out_depth, final_Ts, final_idx)."""
    final_Ts, final_idx, out_img, out_depth = rasterize_depth_ffi(
        colors,
        opacities,
        map_opacities,
        background.reshape(1, 3),
        xys,
        depths,
        radii,
        conics,
        cum_tiles_hit,
        n,
        H,
        W,
        output_dims=(H, W),
    )
    return out_img, out_depth, final_Ts, final_idx


def _rasterize_depth_fwd(
    colors: jax.Array,
    opacities: jax.Array,
    map_opacities: jax.Array,
    background: jax.Array,
    xys: jax.Array,
    depths: jax.Array,
    radii: jax.Array,
    conics: jax.Array,
    cum_tiles_hit: jax.Array,
    n: int,
    H: int,
    W: int,
) -> tuple[tuple[jax.Array, jax.Array, jax.Array, jax.Array], tuple[jax.Array, ...]]:
    """Forward pass of _rasterize_depth, reusing the primitive and keeping its residuals."""
    out = _rasterize_depth(
        colors,
        opacities,
        map_opacities,
        background,
        xys,
        depths,
        radii,
        conics,
        cum_tiles_hit,
        n,
        H,
        W,
    )
    _, _, final_Ts, final_idx = out
    residuals = (
        colors,
        opacities,
        map_opacities,
        background,
        xys,
        depths,
        radii,
        conics,
        cum_tiles_hit,
        final_Ts,
        final_idx,
    )
    return out, residuals


def _rasterize_depth_bwd(
    n: int, H: int, W: int, residuals: tuple[jax.Array, ...], cotangents: tuple[jax.Array, ...]
) -> tuple[jax.Array | None, ...]:
    """Backward pass of _rasterize_depth."""
    (
        colors,
        opacities,
        map_opacities,
        background,
        xys,
        depths,
        radii,
        conics,
        cum_tiles_hit,
        final_Ts,
        final_idx,
    ) = residuals
    # final_Ts feeds the alpha map and the expected depth normalization, so it carries a cotangent
    v_img, v_depth_img, v_final_Ts, _ = cotangents
    v_colors, v_opacity, v_xy, v_conic, v_depths = rasterize_bwd_depth_ffi(
        colors,
        opacities,
        map_opacities,
        background.reshape(1, 3),
        xys,
        depths,
        radii,
        conics,
        cum_tiles_hit,
        final_Ts,
        final_idx,
        v_img,
        v_depth_img,
        v_final_Ts,
        n,
        H,
        W,
        output_dims=n,
    )
    # Unlike the plain rasterize, depths carries a nonzero cotangent
    return v_colors, v_opacity, None, None, v_xy, v_depths, None, v_conic, None


_rasterize_depth.defvjp(_rasterize_depth_fwd, _rasterize_depth_bwd)
