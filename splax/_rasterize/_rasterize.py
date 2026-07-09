"""Differentiable rasterization stage.

``rasterize`` and ``rasterize_depth`` blend the projected gaussians into an image (and, for the
depth variant, the alpha-blended expected depth map) by composing the Warp blend kernels from
``splax._rasterize._kernels`` with a ``jax.custom_vjp``. The forward keeps the blend residuals
final_Ts and final_idx alive so the backward can walk each tile back to front and reconstruct the
transmittance. The sort and bin structures are not saved, the backward recomputes them
deterministically from the saved cum_tiles_hit.
"""

from __future__ import annotations

from functools import partial
from typing import cast

import jax
import jax.numpy as jnp

from splax._rasterize._kernels import (
    _rasterize_bwd_depth_ffi,
    _rasterize_bwd_ffi,
    _rasterize_depth_ffi,
    _rasterize_ffi,
)


def _rasterize_call(
    colors: jax.Array,
    opacities: jax.Array,
    background: jax.Array,
    xys: jax.Array,
    depths: jax.Array,
    radii: jax.Array,
    conics: jax.Array,
    cum_tiles_hit: jax.Array,
    n: int,
    H: int,
    W: int,
    map_opacities: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    # map_opacities is the raw opacity for the key emission and must match the
    # projection that produced cum_tiles_hit. It defaults to opacities, which is
    # the plain path. In antialiased mode opacities is compensated for the blend
    # while map_opacities stays raw, so the key total still matches.
    if map_opacities is None:
        map_opacities = opacities
    out = _rasterize_ffi(
        colors,
        opacities.reshape(n),
        map_opacities.reshape(n),
        background.reshape(1, 3),
        xys,
        depths.reshape(n),
        radii.reshape(n).astype(jnp.int32),
        conics,
        cum_tiles_hit.reshape(n).astype(jnp.int32),
        int(n),
        int(H),
        int(W),
        output_dims=(H, W),
    )
    return cast("tuple[jax.Array, jax.Array, jax.Array]", out)


@partial(jax.custom_vjp, nondiff_argnums=(9, 10, 11))
def _rasterize_differentiable(
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
) -> jax.Array:
    _final_Ts, _final_idx, out_img = _rasterize_call(
        colors,
        opacities,
        background,
        xys,
        depths,
        radii,
        conics,
        cum_tiles_hit,
        n,
        H,
        W,
        map_opacities,
    )
    return out_img


def _rasterize_fwd_rule(
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
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
    final_Ts, final_idx, out_img = _rasterize_call(
        colors,
        opacities,
        background,
        xys,
        depths,
        radii,
        conics,
        cum_tiles_hit,
        n,
        H,
        W,
        map_opacities,
    )
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
    return out_img, residuals


def _rasterize_bwd_rule(
    n: int, H: int, W: int, residuals: tuple[jax.Array, ...], v_img: jax.Array
) -> tuple[jax.Array | None, ...]:
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
    v_colors, v_opacity, v_xy, v_conic = _rasterize_bwd_ffi(
        colors,
        opacities.reshape(n),
        map_opacities.reshape(n),
        background.reshape(1, 3),
        xys,
        depths.reshape(n),
        radii.reshape(n).astype(jnp.int32),
        conics,
        cum_tiles_hit.reshape(n).astype(jnp.int32),
        final_Ts,
        final_idx,
        v_img,
        int(n),
        int(H),
        int(W),
        output_dims=n,
    )
    v_opacity = v_opacity.reshape(opacities.shape)
    # Cotangents for (colors, opacities, map_opacities, background, xys, depths,
    # radii, conics, cum_tiles_hit). map_opacities feeds only the integer key
    # emission, so it is non-diff like background, depths, radii, and the cumsum.
    return (v_colors, v_opacity, None, None, v_xy, None, None, v_conic, None)


_rasterize_differentiable.defvjp(_rasterize_fwd_rule, _rasterize_bwd_rule)


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
) -> jax.Array:
    """Blend projected gaussians into an (H, W, 3) image.

    Differentiable with respect to colors, opacities, xys, and conics via
    jax.custom_vjp. background, depths, radii, and cum_tiles_hit are non-diff.
    Without gradients the primal is identical to the forward-only path, so pure
    inference does not regress.

    The key emission walks the same opacity-aware ellipse as the projection that
    produced cum_tiles_hit, so the inputs must come from splax.project.
    map_opacities is the raw opacity for the key emission in antialiased mode,
    where opacities is the compensated blend opacity. It defaults to opacities.
    """
    n = colors.shape[0]
    H, W = img_shape
    if map_opacities is None:
        map_opacities = opacities
    return _rasterize_differentiable(
        colors,
        opacities,
        map_opacities,
        background,
        xys,
        depths,
        radii,
        conics,
        cum_tiles_hit,
        int(n),
        int(H),
        int(W),
    )


def _rasterize_depth_call(
    colors: jax.Array,
    opacities: jax.Array,
    background: jax.Array,
    xys: jax.Array,
    depths: jax.Array,
    radii: jax.Array,
    conics: jax.Array,
    cum_tiles_hit: jax.Array,
    n: int,
    H: int,
    W: int,
    map_opacities: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    if map_opacities is None:
        map_opacities = opacities
    out = _rasterize_depth_ffi(
        colors,
        opacities.reshape(n),
        map_opacities.reshape(n),
        background.reshape(1, 3),
        xys,
        depths.reshape(n),
        radii.reshape(n).astype(jnp.int32),
        conics,
        cum_tiles_hit.reshape(n).astype(jnp.int32),
        int(n),
        int(H),
        int(W),
        output_dims=(H, W),
    )
    return cast("tuple[jax.Array, jax.Array, jax.Array, jax.Array]", out)


@partial(jax.custom_vjp, nondiff_argnums=(9, 10, 11))
def _rasterize_depth_differentiable(
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
) -> tuple[jax.Array, jax.Array]:
    _final_Ts, _final_idx, out_img, out_depth = _rasterize_depth_call(
        colors,
        opacities,
        background,
        xys,
        depths,
        radii,
        conics,
        cum_tiles_hit,
        n,
        H,
        W,
        map_opacities,
    )
    return out_img, out_depth


def _rasterize_depth_fwd_rule(
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
) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, ...]]:
    final_Ts, final_idx, out_img, out_depth = _rasterize_depth_call(
        colors,
        opacities,
        background,
        xys,
        depths,
        radii,
        conics,
        cum_tiles_hit,
        n,
        H,
        W,
        map_opacities,
    )
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
    return (out_img, out_depth), residuals


def _rasterize_depth_bwd_rule(
    n: int,
    H: int,
    W: int,
    residuals: tuple[jax.Array, ...],
    cotangents: tuple[jax.Array, jax.Array],
) -> tuple[jax.Array | None, ...]:
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
    v_img, v_depth_img = cotangents
    v_colors, v_opacity, v_xy, v_conic, v_depths = _rasterize_bwd_depth_ffi(
        colors,
        opacities.reshape(n),
        map_opacities.reshape(n),
        background.reshape(1, 3),
        xys,
        depths.reshape(n),
        radii.reshape(n).astype(jnp.int32),
        conics,
        cum_tiles_hit.reshape(n).astype(jnp.int32),
        final_Ts,
        final_idx,
        v_img,
        v_depth_img,
        int(n),
        int(H),
        int(W),
        output_dims=n,
    )
    v_opacity = v_opacity.reshape(opacities.shape)
    v_depths = v_depths.reshape(depths.shape)
    # Unlike the plain rasterize, depths carries a nonzero cotangent that flows
    # through project's backward to the geometry and camera pose.
    return (v_colors, v_opacity, None, None, v_xy, v_depths, None, v_conic, None)


_rasterize_depth_differentiable.defvjp(_rasterize_depth_fwd_rule, _rasterize_depth_bwd_rule)


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
    """Blend gaussians into (image, expected_depth).

    Identical to rasterize but additionally renders the alpha-blended expected
    depth map with the same visibility weights as the color blend, used for
    sparse-point depth regularization. The depths input carries a nonzero
    cotangent that flows through splax.project's backward to the gaussian
    geometry and camera pose. This is a separate kernel, so the plain render
    never pays for the extra channel.
    """
    n = colors.shape[0]
    H, W = img_shape
    if map_opacities is None:
        map_opacities = opacities
    return _rasterize_depth_differentiable(
        colors,
        opacities,
        map_opacities,
        background,
        xys,
        depths,
        radii,
        conics,
        cum_tiles_hit,
        int(n),
        int(H),
        int(W),
    )
