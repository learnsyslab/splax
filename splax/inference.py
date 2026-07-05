"""Grad-free rendering entry point.

splax.inference.render shares every Warp kernel with splax.training.render and
differs only in the JAX wrapping. It calls the projection and rasterization FFI
primals directly, so there is no custom_vjp on the trace and jax.grad through it
raises. The blend residuals final_Ts and final_idx are discarded, and because
nothing downstream reads them XLA dead-code-eliminates them from the compiled
program. The training forward must keep them alive as backward residuals.
"""

from __future__ import annotations

import jax

from splax._project import _project_call, opacity_compensation
from splax._rasterize import _rasterize_call


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
    c: tuple[float, float],
    glob_scale: float = 1.0,
    clip_thresh: float = 0.01,
    antialiased: bool = False,
) -> jax.Array:
    """Render a splat in inference mode without autodiff support.

    Same forward result as splax.training.render, but guaranteed grad-free. With
    antialiased=True the Mip-Splatting opacity compensation is applied to the
    blend opacity, matching a model trained with the antialiased training render.
    """
    n = means3d.shape[0]
    H, W = img_shape

    opac = opacities.reshape(n)
    xys, depths, radii, conics, _num_tiles_hit, cum_tiles_hit = _project_call(
        means3d,
        scales,
        quats,
        viewmat,
        opac,
        n,
        img_shape,
        f,
        c,
        float(glob_scale),
        float(clip_thresh),
    )

    # In antialiased mode the blend opacity is compensated while the key emission
    # stays on the raw opacity, so the sort offsets remain valid.
    blend_opac = opacities
    map_opac = None
    if antialiased:
        rho = opacity_compensation(conics, radii)
        blend_opac = opacities * rho.reshape(opacities.shape)
        map_opac = opacities
    _final_Ts, _final_idx, out_img = _rasterize_call(
        colors,
        blend_opac,
        background,
        xys,
        depths,
        radii,
        conics,
        cum_tiles_hit,
        n,
        H,
        W,
        map_opac,
    )
    return out_img


__all__ = ["render"]
