"""splax.training: the differentiable rendering entry point.

``splax.training.render`` is the gradient-carrying twin of
``splax.inference.render``. It composes the ``jax.custom_vjp`` projection
(``splax.project``) and rasterization (``splax.rasterize``) primitives, so
``jax.grad`` / ``jax.value_and_grad`` flow through it w.r.t. means3d, scales,
quats, colors and opacities (viewmat and background are constants by default).
This is the path ``scripts/train_lego.py`` fits with.

It shares every Warp kernel with the inference path; the only difference is that
the forward custom_vjp rule keeps ``final_Ts`` / ``final_idx`` alive as backward
residuals. The tile intersection is the tight O6 path, whose gradients are
finite-difference validated (``test_warp_grad.py``); the legacy ``tight=False``
emission -- which the gsplat grad-parity test uses -- is still reachable via
``splax.rasterize(..., tight=False)`` directly.

Batched gradients are batch-native: the backward FFIs are
``vmap_method="expand_dims"``, so ``jax.vmap(jax.grad(render))`` runs a single
batched backward launch (no per-sample Python loop) for every ``diff_wrt``
selection, matching per-sample sequential grad. Broadcast (shared-across-batch)
inputs get their gradients summed over the batch axis; per-image inputs (e.g. a
batch of camera poses under ``diff_wrt=("viewmat",)``) get per-image gradients.
Batched *inference* is unaffected -- use ``splax.inference.render``.
"""

from __future__ import annotations

import jax

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
    c: tuple[float, float],
    glob_scale: float,
    clip_thresh: float,
    block_size: int = 16,
    diff_wrt: tuple[str, ...] = ("gaussians",),
    antialiased: bool = False,
    render_depth: bool = False,
) -> tuple[jax.Array, jax.Array | None]:
    """Differentiable renderer: Warp projection + rasterization via custom_vjp.

    Returns ``(image, depths)`` where ``depths`` is ``None`` unless
    ``render_depth=True``. The image matches ``splax.inference.render``'s
    forward result, but is differentiable w.r.t. means3d, scales, quats, colors,
    opacities.

    ``diff_wrt`` selects the projection backward variant and hence which
    gradients flow through the *camera pose* (viewmat):

    - ``("gaussians",)`` (default): grads w.r.t. means3d/scales/quats/colors/
      opacities; viewmat is a constant.
    - ``("viewmat",)``: only the camera-pose gradient. The gaussian projection grad
      chains and their atomics are skipped, so post-training camera-pose
      optimization pays only for the camera gradient. The gaussian inputs receive
      *no* cotangent (grad w.r.t. them raises), matching the intent that they are
      frozen. Colors/opacities are still differentiated through rasterize, so a
      pure pose loss ignores them (their grads are unused).
    - ``("gaussians", "viewmat")``: both the gaussian and the camera-pose grads;
      the gaussian grads are bit-identical to ``("gaussians",)``.

    ``antialiased=True`` enables the Mip-Splatting opacity compensation (gsplat
    ``rasterize_mode="antialiased"``): the per-gaussian factor ρ =
    √(det Σ₂D / det(Σ₂D+εI)) — the det ratio of the undilated over the ε=0.3-dilated
    2D covariance projection already applies — is multiplied into the opacity for the
    blend, cancelling the artificial area inflation thin gaussians get from the
    dilation. ρ is computed from the projection's ``conics`` output
    (``opacity_compensation``), so its gradient chains back to scales/quats/means
    through the existing conic→covariance vjp with no Warp-kernel change; the tile
    intersection still uses the raw opacity (``map_opacities``) so the sort offsets
    stay valid. Default ``False`` is byte-identical to the pre-antialiased path.

    ``render_depth=True`` (survey T2) fills the depth slot with an alpha-blended
    expected-depth map D = Σ wᵢ dᵢ, for COLMAP sparse-point depth regularization
    (gsplat ``depth_loss``). The depth channel is differentiable and routes a
    nonzero cotangent through ``depths`` to the gaussian geometry / camera pose.
    Default ``False`` returns ``(image, None)`` with an image byte-identical to
    the plain path (the depth channel is a separate Warp kernel, no extra cost).

    Background is always a constant. For pure inference with no autodiff overhead
    use ``splax.inference.render``.
    """
    xys, depths, radii, conics, _num_tiles_hit, cum_tiles_hit = project(
        means3d,
        scales,
        quats,
        viewmat,
        img_shape=img_shape,
        f=f,
        c=c,
        glob_scale=glob_scale,
        clip_thresh=clip_thresh,
        block_width=block_size,
        opacities=opacities,
        diff_wrt=diff_wrt,
    )

    blend_opacities = opacities
    map_opacities = None
    if antialiased:
        rho = opacity_compensation(conics, radii)
        blend_opacities = opacities * rho.reshape(opacities.shape)
        map_opacities = opacities  # tile count stays on the raw opacity

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
            block_width=block_size,
            tight=True,
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
        block_width=block_size,
        tight=True,
        map_opacities=map_opacities,
    )
    return img, None


__all__ = ["render"]
