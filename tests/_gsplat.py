"""gsplat reference shims for the splax parity tests.

We compare our results to gsplat (https://github.com/nerfstudio-project/gsplat) to cross-check that
they produce the correct results. The API of gsplat differs slightly, so we wrap the respective
gsplat functions to return the same as ``splax.project`` / ``splax.render``.

Convention differences:

  - Backend: gsplat uses torch.
  - Parameterization: the shims take the activated arrays via ``splax.io.apply_activations``.
  - viewmat: gsplat always uses batched camera axes.
  - Intrinsics: gsplat takes a 3x3 K matrix rather than separate f and c values.
  - glob_scale: gsplat is missing a global scale.
  - Camera z clipping: gsplat uses `near_plane` instead of `clip_thresh`.
  - Alpha: gsplat returns the accumulated alpha with a trailing singleton channel, splax as (H, W).
  - Depth: gsplat clamps the denominator at 1e-10, which lands on splax's depth 0.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import gsplat
import numpy as np
import torch

if TYPE_CHECKING:
    import jax


def cuda_tensor(a: jax.Array | np.ndarray) -> torch.Tensor:
    """Convert a jax/numpy array to a float32 CUDA torch tensor."""
    # Older torch versions do not work correctly with direct jax->torch DLPack conversion
    return torch.as_tensor(np.array(a, dtype=np.float32), dtype=torch.float32, device="cuda")


def intrinsics(f: tuple[float, float], c: tuple[float, float]) -> np.ndarray:
    fx, fy = f
    cx, cy = c
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], np.float32)


def project(
    means: jax.Array | np.ndarray,
    scales: jax.Array | np.ndarray,
    quats: jax.Array | np.ndarray,
    viewmat: jax.Array | np.ndarray,
    *,
    img_shape: tuple[int, int],
    f: tuple[float, float],
    c: tuple[float, float],
    glob_scale: float = 1.0,
    clip_thresh: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Gsplat ``fully_fused_projection`` as numpy (radii, means2d, depths, conics).

    ``radii`` is a per-axis pixel radius, so mask visibility with ``(radii > 0).any(-1)``.
    """
    H, W = img_shape
    radii, means2d, depths, conics, _ = gsplat.fully_fused_projection(
        cuda_tensor(means),
        None,
        cuda_tensor(quats),
        cuda_tensor(scales) * float(glob_scale),
        cuda_tensor(viewmat)[None],
        cuda_tensor(intrinsics(f, c))[None],
        W,
        H,
        eps2d=0.3,
        near_plane=float(clip_thresh),
        packed=False,
        calc_compensations=False,
    )
    return tuple(x[0].detach().cpu().numpy() for x in [radii, means2d, depths, conics])


def render(
    means: jax.Array | np.ndarray,
    scales: jax.Array | np.ndarray,
    quats: jax.Array | np.ndarray,
    colors: jax.Array | np.ndarray,
    opacities: jax.Array | np.ndarray,
    *,
    viewmat: jax.Array | np.ndarray,
    background: jax.Array | np.ndarray,
    img_shape: tuple[int, int],
    f: tuple[float, float],
    c: tuple[float, float],
    glob_scale: float = 1.0,
    clip_thresh: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Gsplat ``rasterization`` in splax.render's terms.

    Returns:
        The numpy image ``(H, W, 3)`` on the requested background and the accumulated alpha
        ``(H, W)``.
    """
    H, W = img_shape
    out, alpha, _ = gsplat.rasterization(
        cuda_tensor(means),
        cuda_tensor(quats),
        cuda_tensor(scales) * float(glob_scale),
        cuda_tensor(opacities),
        cuda_tensor(colors),
        cuda_tensor(viewmat)[None],
        cuda_tensor(intrinsics(f, c))[None],
        W,
        H,
        near_plane=float(clip_thresh),
        eps2d=0.3,
        render_mode="RGB",
    )
    # gsplat returns colors composited over black plus the accumulated alpha. Put it on the
    # requested background exactly as splax.render does. gsplat carries a trailing singleton on the
    # alpha, splax leaves it out.
    img = out[0] + (1.0 - alpha[0]) * cuda_tensor(background).reshape(3)
    return img.detach().cpu().numpy(), alpha[0, ..., 0].detach().cpu().numpy()


def render_depth(
    means: jax.Array | np.ndarray,
    scales: jax.Array | np.ndarray,
    quats: jax.Array | np.ndarray,
    colors: jax.Array | np.ndarray,
    opacities: jax.Array | np.ndarray,
    *,
    viewmat: jax.Array | np.ndarray,
    background: jax.Array | np.ndarray,
    img_shape: tuple[int, int],
    f: tuple[float, float],
    c: tuple[float, float],
    glob_scale: float = 1.0,
    clip_thresh: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Gsplat ``rasterization`` in splax.rasterize_depth's terms.

    Returns:
        The numpy image ``(H, W, 4)`` on the requested background, RGB in the first three channels
        and the expected depth in camera units in the fourth, and the accumulated alpha ``(H, W)``.
    """
    H, W = img_shape
    out, alpha, _ = gsplat.rasterization(
        cuda_tensor(means),
        cuda_tensor(quats),
        cuda_tensor(scales) * float(glob_scale),
        cuda_tensor(opacities),
        cuda_tensor(colors),
        cuda_tensor(viewmat)[None],
        cuda_tensor(intrinsics(f, c))[None],
        W,
        H,
        near_plane=float(clip_thresh),
        eps2d=0.3,
        render_mode="RGB+ED",
    )
    # gsplat returns colors composited over black plus the accumulated alpha. Put it on the
    # requested background exactly as splax.render does. The depth rides in the last channel and
    # carries no background, it is a coverage-normalized camera depth rather than a colour.
    rgb = out[0, ..., :3] + (1.0 - alpha[0]) * cuda_tensor(background).reshape(3)
    img = torch.cat([rgb, out[0, ..., 3:]], dim=-1)
    return img.detach().cpu().numpy(), alpha[0, ..., 0].detach().cpu().numpy()


def viewmat_grad(
    means: jax.Array | np.ndarray,
    scales: jax.Array | np.ndarray,
    quats: jax.Array | np.ndarray,
    colors: jax.Array | np.ndarray,
    opacities: jax.Array | np.ndarray,
    *,
    viewmat: jax.Array | np.ndarray,
    target: np.ndarray,
    background: jax.Array | np.ndarray,
    img_shape: tuple[int, int],
    f: tuple[float, float],
    c: tuple[float, float],
    glob_scale: float = 1.0,
    clip_thresh: float = 0.01,
) -> np.ndarray:
    """Gsplat gradient of ``mean((render(viewmat) - target) ** 2)`` wrt the viewmat.

    Returns the (4, 4) numpy gradient.
    """
    H, W = img_shape
    viewmat_t = cuda_tensor(viewmat).requires_grad_(True)
    out, alpha, _ = gsplat.rasterization(
        cuda_tensor(means),
        cuda_tensor(quats),
        cuda_tensor(scales) * float(glob_scale),
        cuda_tensor(opacities),
        cuda_tensor(colors),
        viewmat_t[None],
        cuda_tensor(intrinsics(f, c))[None],
        W,
        H,
        near_plane=float(clip_thresh),
        eps2d=0.3,
        render_mode="RGB",
    )
    img = out[0] + (1.0 - alpha[0]) * cuda_tensor(background).reshape(3)
    ((img - cuda_tensor(target)) ** 2).mean().backward()
    return viewmat_t.grad.detach().cpu().numpy()


def grad(
    means: jax.Array | np.ndarray,
    scales: jax.Array | np.ndarray,
    quats: jax.Array | np.ndarray,
    colors: jax.Array | np.ndarray,
    opacities: jax.Array | np.ndarray,
    *,
    viewmat: jax.Array | np.ndarray,
    background: jax.Array | np.ndarray,
    img_shape: tuple[int, int],
    f: tuple[float, float],
    c: tuple[float, float],
    glob_scale: float = 1.0,
    clip_thresh: float = 0.01,
    weight: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Gsplat grads wrt (means, scales, quats, colors, opacities)."""
    H, W = img_shape
    means_t = cuda_tensor(means).requires_grad_(True)
    scales_t = cuda_tensor(scales).requires_grad_(True)
    quats_t = cuda_tensor(quats).requires_grad_(True)
    colors_t = cuda_tensor(colors).requires_grad_(True)
    opac_t = cuda_tensor(opacities).requires_grad_(True)

    out, alpha, _ = gsplat.rasterization(
        means_t,
        quats_t,
        scales_t * float(glob_scale),  # gsplat only has a single scale factor
        opac_t,
        colors_t,
        cuda_tensor(viewmat)[None],
        cuda_tensor(intrinsics(f, c))[None],
        W,
        H,
        near_plane=float(clip_thresh),
        eps2d=0.3,
        render_mode="RGB",
    )
    img = out[0] + (1.0 - alpha[0]) * cuda_tensor(background).reshape(3)
    loss = img.sum() if weight is None else (cuda_tensor(weight) * img**2).mean()
    loss.backward()

    grads = (means_t, scales_t, quats_t, colors_t)
    out_g = [x.grad.detach().cpu().numpy() for x in grads]
    out_g.append(opac_t.grad.detach().cpu().numpy())
    return tuple(out_g)
