"""Spherical harmonics color evaluation.

``spherical_harmonics`` turns per-gaussian coefficients and a view direction into the RGB color the
rasterizer blends, giving gaussians a color that changes with the viewing angle.

We could implement this in pure JAX, but the Warp kernels in ``splax._harmonics._kernels`` are
slightly faster and more memory efficient. For batches, they allow us to sum gradients of operands
shared across the batch directly in the kernel.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from splax._harmonics._kernels import (
    C0,
    COEFFICIENTS,
    spherical_harmonics_bwd_ffi,
    spherical_harmonics_ffi,
)


def spherical_harmonics(coeffs: jax.Array, directions: jax.Array) -> jax.Array:
    """Evaluate view-dependent colors from spherical harmonics coefficients.

    Args:
        coeffs: Coefficients ordered by band, shape ``(N, K, 3)`` with ``K`` one of 1, 4, 9, or 16,
            the coefficient counts of degrees 0 to 3. The first is the base color.
        directions: View directions from the camera to the gaussian centers, shape ``(N, 3)``.

    Returns:
        RGB colors clamped to non-negative values, shape ``(N, 3)``.
    """
    assert coeffs.ndim == 3, f"coeffs must be (N, K, 3), got shape {coeffs.shape}"
    n_coeffs = coeffs.shape[-2]
    assert n_coeffs in COEFFICIENTS, f"coefficients must be one of {COEFFICIENTS}, got {n_coeffs}"
    if n_coeffs == 1:  # static color, no view dependence
        return jnp.maximum(coeffs[..., 0, :] * C0 + 0.5, 0.0)
    n = coeffs.shape[0]
    return spherical_harmonics_ffi(
        coeffs, directions, n, output_dims=n, vjp=_spherical_harmonics_vjp
    )[0]


def _spherical_harmonics_vjp(
    arrays: tuple[jax.Array, ...], cotangents: tuple[jax.Array, ...], mask: tuple[bool, ...], n: int
) -> tuple[jax.Array, ...]:
    """A custom backward kernel that sums gradients of operands shared across the batch."""
    dims = {"v_coeffs": (n, arrays[0].shape[-2]), "v_directions": n}
    summed = tuple(not m for m in mask[:2])
    return spherical_harmonics_bwd_ffi(
        *arrays, *cotangents, n, output_dims=dims, mask=mask, summed=summed
    )
