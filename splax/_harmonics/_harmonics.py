"""Spherical harmonics color evaluation.

``spherical_harmonics`` turns per-gaussian coefficients and a view direction into the RGB color the
rasterizer blends, giving gaussians a color that changes with the viewing angle.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

# Real spherical harmonics normalization constants up to degree 3, one tuple per band.
C0 = 0.5 / np.sqrt(np.pi)
C1 = 0.5 * np.sqrt(3.0 / np.pi)
C2 = (
    0.5 * np.sqrt(15.0 / np.pi),
    -0.5 * np.sqrt(15.0 / np.pi),
    0.25 * np.sqrt(5.0 / np.pi),
    -0.5 * np.sqrt(15.0 / np.pi),
    0.25 * np.sqrt(15.0 / np.pi),
)
C3 = (
    -0.25 * np.sqrt(35.0 / (2.0 * np.pi)),
    0.5 * np.sqrt(105.0 / np.pi),
    -0.25 * np.sqrt(21.0 / (2.0 * np.pi)),
    0.25 * np.sqrt(7.0 / np.pi),
    -0.25 * np.sqrt(21.0 / (2.0 * np.pi)),
    0.25 * np.sqrt(105.0 / np.pi),
    -0.25 * np.sqrt(35.0 / (2.0 * np.pi)),
)

# Coefficient counts of degrees 0 to 3. The index into the tuple is the degree.
COEFFICIENTS = tuple((d + 1) ** 2 for d in range(4))


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
    degree = COEFFICIENTS.index(n_coeffs)
    if degree == 0:  # static color, no view dependence
        return jnp.maximum(coeffs[..., 0, :] * C0 + 0.5, 0.0)
    norm_directions = directions / jnp.linalg.norm(directions, axis=-1, keepdims=True)
    x, y, z = jnp.split(norm_directions, 3, axis=-1)
    basis = [jnp.broadcast_to(jnp.float32(C0), x.shape), -C1 * y, C1 * z, -C1 * x]
    if degree > 1:
        xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
        basis += [C2[0] * xy, C2[1] * yz, C2[2] * (2.0 * zz - xx - yy)]
        basis += [C2[3] * xz, C2[4] * (xx - yy)]
    if degree > 2:
        basis += [C3[0] * y * (3.0 * xx - yy), C3[1] * xy * z, C3[2] * y * (4.0 * zz - xx - yy)]
        basis += [C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy), C3[4] * x * (4.0 * zz - xx - yy)]
        basis += [C3[5] * z * (xx - yy), C3[6] * x * (xx - 3.0 * yy)]
    colors = (jnp.concatenate(basis, axis=-1)[..., None] * coeffs).sum(axis=1)
    return jnp.maximum(colors + 0.5, 0.0)
