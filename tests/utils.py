"""Scene and camera builders shared across the splax test suite.

``scene`` draws a random splat, in either of the two regimes the suite needs. ``camera`` builds the
matching intrinsics for a square-focal pinhole.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial.transform import RigidTransform as TF
from scipy.spatial.transform import Rotation as R

import splax
import splax._rasterize._sort._sort as _sort

VIEWMAT = jnp.array([[1, 0, 0, 0.2], [0, 1, 0, -0.1], [0, 0, 1, 5], [0, 0, 0, 1]], jnp.float32)
# Three views panning along x, the B=3 batch the vmap tests compare against their unbatched loop.
VIEWS = jnp.stack([VIEWMAT.at[0, 3].set(dx) for dx in (0.0, 0.3, -0.2)])


def scene(
    n: int, seed: int = 0, *, dense: bool = False
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Draw a random splat with a random background color.

    Args:
        n: Number of gaussians.
        seed: Seed for the parameter draws.
        dense: Draw many small gaussians over a unit ball across the full opacity range, the
            regime the parity and tile-emission tests exercise. Otherwise the gaussians are large,
            soft, and tightly clustered, which suits the gradient and finite-difference checks.

    Returns:
        means ``(n, 3)``, scales ``(n, 3)``, quats ``(n, 4)``, colors ``(n, 3)``, opacities
        ``(n,)``, and a background color ``(3,)``.
    """
    spread = 1.0 if dense else 0.5
    scale_lo, scale_hi = (0.005, 0.05) if dense else (0.02, 0.08)
    opacity_lo, opacity_hi = (0.0, 1.0) if dense else (0.1, 0.6)
    k = jax.random.split(jax.random.key(seed), 6)
    means = jax.random.normal(k[0], (n, 3)) * spread
    scales = jax.random.uniform(k[1], (n, 3), minval=scale_lo, maxval=scale_hi)
    quats = jax.random.normal(k[2], (n, 4))
    quats = quats / jnp.linalg.norm(quats, axis=-1, keepdims=True)
    colors = jax.random.uniform(k[3], (n, 3))
    opacities = jax.random.uniform(k[4], (n,), minval=opacity_lo, maxval=opacity_hi)
    background = jax.random.uniform(k[5], (3,))
    return means, scales, quats, colors, opacities, background


def camera(H: int, W: int) -> dict:
    """Build the intrinsics keyword arguments for an ``H x W`` image, focal H, centered."""
    return {"img_shape": (H, W), "f": (float(H), float(H)), "c": (W // 2, H // 2)}


def projected(
    n: int, H: int, W: int, seed: int = 0, *, dense: bool = True
) -> tuple[jax.Array, ...]:
    """Project a random splat into the positional argument tuple ``splax.rasterize`` consumes.

    Returns:
        ``(colors, opacities, background, xys, depths, radii, conics, cum_tiles_hit)``.
    """
    means, scales, quats, colors, opacities, background = scene(n, seed, dense=dense)
    xys, depths, radii, conics, _nth, cum = splax.project(
        means, scales, quats, VIEWMAT, opacities=opacities, **camera(H, W)
    )
    return colors, opacities, background, xys, depths, radii, conics, cum


def poses(batch: int, seed: int = 0) -> jax.Array:
    """Draw ``batch`` distinct camera poses around ``VIEWMAT``, with an exact bottom row."""
    d = 0.02 * jax.random.normal(jax.random.key(seed), (batch, 4, 4))
    vms = jnp.broadcast_to(VIEWMAT, (batch, 4, 4)) + d
    return vms.at[:, 3, :].set(jnp.array([0.0, 0.0, 0.0, 1.0]))


def manual_move(
    means: jax.Array, quats: jax.Array, T: np.ndarray, start: int, stop: int
) -> tuple[jax.Array, jax.Array]:
    """Apply a rigid transform to a slice of the splat arrays in JAX, the projection's reference."""
    transform = TF.from_matrix(jnp.asarray(T))
    rotated = transform.rotation * R.from_quat(quats[start:stop], scalar_first=True)
    m2 = means.at[start:stop].set(transform.apply(means[start:stop]))
    q2 = quats.at[start:stop].set(rotated.as_quat(scalar_first=True))
    return m2, q2


def rasterize_both_keymodes(args: tuple[jax.Array, ...], H: int, W: int) -> tuple[np.ndarray, ...]:
    """Rasterize the same inputs with the packed 32-bit key and the 64-bit key."""
    orig = _sort._use_32bit_keys
    try:
        splax.clear_cache()
        _sort._use_32bit_keys = lambda depth_bits: depth_bits >= 16  # ty: ignore[invalid-assignment]
        packed = np.asarray(splax.rasterize(*args, img_shape=(H, W)))
        splax.clear_cache()
        _sort._use_32bit_keys = lambda depth_bits: False  # ty: ignore[invalid-assignment]
        wide = np.asarray(splax.rasterize(*args, img_shape=(H, W)))
    finally:
        _sort._use_32bit_keys = orig
        splax.clear_cache()
    return packed, wide
