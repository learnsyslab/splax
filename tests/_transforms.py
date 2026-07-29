"""Scene, camera, and reference builders shared by the rigid transform tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import jax
import jax.numpy as jnp
from scipy.spatial.transform import RigidTransform as TF
from scipy.spatial.transform import Rotation as R

if TYPE_CHECKING:
    import numpy as np


class KW(TypedDict):
    viewmat: jax.Array
    background: jax.Array
    img_shape: tuple[int, int]
    f: tuple[float, float]
    c: tuple[float, float]


def scene(n: int, seed: int = 0) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Build a random splat."""
    k = jax.random.split(jax.random.key(seed), 5)
    means = jax.random.normal(k[0], (n, 3)) * 0.5
    scales = jax.random.uniform(k[1], (n, 3), minval=0.02, maxval=0.08)
    quats = jax.random.normal(k[2], (n, 4))
    quats = quats / jnp.linalg.norm(quats, axis=-1, keepdims=True)
    colors = jax.random.uniform(k[3], (n, 3))
    opac = jax.random.uniform(k[4], (n,), minval=0.1, maxval=0.6)
    return means, scales, quats, colors, opac


def kw(H: int, W: int) -> KW:
    """Build the render keyword arguments for an ``H x W`` image."""
    vm = jnp.array([[1, 0, 0, 0.2], [0, 1, 0, -0.1], [0, 0, 1, 5], [0, 0, 0, 1]], jnp.float32)
    return {
        "viewmat": vm,
        "background": jnp.zeros(3),
        "img_shape": (H, W),
        "f": (float(H), float(H)),
        "c": (W // 2, H // 2),
    }


def manual_move(
    means: jax.Array, quats: jax.Array, T: np.ndarray, start: int, stop: int
) -> tuple[jax.Array, jax.Array]:
    """Reference transform of a slice, applied to the splat arrays in JAX."""
    transform = TF.from_matrix(jnp.asarray(T))
    rotated = transform.rotation * R.from_quat(quats[start:stop], scalar_first=True)
    m2 = means.at[start:stop].set(transform.apply(means[start:stop]))
    q2 = quats.at[start:stop].set(rotated.as_quat(scalar_first=True))
    return m2, q2
