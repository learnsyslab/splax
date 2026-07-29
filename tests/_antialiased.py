"""Scene and camera builders shared by the anti aliasing unit and integration tests."""

from __future__ import annotations

from typing import TypedDict

import jax
import jax.numpy as jnp


def scene(
    n: int, H: int, W: int, seed: int = 0
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Build a random splat with its background color and viewmat."""
    key = jax.random.key(seed)
    k = jax.random.split(key, 6)
    means = jax.random.normal(k[0], (n, 3)) * 0.5
    scales = jax.random.uniform(k[1], (n, 3), minval=0.02, maxval=0.08)
    quats = jax.random.normal(k[2], (n, 4))
    quats = quats / jnp.linalg.norm(quats, axis=-1, keepdims=True)
    colors = jax.random.uniform(k[3], (n, 3))
    opac = jax.random.uniform(k[4], (n,), minval=0.1, maxval=0.6)
    bg = jax.random.uniform(k[5], (3,))
    vm = jnp.array([[1, 0, 0, 0.2], [0, 1, 0, -0.1], [0, 0, 1, 5], [0, 0, 0, 1]], jnp.float32)
    return means, scales, quats, colors, opac, bg, vm


class PK(TypedDict):
    img_shape: tuple[int, int]
    f: tuple[float, float]
    c: tuple[float, float]
    glob_scale: float
    clip_thresh: float


def pk(H: int, W: int) -> PK:
    """Build the projection keyword arguments for an ``H x W`` image."""
    return {
        "img_shape": (H, W),
        "f": (float(H), float(H)),
        "c": (W // 2, H // 2),
        "glob_scale": 1.0,
        "clip_thresh": 0.01,
    }
