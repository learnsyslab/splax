"""Scene builders and key-mode helpers shared by the rasterization tests."""

from __future__ import annotations

from typing import TypedDict

import jax
import jax.numpy as jnp
import numpy as np

import splax
import splax._rasterize._sort._sort as _sort


class RastKW(TypedDict):
    img_shape: tuple[int, int]


class RenderKW(TypedDict):
    viewmat: jax.Array
    background: jax.Array
    img_shape: tuple[int, int]
    f: tuple[float, float]
    c: tuple[float, float]
    glob_scale: float
    clip_thresh: float


def render_scene(
    n: int, H: int, W: int, seed: int
) -> tuple[tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array], RenderKW]:
    """Build a random splat with the render keyword arguments for an ``H x W`` image."""
    key = jax.random.key(seed)
    k = jax.random.split(key, 6)
    means = jax.random.normal(k[0], (n, 3))
    scales = jax.random.uniform(k[1], (n, 3), minval=0.005, maxval=0.05)
    quats = jax.random.normal(k[2], (n, 4))
    quats = quats / jnp.linalg.norm(quats, axis=-1, keepdims=True)
    colors = jax.random.uniform(k[3], (n, 3))
    opac = jax.random.uniform(k[4], (n,))
    background = jax.random.uniform(k[5], (3,))
    vm = jnp.array([[1, 0, 0, 0.2], [0, 1, 0, -0.1], [0, 0, 1, 5], [0, 0, 0, 1]], jnp.float32)
    kw: RenderKW = {
        "viewmat": vm,
        "background": background,
        "img_shape": (H, W),
        "f": (float(H), float(H)),
        "c": (W // 2, H // 2),
        "glob_scale": 1.0,
        "clip_thresh": 0.01,
    }
    return (means, scales, quats, colors, opac), kw


def rasterize_both_keymodes(
    args: tuple[jax.Array, ...], kw: RastKW
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize the same inputs with the packed 32-bit key and the 64-bit key."""
    orig = _sort._use_32bit_keys
    try:
        splax.clear_cache()
        _sort._use_32bit_keys = lambda depth_bits: depth_bits >= 16  # ty: ignore[invalid-assignment]
        packed = np.asarray(splax.rasterize(*args, **kw))
        splax.clear_cache()
        _sort._use_32bit_keys = lambda depth_bits: False  # ty: ignore[invalid-assignment]
        wide = np.asarray(splax.rasterize(*args, **kw))
    finally:
        _sort._use_32bit_keys = orig
        splax.clear_cache()
    return packed, wide
