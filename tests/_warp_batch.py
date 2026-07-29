"""Scene, camera, and render helpers shared by the native batching tests."""

from __future__ import annotations

from typing import TypedDict

import jax
import jax.numpy as jnp
import pytest

import splax
import splax._rasterize._sort._sort as _sort


@pytest.fixture(autouse=True)
def faithful_64bit_keys(monkeypatch: pytest.MonkeyPatch):
    """Pin the 64-bit sort key for the bit-exact batch-native assertions.

    Batch-native == stack-of-unbatched is bit-exact only for the 64-bit key, whose per-image
    (tile, depth) order is independent of B. The default packed 32-bit key sizes its depth field
    as 31 - image_bits - tile_bits, which shrinks as B grows, so a batched render quantizes depth
    slightly coarser than the B=1 reference and matches only up to a perceptual bound.
    """
    monkeypatch.setattr(_sort, "_use_32bit_keys", lambda depth_bits: False)


class KW(TypedDict):
    background: jax.Array
    img_shape: tuple[int, int]
    f: tuple[float, float]
    c: tuple[float, float]
    glob_scale: float
    clip_thresh: float


class ProjKW(TypedDict):
    img_shape: tuple[int, int]
    f: tuple[float, float]
    c: tuple[float, float]
    glob_scale: float
    clip_thresh: float


def rand_scene(
    n: int, seed: int = 0
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Build a random splat."""
    k = jax.random.split(jax.random.key(seed), 5)
    means = jax.random.normal(k[0], (n, 3))
    scales = jax.random.uniform(k[1], (n, 3), minval=0.005, maxval=0.05)
    quats = jax.random.normal(k[2], (n, 4))
    quats = quats / jnp.linalg.norm(quats, axis=-1, keepdims=True)
    colors = jax.random.uniform(k[3], (n, 3))
    opacities = jax.random.uniform(k[4], (n,))
    return means, scales, quats, colors, opacities


def viewmat(dx: float, dy: float = -0.1, dz: float = 5.0) -> jax.Array:
    """Build a translation-only world-to-camera matrix."""
    return jnp.array([[1, 0, 0, dx], [0, 1, 0, dy], [0, 0, 1, dz], [0, 0, 0, 1]], jnp.float32)


N = 8_000
H = W = 128
CAMERA: KW = {
    "background": jnp.zeros(3),
    "img_shape": (H, W),
    "f": (float(H), float(H)),
    "c": (W // 2, H // 2),
    "glob_scale": 1.0,
    "clip_thresh": 0.01,
}
PROJ_CAMERA: ProjKW = {
    "img_shape": (H, W),
    "f": (float(H), float(H)),
    "c": (W // 2, H // 2),
    "glob_scale": 1.0,
    "clip_thresh": 0.01,
}
VIEWS = jnp.stack([viewmat(0.0), viewmat(0.3), viewmat(-0.2)])  # B=3


def render(
    m: jax.Array, s: jax.Array, q: jax.Array, c: jax.Array, o: jax.Array, vm: jax.Array
) -> jax.Array:
    """Render one view of the splat through the shared camera."""
    return splax.render(m, s, q, c, o, viewmat=vm, **CAMERA)[0]
