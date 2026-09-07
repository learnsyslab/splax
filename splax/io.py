"""PLY import/export and the parameter conversions around it.

``load_ply`` and ``write_ply`` load and save the unconstrained parameters.

``apply_activations`` and ``invert_activations`` map the scales and opacities to and from the arrays
the primitives consume, ``sh_to_rgb`` and ``rgb_to_sh`` the color.

``fetch`` downloads remote assets into a local cache and returns the cached path, so examples and
tests can pull scenes on demand.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from plyfile import PlyData, PlyElement

from splax._harmonics import C0, COEFFICIENTS


@jax.jit
def sh_to_rgb(base: jax.Array | np.ndarray) -> jax.Array:
    """Map the base color coefficient ``(N, 3)`` to unclamped RGB, where ``0`` is mid grey."""
    return jnp.maximum(base * C0 + 0.5, 0.0)


@jax.jit
def rgb_to_sh(colors: jax.Array | np.ndarray) -> jax.Array:
    """Map RGB in ``[0, 1]`` to the base color coefficient ``(N, 3)``."""
    return (colors - 0.5) / C0


@jax.jit
def apply_activations(
    log_scales: jax.Array | np.ndarray, logit_opacities: jax.Array | np.ndarray
) -> tuple[jax.Array, jax.Array]:
    """Map the stored geometry parameters to the arrays ``project`` and ``rasterize`` consume.

    Args:
        log_scales: Log of the per-axis scales, shape ``(N, 3)``.
        logit_opacities: Opacity logits, shape ``(N,)``.

    Returns:
        scales ``(N, 3)`` and opacities ``(N,)`` in ``[0, 1]``.
    """
    return jnp.exp(log_scales), jax.nn.sigmoid(logit_opacities)


@jax.jit
def invert_activations(
    scales: jax.Array | np.ndarray, opacities: jax.Array | np.ndarray
) -> tuple[jax.Array, jax.Array]:
    """Map the activated geometry arrays to the stored parameters.

    The conversion is the inverse of ``apply_activations`` in exact arithmetic. In float32 both
    activations are lossy, so a value that has to survive repeated round trips belongs in the
    parameters. ``rgb_to_sh`` inverts the color map.

    Args:
        scales: Positive per-axis scales, shape ``(N, 3)``.
        opacities: Opacities in ``[0, 1]``, shape ``(N,)``.

    Returns:
        log_scales ``(N, 3)`` and logit_opacities ``(N,)``.
    """
    return jnp.log(scales), jax.scipy.special.logit(opacities)


def fetch(
    url: str, *, cache: Path | None = None, force: bool = False, allow_unchecked: bool = False
) -> Path:
    """Download ``url`` into a local cache and return the path to the cached file.

    A cached file is reused only while its stored ETag still matches the remote. When the remote
    sends no ETag, the asset is downloaded on every call. Fetching with the ``force`` parameter
    ensures a fresh download. The cache directory defaults to ``$SPLAX_CACHE`` if set, else
    ``$XDG_CACHE_HOME/splax`` if set, else ``~/.cache/splax``.

    Args:
        url: URL to download.
        cache: Cache directory, overriding the environment-based default.
        force: Re-download and overwrite the cached copy even if it exists.
        allow_unchecked: Serve a cached file as-is instead of revalidating it against the remote.

    Returns:
        Path to the cached file.
    """
    if cache is None:
        xdg = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        cache = Path(os.environ["SPLAX_CACHE"]) if "SPLAX_CACHE" in os.environ else xdg / "splax"
    assert isinstance(cache, Path), f"cache must be a Path, got {type(cache)}"
    name = Path(urllib.parse.urlparse(url).path).name
    path = cache / (hashlib.sha256(url.encode()).hexdigest()[:16] + "-" + name)
    token_path = cache / (path.name + ".etag")
    if not force and allow_unchecked and path.exists():
        return path
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as resp:
        etag = resp.headers.get("ETag")
    if not force and path.exists() and token_path.exists() and token_path.read_text() == etag:
        return path
    cache.mkdir(parents=True, exist_ok=True)
    # Download to a temp file and atomically swap it in, so path is never left half-written.
    tmp = tempfile.NamedTemporaryFile(dir=cache, delete=False)
    try:
        with tmp, urllib.request.urlopen(url) as src:
            shutil.copyfileobj(src, tmp)
        os.replace(tmp.name, path)
    finally:
        Path(tmp.name).unlink(missing_ok=True)
    if etag is not None:
        token_path.write_text(etag)
    return path


def load_ply(path: Path) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Read a 3DGS ``.ply`` into the five parameter arrays ``render`` consumes.

    Args:
        path: Path to a 3DGS ``.ply`` file containing the fields ``x``, ``y``, ``z``,
            ``scale_0..2``, ``rot_0..3``, ``f_dc_0..2``, ``opacity``, and optionally a block of
            ``f_rest_j`` with the higher harmonics.

    Returns:
        means (N, 3), log_scales (N, 3), quats (N, 4), sh_colors (N, K, 3), logit_opacities (N,) as
        float32 jax arrays, where K is the SH coefficient count.
    """
    v = PlyData.read(str(path))["vertex"]
    means = jnp.asarray(np.stack([v["x"], v["y"], v["z"]], axis=-1), jnp.float32)
    log_scales = jnp.asarray(np.stack([v[f"scale_{i}"] for i in range(3)], axis=-1), jnp.float32)
    quats = jnp.asarray(np.stack([v[f"rot_{i}"] for i in range(4)], axis=-1), jnp.float32)
    base = np.stack([v[f"f_dc_{i}"] for i in range(3)], axis=-1, dtype=np.float32)[:, None]
    rest = _load_rest(v)
    sh_colors = base if rest is None else np.concatenate([base, rest], axis=1)
    logit_opacities = jnp.asarray(v["opacity"], jnp.float32)
    return means, log_scales, quats, jnp.asarray(sh_colors), logit_opacities


def _load_rest(vertex: PlyElement) -> np.ndarray | None:
    """Read the higher-order coefficients of a 3DGS ``.ply`` as an (N, K - 1, 3) array."""
    stored = sum(p.name.startswith("f_rest_") for p in vertex.properties)
    if not stored:  # No higher-order coefficients available
        return None
    higher, remainder = divmod(stored, 3)
    assert not remainder, f"f_rest holds {stored} fields, not three equal channel blocks"
    assert higher + 1 in COEFFICIENTS, f"{higher + 1} coefficients is not a whole set of bands"
    rest = np.stack([vertex[f"f_rest_{j}"] for j in range(stored)], axis=-1, dtype=np.float32)
    return rest.reshape(-1, 3, higher).swapaxes(1, 2)  # Transpose from channel to band-major


def write_ply(
    path: Path,
    means: jax.Array | np.ndarray,
    log_scales: jax.Array | np.ndarray,
    quats: jax.Array | np.ndarray,
    sh_colors: jax.Array | np.ndarray,
    logit_opacities: jax.Array | np.ndarray,
):
    """Write splat parameters to a 3DGS ``.ply``.

    Args:
        path: Path to the output ``.ply`` file.
        means: World positions, shape ``(N, 3)``.
        log_scales: Log of the per-axis scales, shape ``(N, 3)``.
        quats: wxyz quaternions, shape ``(N, 4)``.
        sh_colors: SH color coefficients, shape ``(N, K, 3)``.
        logit_opacities: Opacity logits, shape ``(N,)``.
    """
    means = np.asarray(means, np.float32)
    log_scales = np.asarray(log_scales, np.float32)
    quats = np.asarray(quats, np.float32)
    sh_colors = np.asarray(sh_colors, np.float32)
    logit_opacities = np.asarray(logit_opacities, np.float32)
    n = means.shape[0]
    rest = sh_colors[:, 1:].swapaxes(1, 2).reshape(n, -1)  # Transpose back to channel-major
    columns = [means, np.zeros((n, 3)), sh_colors[:, 0]]
    fields = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    if rest.shape[1]:
        columns.append(rest)
        fields += [f"f_rest_{j}" for j in range(rest.shape[1])]
    data = np.column_stack([*columns, logit_opacities, log_scales, quats])
    fields += ["opacity"] + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)]
    verts = np.empty(n, dtype=[(f, "f4") for f in fields])
    for field, column in zip(fields, data.T, strict=True):
        verts[field] = column
    PlyData([PlyElement.describe(verts, "vertex")], text=False).write(str(path))


def prune(
    means: jax.Array,
    log_scales: jax.Array,
    quats: jax.Array,
    sh_colors: jax.Array,
    logit_opacities: jax.Array,
    *,
    radius: float = 3.0,
    opacity: float = 0.01,
    scale: float = 10.0,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Prune gaussians that are significantly outside the scene.

    Training creates stray gaussians outside the rendered scene, near transparent gaussians, and
    oversized floaters. We prune all three based off the 99th percentile of the scene's statistics.

    Args:
        means: World positions, shape ``(N, 3)``.
        log_scales: Log of the per-axis scales, shape ``(N, 3)``.
        quats: wxyz quaternions, shape ``(N, 4)``.
        sh_colors: SH color coefficients, shape ``(N, K, 3)``.
        logit_opacities: Opacity logits, shape ``(N,)``.
        radius: Multiple of the 99th percentile distance from the median center beyond which a
            gaussian counts as a stray.
        opacity: Opacity below which a gaussian is dropped.
        scale: Multiple of the 99th percentile largest axis scale above which a gaussian is dropped.

    Returns:
        The five parameter arrays restricted to the kept gaussians.
    """
    positions = np.asarray(means)
    distance = np.linalg.norm(positions - np.median(positions, axis=0), axis=1)
    extent = np.exp(np.asarray(log_scales)).max(axis=1)
    alpha = 1.0 / (1.0 + np.exp(-np.asarray(logit_opacities)))
    keep = distance <= radius * np.percentile(distance, 99)
    keep &= extent <= scale * np.percentile(extent, 99)
    keep &= alpha >= opacity
    return means[keep], log_scales[keep], quats[keep], sh_colors[keep], logit_opacities[keep]
