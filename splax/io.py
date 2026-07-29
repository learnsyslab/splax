"""PLY import/export and the parameter conversions around it.

``load_ply`` and ``write_ply`` load and save the unconstrained parameters.

``apply_activations`` and ``invert_activations`` map between those parameters and the activated
arrays, the linear scales, RGB colors, and ``[0, 1]`` opacities the primitives consume.
``sh_to_rgb`` and ``rgb_to_sh`` map the colors on their own.

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

# Value of the degree-0 SH basis function Y00, the constant term of the expansion splax renders.
_C0 = 0.5 / np.sqrt(np.pi)


def sh_to_rgb(sh_colors: jax.Array | np.ndarray) -> jax.Array:
    """Map degree-0 SH coefficients to RGB in ``[0, 1]``, where ``0`` is mid grey."""
    return jnp.clip(sh_colors * _C0 + 0.5, 0.0, 1.0)  # files may store out-of-range coefficients


def rgb_to_sh(colors: jax.Array | np.ndarray) -> jax.Array:
    """Map RGB in ``[0, 1]`` to degree-0 SH coefficients."""
    return (colors - 0.5) / _C0


def apply_activations(
    log_scales: jax.Array | np.ndarray,
    sh_colors: jax.Array | np.ndarray,
    logit_opacities: jax.Array | np.ndarray,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Map stored parameters to the activated arrays ``project`` and ``rasterize`` consume.

    Args:
        log_scales: Log of the per-axis scales, shape ``(N, 3)``.
        sh_colors: Degree-0 SH color coefficients, shape ``(N, 3)``.
        logit_opacities: Opacity logits, shape ``(N,)``.

    Returns:
        scales ``(N, 3)``, colors ``(N, 3)`` in ``[0, 1]``, and opacities ``(N,)`` in ``[0, 1]``.
    """
    return jnp.exp(log_scales), sh_to_rgb(sh_colors), jax.nn.sigmoid(logit_opacities)


def invert_activations(
    scales: jax.Array | np.ndarray,
    colors: jax.Array | np.ndarray,
    opacities: jax.Array | np.ndarray,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Map activated arrays to the stored parameters.

    The conversion is the inverse of ``apply_activations`` in exact arithmetic. In float32 the
    scale and opacity activations are lossy, so a value that has to survive repeated round trips
    belongs in the parameters rather than in activated form.

    Args:
        scales: Positive per-axis scales, shape ``(N, 3)``.
        colors: RGB in ``[0, 1]``, shape ``(N, 3)``.
        opacities: Opacities in ``[0, 1]``, shape ``(N,)``.

    Returns:
        log_scales ``(N, 3)``, sh_colors ``(N, 3)``, and logit_opacities ``(N,)``.
    """
    logits = jax.scipy.special.logit(opacities)
    return jnp.log(scales), rgb_to_sh(colors), logits


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
            ``scale_0..2``, ``rot_0..3``, ``f_dc_0..2``, and ``opacity``.

    Returns:
        means (N, 3), log_scales (N, 3), quats (N, 4), sh_colors (N, 3), logit_opacities (N,) as
        float32 jax arrays.
    """
    v = PlyData.read(str(path))["vertex"]
    means = jnp.asarray(np.stack([v["x"], v["y"], v["z"]], axis=-1), jnp.float32)
    log_scales = jnp.asarray(np.stack([v[f"scale_{i}"] for i in range(3)], axis=-1), jnp.float32)
    quats = jnp.asarray(np.stack([v[f"rot_{i}"] for i in range(4)], axis=-1), jnp.float32)
    sh_colors = jnp.asarray(np.stack([v[f"f_dc_{i}"] for i in range(3)], axis=-1), jnp.float32)
    logit_opacities = jnp.asarray(v["opacity"], jnp.float32)
    return means, log_scales, quats, sh_colors, logit_opacities


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
        sh_colors: Degree-0 SH color coefficients, shape ``(N, 3)``.
        logit_opacities: Opacity logits, shape ``(N,)``.
    """
    means = np.asarray(means, np.float32)
    log_scales = np.asarray(log_scales, np.float32)
    quats = np.asarray(quats, np.float32)
    sh_colors = np.asarray(sh_colors, np.float32)
    logit_opacities = np.asarray(logit_opacities, np.float32)
    n = means.shape[0]
    data = np.column_stack([means, np.zeros((n, 3)), sh_colors, logit_opacities, log_scales, quats])
    fields = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity"]
    fields += [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)]
    verts = np.empty(n, dtype=[(f, "f4") for f in fields])
    for field, column in zip(fields, data.T, strict=True):
        verts[field] = column
    PlyData([PlyElement.describe(verts, "vertex")], text=False).write(str(path))
