"""PLY import/export for splats in render space.

``load_ply`` reads a 3DGS ``.ply`` and maps the stored (activation-space) fields
to render-space inputs:

    scales  = exp(scale_i)                     # log-scale     -> scale
    quats   = normalize(rot_i)                 # raw quat       -> unit wxyz
    colors  = clip(f_dc_i * C0 + 0.5, 0, 1)    # SH deg-0 coeff -> rgb
    opac    = sigmoid(opacity)                 # logit          -> [0, 1]

``write_ply`` takes the *render-space* splats (the same tensors ``render``
consumes) and writes the inverse activation-space fields so a subsequent
``load_ply`` reproduces them. Normals are zeroed and ``f_rest`` (higher-order SH)
is omitted. ``load_ply`` reads neither, so a written file round-trips exactly
through it (SH degree 0 only, which is all splax renders).

``fetch`` downloads remote assets (e.g. scene ``.ply`` files) into a local cache
and returns the cached path, so examples and tests can pull scenes on demand.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.parse
import urllib.request
from contextlib import ExitStack
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from plyfile import PlyData, PlyElement

# SH degree-0 basis constant, shared with load_ply (colors = f_dc * C0 + 0.5).
_C0 = 0.28209479177387814


def fetch(
    url: str, *, sha256: str | None = None, cache: Path | None = None, force: bool = False
) -> Path:
    """Download ``url`` into a local cache and return the path to the cached file.

    If the file is already cached and ``force`` is False, it is returned without
    touching the network, so ``sha256`` is only verified on actual downloads, not
    on cache hits. The cache directory defaults to ``$SPLAX_CACHE`` if set, else
    ``$XDG_CACHE_HOME/splax`` if set, else ``~/.cache/splax``; ``cache`` overrides
    all three. Cached files are named ``sha256(url)[:16] + "-" + basename`` so
    entries are unique per URL yet human-recognizable, and downloads are atomic
    (streamed to a temp file in the cache directory, then renamed into place).

    Args:
        url: URL to download.
        sha256: Expected hex digest of the downloaded bytes, checked on download.
        cache: Cache directory, overriding the environment-based default.
        force: Re-download and overwrite the cached copy even if it exists.

    Returns:
        Path to the cached file.

    Raises:
        ValueError: If ``sha256`` is given and the downloaded bytes don't match.
    """
    if cache is None:
        loc = os.environ.get("SPLAX_CACHE", os.environ.get("XDG_CACHE_HOME"))
        cache = Path(loc) if loc is not None else Path.home() / ".cache/splax"
    assert isinstance(cache, Path), f"cache must be a Path, got {type(cache)}"
    name = Path(urllib.parse.urlparse(url).path).name or "download"
    path = cache / (hashlib.sha256(url.encode()).hexdigest()[:16] + "-" + name)
    if path.exists() and not force:
        return path
    cache.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(dir=cache, delete=False)
    with ExitStack() as stack:  # Ensure tmp is deleted when an exception occurs before os.replace
        stack.callback(Path(tmp.name).unlink, missing_ok=True)
        with tmp, urllib.request.urlopen(url) as src:
            shutil.copyfileobj(src, tmp)
        if sha256 is not None:
            digest = hashlib.sha256(Path(tmp.name).read_bytes()).hexdigest()
            if digest != sha256:
                raise ValueError(f"sha256 mismatch for {url}: expected {sha256}, got {digest}")
        os.replace(tmp.name, path)
        stack.pop_all()
    return path


def load_ply(path: str | Path) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Read a 3DGS ``.ply`` into the five render-space arrays ``render`` consumes.

    Returns (means, scales, quats, colors, opacities) as float32 jax arrays,
    shapes (N, 3), (N, 3), (N, 4), (N, 3), (N, 1). SH degree 0 only (f_rest
    ignored). Exact inverse of ``write_ply``.
    """
    v = PlyData.read(path)["vertex"]
    means = jnp.asarray(np.stack([v["x"], v["y"], v["z"]], axis=-1), jnp.float32)
    scales = jnp.asarray(
        np.exp(np.stack([v[f"scale_{i}"] for i in range(3)], axis=-1)), jnp.float32
    )
    quats = jnp.asarray(np.stack([v[f"rot_{i}"] for i in range(4)], axis=-1), jnp.float32)
    quats /= jnp.linalg.norm(quats, axis=-1, keepdims=True)
    sh0 = jnp.asarray(np.stack([v[f"f_dc_{i}"] for i in range(3)], axis=-1), jnp.float32)
    colors = jnp.clip(sh0 * _C0 + 0.5, 0.0, 1.0)
    opacities = 1.0 / (1.0 + jnp.exp(-v["opacity"]))[..., None]
    return means, scales, quats, colors, opacities


def write_ply(
    path: str | Path,
    means: jax.Array | np.ndarray,
    scales: jax.Array | np.ndarray,
    quats: jax.Array | np.ndarray,
    colors: jax.Array | np.ndarray,
    opacities: jax.Array | np.ndarray,
) -> None:
    """Write render-space splats to a 3DGS ``.ply`` (inverse of ``load_ply``).

    Args mirror ``render``'s first five positional arguments:
      means      (N, 3) world positions
      scales     (N, 3) positive per-axis scales (stored as log)
      quats      (N, 4) wxyz quaternions (normalized on write)
      colors     (N, 3) rgb in [0, 1] (stored as SH deg-0 coeff (rgb-0.5)/C0)
      opacities  (N, 1) or (N,) opacity in [0, 1] (stored as logit)

    Fields written: x,y,z, nx,ny,nz (zero), f_dc_0..2, opacity, scale_0..2,
    rot_0..3, exactly the set ``load_ply`` consumes.
    """
    means = np.asarray(means, np.float32)
    scales = np.asarray(scales, np.float32)
    quats = np.asarray(quats, np.float32)
    colors = np.asarray(colors, np.float32)
    opacities = np.asarray(opacities, np.float32).reshape(-1)
    n = means.shape[0]

    log_scales = np.log(scales)
    quats = quats / np.linalg.norm(quats, axis=-1, keepdims=True)
    f_dc = (colors - 0.5) / _C0
    # logit: inverse of sigmoid. Clip away from {0,1} to keep it finite.
    opac = np.clip(opacities, 1e-7, 1.0 - 1e-7)
    opacity = np.log(opac / (1.0 - opac))

    fields = [
        "x",
        "y",
        "z",
        "nx",
        "ny",
        "nz",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    ]
    verts = np.empty(n, dtype=[(f, "f4") for f in fields])
    verts["x"], verts["y"], verts["z"] = means[:, 0], means[:, 1], means[:, 2]
    verts["nx"] = verts["ny"] = verts["nz"] = 0.0
    verts["f_dc_0"], verts["f_dc_1"], verts["f_dc_2"] = f_dc[:, 0], f_dc[:, 1], f_dc[:, 2]
    verts["opacity"] = opacity
    verts["scale_0"], verts["scale_1"], verts["scale_2"] = (
        log_scales[:, 0],
        log_scales[:, 1],
        log_scales[:, 2],
    )
    verts["rot_0"], verts["rot_1"], verts["rot_2"], verts["rot_3"] = (
        quats[:, 0],
        quats[:, 1],
        quats[:, 2],
        quats[:, 3],
    )

    el = PlyElement.describe(verts, "vertex")
    PlyData([el], text=False).write(str(path))
