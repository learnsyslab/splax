"""COLMAP sparse reconstruction ingestion.

Binary readers for ``cameras.bin`` / ``images.bin`` / ``points3D.bin`` and the fixed-N splat
initialization from the sparse point cloud.
"""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING, BinaryIO

import jax.numpy as jnp
import numpy as np
from scipy.spatial import KDTree

if TYPE_CHECKING:
    from pathlib import Path

    import jax

_CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}


def _r(f: BinaryIO, fmt: str) -> tuple:
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))


def read_cameras(path: str | Path) -> dict[int, tuple[str, int, int, tuple[float, ...]]]:
    """Return {camera_id: (model_name, w, h, params-tuple)}."""
    cams = {}
    with open(path, "rb") as f:
        (n,) = _r(f, "Q")
        for _ in range(n):
            cid, model, w, h = _r(f, "iiQQ")
            name, npar = _CAMERA_MODELS[model]
            params = _r(f, "d" * npar)
            cams[cid] = (name, w, h, params)
    return cams


def read_images(path: str | Path) -> list[dict]:
    """Return list of dicts {id, qvec, tvec, camera_id, name, obs_xy, obs_pid}.

    ``obs_xy`` (K,2 float64) / ``obs_pid`` (K, int64) are the per-image 2D keypoint
    observations that have a valid triangulated 3D point. These are the COLMAP sparse points visible
    in this view, used for depth regularization. Views with no depth loss simply ignore them.
    """
    imgs = []
    with open(path, "rb") as f:
        (n,) = _r(f, "Q")
        for _ in range(n):
            iid, qw, qx, qy, qz, tx, ty, tz, camid = _r(f, "idddddddi")
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            (np2d,) = _r(f, "Q")
            buf = f.read(np2d * 24)  # per point2D: x,y (double) + point3D_id (int64)
            if np2d:
                rec = np.frombuffer(buf, dtype=np.uint8).reshape(np2d, 24)
                xy = rec[:, :16].copy().view(np.float64).reshape(np2d, 2)
                pid = rec[:, 16:].copy().view(np.int64).reshape(np2d)
                keep = pid >= 0
                obs_xy, obs_pid = xy[keep], pid[keep]
            else:
                obs_xy = np.zeros((0, 2), np.float64)
                obs_pid = np.zeros((0,), np.int64)
            imgs.append(
                {
                    "id": iid,
                    "qvec": np.array([qw, qx, qy, qz]),
                    "tvec": np.array([tx, ty, tz]),
                    "camera_id": camid,
                    "name": name.decode(),
                    "obs_xy": obs_xy,
                    "obs_pid": obs_pid,
                }
            )
    imgs.sort(key=lambda d: d["name"])
    return imgs


def read_points3D(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (xyz (M,3) float64, rgb (M,3) uint8, ids (M,) int64, track_lens (M,) int64)."""
    xyz, rgb, ids, track_lens = [], [], [], []
    with open(path, "rb") as f:
        (n,) = _r(f, "Q")
        for _ in range(n):
            pid, x, y, z, rr, gg, bb, err = _r(f, "QdddBBBd")
            (tl,) = _r(f, "Q")
            f.read(tl * 8)  # track: (image_id int32, point2D_idx int32) * tl
            xyz.append((x, y, z))
            rgb.append((rr, gg, bb))
            ids.append(pid)
            track_lens.append(tl)
    return (
        np.asarray(xyz, np.float64),
        np.asarray(rgb, np.uint8),
        np.asarray(ids, np.int64),
        np.asarray(track_lens, np.int64),
    )


def knn_scales(xyz: np.ndarray, k: int = 3, cap: float | None = None) -> np.ndarray:
    """Log-scale init = log(mean distance to k nearest neighbours)."""
    tree = KDTree(xyz)
    d, _ = tree.query(xyz, k=k + 1)  # includes self at dist 0
    dist = d[:, 1:].mean(axis=1)
    dist = np.clip(dist, 1e-4, cap if cap else np.inf)
    return np.log(dist).astype(np.float32)


def init_from_points(
    xyz: np.ndarray,
    rgb: np.ndarray,
    n: int,
    opa: float,
    seed: int = 0,
    weights: np.ndarray | None = None,
) -> dict[str, jax.Array]:
    """Fixed-N init from the sparse cloud (pad by jittered duplication / subsample)."""
    rng = np.random.default_rng(seed)
    m = xyz.shape[0]
    prob = None
    if weights is not None:
        prob = np.log1p(np.asarray(weights, np.float64))
        prob = np.clip(prob, 0.0, None)
        total = float(prob.sum())
        if total > 0:
            prob = prob / total
        else:
            prob = None
    # cap init gaussian size (normalized units, cameras sit at dist ~1) so a few isolated outlier
    # points don't seed giant gaussians.
    cap = 0.3
    if m >= n:
        sel = rng.choice(m, n, replace=False, p=prob)
        xyz_n, rgb_n = xyz[sel], rgb[sel]
        log_scales = knn_scales(xyz_n, cap=cap)
    else:
        pad = n - m
        src = rng.choice(m, pad, replace=True, p=prob)
        base_ls = knn_scales(xyz, cap=cap)  # (m,) at the SPARSE m-point density
        # N-aware scale correction. knn_scales is the mean nearest-neighbour distance at the
        # *sparse* density (m points spread through the scene volume V). Padding to n>m gaussians
        # raises the density to n/V, and for a roughly uniform cloud the mean NN spacing scales as
        # density^(-1/3). The per-gaussian scale at the target density is thus smaller by a factor
        # cbrt(n/m). We correct in log space by subtracting (1/3)ln(n/m) from every knn log-scale.
        # The jitter that spreads the padded copies uses the corrected (smaller) scale too, so the
        # seeded points sit at the target spacing. Only fires when padding (n>m).
        base_ls = base_ls - np.log(n / m) / 3.0
        jitter = rng.normal(size=(pad, 3)).astype(np.float32) * np.exp(base_ls[src])[:, None]
        xyz_n = np.concatenate([xyz, xyz[src] + jitter], 0)
        rgb_n = np.concatenate([rgb, rgb[src]], 0)
        log_scales = np.concatenate([base_ls, base_ls[src]], 0)
    colors = np.clip(rgb_n.astype(np.float32) / 255.0, 1e-4, 1 - 1e-4)
    colors_logit = np.log(colors / (1 - colors))
    quats = rng.normal(size=(n, 4)).astype(np.float32)
    opac_logit = np.full((n, 1), float(np.log(opa / (1 - opa))), np.float32)
    return {
        "means": jnp.asarray(xyz_n.astype(np.float32)),
        "log_scales": jnp.asarray(log_scales[:, None].repeat(3, 1)),
        "quats": jnp.asarray(quats),
        "colors_logit": jnp.asarray(colors_logit),
        "opac_logit": jnp.asarray(opac_logit),
    }
