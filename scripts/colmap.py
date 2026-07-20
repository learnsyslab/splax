"""COLMAP sparse reconstruction ingestion.

Loads a sparse model through pycolmap and initializes a fixed-N splat from the point cloud.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import pycolmap
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation
from scipy.special import logit

if TYPE_CHECKING:
    from pathlib import Path

    import jax

Points = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def read_reconstruction(
    path: str | Path,
) -> tuple[dict[int, tuple[str, int, int, tuple[float, ...]]], list[dict], Points]:
    """Read a COLMAP sparse model directory.

    Args:
        path: Path to the sparse model directory holding ``cameras``, ``images``, and ``points3D``.

    Returns:
        cams: Mapping of camera id to ``(model_name, width, height, params)``.
        images: List of per-image dicts with keys ``id``, ``qvec`` (wxyz), ``tvec``, ``camera_id``,
            ``name``, ``obs_xy`` (K, 2 float64), and ``obs_pid`` (K, int64), sorted by image name.
            The observations are the 2D keypoints with a valid triangulated 3D point, used for depth
            regularization. Views with no depth loss simply ignore them.
        points: Positions (M, 3) float64, colors (M, 3) uint8, point ids (M,) int64, and track
            lengths (M,) int64.
    """
    rec = pycolmap.Reconstruction(str(path))
    cams = {
        cid: (c.model_name, c.width, c.height, tuple(c.params)) for cid, c in rec.cameras.items()
    }
    images = []
    for iid, im in rec.images.items():
        pose = im.cam_from_world()
        valid = [p for p in im.points2D if p.has_point3D()]
        images.append(
            {
                "id": iid,
                "qvec": Rotation.from_matrix(pose.rotation.matrix()).as_quat(scalar_first=True),
                "tvec": np.asarray(pose.translation),
                "camera_id": im.camera_id,
                "name": im.name,
                "obs_xy": np.array([p.xy for p in valid], np.float64).reshape(-1, 2),
                "obs_pid": np.array([p.point3D_id for p in valid], np.int64),
            }
        )
    images.sort(key=lambda d: d["name"])
    items = list(rec.points3D.items())
    points = (
        np.array([p.xyz for _, p in items], np.float64).reshape(-1, 3),
        np.array([p.color for _, p in items], np.uint8).reshape(-1, 3),
        np.array([pid for pid, _ in items], np.int64),
        np.array([p.track.length() for _, p in items], np.int64),
    )
    return cams, images, points


def knn_scales(xyz: np.ndarray, cap: float, k: int = 3) -> np.ndarray:
    """Log-scale init from the mean distance to the k nearest neighbours.

    Args:
        xyz: Point positions, shape ``(M, 3)``.
        cap: Upper bound on the distance before the log.
        k: Number of neighbours to average over.

    Returns:
        Log scales, shape ``(M,)``, as float32.
    """
    d, _ = KDTree(xyz).query(xyz, k=k + 1)  # includes self at dist 0
    dist = np.clip(d[:, 1:].mean(axis=1), 1e-4, cap)  # floor: duplicate points would log to -inf
    return np.log(dist).astype(np.float32)


def init_from_points(
    xyz: np.ndarray,
    rgb: np.ndarray,
    n: int,
    opacity: float,
    seed: int = 0,
    weights: np.ndarray | None = None,
) -> dict[str, jax.Array]:
    """Initialize a fixed-N splat from the sparse cloud.

    Subsamples when the cloud has more than ``n`` points and pads by jittered duplication when it
    has fewer.

    Args:
        xyz: Sparse point positions, shape ``(M, 3)``.
        rgb: Point colors as uint8, shape ``(M, 3)``.
        n: Number of gaussians to initialize.
        opacity: Initial opacity of every gaussian.
        seed: Seed for the subsample and padding draws.
        weights: Positive per-point sampling weights such as track lengths, shape ``(M,)``.

    Returns:
        Parameter dict with the ``render_log`` arrays ``means``, ``log_scales``, ``quats``,
        ``colors_logit``, and ``opac_logit``.
    """
    rng = np.random.default_rng(seed)
    m = xyz.shape[0]
    prob = None
    if weights is not None:
        prob = np.log1p(weights)
        prob = prob / prob.sum()
    # cap init gaussian size (normalized units, cameras sit at dist ~1) so a few isolated outlier
    # points don't seed giant gaussians.
    cap = 0.3
    if m >= n:
        sel = rng.choice(m, n, replace=False, p=prob)
        xyz_n, rgb_n = xyz[sel], rgb[sel]
        log_scales = knn_scales(xyz_n, cap)
    else:
        pad = n - m
        src = rng.choice(m, pad, replace=True, p=prob)
        base_ls = knn_scales(xyz, cap)  # (m,) at the SPARSE m-point density
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
    # logit is infinite at exactly 0 and 1, which pure black/white uint8 colors hit
    colors_logit = logit(np.clip(rgb_n / 255.0, 1e-4, 1.0 - 1e-4))
    return {
        "means": jnp.asarray(xyz_n, jnp.float32),
        "log_scales": jnp.asarray(log_scales[:, None].repeat(3, 1)),
        "quats": jnp.asarray(rng.normal(size=(n, 4)), jnp.float32),
        "colors_logit": jnp.asarray(colors_logit, jnp.float32),
        "opac_logit": jnp.full((n, 1), logit(opacity), jnp.float32),
    }
