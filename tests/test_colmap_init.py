"""N-aware init-scale correction for the ``colmap`` training-toolkit module.

CPU-only, data-independent (synthetic point cloud): checks the density-ratio
scale correction is applied iff the fixed-N init pads the sparse cloud (n>m),
and with exactly the (1/3)ln(n/m) log-space magnitude. See ``init_from_points``.
"""

from __future__ import annotations

import numpy as np
from colmap import init_from_points, knn_scales


def _cloud(m: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    xyz = rng.uniform(-1.0, 1.0, size=(m, 3)).astype(np.float32)
    rgb = rng.integers(0, 256, size=(m, 3)).astype(np.uint8)
    return xyz, rgb


def test_padding_applies_density_ratio_correction() -> None:
    """When n>m, every original knn log-scale is lowered by exactly (1/3)ln(n/m)."""
    m, n = 500, 4000
    xyz, rgb = _cloud(m)
    p = init_from_points(xyz, rgb, n, 0.1, seed=0)
    ls = np.asarray(p["log_scales"])
    # the correction is applied to the whole knn-derived cloud, the first m rows are the
    # original points, so compare them against the uncorrected knn scales (cap=0.3 as in
    # init_from_points) minus the expected (1/3)ln(n/m) offset.
    base = knn_scales(xyz, cap=0.3)  # (m,) uncorrected
    expected = (base - np.log(n / m) / 3.0)[:, None].repeat(3, 1)
    assert np.allclose(ls[:m], expected, atol=1e-5)
    # all three columns share the per-gaussian scale
    assert np.allclose(ls[:, 0], ls[:, 1]) and np.allclose(ls[:, 0], ls[:, 2])


def test_correction_scales_with_padding_ratio() -> None:
    """Offset tracks the padding ratio: n=8m is 3x lower than n=m in log space."""
    m = 500
    xyz, rgb = _cloud(m)
    base = knn_scales(xyz, cap=0.3)
    # small pad (n just above m) vs large pad (n=8m): the mean original-block log-scale
    # drops by exactly the difference of the two (1/3)ln(n/m) corrections.
    p_small = init_from_points(xyz, rgb, m + 1, 0.1, seed=0)
    p_large = init_from_points(xyz, rgb, 8 * m, 0.1, seed=0)
    off_small = base.mean() - np.asarray(p_small["log_scales"])[:m, 0].mean()
    off_large = base.mean() - np.asarray(p_large["log_scales"])[:m, 0].mean()
    assert np.isclose(off_small, np.log((m + 1) / m) / 3.0, atol=1e-5)
    assert np.isclose(off_large, np.log(8) / 3.0, atol=1e-5)


def test_subsample_branch_has_no_correction() -> None:
    """Check that subsampling skips density correction when n is not padded."""
    m, n = 4000, 500
    xyz, rgb = _cloud(m)
    p = init_from_points(xyz, rgb, n, 0.1, seed=0)
    # reproduce the subsample selection with the same rng draw order as init_from_points
    rng = np.random.default_rng(0)
    sel = rng.choice(m, n, replace=False)
    expected = knn_scales(xyz[sel], cap=0.3)[:, None].repeat(3, 1)
    assert np.allclose(np.asarray(p["log_scales"]), expected, atol=1e-5)
