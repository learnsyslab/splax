"""Test the fixed-N splat initialization from a sparse point cloud."""

from __future__ import annotations

import numpy as np
import pytest
from colmap import init_from_points
from scipy.spatial import KDTree
from scipy.special import expit

pytestmark = pytest.mark.colmap


def _cloud(m: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    xyz = rng.uniform(-1.0, 1.0, size=(m, 3)).astype(np.float32)
    rgb = rng.integers(0, 256, size=(m, 3)).astype(np.uint8)
    return xyz, rgb


@pytest.mark.parametrize("m,n", [(4000, 500), (500, 4000)])
def test_scales_match_the_realized_point_density(m: int, n: int):
    """Test that the initial scale matches the spacing of the points it initializes."""
    xyz, rgb = _cloud(m)
    p = init_from_points(xyz, rgb, n, 0.1)
    means = np.asarray(p["means"])
    scales = np.exp(np.asarray(p["log_scales"])[:, 0])
    distances, _ = KDTree(means).query(means, k=4)  # k=4 includes the point itself at distance 0
    # padding duplicates points, so the splat is denser than the cloud it was seeded from and the
    # scale has to follow the density it ends up at, not the one it was measured on
    ratio = scales.mean() / distances[:, 1:].mean()
    assert 0.9 < ratio < 1.15, f"scale is {ratio:.2f}x the spacing of the initialized points"


def test_scales_are_isotropic():
    """Test that every gaussian is seeded with one scale shared across its three axes."""
    log_scales = np.asarray(init_from_points(*_cloud(500), 4000, 0.1)["log_scales"])
    assert (log_scales == log_scales[:, :1]).all()


def test_padding_keeps_the_cloud_and_seeds_the_copies_next_to_it():
    """Test that padding preserves the sparse points and seeds the copies next to them."""
    m, n = 500, 4000
    xyz, rgb = _cloud(m)
    p = init_from_points(xyz, rgb, n, 0.1)
    means = np.asarray(p["means"])
    scales = np.exp(np.asarray(p["log_scales"])[:, 0])
    np.testing.assert_array_equal(means[:m], xyz)
    distance, _ = KDTree(xyz).query(means[m:], k=1)
    assert (distance < 6.0 * scales[m:]).all(), "padded copies drift away from the cloud"


def test_subsampling_draws_distinct_input_points():
    """Test that subsampling returns input points and draws each of them at most once."""
    m, n = 4000, 500
    xyz, rgb = _cloud(m)
    means = np.asarray(init_from_points(xyz, rgb, n, 0.1)["means"])
    distance, _ = KDTree(xyz).query(means, k=1)
    assert not distance.any(), "subsampled means are not input points"
    assert len(np.unique(means, axis=0)) == n


def test_colors_and_opacity_round_trip():
    """Test that the logit parameters decode back to the point colors and the given opacity."""
    xyz, rgb = _cloud(4000)
    p = init_from_points(xyz, rgb, 500, 0.37)
    _, index = KDTree(xyz).query(np.asarray(p["means"]), k=1)
    colors = expit(np.asarray(p["colors_logit"])) * 255.0
    assert np.abs(colors - rgb[index]).max() < 0.1, "colors do not follow their points"
    assert np.allclose(expit(np.asarray(p["opac_logit"])), 0.37)


def test_weights_bias_the_subsample():
    """Test that sampling weights pull the draw onto the points that carry them."""
    m, n, hot = 2000, 500, 200
    xyz, rgb = _cloud(m)
    weights = np.full(m, 1.0, np.float32)
    weights[:hot] = 1e4
    tree = KDTree(xyz)
    biased = init_from_points(xyz, rgb, n, 0.1, weights=weights)
    _, weighted = tree.query(np.asarray(biased["means"]))
    _, uniform = tree.query(np.asarray(init_from_points(xyz, rgb, n, 0.1)["means"]))
    assert np.mean(weighted < hot) > 2.0 * np.mean(uniform < hot)


def test_isolated_outlier_gets_a_bounded_scale():
    """Test that a point far outside the cloud is not seeded with a scene-sized gaussian."""
    xyz, rgb = _cloud(500)
    xyz = np.concatenate([xyz, np.array([[50.0, 50.0, 50.0]], np.float32)])
    rgb = np.concatenate([rgb, np.array([[1, 2, 3]], np.uint8)])
    p = init_from_points(xyz, rgb, 4000, 0.1)
    extent = np.ptp(xyz[:-1], axis=0).max()
    assert np.exp(np.asarray(p["log_scales"])).max() < 0.25 * extent
