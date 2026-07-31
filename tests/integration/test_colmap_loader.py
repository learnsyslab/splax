"""Test the COLMAP reconstruction loader and the splat it initializes from the points."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from colmap import init_from_points, read_reconstruction
from scipy.spatial.transform import Rotation

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.colmap


def test_parsers_and_conventions(drone_sparse: Path):
    """Read a real reconstruction into cross-referencing cameras, images, and points."""
    cams, imgs, (xyz, rgb, ids, _) = read_reconstruction(drone_sparse)

    assert len(cams) >= 1 and len(imgs) > 0 and xyz.shape[0] > 0
    assert rgb.shape == xyz.shape and ids.shape == (xyz.shape[0],)
    # images sorted by name, every image references a known camera
    assert [im["name"] for im in imgs] == sorted(im["name"] for im in imgs)
    assert all(im["camera_id"] in cams for im in imgs)
    # per-image 2D observations: valid point ids reference known points
    known = set(int(p) for p in ids)
    obs = imgs[0]["obs_pid"]
    assert imgs[0]["obs_xy"].shape == (obs.shape[0], 2)
    assert obs.shape[0] == 0 or all(int(p) in known for p in obs[:50])


def test_pose_reprojects_the_observations(drone_sparse: Path):
    """Reproject an image's own observed points through its pose onto their 2D keypoints."""
    cams, imgs, (xyz, _, ids, _) = read_reconstruction(drone_sparse)
    image = imgs[0]
    fx, fy, cx, cy = cams[image["camera_id"]][3][:4]
    row = {int(pid): i for i, pid in enumerate(ids)}
    points = xyz[[row[int(pid)] for pid in image["obs_pid"]]]

    # The projection only closes under the scalar-first quaternion the loader documents. Reading it
    # scalar-last puts a tenth of the points behind the camera and the rest hundreds of pixels off.
    rotation = Rotation.from_quat(image["qvec"], scalar_first=True).as_matrix()
    camera = points @ rotation.T + image["tvec"]
    assert (camera[:, 2] > 0).all(), "an observed point sits behind the camera that observed it"
    pixels = np.stack([fx, fy]) * camera[:, :2] / camera[:, 2:] + np.stack([cx, cy])

    # The camera carries mild lens distortion this pinhole projection ignores, so the keypoints are
    # met to about a pixel rather than exactly.
    error = np.linalg.norm(pixels - image["obs_xy"], axis=1)
    assert np.median(error) < 1.0, f"median reprojection error {np.median(error):.2f} px"
    assert np.percentile(error, 95) < 5.0, f"95th percentile {np.percentile(error, 95):.2f} px"


def test_point_init_pads_a_real_reconstruction(drone_sparse: Path):
    """Keep every point of a real reconstruction and seed the padding beside them."""
    _, _, (xyz, rgb, _, _) = read_reconstruction(drone_sparse)
    points, n = xyz[:3000].astype(np.float32), 8000
    p = init_from_points(points, rgb[:3000], n, 0.1, seed=0)

    means = p["means"]
    assert means.shape == (n, 3), "the initializer did not pad up to the requested count"
    np.testing.assert_array_equal(means[: len(points)], points)
    assert np.all(np.isfinite(means)) and np.all(np.isfinite(p["log_scales"]))
