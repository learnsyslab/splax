"""Join two splats into one scene and drive one of them with a rigid transform.

Two copies of the lego splat are concatenated into a single set of arrays. The second copy is
declared as a movable slice, so it follows a pose while the first stays put. The splat is never
copied per frame, only the ``(1, 4, 4)`` transform changes.

Usage:
  python examples/compose_splats.py
  python examples/compose_splats.py --out orbit.gif --frames 60
"""

from __future__ import annotations

import argparse
import logging
import os
from functools import partial
from pathlib import Path

import imageio.v3 as iio
import jax
import jax.numpy as jnp
import numpy as np

os.environ.setdefault("SCIPY_ARRAY_API", "1")

from scipy.spatial.transform import RigidTransform, Rotation

import splax

logger = logging.getLogger(__name__)
EXAMPLES = Path(__file__).parent
BASE = "https://huggingface.co/datasets/amacati/splax-test-data/resolve/main"


def orbit_pose(centre: np.ndarray, offset: np.ndarray, angle: float) -> np.ndarray:
    """Yaw a splat about its own centre and carry it around by ``offset``.

    Args:
        centre: World position the splat rotates about, shape ``(3,)``.
        offset: World translation applied after the rotation, shape ``(3,)``.
        angle: Yaw about the world up axis, in radians.

    Returns:
        A ``(4, 4)`` world-space rigid transform.
    """
    to_origin = RigidTransform.from_translation(-centre)
    spin = RigidTransform.from_rotation(Rotation.from_euler("z", angle))
    return (RigidTransform.from_translation(centre + offset) * spin * to_origin).as_matrix()


def main(out: Path, res: int, fov: float, frames: int, distance: float, up: tuple):
    splat = splax.io.load_ply(splax.io.fetch(f"{BASE}/scenes/lego.ply"))
    n = splat[0].shape[0]
    centre = np.asarray(splat[0].mean(axis=0))
    radius = float(jnp.linalg.norm(splat[0] - centre, axis=-1).max())
    logger.info(f"loaded {n} gaussians, composing a scene of {2 * n}")

    # One splat holding both copies back to back. Gaussians [n, 2n) follow transform 0, the rest
    # stay static, so the two halves share every kernel launch.
    scene = tuple(jnp.concatenate([array, array]) for array in splat)
    slices = ((n, 2 * n),)

    reach = radius * (1.0 + distance)
    eye = centre + 2.5 * reach * np.array([1.0, -1.0, 0.5]) / np.linalg.norm([1.0, -1.0, 0.5])
    focal = 0.5 * res / np.tan(0.5 * np.deg2rad(fov))
    render = jax.jit(
        partial(
            splax.render,
            viewmat=jnp.asarray(splax.utils.look_at(eye, centre, up)),
            background=jnp.ones(3),
            img_shape=(res, res),
            f=(focal, focal),
            gaussian_slices=slices,
        )
    )

    images = []
    for angle in np.linspace(0.0, 2 * np.pi, frames, endpoint=False):
        offset = radius * distance * np.array([np.cos(angle), np.sin(angle), 0.0])
        pose = orbit_pose(centre, offset, angle)
        img, _ = render(*scene, gaussian_transforms=jnp.asarray(pose)[None])
        images.append(np.asarray(jnp.clip(img, 0.0, 1.0) * 255, np.uint8))
    iio.imwrite(out, images, duration=frames // 30, loop=0)
    logger.info(f"wrote {out} with {frames} frames")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=EXAMPLES / "compose.gif")
    parser.add_argument("--res", type=int, default=512)
    parser.add_argument("--fov", type=float, default=40.0, help="horizontal field of view (deg)")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--distance", type=float, default=1.5, help="orbit radius, in splat radii")
    parser.add_argument("--up", type=float, nargs=3, default=(0.0, 0.0, 1.0))
    args = parser.parse_args()
    main(args.out, args.res, args.fov, args.frames, args.distance, tuple(args.up))
