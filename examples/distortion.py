"""Render a splat through a distorted lens next to its pinhole image.

The scene is rendered twice, once with an ideal pinhole camera and once with OpenCV distortion
coefficients, which the projection applies directly. Both go into one before/after PNG, pinhole on
the left.

Usage:
  python examples/distortion.py
  python examples/distortion.py --dist 0.4 0.2 0 0 0 --out pincushion.png
"""

from __future__ import annotations

import argparse
import logging
from functools import partial
from pathlib import Path

import imageio.v3 as iio
import jax
import jax.numpy as jnp
import numpy as np

import splax

logger = logging.getLogger(__name__)
EXAMPLES = Path(__file__).parent
BASE = "https://huggingface.co/datasets/amacati/splax-test-data/resolve/main"
RES = 800
DIRECTION = (1.0, -1.0, 0.55)
UP = (0.0, 0.0, 1.0)


def frame_camera(means: jax.Array) -> np.ndarray:
    """Look at the splat centre from ``DIRECTION``, close enough for the lens to bend the edges.

    Args:
        means: Gaussian centers, shape ``(N, 3)``.

    Returns:
        A ``(4, 4)`` world-to-camera matrix.
    """
    centre = np.asarray(means.mean(axis=0))
    radius = float(jnp.linalg.norm(means - centre, axis=-1).max())
    offset = np.asarray(DIRECTION, float)
    return splax.utils.look_at(centre + 1.25 * radius * offset / np.linalg.norm(offset), centre, UP)


def main(out: Path, fov: float, dist: tuple):
    splats = splax.io.load_ply(splax.io.fetch(f"{BASE}/scenes/lego.ply"))
    logger.info(f"loaded {splats[0].shape[0]} gaussians")

    focal = 0.5 * RES / np.tan(0.5 * np.deg2rad(fov))
    render = jax.jit(
        partial(
            splax.render,
            *splats,
            viewmat=jnp.asarray(frame_camera(splats[0])),
            background=jnp.ones(3),
            img_shape=(RES, RES),
            f=(focal, focal),
        ),
        static_argnames="dist",
    )

    img = jnp.concatenate([render()[0], render(dist=dist)[0]], axis=1)
    iio.imwrite(out, np.asarray(jnp.clip(img, 0.0, 1.0) * 255, np.uint8))
    logger.info(f"wrote {out}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=EXAMPLES / "distortion.png")
    parser.add_argument("--fov", type=float, default=60.0, help="horizontal field of view (deg)")
    parser.add_argument(
        "--dist",
        type=float,
        nargs=5,
        default=(-0.45, 0.15, 0.0, 0.0, 0.0),
        help="OpenCV distortion coefficients k1 k2 p1 p2 k3",
    )
    args = parser.parse_args()
    main(args.out, args.fov, tuple(args.dist))
