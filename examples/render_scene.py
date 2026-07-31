"""Render a splat from the splax test-data repository to an image.

The scene is downloaded into the local splax cache on first use and reused afterwards. The camera is
framed from the splat itself, so no per-scene metadata is needed.

Usage:
  python examples/render_scene.py
  python examples/render_scene.py --scene lego --out lego.png --res 800
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


def frame_camera(means: jax.Array, direction: tuple, up: tuple) -> np.ndarray:
    """Look at the splat centre from ``direction``, backed off far enough to frame all of it.

    Args:
        means: Gaussian centers, shape ``(N, 3)``.
        direction: World-space direction from the centre towards the camera.
        up: World up direction.

    Returns:
        A ``(4, 4)`` world-to-camera matrix.
    """
    centre = np.asarray(means.mean(axis=0))
    radius = float(jnp.linalg.norm(means - centre, axis=-1).max())
    offset = np.asarray(direction, float)
    return splax.utils.look_at(centre + 2.5 * radius * offset / np.linalg.norm(offset), centre, up)


def main(scene: str, out: Path, res: int, fov: float, direction: tuple, up: tuple):
    splats = splax.io.load_ply(splax.io.fetch(f"{BASE}/scenes/{scene}.ply"))
    logger.info(f"loaded {splats[0].shape[0]} gaussians from {scene}.ply")

    focal = 0.5 * res / np.tan(0.5 * np.deg2rad(fov))
    render = jax.jit(
        partial(splax.render, img_shape=(res, res), f=(focal, focal), background=jnp.ones(3))
    )

    img, _ = render(*splats, viewmat=jnp.asarray(frame_camera(splats[0], direction, up)))
    iio.imwrite(out, np.asarray(jnp.clip(img, 0.0, 1.0) * 255, np.uint8))
    logger.info(f"wrote {out}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="lego", help="scene name under scenes/ in the dataset")
    parser.add_argument("--out", type=Path, default=EXAMPLES / "render.png")
    parser.add_argument("--res", type=int, default=800)
    parser.add_argument("--fov", type=float, default=40.0, help="horizontal field of view (deg)")
    parser.add_argument("--direction", type=float, nargs=3, default=(1.0, -1.0, 0.6))
    parser.add_argument("--up", type=float, nargs=3, default=(0.0, 0.0, 1.0))
    args = parser.parse_args()
    main(args.scene, args.out, args.res, args.fov, tuple(args.direction), tuple(args.up))
