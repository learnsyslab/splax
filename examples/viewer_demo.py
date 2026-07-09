"""Minimal ``splax.viewer`` demo with a static scene splat plus a moving object splat.

The scene splat is uploaded once and stays static. The object splat is moved along a circle every
frame via ``Viewer.update_pose``.

Usage:
  python examples/viewer_demo.py
  python examples/viewer_demo.py --scene data/scenes/room.ply \
      --object data/scenes/drone.ply --port 8080

Open http://localhost:8080 in a browser, then stop with Ctrl+C.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("SCIPY_ARRAY_API", "1")

from scipy.spatial.transform import Rotation as R

import splax
from splax.viewer import Viewer

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).parents[1]


def main(scene: Path, obj: Path, port: int, radius: float, height: float, freq: float):

    viewer = Viewer(port=port)
    logger.info(f"Loading scene splat from {scene}")
    viewer.add_splats("scene", *splax.io.load_ply(scene))
    logger.info(f"Loading object splat from {obj}")
    viewer.add_splats("object", *splax.io.load_ply(obj), position=(radius, 0.0, height))
    logger.info(f"Viewer running at http://localhost:{port} -- Ctrl+C to stop")

    t_start = time.time()
    try:
        while True:
            angle = 2 * np.pi * freq * (time.time() - t_start)
            pos = (radius * np.cos(angle), radius * np.sin(angle), height)
            # Yaw along the direction of travel: rotation of angle + pi/2 around +z (wxyz).
            quat = R.from_euler("z", angle + np.pi / 2).as_quat(scalar_first=True)
            viewer.update_pose("object", pos, quat)
            time.sleep(1 / 30)
    except KeyboardInterrupt:
        viewer.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=REPO_ROOT / "data/scenes/room.ply")
    parser.add_argument("--object", type=Path, default=REPO_ROOT / "data/scenes/drone.ply")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--radius", type=float, default=1.0, help="Circle radius (m)")
    parser.add_argument("--height", type=float, default=1.0, help="Flight height (m)")
    parser.add_argument("--freq", type=float, default=0.1, help="Circle frequency (Hz)")
    args = parser.parse_args()
    main(args.scene, args.object, args.port, args.radius, args.height, args.freq)
