"""Serve a splat scene with a flying drone through ``splax.viewer``.

The hall splat is uploaded once and stays static. The drone splat is flown around a circle every
frame with ``Viewer.update_pose``, which moves it without re-uploading its gaussians.

Usage:
  python examples/viewer_demo.py
  python examples/viewer_demo.py --radius 1.0 --height 1.5 --port 8080

Open http://localhost:8080 in a browser, then stop with Ctrl+C.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from typing import TYPE_CHECKING

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("SCIPY_ARRAY_API", "1")

from scipy.spatial.transform import Rotation

import splax
from splax.viewer import Viewer

if TYPE_CHECKING:
    import viser

logger = logging.getLogger(__name__)
BASE = "https://huggingface.co/datasets/amacati/splats/resolve/main"
# The hall spans roughly 39 x 12 x 7 m with its floor near z = 0, so the drone flies over
# the middle of it.
CENTRE = np.array([1.0, 0.5, 0.0])


def main(hall: str, drone: str, port: int, radius: float, height: float, freq: float):
    viewer = Viewer(port=port)
    logger.info(f"loading {hall}")
    viewer.add_splats("hall", *splax.io.load_ply(splax.io.fetch(f"{BASE}/{hall}.ply")))
    logger.info(f"loading {drone}")
    viewer.add_splats("drone", *splax.io.load_ply(splax.io.fetch(f"{BASE}/{drone}.ply")))

    # Focus on the drone
    focus = CENTRE + np.array([0.0, 0.0, height])
    stand = (radius + 0.6) / np.sqrt(2.0)

    @viewer.server.on_client_connect
    def _(client: viser.ClientHandle):
        client.camera.position = focus + np.array([stand, -stand, 0.25])
        client.camera.look_at = focus

    logger.info(f"viewer running at http://localhost:{port} -- Ctrl+C to stop")
    start = time.time()
    try:
        while time.time() - start < args.duration:
            angle = 2 * np.pi * freq * (time.time() - start)
            position = focus + radius * np.array([np.cos(angle), np.sin(angle), 0.0])
            # Yaw along the direction of travel, a quarter turn ahead of the orbit angle.
            wxyz = Rotation.from_euler("z", angle + np.pi / 2).as_quat(scalar_first=True)
            viewer.update_pose("drone", position, wxyz)
            time.sleep(1 / 30)
    except KeyboardInterrupt:
        viewer.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hall", default="robot_hall", help="static scene in the splat repository")
    parser.add_argument("--drone", default="cf21B_500", help="flying object in the repository")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--radius", type=float, default=0.8, help="flight circle radius (m)")
    parser.add_argument("--height", type=float, default=1.2, help="flight height (m)")
    parser.add_argument("--freq", type=float, default=0.1, help="circle frequency (Hz)")
    parser.add_argument("--duration", type=float, default=30.0, help="demo duration (s)")
    args = parser.parse_args()
    main(args.hall, args.drone, args.port, args.radius, args.height, args.freq)
