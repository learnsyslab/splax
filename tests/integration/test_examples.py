"""Test that the example scripts run end to end against the remote splats."""

from __future__ import annotations

import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pytest

from splax.io import fetch

EXAMPLES = Path(__file__).parents[2] / "examples"
TIMEOUT = 10.0
SERVE = 1.0

TEST_DATA = "https://huggingface.co/datasets/amacati/splax-test-data/resolve/main"
SPLATS = "https://huggingface.co/datasets/amacati/splats/resolve/main"
SOURCES = {"render_scene.py": TEST_DATA, "compose_splats.py": TEST_DATA, "viewer_demo.py": SPLATS}
ASSETS = (f"{TEST_DATA}/scenes/lego.ply", f"{SPLATS}/robot_hall.ply", f"{SPLATS}/cf21B_500.ply")


@pytest.fixture(scope="module", autouse=True)
def cached_assets():
    """Pull the splats the examples fetch into the cache before any of them is timed."""
    # The URLs live in the scripts, so a cold download would otherwise land inside TIMEOUT.
    for script, base in SOURCES.items():
        assert f'"{base}"' in (EXAMPLES / script).read_text(), f"{script} moved off {base}"
    for url in ASSETS:
        fetch(url)


def _run(script: str, *args: str) -> subprocess.CompletedProcess:
    """Run an example script to completion and return the finished process."""
    cmd = [sys.executable, str(EXAMPLES / script), *args]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT, check=False)


def test_render_scene_example(tmp_path: Path):
    """Render the lego scene through the example script."""
    out = tmp_path / "render.png"
    proc = _run("render_scene.py", "--out", str(out), "--res", "64")

    assert proc.returncode == 0, proc.stderr
    img = iio.imread(out)
    assert img.shape == (64, 64, 3)
    assert img.std() > 0.0, "the example wrote a blank image"
    out.unlink()


def test_compose_splats_example(tmp_path: Path):
    """Move one of two joined lego splats through the example script."""
    out = tmp_path / "compose.gif"
    proc = _run("compose_splats.py", "--out", str(out), "--res", "64", "--frames", "3")

    assert proc.returncode == 0, proc.stderr
    frames = iio.imread(out, index=None).astype(int)
    # Only the moved copy changes between frames. A static transform renders three identical ones,
    # which the GIF writer collapses into a single frame.
    assert frames.shape[0] == 3, "the moved slice never left its start pose"
    assert np.abs(frames[1] - frames[0]).max() > 0, "consecutive frames are identical"
    out.unlink()


def test_viewer_demo_example():
    """Serve the hall and the drone through the viewer example, then shut it down."""
    with socket.socket() as probe:  # let the OS pick a port that is free right now
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    cmd = [sys.executable, str(EXAMPLES / "viewer_demo.py"), "--port", str(port)]
    viewer = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    # The demo serves forever, so a watchdog closes its pipe and ends the scan if it never reports.
    watchdog = threading.Timer(TIMEOUT, viewer.kill)
    watchdog.start()
    try:
        # The line lands after both splats are loaded and uploaded, so it covers the whole demo.
        assert any("viewer running" in line for line in viewer.stdout), "the viewer never served"
        with socket.create_connection(("127.0.0.1", port), timeout=SERVE):
            time.sleep(SERVE)  # fly the drone for a moment before shutting down
        assert viewer.poll() is None, "the viewer died while serving"
    finally:
        watchdog.cancel()
        viewer.terminate()
        viewer.wait(timeout=10.0)
