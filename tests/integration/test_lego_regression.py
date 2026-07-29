"""Render-quality regression gate on the pretrained lego scene.

The gate renders the pretrained lego splat of ~313k gaussians at three held-out test poses and
bounds the PSNR against the ground-truth images from below. It needs no external reference, splax
renders the scene itself. The splat, the poses, and the ground-truth views come from the
``lego_ply``, ``lego_meta`` and ``lego_view`` fixtures.

The poses arrive as NeRF camera-to-world matrices in the OpenGL convention with -z forward, which
``splax.utils.nerf_camera`` turns into the world-to-camera viewmat splax renders from. The focal
length follows from ``camera_angle_x``, the principal point sits at the image center, and the
ground truth is alpha-composited onto the white background the render uses.

The floors are the reference values of 30.89, 31.43 and 32.08 dB less a 0.05 dB slack for float32
blend-order jitter, which splax reproduces to better than 0.01 dB.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import pytest

import splax

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

# Frame index to the reference PSNR in dB at that held-out test pose.
KNOWN_PSNR = {0: 30.89, 25: 31.43, 50: 32.08}
SLACK = 0.05


@pytest.mark.parametrize("frame_idx", [0, 25, 50])
def test_lego_render_psnr_regression(
    frame_idx: int, lego_meta: dict, lego_view: Callable[[str], np.ndarray], lego_ply: Path
):
    splats = splax.io.load_ply(lego_ply)

    frame = lego_meta["frames"][frame_idx]
    gt = lego_view(frame["file_path"])
    H, W = gt.shape[:2]
    gt = gt.astype(np.float32) / 255.0
    gt = gt[..., :3] * gt[..., 3:] + (1.0 - gt[..., 3:])  # composite on white
    viewmat = splax.utils.nerf_camera(frame["transform_matrix"])

    ff = 0.5 * W / np.tan(0.5 * lego_meta["camera_angle_x"])
    img, _ = splax.render(
        *splats,
        viewmat=jnp.asarray(viewmat),
        background=jnp.ones(3),
        img_shape=(H, W),
        f=(float(ff), float(ff)),
        c=(W // 2, H // 2),
    )
    img = np.clip(np.asarray(img), 0.0, 1.0)
    psnr = -10.0 * np.log10(float(np.mean((img - gt) ** 2)))

    floor = KNOWN_PSNR[frame_idx] - SLACK
    assert psnr >= floor, f"frame {frame_idx} PSNR {psnr:.3f} dB below floor {floor:.3f}"
