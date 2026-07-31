"""Test the render quality of the pretrained lego scene against known reference values."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from utils import psnr

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
    """Hold the render quality of the pretrained lego splat at a held-out test pose."""
    splats = splax.io.load_ply(lego_ply)
    frame = lego_meta["frames"][frame_idx]
    gt = lego_view(frame["file_path"]).astype(np.float32) / 255.0
    H, W = gt.shape[:2]
    gt = gt[..., :3] * gt[..., 3:] + (1.0 - gt[..., 3:])  # composite on white
    focal = float(0.5 * W / np.tan(0.5 * lego_meta["camera_angle_x"]))
    viewmat = jnp.asarray(splax.utils.nerf_camera(frame["transform_matrix"]))
    render = jax.jit(
        partial(
            splax.render,
            *splats,
            background=jnp.ones(3),
            img_shape=(H, W),
            f=(focal, focal),
            c=(W // 2, H // 2),
        )
    )

    img = jnp.clip(render(viewmat=viewmat)[0], 0.0, 1.0)
    quality = psnr(img, gt)
    floor = KNOWN_PSNR[frame_idx] - SLACK
    assert quality >= floor, f"frame {frame_idx} PSNR {quality:.3f} dB below floor {floor:.3f}"
