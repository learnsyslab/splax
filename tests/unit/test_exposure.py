"""Test the per-image affine exposure correction used in training."""

from __future__ import annotations

import numpy as np
import pytest
from train_colmap import apply_exposure, init_exposure

pytestmark = pytest.mark.colmap


def test_init_exposure_blocks_leave_the_render_unchanged():
    """Test that the initial transforms carry one block per image and correct nothing."""
    ntr = 7
    blocks = init_exposure(ntr)
    assert blocks.shape == (ntr, 3, 4)
    np.testing.assert_array_equal(blocks, np.broadcast_to(blocks[0], blocks.shape))
    img = np.random.default_rng(0).random((5, 4, 3)).astype(np.float32)
    np.testing.assert_array_equal(apply_exposure(img, blocks[0]), img)


def test_apply_exposure_is_affine_in_the_image():
    """Test that the correction is affine in the render it transforms."""
    rng = np.random.default_rng(1)
    a, b = (rng.random((6, 3, 3)).astype(np.float32) for _ in range(2))
    affine, t = rng.normal(size=(3, 4)).astype(np.float32), np.float32(0.3)
    mixed = apply_exposure(t * a + (1.0 - t) * b, affine)
    parts = t * apply_exposure(a, affine) + (1.0 - t) * apply_exposure(b, affine)
    np.testing.assert_allclose(mixed, parts, atol=1e-5)


def test_apply_exposure_composes():
    """Test that two corrections in sequence equal the single correction they compose to."""
    rng = np.random.default_rng(3)
    img = rng.random((6, 3, 3)).astype(np.float32)
    first, second = (rng.normal(size=(3, 4)).astype(np.float32) for _ in range(2))
    matrix = second[:, :3] @ first[:, :3]
    offset = second[:, :3] @ first[:, 3] + second[:, 3]
    composed = np.concatenate([matrix, offset[:, None]], axis=1)
    sequential = apply_exposure(apply_exposure(img, first), second)
    np.testing.assert_allclose(sequential, apply_exposure(img, composed), atol=1e-5)


def test_apply_exposure_routes_input_channels_by_rows():
    """Test that row i of the transform selects what output channel i reads."""
    img = np.zeros((2, 2, 3), np.float32)
    img[..., 0] = 1.0  # a pure red render
    affine = np.zeros((3, 4), np.float32)
    affine[1, 0] = 1.0  # output green reads input red, everything else reads nothing
    np.testing.assert_array_equal(apply_exposure(img, affine), np.roll(img, 1, axis=-1))


def test_apply_exposure_offset_shifts_every_pixel():
    """Test that the offset column moves every pixel by the same constant."""
    rng = np.random.default_rng(6)
    img = rng.random((4, 5, 3)).astype(np.float32)
    offset = rng.normal(size=(3,)).astype(np.float32)
    affine = np.concatenate([np.eye(3, dtype=np.float32), offset[:, None]], axis=1)
    shift = apply_exposure(img, affine) - img
    np.testing.assert_allclose(shift, np.broadcast_to(offset, img.shape), atol=1e-6)
