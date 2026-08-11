"""Test that a batched training step reproduces the single-view steps it averages."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from train_colmap import init_exposure, make_step

if TYPE_CHECKING:
    from collections.abc import Hashable

pytestmark = pytest.mark.colmap

H = W = 48
INTRINSICS = (48.0, 48.0, 24.0, 24.0)
DIST = (0.0, 0.0, 0.0, 0.0, 0.0)
SSIM_LAMBDA, OPACITY_REG, SCALE_REG = 0.2, 0.01, 0.01


def _params(n: int = 200, seed: int = 0) -> dict[str, jax.Array]:
    """Draw a small splat in the parameter dict the training step optimizes."""
    k = jax.random.split(jax.random.key(seed), 5)
    return {
        "means": jax.random.uniform(k[0], (n, 3), minval=-0.6, maxval=0.6),
        "log_scales": jnp.full((n, 3), jnp.log(0.05)),
        "quats": jax.random.normal(k[1], (n, 4)),
        "colors_logit": jax.random.normal(k[2], (n, 3)) * 0.3,
        "opac_logit": jnp.full((n,), -1.0),
    }


def _view(seed: int) -> tuple[jax.Array, jax.Array]:
    """Draw a ground truth image and the camera pose it was taken from."""
    k = jax.random.split(jax.random.key(100 + seed), 2)
    gt = jax.random.uniform(k[0], (H, W, 3))
    vm = jnp.array(
        [[1, 0, 0, 0.1 * seed], [0, 1, 0, -0.05 * seed], [0, 0, 1, 4.0], [0, 0, 0, 1]], jnp.float32
    )
    return gt, vm


def _optimizer(
    params: dict[str, jax.Array], tx: optax.GradientTransformation
) -> optax.GradientTransformation:
    """Apply one transformation to every parameter of the splat."""
    txs: dict[Hashable, optax.GradientTransformation] = {k: tx for k in params}
    return optax.multi_transform(txs, {k: k for k in params})


def _dummy_pts(batch: int) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Build the empty depth supervision a step without a depth loss still takes."""
    return (
        jnp.zeros((batch, 1, 2), jnp.float32),
        jnp.zeros((batch, 1), jnp.float32),
        jnp.zeros((batch, 1), jnp.float32),
    )


def _updated(
    params: dict[str, jax.Array],
    gts: jax.Array,
    vms: jax.Array,
    pts: tuple[jax.Array, jax.Array, jax.Array],
    *,
    opacity_reg: float = OPACITY_REG,
    scale_reg: float = SCALE_REG,
    depth_loss: bool = False,
) -> dict[str, jax.Array]:
    """Run one plain gradient-descent step over a batch of views and return the new parameters."""
    batch = gts.shape[0]
    opt = _optimizer(params, optax.sgd(1.0))
    step = make_step(
        opt,
        H,
        W,
        INTRINSICS,
        DIST,
        SSIM_LAMBDA,
        opacity_reg,
        scale_reg,
        depth_loss=depth_loss,
        batch=batch,
    )
    bg = jnp.broadcast_to(jnp.ones(3), (batch, 3))
    new, _, _ = step(params, opt.init(params), gts, vms, bg, *pts)
    return new


def test_batch2_averages_the_single_view_updates():
    """Average the two single-view updates into the update a batch of both views applies."""
    params = _params(seed=2)
    gt0, vm0 = _view(1)
    gt1, vm1 = _view(2)

    first = _updated(params, gt0[None], vm0[None], _dummy_pts(1))
    second = _updated(params, gt1[None], vm1[None], _dummy_pts(1))
    batched = _updated(params, jnp.stack([gt0, gt1]), jnp.stack([vm0, vm1]), _dummy_pts(2))

    # Descent moves by minus the gradient, so the mean of the two updates is the update of the mean.
    mean = {key: 0.5 * (first[key] + second[key]) for key in params}
    np.testing.assert_allclose(batched["means"], mean["means"], rtol=2e-3, atol=1e-5)
    np.testing.assert_allclose(batched["log_scales"], mean["log_scales"], rtol=2e-3, atol=1e-5)
    np.testing.assert_allclose(batched["quats"], mean["quats"], rtol=2e-3, atol=1e-5)
    np.testing.assert_allclose(batched["colors_logit"], mean["colors_logit"], rtol=2e-3, atol=1e-5)
    np.testing.assert_allclose(batched["opac_logit"], mean["opac_logit"], rtol=2e-3, atol=1e-5)


def test_regularizers_shrink_only_their_own_parameter():
    """Push every entry of the parameter a regularizer names down, and leave the rest untouched."""
    params = _params(seed=7)
    gt, vm = _view(4)

    pts = _dummy_pts(1)
    off = _updated(params, gt[None], vm[None], pts, opacity_reg=0.0, scale_reg=0.0)
    on_opacity = _updated(params, gt[None], vm[None], pts, scale_reg=0.0)
    on_scale = _updated(params, gt[None], vm[None], pts, opacity_reg=0.0)

    # A regularizer penalizes monotonely, so descent lowers it on every gaussian
    assert (on_opacity["opac_logit"] < off["opac_logit"]).all(), "a gaussian gained opacity"
    np.testing.assert_array_equal(on_opacity["means"], off["means"])
    np.testing.assert_array_equal(on_opacity["log_scales"], off["log_scales"])
    np.testing.assert_array_equal(on_opacity["quats"], off["quats"])
    np.testing.assert_array_equal(on_opacity["colors_logit"], off["colors_logit"])

    assert (on_scale["log_scales"] < off["log_scales"]).all(), "a gaussian grew"
    np.testing.assert_array_equal(on_scale["means"], off["means"])
    np.testing.assert_array_equal(on_scale["quats"], off["quats"])
    np.testing.assert_array_equal(on_scale["colors_logit"], off["colors_logit"])
    np.testing.assert_array_equal(on_scale["opac_logit"], off["opac_logit"])


def test_depth_loss_is_gated_by_the_point_mask():
    """Fold the depth residual into the update only where the mask marks a valid observation."""
    params = _params(seed=8)
    gt, vm = _view(3)
    uv = jax.random.uniform(jax.random.key(11), (1, 4, 2)) * (H - 8) + 4
    target = jnp.full((1, 4), 4.0)

    plain = _updated(params, gt[None], vm[None], _dummy_pts(1))
    masked = partial(_updated, params, gt[None], vm[None], depth_loss=True)
    off = masked((uv, target, jnp.zeros((1, 4))))
    on = masked((uv, target, jnp.ones((1, 4))))

    # Masking every point out leaves the depth term nothing to say, so the update is the one the
    # photometric loss alone produces.
    np.testing.assert_array_equal(off["means"], plain["means"])
    np.testing.assert_array_equal(off["log_scales"], plain["log_scales"])
    np.testing.assert_array_equal(off["quats"], plain["quats"])
    np.testing.assert_array_equal(off["colors_logit"], plain["colors_logit"])
    np.testing.assert_array_equal(off["opac_logit"], plain["opac_logit"])
    # Admitting the points moves the geometry the depth is read off, and never the colors, since
    # the expected depth weights camera distances by visibility alone.
    assert (on["means"] != off["means"]).any(), "the depth residual never reached the means"
    assert (on["log_scales"] != off["log_scales"]).any(), "the depth residual never reached scale"
    assert (on["opac_logit"] != off["opac_logit"]).any(), "the depth residual never reached opacity"
    np.testing.assert_array_equal(on["colors_logit"], off["colors_logit"])


def test_step_drives_the_loss_down():
    """Reduce the photometric loss over a run of steps on a fixed pair of views."""
    params = _params(n=150, seed=3)
    gt0, vm0 = _view(1)
    gt1, vm1 = _view(2)
    gts, vms = jnp.stack([gt0, gt1]), jnp.stack([vm0, vm1])
    opt = _optimizer(params, optax.adam(3e-3))
    step = make_step(opt, H, W, INTRINSICS, DIST, SSIM_LAMBDA, OPACITY_REG, SCALE_REG, batch=2)

    state, bg = opt.init(params), jnp.broadcast_to(jnp.ones(3), (2, 3))
    losses = []
    for _ in range(10):
        params, state, l1 = step(params, state, gts, vms, bg, *_dummy_pts(2))
        losses.append(l1)

    assert jnp.isfinite(jnp.stack(losses)).all(), "the loss left the reals"
    assert losses[-1] < losses[0], f"l1 rose from {losses[0]:.4f} to {losses[-1]:.4f}"


def test_exposure_updates_only_the_referenced_views():
    """Move the exposure rows the batch referenced and leave every other row at identity."""
    params = _params(n=150, seed=6)
    B = 3
    opt = _optimizer(params, optax.sgd(1.0))
    opt_state = opt.init(params)
    exp_tx = optax.sgd(1.0)
    exp_p = {"exp": init_exposure(8)}
    exp_state = exp_tx.init(exp_p)
    step = make_step(
        opt,
        H,
        W,
        INTRINSICS,
        DIST,
        SSIM_LAMBDA,
        OPACITY_REG,
        SCALE_REG,
        depth_loss=True,
        aux_tx=exp_tx,
        exp_opt=True,
        batch=B,
    )
    gts = jax.random.uniform(jax.random.key(7), (B, H, W, 3))
    vms = jnp.broadcast_to(jnp.eye(4).at[2, 3].set(4.0), (B, 4, 4))
    bg = jnp.broadcast_to(jnp.ones(3), (B, 3))
    view_index = jnp.array([0, 3, 5], jnp.int32)
    uv = jax.random.uniform(jax.random.key(8), (B, 4, 2)) * 40 + 4
    depth = jnp.full((B, 4), 4.0)
    mask = jnp.ones((B, 4))
    _, _, new_exp, _, l1 = step(
        params, opt_state, exp_p, exp_state, gts, vms, bg, view_index, uv, depth, mask
    )
    assert jnp.isfinite(l1)
    # only the exposure rows the batch referenced moved, the rest stayed identity
    moved = jnp.abs(new_exp["exp"] - exp_p["exp"]).sum((1, 2)) > 0
    np.testing.assert_array_equal(moved, jnp.zeros(8, bool).at[view_index].set(True))
