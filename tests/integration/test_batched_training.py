"""Batched training step of ``scripts/train_colmap.py``.

``make_step`` vmaps a per-view loss over a static batch of views and averages it, so a batched step
must reproduce the mean of the single-view steps it stands in for. The gradients are read back out
of the real jitted step by giving it an SGD optimizer at learning rate 1, where the update is the
gradient itself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import dm_pix
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from train_colmap import init_exposure, make_step, render_args

from splax import render

if TYPE_CHECKING:
    from collections.abc import Hashable

pytestmark = pytest.mark.colmap

H = W = 48
INTRINSICS = (48.0, 48.0, 24.0, 24.0)
CAMERA = {"img_shape": (H, W), "f": INTRINSICS[:2], "c": INTRINSICS[2:]}
SSIM_LAMBDA, OPACITY_REG, SCALE_REG = 0.2, 0.01, 0.01


def _params(n: int = 200, seed: int = 0) -> dict[str, jax.Array]:
    k = jax.random.split(jax.random.key(seed), 5)
    return {
        "means": jax.random.uniform(k[0], (n, 3), minval=-0.6, maxval=0.6),
        "log_scales": jnp.full((n, 3), jnp.log(0.05)),
        "quats": jax.random.normal(k[1], (n, 4)),
        "colors_logit": jax.random.normal(k[2], (n, 3)) * 0.3,
        "opac_logit": jnp.full((n,), -1.0),
    }


def _view(seed: int) -> tuple[jax.Array, jax.Array]:
    k = jax.random.split(jax.random.key(100 + seed), 2)
    gt = jax.random.uniform(k[0], (H, W, 3))
    vm = jnp.array(
        [[1, 0, 0, 0.1 * seed], [0, 1, 0, -0.05 * seed], [0, 0, 1, 4.0], [0, 0, 0, 1]], jnp.float32
    )
    return gt, vm


def _sgd_opt(params: dict[str, jax.Array]) -> optax.GradientTransformation:
    # lr=1 SGD so apply_updates(p) = p - grad, so grad = p - step(p).
    txs: dict[Hashable, optax.GradientTransformation] = {kk: optax.sgd(1.0) for kk in params}
    return optax.multi_transform(txs, {kk: kk for kk in params})


def _dummy_pts(B: int) -> tuple[jax.Array, jax.Array, jax.Array]:
    return (
        jnp.zeros((B, 1, 2), jnp.float32),
        jnp.zeros((B, 1), jnp.float32),
        jnp.zeros((B, 1), jnp.float32),
    )


def _recover_grad(
    params: dict[str, jax.Array], batch: int, gts: jax.Array, vms: jax.Array
) -> dict[str, np.ndarray]:
    """Run one real make_step (SGD lr=1, no depth/exposure) and recover grad = p - new."""
    opt = _sgd_opt(params)
    opt_state = opt.init(params)
    step = make_step(opt, H, W, INTRINSICS, SSIM_LAMBDA, OPACITY_REG, SCALE_REG, batch=batch)
    bg = jnp.broadcast_to(jnp.ones(3), (batch, 3))
    new, _, _ = step(params, opt_state, gts, vms, bg, *_dummy_pts(batch))
    return {kk: np.asarray(params[kk] - new[kk]) for kk in params}


def test_batch1_matches_single_view():
    """Match the batch=1 gradient against the single-view loss written out by hand."""
    params = _params(seed=1)
    gt, vm = _view(3)

    def single_view_loss(p: dict[str, jax.Array]) -> jax.Array:
        img, _ = render(*render_args(p), viewmat=vm, background=jnp.ones(3), **CAMERA)
        l1 = jnp.mean(jnp.abs(img - gt))
        dssim = 1.0 - dm_pix.ssim(img, gt)
        loss = (1.0 - SSIM_LAMBDA) * l1 + SSIM_LAMBDA * dssim
        loss = loss + OPACITY_REG * jnp.mean(jax.nn.sigmoid(p["opac_logit"]))
        loss = loss + SCALE_REG * jnp.mean(jnp.exp(p["log_scales"]))
        return loss

    g_ref = jax.grad(single_view_loss)(params)
    g_b1 = _recover_grad(params, 1, gt[None], vm[None])
    for kk in params:
        assert np.allclose(np.asarray(g_ref[kk]), g_b1[kk], rtol=1e-4, atol=1e-6), kk


def test_batch2_grad_equals_mean_of_single_view_grads():
    """Average the two single-view gradients into the gradient of the batch=2 step."""
    params = _params(seed=2)
    gt0, vm0 = _view(1)
    gt1, vm1 = _view(2)

    g0 = _recover_grad(params, 1, gt0[None], vm0[None])
    g1 = _recover_grad(params, 1, gt1[None], vm1[None])
    gb = _recover_grad(params, 2, jnp.stack([gt0, gt1]), jnp.stack([vm0, vm1]))
    for kk in params:
        mean_g = 0.5 * (g0[kk] + g1[kk])
        assert np.allclose(gb[kk], mean_g, rtol=2e-3, atol=1e-5), kk


def test_batch2_of_one_view_twice_equals_batch1():
    """Collapse a batch of one repeated view onto the single-view gradient."""
    params = _params(seed=4)
    gt, vm = _view(5)
    g_b1 = _recover_grad(params, 1, gt[None], vm[None])
    g_b2 = _recover_grad(params, 2, jnp.stack([gt, gt]), jnp.stack([vm, vm]))
    for kk in params:
        assert np.allclose(g_b1[kk], g_b2[kk], rtol=2e-3, atol=1e-5), kk


def test_batched_step_runs_under_jit_with_exposure_and_depth():
    """Trace and run the batched step with the depth loss and the exposure optimizer enabled."""
    params = _params(n=150, seed=6)
    B = 3
    opt = _sgd_opt(params)
    opt_state = opt.init(params)
    exp_tx = optax.sgd(1.0)
    exp_p = {"exp": init_exposure(8)}
    exp_state = exp_tx.init(exp_p)
    step = make_step(
        opt,
        H,
        W,
        INTRINSICS,
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
    vi = jnp.array([0, 3, 5], jnp.int32)
    uv = jax.random.uniform(jax.random.key(8), (B, 4, 2)) * 40 + 4
    depth = jnp.full((B, 4), 4.0)
    mask = jnp.ones((B, 4))
    _, _, new_exp, _, l1 = step(
        params, opt_state, exp_p, exp_state, gts, vms, bg, vi, uv, depth, mask
    )
    assert np.isfinite(float(l1))
    # only the touched exposure rows (0,3,5) moved, the rest stayed identity.
    moved = np.abs(np.asarray(new_exp["exp"] - exp_p["exp"])).sum((1, 2)) > 0
    assert moved[[0, 3, 5]].all() and not moved[[1, 2, 4, 6, 7]].any()
