"""Test basic properties of the MCMC relocation for training."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from splax import mcmc


def test_compute_relocation_ratio_one_is_identity():
    opac = jnp.array([1e-4, 1e-3, 0.01, 0.1, 0.5, 0.9, 0.999], jnp.float32)
    scales = jnp.tile(jnp.array([0.05, 0.1, 0.3], jnp.float32), (opac.shape[0], 1))
    binoms = mcmc.make_binoms(51)
    o, s = mcmc.compute_relocation(opac, scales, jnp.ones_like(opac), binoms)
    np.testing.assert_allclose(np.asarray(o), np.asarray(opac), rtol=1e-7)
    np.testing.assert_allclose(np.asarray(s), np.asarray(scales), rtol=1e-7)


def test_relocate_teleports_dead_onto_alive():
    n = 500
    k = jax.random.split(jax.random.key(1), 4)
    means = jax.random.uniform(k[0], (n, 3), minval=-1, maxval=1)
    log_scales = jnp.full((n, 3), jnp.log(0.05))
    quats = jax.random.normal(k[1], (n, 4))
    sh_colors = jax.random.normal(k[2], (n, 3))
    # first 100 dead (opacity ~0), rest alive (opacity ~0.7)
    opac_logit = jnp.concatenate([jnp.full((100,), -20.0), jnp.full((400,), 0.85)])

    binoms = mcmc.make_binoms(51)
    (new_means, _, _, _, new_opac_logit), reset = mcmc.relocate(
        k[3], means, log_scales, quats, sh_colors, opac_logit, binoms, min_opacity=0.005
    )

    # shapes are static
    assert new_means.shape == (n, 3)
    assert new_opac_logit.shape == (n,)
    # every dead gaussian was reset and now has opacity above the dead threshold
    reset = np.asarray(reset)
    assert reset[:100].all()
    new_opac = np.asarray(jax.nn.sigmoid(new_opac_logit))
    assert (new_opac[:100] > 0.005).all()
    # relocated means coincide with some alive source position
    alive_means = np.asarray(means[100:])
    new_means = np.asarray(new_means)
    dist = np.min(np.linalg.norm(alive_means[None, :, :] - new_means[:, None, :], axis=-1), axis=1)
    assert (dist < 1e-4).all(), "relocated means do not coincide with any alive source"


def test_inject_noise_respects_opacity():
    n = 400
    k = jax.random.split(jax.random.key(2), 2)
    means = jnp.zeros((n, 3))
    log_scales = jnp.full((n, 3), jnp.log(0.1))
    quats = jnp.tile(jnp.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
    # half near-transparent, half near-opaque
    opac_logit = jnp.concatenate([jnp.full((200,), -5.0), jnp.full((200,), 8.0)])
    moved = mcmc.inject_noise(k[0], means, log_scales, quats, opac_logit, scaler=100.0)
    disp = np.linalg.norm(np.asarray(moved), axis=1)
    # low-opacity gaussians move, high-opacity ones barely move
    assert disp[:200].mean() > 10 * disp[200:].mean() + 1e-6
    assert moved.shape == (n, 3)
