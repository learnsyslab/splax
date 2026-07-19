"""Fixed-budget MCMC training utilities.

This module ports the core mechanics of "3D Gaussian Splatting as Markov Chain Monte Carlo" by
Kheradmand et al. from gsplat to JAX, with one notable difference: compared to the original, we keep
the gaussian count fixed and do not densify the scene. This allows us to implement everything in
pure JAX.

``relocate`` teleports dead gaussians onto alive ones sampled in proportion to opacity. It corrects
opacity and scale so the gaussians that now share a location preserve the original contribution, and
reports which rows need their optimizer moments reset. ``inject_noise`` perturbs the means with
covariance and opacity weighted Gaussian noise every step, so low-opacity gaussians random-walk to
explore the scene while high-opacity ones stay put.

The relocation math matches gsplat's CUDA relocation kernel, and ``tests/test_mcmc.py`` checks
parity against a direct transcription of it.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from scipy.spatial.transform import Rotation as R

_EPS = jnp.finfo(jnp.float32).eps


def make_binoms(n: int) -> jax.Array:
    """Build the binomial coefficient lookup table ``b[n, k] = C(n, k)``.

    The table follows the gsplat convention, with zeros for ``k > n``.

    Args:
        n: Number of rows and columns of the table.

    Returns:
        Lookup table of shape ``(n, n)``.
    """
    b = [[math.comb(i, k) if k <= i else 0 for k in range(n)] for i in range(n)]
    return jnp.array(b, jnp.float32)


def compute_relocation(
    opacities: jax.Array, scales: jax.Array, ratios: jax.Array, binoms: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Compute the corrected opacity and scale for a relocated multiplicity.

    Each entry receives the opacity and scale that preserve the source gaussian's contribution when
    ``ratio`` gaussians share its location.

    Args:
        opacities: Activation-space opacities of the source gaussians, shape ``(N,)``.
        scales: Activation-space per-axis scales of the sources, shape ``(N, 3)``.
        ratios: Number of gaussians sharing each source's location, shape ``(N,)``.
        binoms: Binomial table from ``make_binoms``.

    Returns:
        Tuple of the new opacities with shape ``(N,)`` and the new scales with shape ``(N, 3)``.
    """
    n_max = binoms.shape[0]
    ratios = jnp.clip(jnp.round(ratios).astype(jnp.int32), 1, n_max)
    new_opac = 1.0 - (1.0 - opacities) ** (1.0 / ratios)
    k = jnp.arange(n_max, dtype=jnp.float32)
    sign_sqrt = ((-1.0) ** k) / jnp.sqrt(k + 1.0)
    cb = jnp.cumsum(binoms, axis=0)
    cbr = cb[ratios - 1]  # (N, n_max)
    powers = new_opac[:, None] ** (k + 1.0)
    denom = jnp.sum(cbr * sign_sqrt[None, :] * powers, axis=1)
    coeff = opacities / denom
    return new_opac, coeff[:, None] * scales


def relocate(
    key: jax.Array,
    means3d: jax.Array,
    log_scales: jax.Array,
    quats: jax.Array,
    logit_colors: jax.Array,
    logit_opacities: jax.Array,
    binoms: jax.Array,
    min_opacity: float = 0.005,
) -> tuple[tuple[jax.Array, ...], jax.Array]:
    """Relocate dead gaussians onto alive ones at a fixed gaussian count.

    A gaussian counts as dead when its activated opacity falls to ``min_opacity`` or below. Every
    gaussian samples a source among the alive ones with probability proportional to opacity. Dead
    gaussians copy the mean, quaternion, and color of their source and take the corrected opacity
    and scale from ``compute_relocation``, while a source chosen by dead gaussians has its own
    opacity and scale corrected for the resulting multiplicity.

    Args:
        key: PRNG key for the source sampling.
        means3d: Gaussian centers, shape ``(N, 3)``.
        log_scales: Log of the per-axis scales, shape ``(N, 3)``.
        quats: Rotations as wxyz quaternions, shape ``(N, 4)``.
        logit_colors: Color logits, shape ``(N, 3)``.
        logit_opacities: Opacity logits, one entry per gaussian.
        binoms: Binomial table from ``make_binoms``.
        min_opacity: Opacity threshold at or below which a gaussian counts as dead.

    Returns:
        Tuple of the new parameter arrays, in the same order as the inputs, and a boolean mask of
        shape ``(N,)`` marking the rows whose optimizer moments the caller should zero.
    """
    n = means3d.shape[0]
    opac = jax.nn.sigmoid(logit_opacities).reshape(n)
    scales = jnp.exp(log_scales)
    dead = opac <= min_opacity
    # categorical samples with p ~ exp(logits), so log-opacities sample proportional to opacity
    logits = jnp.where(dead, -jnp.inf, jax.nn.log_sigmoid(logit_opacities).reshape(n))
    src = jax.random.categorical(key, logits, shape=(n,))
    # counts[s] = number of dead gaussians that chose source s
    counts = jnp.zeros(n, jnp.float32).at[src].add(dead.astype(jnp.float32))
    source = jnp.where(dead, src, jnp.arange(n))  # dead pull from src, others self
    ratio = jnp.where(dead, counts[src], counts) + 1.0
    new_opac, new_scale = compute_relocation(opac[source], scales[source], ratio, binoms)
    new_opac = jnp.clip(new_opac, min_opacity, 1.0 - _EPS)
    new_logit_opac = jnp.log(new_opac / (1.0 - new_opac)).reshape(logit_opacities.shape)
    new_log_scales = jnp.log(new_scale)

    reset = dead | (counts > 0)  # rows that actually changed
    m = reset[:, None]
    out = (
        jnp.where(m, means3d[source], means3d),
        jnp.where(m, new_log_scales, log_scales),
        jnp.where(m, quats[source], quats),
        jnp.where(m, logit_colors[source], logit_colors),
        jnp.where(reset.reshape(logit_opacities.shape), new_logit_opac, logit_opacities),
    )
    return out, reset


def inject_noise(
    key: jax.Array,
    means3d: jax.Array,
    log_scales: jax.Array,
    quats: jax.Array,
    logit_opacities: jax.Array,
    scaler: float,
    min_opacity: float = 0.005,
) -> jax.Array:
    """Add covariance and opacity weighted Gaussian noise to the means.

    Each gaussian draws its noise from its own covariance and attenuates it with a steep sigmoid of
    the opacity, so gaussians with low opacity receive close to the full perturbation while opaque
    ones barely move. The caller anneals the strength through ``scaler``, typically as the means
    learning rate at the current step times a noise factor.

    Args:
        key: PRNG key for the noise.
        means3d: Gaussian centers, shape ``(N, 3)``.
        log_scales: Log of the per-axis scales, shape ``(N, 3)``.
        quats: Rotations as wxyz quaternions, shape ``(N, 4)``.
        logit_opacities: Opacity logits, one entry per gaussian.
        scaler: Global noise strength.
        min_opacity: Opacity the noise gate is centered on, matching ``relocate``'s dead threshold.

    Returns:
        The perturbed means, shape ``(N, 3)``.
    """
    opac = jax.nn.sigmoid(logit_opacities).reshape(means3d.shape[0])
    scales = jnp.exp(log_scales)
    rot = R.from_quat(quats, scalar_first=True).as_matrix()
    m = rot * scales[:, None, :]  # R diag(scale) with Sigma = m m^T
    op_sig = jax.nn.sigmoid(100.0 * (min_opacity - opac))
    noise = jax.random.normal(key, means3d.shape) * (op_sig * scaler)[:, None]
    noise = jnp.einsum("nij,nkj,nk->ni", m, m, noise)  # Sigma @ noise
    return means3d + noise
