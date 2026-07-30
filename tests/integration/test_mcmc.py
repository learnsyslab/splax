"""Parity of ``splax.mcmc.compute_relocation`` against gsplat's CUDA relocation kernel.

Both implement Eq. 9 of the MCMC paper. They agree to a tolerance rather than exactly, bounded by
gsplat's cancellation error in ``1 - (1 - o) ** (1 / n)``, which grows as opacity falls and
multiplicity rises. Ratios are integral, as both callers produce, and would diverge otherwise since
splax rounds where gsplat truncates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import numpy as np
import pytest

from splax import mcmc

if TYPE_CHECKING:
    from types import ModuleType

pytestmark = pytest.mark.gsplat

N_MAX = 51


def test_compute_relocation_vs_gsplat(gsplat_shim: ModuleType):
    """Match the corrected opacities and scales against the CUDA kernel."""
    rng = np.random.default_rng(0)
    n = 200
    opacities = rng.uniform(0.01, 0.99, n).astype(np.float32)
    scales = rng.uniform(0.01, 0.5, (n, 3)).astype(np.float32)
    ratios = rng.integers(1, 12, n).astype(np.float32)
    binoms = mcmc.make_binoms(N_MAX)

    new_opacities, new_scales = jax.jit(mcmc.compute_relocation)(opacities, scales, ratios, binoms)
    ref_opacities, ref_scales = gsplat_shim.relocation(opacities, scales, ratios, binoms)

    np.testing.assert_allclose(np.asarray(new_opacities), ref_opacities, rtol=2e-4, atol=1e-6)
    np.testing.assert_allclose(np.asarray(new_scales), ref_scales, rtol=2e-4, atol=1e-6)


def test_compute_relocation_saturates_at_n_max(gsplat_shim: ModuleType):
    """Match the kernel where the ratio is clamped to the binomial table's last row."""
    n = 64
    rng = np.random.default_rng(7)
    opacities = rng.uniform(0.05, 0.95, n).astype(np.float32)
    scales = rng.uniform(0.02, 0.4, (n, 3)).astype(np.float32)
    ratios = np.full(n, float(N_MAX + 10), np.float32)
    binoms = mcmc.make_binoms(N_MAX)

    new_opacities, new_scales = jax.jit(mcmc.compute_relocation)(opacities, scales, ratios, binoms)
    ref_opacities, ref_scales = gsplat_shim.relocation(opacities, scales, ratios, binoms)

    np.testing.assert_allclose(np.asarray(new_opacities), ref_opacities, rtol=2e-4, atol=1e-6)
    np.testing.assert_allclose(np.asarray(new_scales), ref_scales, rtol=2e-4, atol=1e-6)
