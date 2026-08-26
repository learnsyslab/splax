"""Test the spherical harmonics evaluation against the gsplat reference."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import numpy as np
import pytest

import splax

if TYPE_CHECKING:
    from types import ModuleType


@pytest.mark.gsplat
@pytest.mark.parametrize("degree", [0, 1, 2, 3])
def test_spherical_harmonics_vs_gsplat(degree: int, gsplat_shim: ModuleType):
    """Match the evaluated colors against gsplat at every degree."""
    k = jax.random.split(jax.random.key(degree), 2)
    coeffs = jax.random.normal(k[0], (4096, (degree + 1) ** 2, 3))
    directions = jax.random.normal(k[1], (4096, 3)) * 3.0
    colors = np.asarray(splax.spherical_harmonics(coeffs, directions))
    reference = gsplat_shim.spherical_harmonics(coeffs, directions, degree)
    np.testing.assert_allclose(colors, reference, rtol=0, atol=1e-5)
