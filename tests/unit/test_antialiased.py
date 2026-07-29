"""Test the anti aliased opacity compensation factor in isolation."""

from __future__ import annotations

import numpy as np
import pytest
from _antialiased import pk, scene

import splax


@pytest.mark.unit
def test_compensation_closed_form():
    """ρ from the conic matches the direct det-ratio, bounded to [0,1], culled gaussians give 1."""
    n, H, W = 3000, 128, 128
    means, scales, quats, colors, opac, bg, vm = scene(n, H, W, seed=1)
    xys, depths, radii, conics, _nth, _cum = splax.project(
        means, scales, quats, vm, **pk(H, W), opacities=opac
    )
    rho = np.asarray(splax.opacity_compensation(conics, radii))
    conics = np.asarray(conics).reshape(n, 3)
    radii = np.asarray(radii).reshape(n)
    eps = 0.3
    # Reference: rebuild the dilated Σ₂D from the conic (= its inverse), strip the
    # ε dilation, take the det ratio directly.
    a, b, c = conics[:, 0], conics[:, 1], conics[:, 2]
    live = radii > 0
    det_conic = a * c - b * b
    det_d = np.where(live, 1.0 / np.where(det_conic == 0, 1.0, det_conic), 1.0)
    cxx, cyy, cxy = c * det_d, a * det_d, -b * det_d
    det_o = (cxx - eps) * (cyy - eps) - cxy * cxy
    ref = np.sqrt(np.clip(np.where(live, det_o / det_d, 1.0), 0.0, 1.0))
    assert np.allclose(rho[live], ref[live], atol=1e-5), (
        f"max |ρ - ref| = {np.abs(rho[live] - ref[live]).max():.2e}"
    )
    assert np.all(rho >= 0.0) and np.all(rho <= 1.0), "ρ must lie in [0, 1]"
    assert np.allclose(rho[~live], 1.0), "culled gaussians must get ρ = 1"
    # real gaussians actually get compensated (ρ meaningfully below 1 somewhere)
    assert rho[live].min() < 0.98, "expected some thin gaussians with ρ < 1"
