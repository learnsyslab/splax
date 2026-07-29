"""Test anti aliased rendering against the plain forward and against finite differences."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _antialiased import pk, scene

import splax

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.mark.integration
def test_antialiased_off_matches_plain():
    """Match the plain render when anti aliasing is off."""
    n, H, W = 2500, 110, 110
    means, scales, quats, colors, opac, bg, vm = scene(n, H, W, seed=2)
    kw = pk(H, W)

    off, _ = splax.render(
        means, scales, quats, colors, opac, viewmat=vm, background=bg, antialiased=False, **kw
    )
    plain, _ = splax.render(means, scales, quats, colors, opac, viewmat=vm, background=bg, **kw)
    assert np.array_equal(np.asarray(off), np.asarray(plain)), (
        "antialiased=False must be byte-identical to the plain forward"
    )

    # At the rasterize level, map_opacities=opac vs None gives byte-identical forward + grad.
    xys, depths, radii, conics, _nth, cum = splax.project(
        means, scales, quats, vm, **kw, opacities=opac
    )

    def rast(map_opac: jax.Array | None) -> Callable[[jax.Array], jax.Array]:
        def f(o: jax.Array) -> jax.Array:
            return jnp.mean(
                splax.rasterize(
                    colors,
                    o,
                    bg,
                    xys,
                    depths,
                    radii,
                    conics,
                    cum,
                    img_shape=(H, W),
                    map_opacities=map_opac,
                )
            )

        return f

    g_none = np.asarray(jax.grad(rast(None))(opac))
    g_self = np.asarray(jax.grad(rast(opac))(opac))
    assert np.allclose(g_none, g_self, rtol=2e-3, atol=1e-6), (
        f"map_opacities=opac vs None grad mismatch beyond atomic jitter "
        f"(max|d|={np.abs(g_none - g_self).max():.2e})"
    )


@pytest.mark.integration
def test_antialiased_changes_output():
    n, H, W = 2500, 110, 110
    means, scales, quats, colors, opac, bg, vm = scene(n, H, W, seed=3)
    kw = pk(H, W)
    off = np.asarray(
        splax.render(
            means, scales, quats, colors, opac, viewmat=vm, background=bg, antialiased=False, **kw
        )[0]
    )
    on = np.asarray(
        splax.render(
            means, scales, quats, colors, opac, viewmat=vm, background=bg, antialiased=True, **kw
        )[0]
    )
    assert np.abs(on - off).max() > 1e-3, "antialiased render must differ from plain"


@pytest.mark.integration
def test_antialiased_finite_difference():
    """Check anti aliased gradients with finite differences."""
    n, H, W = 400, 80, 80
    means, scales, quats, colors, opac, bg, vm = scene(n, H, W, seed=7)
    w = jax.random.uniform(jax.random.key(5), (H, W, 3))
    kw = pk(H, W)

    def loss(m: jax.Array, s: jax.Array, q: jax.Array, c: jax.Array, o: jax.Array) -> jax.Array:
        img, _ = splax.render(m, s, q, c, o, viewmat=vm, background=bg, antialiased=True, **kw)
        return jnp.mean(w * img)

    args = (means, scales, quats, colors, opac)
    grads = jax.grad(loss, argnums=(0, 1, 2, 3, 4))(*args)
    dirs = [g / (jnp.linalg.norm(g) + 1e-12) for g in grads]
    analytic = sum(float(jnp.vdot(g, d)) for g, d in zip(grads, dirs))

    eps = 2e-3
    plus = [a + eps * d for a, d in zip(args, dirs)]
    minus = [a - eps * d for a, d in zip(args, dirs)]
    numeric = (float(loss(*plus)) - float(loss(*minus))) / (2 * eps)
    rel = abs(analytic - numeric) / (abs(numeric) + 1e-12)
    assert rel < 8e-2, (
        f"antialiased FD mismatch: analytic {analytic:.6e} vs numeric {numeric:.6e} (rel {rel:.2e})"
    )
