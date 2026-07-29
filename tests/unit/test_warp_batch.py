"""Native batching of splax.project: jax.vmap over the pure-Warp projection.

``splax.project`` carries vmap_method="expand_dims" and launches a single grid over the whole
batch, so a vmapped call must equal ``jnp.stack`` of the per-element unbatched calls bit-exactly.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest
from _warp_batch import (
    PROJ_CAMERA,
    VIEWS,
    N,
    faithful_64bit_keys,  # noqa: F401 (autouse fixture)
    rand_scene,
)

import splax


@pytest.mark.unit
def test_project_vmap_over_viewmats():
    """vmap(project) over B viewmats: per-image outputs are bit-exact vs unbatched.

    All outputs except cum_tiles_hit are per-gaussian and batch-invariant, so they
    match the stacked unbatched projections exactly. cum_tiles_hit is intentionally a
    *global* inclusive prefix sum across the whole batch (gsplat's single-sort
    layout: every image's intersections are laid out contiguously), so it equals the
    global cumsum of the flattened n_tiles_hit rather than the per-image cumsum.
    """
    m, s, q, _c, o = rand_scene(N, seed=1)

    def f(vm: jax.Array) -> tuple:
        return splax.project(m, s, q, vm, opacities=o, **PROJ_CAMERA)

    B = VIEWS.shape[0]
    batched = jax.vmap(f)(VIEWS)
    # outputs 0..4 = xys, depths, radii, conics, n_tiles_hit, bit-exact per image
    for i in range(B):
        ref = f(VIEWS[i])
        for k in range(5):
            np.testing.assert_array_equal(np.asarray(batched[k][i]), np.asarray(ref[k]))
    # cum_tiles_hit (output 5) is the global inclusive scan of flattened n_tiles_hit
    nth = np.asarray(batched[4]).reshape(-1).astype(np.int64)
    cum = np.asarray(batched[5]).reshape(-1).astype(np.int64)
    np.testing.assert_array_equal(cum, np.cumsum(nth))
