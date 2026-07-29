"""Backward-cost benchmark split by differentiation argument.

Measures the wall time of one forward plus backward render step when differentiating with respect to
the gaussian parameters, the camera viewmats, or the rigid transforms, across scenes and batch
sizes. Comparing two runs (one per implementation) shows how each derivative shifted. Writes
``reports/bench_derivatives_<impl>.json``. Run with:

    pixi run -e tests python benchmark/bench_derivatives.py --impl new
"""

from __future__ import annotations

import os
import timeit

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import json
import multiprocessing
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import warp as wp
from bench_forward import BATCHES, BUILDERS, ITERS, REPEAT, WARMUP

import splax

if TYPE_CHECKING:
    from collections.abc import Callable

    from bench_forward import Scene

OUT_DIR = Path(__file__).resolve().parents[1] / "reports"
DERIVS = ("gaussians", "cameras", "transforms")
N_SLICES = 4


def _target(sc: Scene, batch: int) -> jax.Array:
    """Fixed random target image stack the loss regresses against."""
    arr = np.random.default_rng(0).uniform(size=(batch, sc.res, sc.res, 3)).astype(np.float32)
    return jnp.asarray(arr)


def _slices(n: int) -> tuple[tuple[int, int], ...]:
    """Contiguous non-overlapping slices covering all gaussians, one transform each."""
    step = n // N_SLICES
    return tuple((i * step, (i + 1) * step if i < N_SLICES - 1 else n) for i in range(N_SLICES))


def make_step(sc: Scene, batch: int, deriv: str) -> Callable[[], object]:
    """Build a jitted forward-plus-backward step that differentiates one argument."""
    *params, background = sc.scene
    views = jnp.asarray(sc.viewmats[:batch])
    target = _target(sc, batch)
    camera = {"background": background, "img_shape": (sc.res, sc.res), "f": (sc.focal, sc.focal)}
    n = params[0].shape[0]

    def photo(imgs: jax.Array) -> jax.Array:
        return jnp.mean((imgs - target) ** 2)

    if deriv == "transforms":
        slices = _slices(n)
        tfs = jnp.broadcast_to(jnp.eye(4, dtype=jnp.float32), (len(slices), 4, 4))

        def loss(t: jax.Array, vms: jax.Array) -> jax.Array:
            rv = partial(splax.render, *params, gaussian_transforms=t, gaussian_slices=slices)
            imgs, _ = jax.vmap(lambda vm: rv(viewmat=vm, **camera))(vms)
            return photo(imgs)

        step = jax.jit(jax.value_and_grad(loss))
        return lambda: jax.block_until_ready(step(tfs, views))

    def loss(p: list[jax.Array], vms: jax.Array) -> jax.Array:
        imgs, _ = jax.vmap(partial(splax.render, *p, **camera))(viewmat=vms)
        return photo(imgs)

    argnums = 0 if deriv == "gaussians" else 1
    step = jax.jit(jax.value_and_grad(loss, argnums=argnums))
    return lambda: jax.block_until_ready(step(list(params), views))


def measure(sc: Scene, batch: int, deriv: str) -> dict:
    """Time one derivative at one batch and read the JAX plus Warp allocator peak."""
    call = make_step(sc, batch, deriv)
    for _ in range(WARMUP):
        call()
    ms = (min(timeit.Timer(call).repeat(repeat=REPEAT, number=ITERS)) / ITERS) * 1e3
    peak = jax.devices()[0].memory_stats()["peak_bytes_in_use"]
    peak += wp.get_mempool_used_mem_high(wp.get_device())
    return {"time_ms": ms, "peak_bytes": peak}


def run_scene(name: str) -> dict:
    """Sweep every batch and derivative for one scene in an isolated process."""
    sc = BUILDERS[name]()
    n = sc.scene[0].shape[0]
    print(f"\n== {sc.name}: {n:,} gaussians, {sc.res}x{sc.res} ==")
    print(f"{'batch':>6} " + " ".join(f"{d:>12}" for d in DERIVS))
    oom = dict.fromkeys(DERIVS, False)  # once a derivative OOMs, larger batches keep the NaN
    rows = []
    for batch in BATCHES:
        row: dict = {"batch": batch}
        for d in DERIVS:
            cell = {"time_ms": float("nan"), "peak_bytes": float("nan")}
            if not oom[d]:
                try:
                    cell = measure(sc, batch, d)
                except jax.errors.JaxRuntimeError:
                    oom[d] = True
            row[d] = cell
        rows.append(row)
        print(f"{batch:>6} " + " ".join(f"{row[d]['time_ms']:>10.3f}ms" for d in DERIVS))
    return {"name": sc.name, "n_gaussians": n, "res": sc.res, "rows": rows}


def main():
    """Run every scene in an isolated process and write the per-implementation JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--impl", required=True, help="implementation label, e.g. old or new")
    parser.add_argument("--scenes", nargs="+", default=list(BUILDERS), help="subset of scenes")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ctx = multiprocessing.get_context("spawn")
    scenes = []
    for name in args.scenes:
        with ctx.Pool(1) as pool:
            scenes.append(pool.apply(run_scene, (name,)))

    import jax as _jax

    data = {
        "meta": {
            "impl": args.impl,
            "generated": datetime.now(timezone.utc).isoformat(),
            "jax_version": _jax.__version__,
            "warp_version": wp.__version__,
            "warmup": WARMUP,
            "iters": ITERS,
            "batches": BATCHES,
            "derivatives": list(DERIVS),
            "n_slices": N_SLICES,
            "metric": "best-of-repeat mean of one forward plus backward step, milliseconds",
        },
        "scenes": scenes,
    }
    out = OUT_DIR / f"bench_derivatives_{args.impl}.json"
    out.write_text(json.dumps(data, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
