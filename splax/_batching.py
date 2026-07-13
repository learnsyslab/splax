"""Nested jax.vmap for the Warp FFI callables.

Warp's jax_callable batches a single leading axis (vmap_method="expand_dims"). Under nested vmap an
operand mapped over one axis but broadcast over another collapses to a partial batch, so the kernel
launches A*B images and reads it out of bounds. nested_vmap wraps a callable in a primitive whose
batching rule folds every vmap axis into one flat batch (operands mapped over no axis stay shared),
then lowers to a single vmap. Wrapping every projection and rasterization FFI gives project,
rasterize, render, and their gradients nested vmap from one place.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax.extend.core import Primitive
from jax.interpreters import batching, mlir

if TYPE_CHECKING:
    from typing import Callable

    Static = int | float | bool
    OutputDims = int | tuple[int, ...] | dict[str, int | tuple[int, int]]
    FrozenDims = int | tuple[int, ...] | frozenset[tuple[str, int | tuple[int, int]]]


def _tile(x: jax.Array, sz: int, cur: int) -> jax.Array:
    base = x.shape[2:]
    return jnp.broadcast_to(x, (sz, cur, *base)).reshape(sz * cur, *base)


def nested_vmap(ffi: Callable, n_arrays: int, name: str) -> Callable:
    """Wrap a Warp jax_callable so nested jax.vmap flattens to its single-axis batch.

    Args:
        ffi: The jax_callable to wrap, called as ``ffi(*arrays, *statics, output_dims=...)``.
        n_arrays: Number of leading array operands. The remaining positional args are static.
        name: Name for the wrapping primitive.
    """
    prim = Primitive(name)
    prim.multiple_results = True

    def call(statics: tuple[Static, ...], out_dims: FrozenDims) -> Callable:
        # dict output_dims -> frozenset for a hashable param. Warp reads by name so order is moot
        dims = dict(out_dims) if isinstance(out_dims, frozenset) else out_dims
        return lambda *arrays: tuple(ffi(*arrays, *statics, output_dims=dims))

    def impl(
        *arrays: jax.Array,
        mask: tuple[bool, ...],
        statics: tuple[Static, ...],
        out_dims: FrozenDims,
    ) -> tuple[jax.Array, ...]:
        fn = call(statics, out_dims)
        if not any(mask):
            return fn(*arrays)
        return jax.vmap(fn, in_axes=tuple(0 if m else None for m in mask))(*arrays)

    def abstract(
        *avals: jax.core.ShapedArray,
        mask: tuple[bool, ...],
        statics: tuple[Static, ...],
        out_dims: FrozenDims,
    ) -> tuple[jax.core.ShapedArray, ...]:
        specs = [
            jax.ShapeDtypeStruct(a.shape[1:] if m else a.shape, a.dtype)
            for a, m in zip(avals, mask)
        ]
        b = next((a.shape[0] for a, m in zip(avals, mask) if m), None)
        outs = jax.eval_shape(call(statics, out_dims), *specs)
        return tuple(
            jax.core.ShapedArray((b, *o.shape) if b is not None else o.shape, o.dtype) for o in outs
        )

    def batch(
        args: tuple[jax.Array, ...],
        dims: tuple[int | None, ...],
        *,
        mask: tuple[bool, ...],
        statics: tuple[Static, ...],
        out_dims: FrozenDims,
    ) -> tuple[tuple[jax.Array, ...], tuple[int, ...]]:
        moved = [a if d is None else jnp.moveaxis(a, d, 0) for a, d in zip(args, dims)]
        sz = next(m.shape[0] for m, d in zip(moved, dims) if d is not None)
        prior = any(mask)
        cur = 1
        if prior:
            i = next(k for k, mk in enumerate(mask) if mk)
            cur = moved[i].shape[1] if dims[i] is not None else moved[i].shape[0]
        new_mask = tuple(bool(mk) or d is not None for mk, d in zip(mask, dims))
        flat = []
        for m, mk, d in zip(moved, mask, dims):
            if mk and d is not None:
                flat.append(m.reshape(sz * cur, *m.shape[2:]))
            elif mk:
                flat.append(_tile(m[None], sz, cur))
            elif d is not None:
                flat.append(_tile(m[:, None], sz, cur))
            else:
                flat.append(m)
        outs = prim.bind(*flat, mask=new_mask, statics=statics, out_dims=out_dims)
        split = lambda o: o.reshape(sz, cur, *o.shape[1:]) if prior else o.reshape(sz, *o.shape[1:])  # noqa: E731
        return tuple(split(o) for o in outs), (0,) * len(outs)

    prim.def_impl(impl)
    prim.def_abstract_eval(abstract)
    mlir.register_lowering(prim, mlir.lower_fun(impl, multiple_results=True))
    batching.primitive_batchers[prim] = batch

    def wrapped(*args: jax.Array | Static, output_dims: OutputDims) -> list[jax.Array]:
        arrays, statics = args[:n_arrays], tuple(args[n_arrays:])
        out_dims = frozenset(output_dims.items()) if isinstance(output_dims, dict) else output_dims
        return prim.bind(*arrays, mask=(False,) * n_arrays, statics=statics, out_dims=out_dims)

    return wrapped
