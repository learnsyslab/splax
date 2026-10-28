"""Nested jax.vmap for the Warp FFI callables.

Our Warp kernels only operate on one batch dimension. We therefore have to flatten nested vmap axes
into one flat batch. Furthermore, nested vmapping of an operand mapped over one axis but broadcast
over another collapses to a partial batch, which our kernels cannot handle. We therefore have to
tile the broadcasted operand to match the batch size.

Note that this is not equivalent to vmap_method="broadcast_all" for two reasons. First, we flatten
vmaps to make them compatible with our kernels. Second, and more importantly, we only tile arguments
that are at least vmapped over once. Unbatched arguments, i.e. the splat parameters, remain shared
across the batch, avoiding excessive memory allocation.

Kernels can also register their own vjp to allow for more efficient gradient computation and
accumulation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax.experimental.hijax import HiPrim, ShapedArray
from warp import JaxCallableGraphMode, jax_callable

if TYPE_CHECKING:
    from typing import Any, Callable

    Static = int | float | bool
    OutputDims = int | tuple[int, ...] | dict[str, int | tuple[int, int]]


def nested_vmap(ffi: Any, n_arrays: int, name: str) -> Callable:
    """Wrap a Warp jax_callable so nested jax.vmap flattens to its single-axis batch.

    Args:
        ffi: The jax_callable to wrap, called as ``ffi(*arrays, *statics, output_dims=...)``.
        n_arrays: Number of leading array operands. The remaining positional args are static.
        name: Name for the wrapping primitive.

    Returns:
        The wrapped callable. It optionally takes a ``vjp`` that maps the arrays, the cotangents,
        their batch mask and the statics to one gradient per array.
    """
    # Warp graphs only work for CUDA, so we fall back to no graph mode on the CPU
    cpu_ffi = jax_callable(
        ffi.func, num_outputs=ffi.num_outputs, graph_mode=JaxCallableGraphMode.NONE
    )

    def bind(
        *args: jax.Array | Static,
        output_dims: OutputDims,
        vjp: Callable | None = None,
        mask: tuple[bool, ...] | None = None,
        summed: tuple[bool, ...] | None = None,
    ) -> tuple[jax.Array, ...]:
        arrays, statics = args[:n_arrays], tuple(args[n_arrays:])
        mask = mask or (False,) * n_arrays
        avals = [jax.typeof(a) for a in arrays]
        b = next((a.shape[0] for a, m in zip(avals, mask) if m), None)
        out_dims = frozenset(output_dims.items()) if isinstance(output_dims, dict) else output_dims
        params = {
            "name": name,
            "mask": mask,
            "out_blocks": tuple(None if s else b for s in summed or (False,) * ffi.num_outputs),
            "_ffi": ffi,
            "_cpu_ffi": cpu_ffi,
            "_statics": statics,
            "_out_dims": out_dims,
            "_vjp": vjp,
        }
        return NestedVmap(avals, params)(*arrays)

    return bind


def _call(params: dict) -> Callable:
    """Return the FFI call of one application, in no graph mode on the CPU."""
    ffi, cpu_ffi, statics = params["_ffi"], params["_cpu_ffi"], params["_statics"]
    # dict output_dims -> frozenset for hashability. Warp reads by name so order doesn't matter.
    out_dims = params["_out_dims"]
    dims = dict(out_dims) if isinstance(out_dims, frozenset) else out_dims
    return lambda *arrays: jax.lax.platform_dependent(
        *arrays,
        cpu=lambda *arrays: tuple(cpu_ffi(*arrays, *statics, output_dims=dims)),
        default=lambda *arrays: tuple(ffi(*arrays, *statics, output_dims=dims)),
    )


class NestedVmap(HiPrim):
    """A primitive for accumulating nested batches into a single batch dimension for FFI calls."""

    mask: tuple[bool, ...]
    out_blocks: tuple[int | None, ...]  # Batch size of each output. None if unbatched
    _ffi: Any
    _statics: tuple[Static, ...]
    _vjp: Callable

    def __init__(self, avals: list[ShapedArray], params: dict):
        mask = params["mask"]
        specs = [
            jax.ShapeDtypeStruct(a.shape[1:] if m else a.shape, a.dtype)
            for a, m in zip(avals, mask)
        ]
        outs = jax.eval_shape(_call(params), *specs)
        self.in_avals = tuple(avals)
        self.out_aval = tuple(
            ShapedArray(o.shape if k is None else (k, *o.shape), o.dtype)
            for o, k in zip(outs, params["out_blocks"])
        )
        self.params = params
        super().__init__()

    def expand(self, *arrays: jax.Array) -> tuple[jax.Array, ...]:
        # The kernels see the batch flattened into the leading axis of each operand and output
        flat = [a.reshape(-1, *a.shape[2:]) if m else a for a, m in zip(arrays, self.mask)]
        shapes = [
            (o.shape[0] * o.shape[1], *o.shape[2:]) if k else o.shape
            for o, k in zip(self.out_aval, self.out_blocks)
        ]
        dims = {arg.name: s for arg, s in zip(self._ffi.output_args, shapes)}
        outs = _call({**self.params, "_out_dims": frozenset(dims.items())})(*flat)
        return tuple(o.reshape(a.shape) for o, a in zip(outs, self.out_aval))

    def batch(
        self, axis_data: Any, args: tuple[jax.Array, ...], dims: tuple[int | None, ...]
    ) -> tuple[tuple[jax.Array, ...], tuple[int | None, ...]]:
        if all(d is None for d in dims):  # early exit for vmap with no new batch dimensions
            return self(*args), (None,) * len(self.out_aval)
        moved = [a if d is None else jnp.moveaxis(a, d, 0) for a, d in zip(args, dims)]
        sz = next(m.shape[0] for m, d in zip(moved, dims) if d is not None)
        prior = any(self.mask)
        cur = 1
        if prior:  # We are inside a nested vmap, so we have to fold with the last batch size
            i = next(k for k, mk in enumerate(self.mask) if mk)
            cur = moved[i].shape[1] if dims[i] is not None else moved[i].shape[0]
        new_mask = tuple(mk or d is not None for mk, d in zip(self.mask, dims))
        flat = []
        for m, mk, d in zip(moved, self.mask, dims):
            if mk and d is not None:  # Has been folded before AND has a new batch dimension
                flat.append(m.reshape(sz * cur, *m.shape[2:]))
            elif mk:  # Has been folded before but has no new batch dimension, so we need to extend
                flat.append(jnp.tile(m, (sz, *(1,) * (m.ndim - 1))))
            elif d is not None:  # Not folded before, but has a new batch dimension.
                flat.append(jnp.repeat(m, cur, axis=0))  # Retroactively repeat across prior batches
            else:  # Not folded and no batch dimension, continue to share across the batch
                flat.append(m)
        # A new batch axis is added to every output. Either it's sz * k if we have a prior batch
        # with k elements, or it's sz if k is None.
        out_blocks = tuple(sz * (k or 1) for k in self.out_blocks)
        params = {**self.params, "mask": new_mask, "out_blocks": out_blocks}
        outs = NestedVmap([jax.typeof(a) for a in flat], params)(*flat)
        split = [
            o if k is None else o.reshape(sz, k, *o.shape[1:])
            for o, k in zip(outs, self.out_blocks)
        ]
        return tuple(split), (0,) * len(outs)

    def vjp_fwd(
        self, nzs_in: tuple[bool, ...], *arrays: jax.Array
    ) -> tuple[tuple[jax.Array, ...], tuple[jax.Array, ...]]:
        return self(*arrays), arrays

    def vjp_bwd_retval(
        self, arrays: tuple[jax.Array, ...], cotangents: tuple[jax.Array, ...]
    ) -> tuple[jax.Array, ...]:
        mask = (*self.mask, *(k is not None for k in self.out_blocks))
        return self._vjp(arrays, cotangents, mask, *self._statics)
