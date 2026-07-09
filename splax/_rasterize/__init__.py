"""Rasterization stage.

The public API is re-exported here from ``splax._rasterize._rasterize``, the differentiable
rasterization module. The Warp kernels and their JAX FFI callables live in
``splax._rasterize._kernels``.
"""

from splax._rasterize._rasterize import _rasterize_call, rasterize, rasterize_depth

__all__ = ["_rasterize_call", "rasterize", "rasterize_depth"]
