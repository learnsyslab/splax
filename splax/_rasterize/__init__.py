"""Rasterization stage.

Both ``rasterize`` and ``rasterize_depth`` are JAX wrappers with custom jvps backed by Warp kernels.
"""

from splax._rasterize._rasterize import _rasterize_call, rasterize, rasterize_depth

__all__ = ["_rasterize_call", "rasterize", "rasterize_depth"]
