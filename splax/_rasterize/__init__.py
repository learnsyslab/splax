"""Rasterization stage.

Both ``rasterize`` and ``rasterize_depth`` are JAX wrappers with custom vjps backed by Warp kernels.
"""

from splax._rasterize._rasterize import rasterize, rasterize_depth

__all__ = ["rasterize", "rasterize_depth"]
