"""splax, an NVIDIA Warp gaussian-splatting pipeline for JAX.

Projection, rasterization, and their backward passes run as Warp kernels behind JAX FFI calls, so
scenes render fast and fit with jax.grad.

splax.inference.render is the grad-free forward without custom_vjp or blend residuals, the fast path
for serving a baked scene. splax.training.render is the differentiable forward. splax.render aliases
the differentiable version.
"""

__version__ = "0.1.0"

from splax import inference, io, mcmc
from splax._intersect import clear_scratch
from splax._project import opacity_compensation, project
from splax._rasterize import rasterize, rasterize_depth
from splax.training import render

__all__ = [
    "clear_scratch",
    "opacity_compensation",
    "project",
    "rasterize",
    "rasterize_depth",
    "render",
    "mcmc",
    "io",
    "inference",
]
