"""splax, an NVIDIA Warp gaussian-splatting pipeline for JAX.

Projection, rasterization, and their backward passes run as Warp kernels behind
JAX FFI calls, so scenes render fast and fit with jax.grad. Two entry points
share every Warp kernel and differ only in the JAX wrapping.

splax.inference.render is the grad-free forward without custom_vjp or blend
residuals, the fast path for serving a baked scene. splax.training.render is the
differentiable forward. splax.render aliases the differentiable one.

splax.project and splax.rasterize are the low-level custom_vjp primitives,
splax.mcmc holds the fixed-budget MCMC training ops, and splax.io reads and
writes 3DGS ply files.
"""

__version__ = "0.1.0"

from splax import distillation, inference, mcmc, training
from splax._intersect import clear_scratch
from splax._project import opacity_compensation, project
from splax._rasterize import rasterize, rasterize_depth
from splax.io import load_ply, write_ply
from splax.training import render

__all__ = [
    "clear_scratch",
    "distillation",
    "inference",
    "mcmc",
    "opacity_compensation",
    "project",
    "rasterize",
    "rasterize_depth",
    "load_ply",
    "render",
    "training",
    "write_ply",
]
