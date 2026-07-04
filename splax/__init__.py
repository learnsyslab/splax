"""splax: NVIDIA Warp gaussian-splatting pipeline for JAX.

Projection, rasterization, and their backward passes (``jax.custom_vjp``) run as
Warp kernels behind JAX FFI calls, so scenes render fast and fit with
``jax.grad``. Two guaranteed, documented entry points share every Warp kernel
and differ only in JAX-level wrapping:

- ``splax.inference.render`` -- pure, grad-free forward. No custom_vjp, no blend
  residuals kept, tight O6 tile intersection, vmap-batchable. The fast path for
  serving a baked scene.
- ``splax.training.render`` -- differentiable forward (custom_vjp + residuals),
  the path ``scripts/train_lego.py`` fits with.

``splax.render`` aliases the **differentiable** ``splax.training.render``; new
inference-only code should prefer ``splax.inference.render`` for the guaranteed
grad-free contract.

``splax.project`` / ``splax.rasterize`` / ``splax.rasterize_depth`` are the
low-level custom_vjp primitives; ``splax.mcmc`` holds the fixed-budget MCMC
training ops and ``splax.io.write_ply`` exports fitted splats.
"""

__version__ = "0.1.0"

from splax import inference, mcmc, training
from splax._project import opacity_compensation, project
from splax._rasterize import clear_scratch, rasterize, rasterize_depth
from splax.io import load_ply, write_ply
from splax.training import render
from splax import distillation

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
