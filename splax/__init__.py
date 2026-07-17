"""splax, an NVIDIA Warp gaussian-splatting pipeline for JAX.

Projection, rasterization, and their backward passes run as Warp kernels behind JAX FFI calls, so
scenes render fast and fit with jax.grad.

splax.inference.render is the grad-free forward without custom_vjp or blend residuals, the fast path
for serving a baked scene. splax.training.render is the differentiable forward. splax.render aliases
the differentiable version.
"""

__version__ = "0.1.0"

import os
import sys

# SciPy array API check. We use the most recent array API features, which require the
# SCIPY_ARRAY_API environment variable to be set to "1". This flag MUST be set before importing
# scipy, because scipy's C extensions cannot be unloaded once they have been imported. Therefore, we
# have to error out if the flag is not set. Otherwise, we immediately import scipy to ensure that no
# other package sets the flag to a different value before importing scipy.

if "scipy" in sys.modules and os.environ.get("SCIPY_ARRAY_API") != "1":
    msg = """scipy has already been imported and the 'SCIPY_ARRAY_API' environment variable has not
    been set. Please restart your Python session and set SCIPY_ARRAY_API="1" before importing any
    packages that depend on scipy, or import this package first to automatically set the flag."""
    raise RuntimeError(msg)

os.environ["SCIPY_ARRAY_API"] = "1"
import scipy  # noqa: F401, ensure scipy uses array API features

from splax import colmap, inference, io, mcmc, training
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
    "training",
    "colmap",
]
