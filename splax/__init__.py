"""splax, an NVIDIA Warp gaussian-splatting pipeline for JAX.

Projection, rasterization, and their backward passes run as Warp kernels behind JAX FFI calls, so
scenes render fast and fit with jax.grad.

splax.render is the rendering entry point. It is differentiable with respect to the gaussian
parameters, the camera pose, and per-object rigid transforms.

Gaussians are held as unconstrained parameters, i.e. log scales, degree-0 SH colors, and logit
opacities. The kernel-facing primitives splax.project and splax.rasterize take the activated arrays
instead, and splax.io.apply_activations and splax.io.invert_activations convert between the two.
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

from splax import io, mcmc, utils
from splax._cache import clear_cache
from splax._project import opacity_compensation, project
from splax._rasterize import rasterize, rasterize_depth
from splax._render import render

__all__ = [
    "clear_cache",
    "opacity_compensation",
    "project",
    "rasterize",
    "rasterize_depth",
    "render",
    "mcmc",
    "io",
    "utils",
]
