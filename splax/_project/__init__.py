"""Projection stage.

The public API is re-exported from ``splax._project._project``, the differentiable projection
module. The Warp kernels and their JAX FFI callables live in ``splax._project._kernels``.
"""

from splax._project._project import _project_call, opacity_compensation, project

__all__ = ["_project_call", "opacity_compensation", "project"]
