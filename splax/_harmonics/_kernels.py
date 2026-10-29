"""Warp spherical harmonics kernels and their JAX FFI callables."""

from __future__ import annotations

import math

import warp as wp
from warp import JaxCallableGraphMode, jax_callable

from splax._batching import nested_vmap

wp.set_module_options({"fast_math": True})  # fastmath significantly accelerates the kernels

# Real spherical harmonics normalization constants up to degree 3.
C0 = 0.5 / math.sqrt(math.pi)
C1 = 0.5 * math.sqrt(3.0 / math.pi)
C2 = (
    0.5 * math.sqrt(15.0 / math.pi),
    -0.5 * math.sqrt(15.0 / math.pi),
    0.25 * math.sqrt(5.0 / math.pi),
    -0.5 * math.sqrt(15.0 / math.pi),
    0.25 * math.sqrt(15.0 / math.pi),
)
C3 = (
    -0.25 * math.sqrt(35.0 / (2.0 * math.pi)),
    0.5 * math.sqrt(105.0 / math.pi),
    -0.25 * math.sqrt(21.0 / (2.0 * math.pi)),
    0.25 * math.sqrt(7.0 / math.pi),
    -0.25 * math.sqrt(21.0 / (2.0 * math.pi)),
    0.25 * math.sqrt(105.0 / math.pi),
    -0.25 * math.sqrt(35.0 / (2.0 * math.pi)),
)

# Coefficient counts of degrees 0 to 3. The index into the tuple is the degree.
COEFFICIENTS = tuple((d + 1) ** 2 for d in range(4))


class _vec16(wp.types.vector(length=16, dtype=wp.float32)):
    """Basis values of all bands up to degree 3."""


class _mat16x3(wp.types.matrix(shape=(16, 3), dtype=wp.float32)):
    """Gradients of all coefficients up to degree 3."""


# region forward kernels


def _spherical_harmonics_warp(
    coeffs: wp.array2d[wp.vec3],
    directions: wp.array[wp.vec3],
    n_gaussians: int,
    # outputs
    colors: wp.array[wp.vec3],
):
    """Launch the color kernel, deriving the batch selectors from runtime shapes."""
    n = n_gaussians
    sel_coeffs = coeffs.shape[0] > n
    sel_dirs = directions.shape[0] > n
    inputs = [coeffs, directions, n, coeffs.shape[1], sel_coeffs, sel_dirs]
    wp.launch(_spherical_harmonics_kernel, dim=colors.shape[0], inputs=inputs, outputs=[colors])


spherical_harmonics_ffi = nested_vmap(
    jax_callable(_spherical_harmonics_warp, graph_mode=JaxCallableGraphMode.WARP),
    n_arrays=2,
    name="spherical_harmonics",
)


@wp.kernel
def _spherical_harmonics_kernel(
    coeffs: wp.array2d[wp.vec3],
    directions: wp.array[wp.vec3],
    n_gaussians: wp.int32,
    n_coeffs: wp.int32,
    sel_coeffs: wp.bool,
    sel_dirs: wp.bool,
    # outputs
    colors: wp.array[wp.vec3],
):
    idx = wp.tid()
    gid = idx % n_gaussians  # gaussian id. Used as index when the input is broadcast
    Y = _basis(wp.normalize(directions[wp.where(sel_dirs, idx, gid)]), n_coeffs)
    c_idx = wp.where(sel_coeffs, idx, gid)
    rgb = wp.vec3(0.5)
    for k in range(16):
        if k < n_coeffs:
            rgb += Y[k] * coeffs[c_idx, k]
    colors[idx] = wp.max(rgb, wp.vec3(0.0))


# region backward kernels


def _spherical_harmonics_bwd_warp(
    coeffs: wp.array2d[wp.vec3],
    directions: wp.array[wp.vec3],
    v_colors: wp.array[wp.vec3],
    n_gaussians: int,
    # outputs
    v_coeffs: wp.array2d[wp.vec3],
    v_directions: wp.array[wp.vec3],
):
    """Launch the color backward, deriving the batch selectors and summed views from shapes."""
    n = n_gaussians
    B = max(coeffs.shape[0], directions.shape[0], v_colors.shape[0]) // n
    # Each gradient sums a span of contiguous views, 1 for batched operands
    span_coeffs = B * n // v_coeffs.shape[0]
    span_dirs = B * n // v_directions.shape[0]
    span = max(span_coeffs, span_dirs)  # views per thread
    sel_coeffs = coeffs.shape[0] > n
    sel_dirs = directions.shape[0] > n
    sel_vcolors = v_colors.shape[0] > n
    inputs = [
        coeffs,
        directions,
        v_colors,
        n,
        coeffs.shape[1],
        span,
        span_coeffs,
        span_dirs,
        sel_coeffs,
        sel_dirs,
        sel_vcolors,
    ]
    wp.launch(
        _spherical_harmonics_bwd_kernel,
        dim=B // span * n,
        inputs=inputs,
        outputs=[v_coeffs, v_directions],
    )


spherical_harmonics_bwd_ffi = nested_vmap(
    jax_callable(
        _spherical_harmonics_bwd_warp, num_outputs=2, graph_mode=JaxCallableGraphMode.WARP
    ),
    n_arrays=3,
    name="spherical_harmonics_bwd",
)


@wp.kernel
def _spherical_harmonics_bwd_kernel(
    coeffs: wp.array2d[wp.vec3],
    directions: wp.array[wp.vec3],
    v_colors: wp.array[wp.vec3],
    n_gaussians: wp.int32,
    n_coeffs: wp.int32,
    span: wp.int32,
    span_coeffs: wp.int32,
    span_dirs: wp.int32,
    sel_coeffs: wp.bool,
    sel_dirs: wp.bool,
    sel_vcolors: wp.bool,
    # outputs
    v_coeffs: wp.array2d[wp.vec3],
    v_directions: wp.array[wp.vec3],
):
    tid = wp.tid()
    n = n_gaussians
    gid = tid % n
    sum_coeffs = _mat16x3()
    sum_dirs = wp.vec3(0.0)
    for view in range(tid // n * span, (tid // n + 1) * span):
        idx = view * n + gid
        d = directions[wp.where(sel_dirs, idx, gid)]
        inv_norm = 1.0 / wp.length(d)
        u = d * inv_norm
        Y = _basis(u, n_coeffs)
        # Recompute the color to find channels that were clipped
        c_idx = wp.where(sel_coeffs, idx, gid)
        rgb = wp.vec3(0.5)
        for k in range(16):
            if k < n_coeffs:
                rgb += Y[k] * coeffs[c_idx, k]
        g = v_colors[wp.where(sel_vcolors, idx, gid)]
        g = wp.vec3(
            wp.where(rgb[0] > 0.0, g[0], 0.0),
            wp.where(rgb[1] > 0.0, g[1], 0.0),
            wp.where(rgb[2] > 0.0, g[2], 0.0),
        )
        v_Y = _vec16()
        for k in range(16):
            if k < n_coeffs:
                sum_coeffs[k] += Y[k] * g
                v_Y[k] = wp.dot(coeffs[c_idx, k], g)
        v_u = _basis_vjp(u, v_Y, n_coeffs)
        sum_dirs += (v_u - u * wp.dot(u, v_u)) * inv_norm  # through the normalization
        if (view + 1) % span_coeffs == 0:
            for k in range(16):
                if k < n_coeffs:
                    v_coeffs[view // span_coeffs * n + gid, k] = sum_coeffs[k]
            sum_coeffs = _mat16x3()
        if (view + 1) % span_dirs == 0:
            v_directions[view // span_dirs * n + gid] = sum_dirs
            sum_dirs = wp.vec3(0.0)


# region helpers


@wp.func
def _basis(u: wp.vec3, n_coeffs: wp.int32) -> _vec16:
    """Evaluate the SH basis of the first ``n_coeffs`` coefficients at a unit direction."""
    x = u[0]
    y = u[1]
    z = u[2]
    Y = _vec16()
    Y[0] = wp.static(C0)
    Y[1] = -wp.static(C1) * y
    Y[2] = wp.static(C1) * z
    Y[3] = -wp.static(C1) * x
    if n_coeffs > 4:
        xx = x * x
        yy = y * y
        zz = z * z
        Y[4] = wp.static(C2[0]) * x * y
        Y[5] = wp.static(C2[1]) * y * z
        Y[6] = wp.static(C2[2]) * (2.0 * zz - xx - yy)
        Y[7] = wp.static(C2[3]) * x * z
        Y[8] = wp.static(C2[4]) * (xx - yy)
        if n_coeffs > 9:
            Y[9] = wp.static(C3[0]) * y * (3.0 * xx - yy)
            Y[10] = wp.static(C3[1]) * x * y * z
            Y[11] = wp.static(C3[2]) * y * (4.0 * zz - xx - yy)
            Y[12] = wp.static(C3[3]) * z * (2.0 * zz - 3.0 * xx - 3.0 * yy)
            Y[13] = wp.static(C3[4]) * x * (4.0 * zz - xx - yy)
            Y[14] = wp.static(C3[5]) * z * (xx - yy)
            Y[15] = wp.static(C3[6]) * x * (xx - 3.0 * yy)
    return Y


@wp.func
def _basis_vjp(u: wp.vec3, v_Y: _vec16, n_coeffs: wp.int32) -> wp.vec3:
    """Compute the gradient of the SH basis with respect to its direction."""
    x = u[0]
    y = u[1]
    z = u[2]
    v = wp.static(C1) * wp.vec3(-v_Y[3], -v_Y[1], v_Y[2])
    if n_coeffs > 4:
        xx = x * x
        yy = y * y
        zz = z * z
        v += v_Y[4] * wp.static(C2[0]) * wp.vec3(y, x, 0.0)
        v += v_Y[5] * wp.static(C2[1]) * wp.vec3(0.0, z, y)
        v += v_Y[6] * wp.static(C2[2]) * wp.vec3(-2.0 * x, -2.0 * y, 4.0 * z)
        v += v_Y[7] * wp.static(C2[3]) * wp.vec3(z, 0.0, x)
        v += v_Y[8] * wp.static(C2[4]) * wp.vec3(2.0 * x, -2.0 * y, 0.0)
        if n_coeffs > 9:
            v += v_Y[9] * wp.static(C3[0]) * wp.vec3(6.0 * x * y, 3.0 * (xx - yy), 0.0)
            v += v_Y[10] * wp.static(C3[1]) * wp.vec3(y * z, x * z, x * y)
            v += (
                v_Y[11]
                * wp.static(C3[2])
                * wp.vec3(-2.0 * x * y, 4.0 * zz - xx - 3.0 * yy, 8.0 * y * z)
            )
            v += (
                v_Y[12]
                * wp.static(C3[3])
                * wp.vec3(-6.0 * x * z, -6.0 * y * z, 6.0 * zz - 3.0 * xx - 3.0 * yy)
            )
            v += (
                v_Y[13]
                * wp.static(C3[4])
                * wp.vec3(4.0 * zz - 3.0 * xx - yy, -2.0 * x * y, 8.0 * x * z)
            )
            v += v_Y[14] * wp.static(C3[5]) * wp.vec3(2.0 * x * z, -2.0 * y * z, xx - yy)
            v += v_Y[15] * wp.static(C3[6]) * wp.vec3(3.0 * (xx - yy), -6.0 * x * y, 0.0)
    return v
