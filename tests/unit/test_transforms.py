"""Per-object rigid transform projection semantics and input validation.

The projection kernel applies a 4x4 world-space transform to a slice of the splat on the fly.
Checked here:

  1. Forward correctness against a manual reference that pre-transforms the slice's means and quats
     in JAX. The two formulations are mathematically equal but round differently, so projection
     outputs are compared tightly, with no radii or visibility flips.
  2. Invalid slices and mismatched shapes raise immediately.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from _transforms import kw, manual_move, scene
from scipy.spatial.transform import RigidTransform as TF
from scipy.spatial.transform import Rotation as R

import splax


@pytest.mark.unit
def test_projection_matches_manual_transform():
    """Kernel transform vs pre-transformed inputs, same projection outputs.

    The kernel rotates the covariance factor while the reference rotates the
    quaternion, mathematically equal with different rounding. Projected centers,
    depths, and conics must agree tightly, with zero radii or visibility flips.
    """
    n = 4000
    means, scales, quats, _colors, opac = scene(n, seed=2)
    camera = kw(128, 128)
    rot = R.from_euler("xyz", [0.26, -0.17, 0.52])
    T = TF.from_components((0.3, -0.2, 0.1), rot).as_matrix().astype(np.float32)
    tf_ids = jnp.full((n,), -1, jnp.int32).at[:1000].set(0)
    a = splax.project(
        means,
        scales,
        quats,
        camera["viewmat"],
        opacities=opac,
        img_shape=camera["img_shape"],
        f=camera["f"],
        c=camera["c"],
        gaussian_transforms=jnp.asarray(T)[None],
        transform_ids=tf_ids,
    )
    m2, q2 = manual_move(means, quats, T, 0, 1000)
    b = splax.project(
        m2,
        scales,
        q2,
        camera["viewmat"],
        opacities=opac,
        img_shape=camera["img_shape"],
        f=camera["f"],
        c=camera["c"],
    )

    ra, rb = np.asarray(a[2]).ravel(), np.asarray(b[2]).ravel()
    np.testing.assert_array_equal(ra > 0, rb > 0)
    live = ra > 0
    np.testing.assert_allclose(np.asarray(a[0])[live], np.asarray(b[0])[live], atol=5e-2)
    np.testing.assert_allclose(
        np.asarray(a[1]).ravel()[live], np.asarray(b[1]).ravel()[live], atol=1e-3
    )
    np.testing.assert_allclose(np.asarray(a[3])[live], np.asarray(b[3])[live], atol=1e-3)


@pytest.mark.unit
def test_invalid_transform_inputs_raise():
    n = 1000
    means, scales, quats, colors, opac = scene(n, seed=6)
    camera = kw(64, 64)
    eye = jnp.eye(4, dtype=jnp.float32)[None]

    with pytest.raises(ValueError, match="together"):
        splax.render(means, scales, quats, colors, opac, **camera, gaussian_transforms=eye)
    with pytest.raises(ValueError, match="does not match"):
        splax.render(
            means,
            scales,
            quats,
            colors,
            opac,
            **camera,
            gaussian_transforms=eye,
            gaussian_slices=((0, 100), (200, 300)),
        )
    with pytest.raises(ValueError, match="outside"):
        splax.render(
            means,
            scales,
            quats,
            colors,
            opac,
            **camera,
            gaussian_transforms=eye,
            gaussian_slices=((900, 1100),),
        )
    with pytest.raises(ValueError, match="overlap"):
        splax.render(
            means,
            scales,
            quats,
            colors,
            opac,
            **camera,
            gaussian_transforms=jnp.broadcast_to(eye[0], (2, 4, 4)),
            gaussian_slices=((0, 500), (400, 600)),
        )
