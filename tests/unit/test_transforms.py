"""Test the ability to apply rigid transforms to a slice of the splat."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial.transform import RigidTransform as TF
from scipy.spatial.transform import Rotation as R
from utils import VIEWMAT, camera, manual_move, scene, scene_params

import splax


def test_projection_matches_manual_transform():
    """Compare the kernel transform to pre-transformed inputs."""
    n = 4000
    rng = np.random.default_rng(0)
    means, scales, quats, _, opac, _ = scene(n, seed=1)
    rot = R.from_quat(rng.normal(size=(4,)))
    T = TF.from_components((0.3, -0.2, 0.1), rot).as_matrix().astype(np.float32)
    tf_ids = jnp.full((n,), -1, jnp.int32).at[:1000].set(0)
    jit_project = jax.jit(splax.project, static_argnames=("f", "c", "img_shape"))
    a = jit_project(
        means,
        scales,
        quats,
        VIEWMAT,
        opacities=opac,
        **camera(128, 128),
        gaussian_transforms=jnp.asarray(T)[None],
        transform_ids=tf_ids,
    )
    m2, q2 = manual_move(means, quats, T, 0, 1000)
    b = jit_project(m2, scales, q2, VIEWMAT, opacities=opac, **camera(128, 128))

    ra, rb = np.asarray(a[2]), np.asarray(b[2])
    np.testing.assert_array_equal(ra > 0, rb > 0)
    live = ra > 0
    np.testing.assert_allclose(np.asarray(a[0])[live], np.asarray(b[0])[live], atol=5e-2)
    np.testing.assert_allclose(np.asarray(a[1])[live], np.asarray(b[1])[live], atol=1e-3)
    np.testing.assert_allclose(np.asarray(a[3])[live], np.asarray(b[3])[live], atol=1e-3)


def test_invalid_transform_inputs_raise():
    """Test if inputs are properly validated."""
    means, log_scales, quats, sh_colors, logit_opac, bg = scene_params(1000, seed=1)
    kw = {"viewmat": VIEWMAT, "background": bg, **camera(64, 64)}
    eye = jnp.eye(4, dtype=jnp.float32)[None]

    with pytest.raises(ValueError, match="together"):
        splax.render(means, log_scales, quats, sh_colors, logit_opac, **kw, gaussian_transforms=eye)
    with pytest.raises(ValueError, match="does not match"):
        splax.render(
            means,
            log_scales,
            quats,
            sh_colors,
            logit_opac,
            **kw,
            gaussian_transforms=eye,
            gaussian_slices=((0, 100), (200, 300)),
        )
    with pytest.raises(ValueError, match="outside"):
        splax.render(
            means,
            log_scales,
            quats,
            sh_colors,
            logit_opac,
            **kw,
            gaussian_transforms=eye,
            gaussian_slices=((900, 1100),),
        )
    with pytest.raises(ValueError, match="overlap"):
        splax.render(
            means,
            log_scales,
            quats,
            sh_colors,
            logit_opac,
            **kw,
            gaussian_transforms=jnp.broadcast_to(eye[0], (2, 4, 4)),
            gaussian_slices=((0, 500), (400, 600)),
        )
