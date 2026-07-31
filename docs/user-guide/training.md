# Training

`jax.grad` and `jax.value_and_grad` flow through [`splax.render`][splax.render] with respect to
means, log scales, quats, SH colors, and logit opacities. The viewmat, background, and rigid
transforms are constants by default. Both returned outputs, the image and the alpha, are
differentiable.

We again first have to load a splat and prepare a view matrix:

```python
from functools import partial

import jax
import jax.numpy as jnp
import splax

SCENE = "https://huggingface.co/datasets/amacati/splax-test-data/resolve/main/scenes/lego.ply"
splats = splax.io.load_ply(splax.io.fetch(SCENE))
means, log_scales, quats, sh_colors, logit_opacities = splats
H, W, fx, fy = 400, 400, 400.0, 400.0
viewmat = splax.utils.look_at(jnp.array((0.0, -3.0, 1.0)), jnp.zeros(3), up=(0.0, 0.0, 1.0))
target = jnp.zeros((H, W, 3))  # your ground truth image
bg, cam = jnp.ones(3), {"img_shape": (H, W), "f": (fx, fy)}
```

```{ .python continuation }
def loss(means, log_scales, quats, sh_colors, logit_opacities):
    img, _ = splax.render(
        means,
        log_scales,
        quats,
        sh_colors,
        logit_opacities,
        viewmat=viewmat,
        background=jnp.ones(3),
        img_shape=(H, W),
        f=(fx, fy),
    )
    return jnp.mean((img - target) ** 2)


grads = jax.grad(loss, argnums=(0, 1, 2, 3, 4))(*splats)
```

## Parameters

The five arrays are unconstrained, so an optimizer can update them without projections. Since they
are the fields a 3DGS `.ply` stores, [`splax.io.load_ply`][splax.io.load_ply] resumes from a
checkpoint and [`splax.io.write_ply`][splax.io.write_ply] saves one without conversions.

## Camera pose and object pose gradients

Gradient selection happens via `jax.grad` and its `argnums`. One backward pass computes every
gradient and hands back the ones that were asked for.

With [rigid transforms](rendering.md#dynamic-scene-composition) active, the `(K, 4, 4)` transforms
are differentiable too, so object poses can be optimized directly.

Because `viewmat` is a keyword argument of [`render`][splax.render], take its gradient by closing
over it in the differentiated position, for example:

```{ .python continuation }
def loss(viewmat):
    img, _ = splax.render(*splats, viewmat=viewmat, background=bg, **cam)
    return jnp.mean((img - target) ** 2)


pose_grad = jax.grad(loss)(viewmat)
```

## Depth channel

`render_depth=True` widens the returned image to `(H, W, 4)`, so `image[..., :3]` is the RGB render
and `image[..., 3]` the expected depth

$$
D = \frac{\sum_i w_i d_i}{\sum_i w_i}
$$

as a camera-space z along the optical axis rather than a Euclidean range. It is differentiable, so a
depth loss reaches the gaussian geometry and the camera pose.

Normalizing by coverage means a pixel grazed by one faint gaussian still reports a full-magnitude
depth, and an uncovered pixel reads `0`. Mask with `alpha` before supervising on depth or exporting
a point cloud. Do not scale the depth by `alpha`, the values are already normalized.

```{ .python continuation }
rgbd, alpha = splax.render(
    means,
    log_scales,
    quats,
    sh_colors,
    logit_opacities,
    viewmat=viewmat,
    background=jnp.ones(3),
    img_shape=(H, W),
    f=(fx, fy),
    render_depth=True,
)
img, depth = rgbd[..., :3], rgbd[..., 3]
confident_depth = depth * (alpha > 0.5)
```

## Coverage mask

`alpha` is the accumulated alpha $\sum_i w_i$ of shape `(H, W)`, returned by every entry point and
differentiable. It is the coverage the gaussians contribute and reads `0` where none of them reach.

## Antialiased mode

`antialiased=True` enables the Mip-Splatting opacity compensation described under
[Rendering](rendering.md#antialiased-mode). Its gradient chains back to scales, quats, and means.

## MCMC training utilities

[`splax.mcmc`][splax.mcmc] ports the [fixed-budget MCMC strategy](https://arxiv.org/abs/2404.09591)
as static-shape JAX ops, so a pipeline that needs fixed array shapes still gets MCMC-style training
without densification that grows `N`.

- [`relocate`][splax.mcmc.relocate] teleports dead low-opacity gaussians onto alive ones and
  corrects opacity and scale for the resulting multiplicity. It returns a reset mask marking rows
  whose optimizer moments to zero.
- [`inject_noise`][splax.mcmc.inject_noise] adds covariance- and opacity-weighted Gaussian noise to
  the means every step, so low-opacity gaussians random-walk to explore while high-opacity ones stay
  put.

## Trainer scripts

Two scripts under `scripts/` are reference training recipes.

- `scripts/train_lego.py` fits the synthetic NeRF lego scene, with per-parameter Adam schedules, an L1 plus D-SSIM loss, and progressive resolution fine-tuning.
- `scripts/train_colmap.py` fits any COLMAP sparse reconstruction, initializing the gaussians from its sparse point cloud. Depth regularization, per-image exposure correction, and batched training steps are available as opt-in flags.
