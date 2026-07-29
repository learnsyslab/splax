# Training

`jax.grad` and `jax.value_and_grad` flow through `splax.render` with respect to
means, log scales, quats, SH colors, and logit opacities. The viewmat, background,
and rigid transforms are constants by default. Both returned outputs, the image and
the alpha, are differentiable.

```python
def loss(means, log_scales, quats, sh_colors, logit_opacities):
    img, _ = splax.render(
        means, log_scales, quats, sh_colors, logit_opacities,
        viewmat=viewmat, background=jnp.ones(3), img_shape=(H, W),
        f=(fx, fy),
    )
    return jnp.mean((img - target) ** 2)

grads = jax.grad(loss, argnums=(0, 1, 2, 3, 4))(*splats)
```

## Parameters

The five arrays are unconstrained, so a plain optimizer such as Adam updates them
directly. They are the fields a 3DGS `.ply` stores, so `splax.io.load_ply` resumes
from a checkpoint and `splax.io.write_ply` saves one, see [IO](io.md).

## Camera pose and object pose gradients

Gradient selection happens purely through `jax.grad` and its `argnums`, and only
the gradients actually requested are computed. Differentiating the `viewmat` alone,
as a pose optimizer does, costs only the camera gradient, and the gaussian gradients
are the same whether requested alone or together with it.

With [rigid transforms](rendering.md#dynamic-scene-composition) active, the `(K, 4, 4)`
transforms are differentiable too, so object poses can be optimized directly.

Because `viewmat` is a keyword argument of `render`, take its gradient by closing
over it in the differentiated position, for example:

```python
def loss(viewmat):
    img, _ = splax.render(*splats, viewmat=viewmat, background=bg, **cam)
    return photometric(img, target)

pose_grad = jax.grad(loss)(viewmat)  # runs the camera-pose accumulator only
```

## Depth channel

`render_depth=True` widens the returned image to `(H, W, 4)`, so `image[..., :3]`
is the RGB render and `image[..., 3]` the coverage-normalized expected depth
`D = sum_i w_i d_i / sum_i w_i`. The depth is a camera-space z along the optical axis rather
than a Euclidean range, and it reads `0` on pixels that no gaussian covers. It is
differentiable and routes a cotangent through the gaussian geometry and camera pose.
A three-channel render does not pay for it. This feeds COLMAP sparse-point depth
regularization.

```python
rgbd, alpha = splax.render(
    means, log_scales, quats, sh_colors, logit_opacities,
    viewmat=viewmat, background=jnp.ones(3), img_shape=(H, W),
    f=(fx, fy), render_depth=True,
)
img, depth = rgbd[..., :3], rgbd[..., 3]
confident_depth = depth * (alpha > 0.5)
```

## Coverage mask

`alpha` is the accumulated alpha `sum_i w_i` of shape `(H, W)`, returned by every entry
point and differentiable. Normalizing the expected depth by coverage means a pixel
grazed by a single faint gaussian still reports a full-magnitude depth, so `alpha` is
what separates a confident depth reading from a barely covered one. Threshold on it
before supervising on depth or exporting a point cloud.

## Antialiased mode

`antialiased=True` enables the Mip-Splatting opacity compensation described under
[Rendering](rendering.md#antialiased-mode). Its gradient chains back to scales,
quats, and means through the conic-to-covariance vjp.

## MCMC training utilities

`splax.mcmc` ports the fixed-budget MCMC strategy (Kheradmand et al. 2024) as
static-shape JAX ops, so a pipeline that needs fixed array shapes still gets
MCMC-style training without densification that grows `N`.

- `relocate` teleports dead low-opacity gaussians onto alive ones and corrects opacity and scale for the resulting multiplicity. It returns a reset mask marking rows whose optimizer moments to zero.
- `inject_noise` adds covariance- and opacity-weighted Gaussian noise to the means every step, so low-opacity gaussians random-walk to explore while high-opacity ones stay put.

## Trainer scripts

Two scripts under `scripts/` are reference training recipes.

- `scripts/train_lego.py` fits the synthetic NeRF lego scene, with per-parameter Adam schedules, an L1 plus D-SSIM loss, and progressive resolution fine-tuning.
- `scripts/train_colmap.py` fits any COLMAP sparse reconstruction, initializing the gaussians from its sparse point cloud. Depth regularization, per-image exposure correction, and batched training steps are available as opt-in flags.
