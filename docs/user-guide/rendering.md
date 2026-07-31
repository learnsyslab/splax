# Rendering

[`splax.render`][splax.render] is the rendering entry point. The call returns an `(image, alpha)`
pair, where `image` is the `(H, W, 3)` render and `alpha` the `(H, W)` accumulated coverage.
Gradients are covered under [Training](training.md).

We first load a splat and prepare a view matrix:

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
eyes = jnp.array([(0.0, -3.0, 1.0), (3.0, 0.0, 1.0)])
viewmats = splax.utils.look_at(eyes, jnp.zeros(3), up=(0.0, 0.0, 1.0))
slices = ((100, 1000), (1000, 1500))
poses = jnp.broadcast_to(jnp.eye(4), (len(slices), 4, 4))
pose_batch = jnp.broadcast_to(poses, (len(viewmats), len(slices), 4, 4))
```

```{ .python continuation }
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
)  # (H, W, 3)
```

## Inputs

[`render`][splax.render] takes unconstrained parameters, so an optimizer can update them directly.

| Argument | Shape | Meaning |
|---|---|---|
| `means3d` | `(N, 3)` | World positions |
| `log_scales` | `(N, 3)` | Log of the per-axis scales |
| `quats` | `(N, 4)` | wxyz quaternions |
| `sh_colors` | `(N, 3)` | Degree-0 SH color coefficients, `0` is mid grey |
| `logit_opacities` | `(N,)` | Opacity logits |

[`splax.io.apply_activations`][splax.io.apply_activations] and
[`splax.io.invert_activations`][splax.io.invert_activations] convert to and from linear scales, RGB
colors, and `[0, 1]` opacities, see [IO](io.md#activated-arrays).

## Camera conventions

`viewmat` is a `(4, 4)` world-to-camera matrix in the OpenCV convention (+z forward, +y down, +x
right), consistent with COLMAP's output files. NeRF and OpenGL poses (-z forward) must be converted
 first with [`splax.utils.nerf_camera`][splax.utils.nerf_camera].

`f` is the focal length `(fx, fy)` in pixels and `c` is the principal point `(cx, cy)` in pixels,
where the optical axis meets the image plane. It defaults to the image center `(W / 2, H / 2)`.
Calibrated real cameras, e.g. with COLMAP intrinsics, provide their own off-center values.
`img_shape` is `(H, W)`. `glob_scale` multiplies every gaussian scale, and `clip_thresh` is the
near-plane depth cutoff.

`img_shape`, `f`, and `c` size the kernel launch, so they are static under `jax.jit`, see
[Jitting](#jitting).

## Backgrounds

`background` is a 3-dimensional RGB color composited behind the splat where transmittance remains.
It is a constant and is not differentiated.

## Antialiased mode

`antialiased=True` applies the
[Mip-Splatting opacity compensation](https://arxiv.org/abs/2311.16493), cancelling the area
inflation that thin gaussians gain from the projection. Use the same setting at inference that a
model was trained with.

## Dynamic scene composition

Composed scenes can move whole sections of gaussians with rigid transforms to immitate moving
objects without copying the splats. `gaussian_transforms` is a `(K, 4, 4)` stack of world-space
transforms and `gaussian_slices` the `K` matching, non-overlapping `(start, stop)` index ranges. The
gaussians in slice `k` move by `gaussian_transforms[k]`. Everything outside the slices stays static.

```{ .python continuation }
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
    gaussian_transforms=poses,  # (K, 4, 4)
    gaussian_slices=((100, 1000), (1000, 1500)),
)
```

Batched dynamics work through `jax.vmap` over the transform stack. Every batch element renders the
same shared splat with its objects at different poses.

```{ .python continuation }
move = partial(
    splax.render,
    means,
    log_scales,
    quats,
    sh_colors,
    logit_opacities,
    viewmat=viewmat,
    background=jnp.ones(3),
    img_shape=(H, W),
    f=(fx, fy),
    gaussian_slices=slices,
)
imgs, _ = jax.vmap(move)(gaussian_transforms=pose_batch)  # (B, K, 4, 4) -> (B, H, W, 3)
```

[Examples](../examples.md#join-two-splats-and-move-one) has a runnable version that joins two splats
and orbits one of them.

Omitting both arguments renders the splat as one static scene. The transforms are differentiable,
see [object pose gradients](training.md#camera-pose-and-object-pose-gradients).

## Jitting

`img_shape`, `f`, and `c` size the kernel launch and `gaussian_slices` indexes it, so all four are
static. Under `jax.jit` either declare them or close over them, otherwise the call raises.

```{ .python continuation }
render_jit = jax.jit(splax.render, static_argnames=("img_shape", "f", "c"))
img, _ = render_jit(
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
```

Closing over them with `functools.partial` leaves the batched argument as the only input, which is
what `jax.vmap` maps over. Keyword arguments map along their leading axis.

```{ .python continuation }
render_at = partial(
    splax.render,
    means,
    log_scales,
    quats,
    sh_colors,
    logit_opacities,
    background=jnp.ones(3),
    img_shape=(H, W),
    f=(fx, fy),
)
imgs, _ = jax.jit(jax.vmap(render_at))(viewmat=viewmats)  # (B, H, W, 3)
```

A static value is baked into the compiled kernel, so a resolution sweep or a change of the slice
layout compiles once per distinct value.

## Low-level primitives

[`splax.render`][splax.render] composes two primitives that are public in their own right.

..Warning:: All low-level primitives consume **activated** arrays, not unconstrained parameters. Use
[`splax.io.apply_activations`][splax.io.apply_activations]/
[`splax.io.invert_activations`][splax.io.invert_activations] to convert the parameters.

- [`splax.project`][splax.project] maps gaussians to the 2D screen-space.
- [`splax.rasterize`][splax.rasterize] blends the projected gaussians into a `(H, W, 3)` image and
  its `(H, W)` alpha.
- [`splax.rasterize_depth`][splax.rasterize_depth] blends into a `(H, W, 4)` image whose fourth
  channel is the expected depth, plus the same `(H, W)` alpha.

Both rasterization primitives must be passed the same opacities [`splax.project`][splax.project] ran
on. Failing to do so will result in crashes or incorrect renderings. Rasterization takes an
`antialiased` keyword, so the compensation needs no separate call.

splax maintains a scratch memory pool for intermediate arrays used by the backend.
[`splax.clear_cache`][splax.clear_cache] releases this memory, for example before switching to a
different workload size.
