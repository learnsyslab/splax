# Rendering

`splax.render` is the rendering entry point. The call returns an `(image, alpha)` pair,
where `image` is the `(H, W, 3)` render and `alpha` the `(H, W)` accumulated coverage.
Gradients are covered under [Training](training.md).

```python
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

`render` takes unconstrained parameters, so an optimizer can update them directly.

| Argument | Shape | Meaning |
|---|---|---|
| `means3d` | `(N, 3)` | World positions |
| `log_scales` | `(N, 3)` | Log of the per-axis scales |
| `quats` | `(N, 4)` | wxyz quaternions |
| `sh_colors` | `(N, 3)` | Degree-0 SH color coefficients, `0` is mid grey |
| `logit_opacities` | `(N,)` | Opacity logits |

`splax.io.apply_activations` and `splax.io.invert_activations` convert to and from
linear scales, RGB colors, and `[0, 1]` opacities, see [IO](io.md#activated-arrays).

## Camera conventions

`viewmat` is a `(4, 4)` world-to-camera matrix in the OpenCV convention (+z
forward, +y down, +x right). This is what COLMAP stores directly. NeRF and
OpenGL poses (-z forward) must be converted first, which
`splax.utils.nerf_camera` does.

`f` is the focal length `(fx, fy)` in pixels and `c` is the principal point
`(cx, cy)` in pixels, where the optical axis meets the image plane. It defaults
to the image center `(W / 2, H / 2)`, which is exact for synthetic cameras.
Calibrated real cameras (COLMAP intrinsics) provide their own off-center values.
`img_shape` is `(H, W)`. `glob_scale` multiplies every gaussian scale, and
`clip_thresh` is the near-plane depth cutoff.

`img_shape`, `f`, and `c` size the kernel launch, so they are static under `jax.jit`, see
[Jitting](#jitting).

## Backgrounds

`background` is a 3-dimensional RGB color composited behind the splat where
transmittance remains. It is a constant and is not differentiated.

## Antialiased mode

`antialiased=True` applies the Mip-Splatting opacity compensation, cancelling the
area inflation that thin gaussians gain from the projection. Use the same setting at
inference that a model was trained with.

## Dynamic scene composition

Composed scenes can move whole sections of gaussians with rigid transforms, for
example a drone splat concatenated onto a room splat. `gaussian_transforms` is a
`(K, 4, 4)` stack of world-space transforms and `gaussian_slices` the `K`
matching non-overlapping `(start, stop)` index ranges. The gaussians in slice `k`
move by `gaussian_transforms[k]` and everything outside the slices stays static.
The splat is never copied.

```python
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

Batched dynamics work through `jax.vmap` over the transform stack. Every batch
element renders the same shared splat with its objects at different poses.

```python
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

Omitting both arguments renders the splat as one static scene. The transforms are
differentiable, see
[object pose gradients](training.md#camera-pose-and-object-pose-gradients).

## Jitting

`img_shape`, `f`, and `c` size the kernel launch and `gaussian_slices` indexes it, so all
four are static. Under `jax.jit` either declare them or close over them, otherwise the call
raises.

```python
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

Closing over them with `functools.partial` leaves the batched argument as the only input,
which is what `jax.vmap` maps over. Keyword arguments map along their leading axis.

```python
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

A static value is baked into the compiled kernel, so a resolution sweep or a change of the
slice layout compiles once per distinct value.

## Low-level primitives

`splax.render` composes two primitives that are public in their own right. They
consume activated arrays rather than parameters.

- `splax.project` maps gaussians to screen-space `(xys, depths, radii, conics, n_tiles_hit, cum_tiles_hit)`.
- `splax.rasterize` blends the projected gaussians into a `(H, W, 3)` image and its `(H, W)` alpha.
- `splax.rasterize_depth` blends into a `(H, W, 4)` image whose fourth channel is the expected depth, plus the same `(H, W)` alpha.

`splax.clear_cache` releases the scratch memory the backend holds between renders,
for example before switching to a very different workload size.
