# IO

`splax.io` reads and writes 3DGS `.ply` files. The stored fields are the
unconstrained parameters described under
[Rendering](rendering.md#inputs).

## Loading

`splax.io.load_ply` reads the vertex fields verbatim and returns
`(means, log_scales, quats, sh_colors, logit_opacities)` as float32 JAX arrays
with shapes `(N, 3)`, `(N, 3)`, `(N, 4)`, `(N, 3)`, `(N,)`.

```python
splats = splax.io.load_ply("scene.ply")
img, _ = splax.render(*splats, viewmat=viewmat, background=jnp.ones(3),
                      img_shape=(H, W), f=(fx, fy))
```

| Array | `.ply` field |
|---|---|
| `means` | `x`, `y`, `z` |
| `log_scales` | `scale_0..2` |
| `quats` | `rot_0..3` |
| `sh_colors` | `f_dc_0..2` |
| `logit_opacities` | `opacity` |

## Writing

`splax.io.write_ply` stores the same five arrays verbatim.

```python
splax.io.write_ply("out.ply", *splats)
```

A scene survives any number of load and write cycles unchanged. splax renders
spherical harmonics of degree 0 only, a single per-gaussian color, so normals are
written as zeros and the higher-order SH field `f_rest` is omitted.

## Activated arrays

`splax.project` and `splax.rasterize` consume activated arrays, the linear scales,
RGB colors, and `[0, 1]` opacities of [Rendering](rendering.md#inputs).

```python
scales, colors, opacities = splax.io.apply_activations(log_scales, sh_colors, logit_opacities)
log_scales, sh_colors, logit_opacities = splax.io.invert_activations(scales, colors, opacities)
```

The two are inverses in exact arithmetic. In float32 the scale and opacity
activations are lossy, so a value that has to survive repeated round trips belongs
in the parameters.
