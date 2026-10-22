# IO

[`splax.io`][splax.io] reads and writes 3DGS `.ply` files. The stored fields are the unconstrained parameters described under [Rendering](rendering.md#inputs).

## Loading

[`splax.io.load_ply`][splax.io.load_ply] reads the vertex fields without additional processing and returns `(means, log_scales, quats, sh_colors, logit_opacities)` as float32 JAX arrays with shapes `(N, 3)`, `(N, 3)`, `(N, 4)`, `(N, K, 3)`, `(N,)`. It takes a path, so remote splats go through [`splax.io.fetch`][splax.io.fetch] first, which downloads into a local cache and revalidates it against the remote on later calls.

```python
import jax.numpy as jnp
import splax

SCENE = "https://huggingface.co/datasets/amacati/splax-test-data/resolve/main/scenes/lego.ply"
splats = splax.io.load_ply(splax.io.fetch(SCENE))
means, log_scales, quats, sh_colors, logit_opacities = splats
H, W, fx, fy = 400, 400, 400.0, 400.0
viewmat = splax.utils.look_at(jnp.array((0.0, -3.0, 1.0)), jnp.zeros(3), up=(0.0, 0.0, 1.0))
img, _ = splax.render(
    *splats, viewmat=viewmat, background=jnp.ones(3), img_shape=(H, W), f=(fx, fy)
)
```

| Array | `.ply` field |
|---|---|
| `means` | `x`, `y`, `z` |
| `log_scales` | `scale_0..2` |
| `quats` | `rot_0..3` |
| `sh_colors` | `f_dc_0..2` and `f_rest_*` |
| `logit_opacities` | `opacity` |

We always load the full SH coefficients. Render a lower degree by slicing, see [Spherical harmonics](rendering.md#spherical-harmonics).

## Writing

[`splax.io.write_ply`][splax.io.write_ply] stores the same five arrays without additional
processing.

<!-- notest: writes a .ply to disk -->
```{ .python notest }
splax.io.write_ply("out.ply", *splats)
```

## Pruning

Training leaves gaussians that do not contribute to the scene, i.e. strays, transparent ones, and oversized floaters. [`splax.io.prune`][splax.io.prune] removes all three,
with thresholds relative to the 99th percentile of the scene's distances and scales. The keywords
`radius`, `opacity`, and `scale` set how strict each one is.

```{ .python continuation }
splats = splax.io.prune(*splats)
```

## Reordering for rendering

Trained scenes usually store their gaussians in an order unrelated to 3D space.
[`splax.io.morton_sort`][splax.io.morton_sort] reorders them, so gaussians
close in space sit close in memory, which accelerates rendering.

```{ .python continuation }
splats = splax.io.morton_sort(*splats)
```

## Activated arrays

[`splax.project`][splax.project] and [`splax.rasterize`][splax.rasterize] consume activated arrays,
the linear scales, RGB colors, and `[0, 1]` opacities of [Rendering](rendering.md#inputs).
[`splax.io.apply_activations`][splax.io.apply_activations] and
[`splax.io.invert_activations`][splax.io.invert_activations] convert the scales and opacities,
[`splax.io.sh_to_rgb`][splax.io.sh_to_rgb] and [`splax.io.rgb_to_sh`][splax.io.rgb_to_sh] the base
color. Higher-order coefficients need a view direction, so they go through
[`splax.spherical_harmonics`][splax.spherical_harmonics].

```{ .python continuation }
scales, opacities = splax.io.apply_activations(log_scales, logit_opacities)
colors = splax.io.sh_to_rgb(sh_colors[:, 0])
log_scales, logit_opacities = splax.io.invert_activations(scales, opacities)
band0 = splax.io.rgb_to_sh(colors)
```
