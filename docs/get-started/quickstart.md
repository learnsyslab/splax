# Quickstart

This page renders a scene, batches over cameras, and takes a gradient. All three share the same five
gaussian arrays: `means` `(N, 3)`, `log_scales` `(N, 3)`, `quats` `(N, 4)` in wxyz order,
`sh_colors` `(N, 3)` degree-0 spherical harmonics, and `logit_opacities` `(N,)`. These are
unconstrained as stored in a `.ply`, and an optimizer can update them without constraints.

## Render a scene

[`splax.io.load_ply`][splax.io.load_ply] reads a 3DGS `.ply` into the five parameter arrays.
[`splax.render`][splax.render] returns an `(image, alpha)` pair, the `(H, W, 3)` image and its
`(H, W)` accumulated coverage.

```python
import jax.numpy as jnp
import splax

SCENE = "https://huggingface.co/datasets/amacati/splax-test-data/resolve/main/scenes/lego.ply"
splats = splax.io.load_ply(splax.io.fetch(SCENE))
H, W, fx, fy = 400, 400, 400.0, 400.0
viewmat = splax.utils.look_at(jnp.array((0.0, -3.0, 1.0)), jnp.zeros(3), up=(0.0, 0.0, 1.0))
img, _ = splax.render(
    *splats, viewmat=viewmat, background=jnp.ones(3), img_shape=(H, W), f=(fx, fy)
)  # (H, W, 3)
```

`viewmat` is a `(4, 4)` world-to-camera matrix in the OpenCV convention (+z forward). `f` is the
focal length `(fx, fy)` and `c` is the principal point `(cx, cy)`.

## Batch over cameras

`jax.vmap` renders a stack of view matrices in one batch.

```{ .python continuation }
import jax
from functools import partial

eyes = jnp.array([(0.0, -3.0, 1.0), (3.0, 0.0, 1.0)])
viewmats = splax.utils.look_at(eyes, jnp.zeros(3), up=(0.0, 0.0, 1.0))

render_at = partial(splax.render, *splats, background=jnp.ones(3), img_shape=(H, W), f=(fx, fy))
frames, _ = jax.vmap(render_at)(viewmat=viewmats)  # (B, H, W, 3)
```

## Take a gradient

[`splax.render`][splax.render] differentiates with respect to all five parameter arrays.

```{ .python continuation }
import jax

target = jnp.zeros((H, W, 3))  # your ground truth image


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

## Next steps

- [Rendering](../user-guide/rendering.md) covers camera conventions, backgrounds, and the
  antialiased flag.
- [Training](../user-guide/training.md) covers camera-pose gradients, the depth channel, and the
  trainer scripts.
- [Batching](../user-guide/batching.md) covers `vmap` semantics for inference and gradients.
