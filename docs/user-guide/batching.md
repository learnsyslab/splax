# Batching

`jax.vmap` over [`splax.render`][splax.render] renders a batch in one go rather than looping in
Python, for both the forward and the backward pass.

## Batched inference

We first prepare a batch of view matrices:

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
```

Wrap [`splax.render`][splax.render] in `jax.vmap` over any batched argument. Mapping over a stack of
view matrices renders one image per camera.

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
frames, _ = jax.vmap(render_at)(viewmat=viewmats)  # (B, H, W, 3)
```


## Batched gradients

`jax.vmap(jax.grad(render))` computes batched gradients for efficient training. The reduction
depends on how an input is batched.

- Broadcast inputs, shared across the batch, get their gradients summed over the batch axis.
- Per-image inputs, for example a batch of camera poses differentiated with
  `jax.grad(loss, argnums=viewmat)`, get per-image gradients.

## Memory trade at large batch

Rendering all `B` cameras together scales the working memory with the batch size, so at large `B`
the peak footprint is higher than looping one camera at a time.
[`splax.clear_cache`][splax.clear_cache] releases it when switching between very different batch
sizes.
