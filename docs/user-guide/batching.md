# Batching

`jax.vmap` over `splax.render` renders a batch in one go rather than looping in
Python, for both the forward and the backward pass.

## Batched inference

Wrap `splax.render` in `jax.vmap` over any batched argument. Mapping
over a stack of view matrices renders one image per camera.

```python
frames = jax.vmap(lambda vm: splax.render(
    means, log_scales, quats, sh_colors, logit_opacities,
    viewmat=vm, background=jnp.ones(3), img_shape=(H, W),
    f=(fx, fy),
)[0])(viewmats)  # (B, H, W, 3)
```


## Batched gradients

`jax.vmap(jax.grad(render))` matches the per-sample sequential gradients. The
reduction depends on how an input is batched.

- Broadcast inputs, shared across the batch, get their gradients summed over the batch axis.
- Per-image inputs, for example a batch of camera poses differentiated with `jax.grad(loss, argnums=viewmat)`, get per-image gradients.

## Memory trade at large batch

Rendering all `B` cameras together scales the working memory with the batch size,
so at large `B` the peak footprint is higher than looping one camera at a time.
`splax.clear_cache` releases it when switching between very different batch sizes.
