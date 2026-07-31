# API Reference

This section is generated from the splax source with
[mkdocstrings](https://mkdocstrings.github.io/).

| Module | Description |
|---|---|
| [`splax`](splax/index.md) | [`render`][splax.render] and the low-level [`project`][splax.project] / [`rasterize`][splax.rasterize] / [`rasterize_depth`][splax.rasterize_depth] primitives |
| [`splax.mcmc`](splax/mcmc.md) | Fixed-budget MCMC training utilities |
| [`splax.io`](splax/io.md) | 3DGS `.ply` load and write, and the parameter conversions around it |

## Parameters and activated arrays

Splat arrays come in two spaces, the unconstrained parameters and the activated arrays, both
described under [Rendering](../user-guide/rendering.md#inputs).

| Function | Space it takes |
|---|---|
| [`splax.render`][splax.render] | parameter |
| [`splax.io.load_ply`][splax.io.load_ply] / [`splax.io.write_ply`][splax.io.write_ply] | parameter, stored verbatim |
| [`splax.mcmc.relocate`][splax.mcmc.relocate] / [`splax.mcmc.inject_noise`][splax.mcmc.inject_noise] | parameter |
| [`splax.project`][splax.project] | activated |
| [`splax.rasterize`][splax.rasterize] / [`splax.rasterize_depth`][splax.rasterize_depth] | activated |
| [`splax.viewer.Viewer.add_splats`][splax.viewer.Viewer.add_splats] | parameter |

[`splax.io.apply_activations`][splax.io.apply_activations] and [`splax.io.invert_activations`][splax.io.invert_activations] convert between the two.

Use the sidebar to browse the full generated reference.
