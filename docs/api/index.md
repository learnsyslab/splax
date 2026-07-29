# API Reference

This section is generated from the splax source with
[mkdocstrings](https://mkdocstrings.github.io/). Private modules and the
distillation module are not included.

| Module | Description |
|---|---|
| `splax` | `render` and the low-level `project` / `rasterize` / `rasterize_depth` primitives |
| `splax.mcmc` | Fixed-budget MCMC training utilities |
| `splax.io` | 3DGS `.ply` load and write, and the parameter conversions around it |

## Parameters and activated arrays

Splat arrays come in two spaces, the unconstrained parameters and the activated arrays, both
described under [Rendering](../user-guide/rendering.md#inputs).

| Function | Space it takes |
|---|---|
| `splax.render` | parameter |
| `splax.io.load_ply` / `splax.io.write_ply` | parameter, stored verbatim |
| `splax.mcmc.relocate` / `splax.mcmc.inject_noise` | parameter |
| `splax.project` | activated |
| `splax.rasterize` / `splax.rasterize_depth` | activated |
| `splax.viewer.Viewer.add_splats` | parameter |

`splax.io.apply_activations` and `splax.io.invert_activations` convert between the two.

Use the sidebar to browse the full generated reference.
