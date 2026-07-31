# Examples

## Render a scene

Fetch a splat and write a single image. The camera is framed from the splat's own extent, so the
script works for any scene in the dataset without per-scene metadata.

```bash
python examples/render_scene.py --scene lego --res 800
```

```python
--8<--
examples/render_scene.py
--8<--
```

## Join two splats and move one

Concatenate two copies of the lego splat into one set of arrays and drive the second with a rigid
transform, writing a GIF of the orbit. Both copies share every kernel launch, and only the
`(1, 4, 4)` transform changes per frame. See
[dynamic scene composition](user-guide/rendering.md#dynamic-scene-composition) for the mechanism.

```bash
python examples/compose_splats.py --frames 60
```

```python
--8<--
examples/compose_splats.py
--8<--
```

## Serve a moving object to a browser

Upload the hall splat once and fly a drone splat around a circle every frame, moving it without
re-uploading its gaussians. See [Viewer](user-guide/viewer.md).

```bash
python examples/viewer_demo.py
```

```python
--8<--
examples/viewer_demo.py
--8<--
```
