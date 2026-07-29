# Viewer

`splax.viewer` serves splats to a web client using [viser](https://viser.studio). It requires the optional `viser` dependency:

```bash
pip install splax[viewer]
```


## Rigid objects

The viewer holds a set of named rigid objects, each with its own gaussians and world
pose. `add_splats` uploads an object's gaussians once and `update_pose` moves it
afterwards.

```python
import splax
from splax.viewer import Viewer

viewer = Viewer(port=8080, up_direction="+z")
viewer.add_splats("scene", *splax.io.load_ply("room.ply"))
viewer.add_splats("drone", *splax.io.load_ply("drone.ply"), position=(0.0, 0.0, 1.0))

for pos, wxyz in trajectory:  # e.g. from a simulator
    viewer.update_pose("drone", pos, wxyz)
```

Open `http://localhost:8080` in a browser to view the scene. `remove` deletes an
object and `close` stops the server. The server runs in a background thread, so keep
the process alive for as long as the viewer should stay reachable.

## Beyond splats

`Viewer.server` exposes the underlying `viser.ViserServer` for anything the wrapper
does not cover, such as GUI elements, meshes, or camera controls. See the
[viser documentation](https://viser.studio) for its full API.
