# Order-Independent Transparency (OIT)

The current public OIT controls live on `forge3d.Scene` and in the viewer IPC
helpers.

## Public Python surface

- `Scene.enable_oit()`
- `Scene.disable_oit()`
- `Scene.is_oit_enabled()`
- `Scene.get_oit_mode()`
- `forge3d.viewer_ipc.set_oit_enabled(...)`
- `forge3d.viewer_ipc.get_oit_mode(...)`

## Native scene example

```python
import forge3d as f3d

scene = f3d.Scene(640, 480)
scene.enable_oit()

if scene.is_oit_enabled():
    print("mode:", scene.get_oit_mode())

rgba = scene.render_rgba()
f3d.numpy_to_png("oit.png", rgba)
```

## Interactive viewer example

```python
from forge3d.viewer_ipc import get_oit_mode, launch_viewer, set_oit_enabled

process, port, sock = launch_viewer(width=1280, height=720)
try:
    set_oit_enabled(sock, enabled=True, mode="auto")
    print(get_oit_mode(sock))
finally:
    from forge3d.viewer_ipc import close_viewer
    close_viewer(sock, process)
```

## Status

- Low-level weighted-OIT demo helpers are not re-exported on the top-level
  `forge3d` package.
- Viewer-based vector overlays still go through IPC.
- If you need experimental native-only OIT helpers, access them through
  `forge3d._native.get_native_module()` and treat them as internal.
