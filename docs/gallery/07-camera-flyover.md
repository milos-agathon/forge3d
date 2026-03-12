# Camera Flyover

![Camera flyover preview](images/07-camera-flyover.png)

The viewer handle is scriptable enough that a flyover is just a frame loop.

## Ingredients

- `forge3d.open_viewer_async()`
- `ViewerHandle.set_orbit_camera()`
- `ViewerHandle.snapshot()`

## Sketch

```python
import forge3d as f3d

with f3d.open_viewer_async(terrain_path=f3d.fetch_dem("rainier")) as viewer:
    for step, phi in enumerate(range(0, 360, 20)):
        viewer.set_orbit_camera(phi_deg=phi, theta_deg=50, radius=5400)
        viewer.snapshot(f"frames/frame-{step:03d}.png")
```
