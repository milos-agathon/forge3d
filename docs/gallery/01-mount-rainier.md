# Mount Rainier

![Mount Rainier preview](images/01-mount-rainier.png)

This is the baseline terrain scene: one DEM, one camera, one sun, one
snapshot.

## Ingredients

- `forge3d.fetch_dem("rainier")`
- `forge3d.open_viewer_async()`
- `ViewerHandle.set_orbit_camera()`
- `ViewerHandle.set_sun()`

## Sketch

```python
import forge3d as f3d

with f3d.open_viewer_async(terrain_path=f3d.fetch_dem("rainier")) as viewer:
    viewer.set_orbit_camera(phi_deg=28, theta_deg=49, radius=5400)
    viewer.set_sun(azimuth_deg=302, elevation_deg=24)
    viewer.snapshot("rainier.png")
```
