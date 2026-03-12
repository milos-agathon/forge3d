# Swiss Land-Cover

![Swiss land-cover preview](images/03-swiss-landcover.png)

This scene is about overlay draping rather than geometry complexity.

## Ingredients

- `forge3d.fetch_dem("swiss")`
- `forge3d.fetch_dataset("swiss-land-cover")`
- `ViewerHandle.load_overlay()`

## Sketch

```python
import forge3d as f3d

with f3d.open_viewer_async(terrain_path=f3d.fetch_dem("swiss")) as viewer:
    viewer.load_overlay("land-cover", f3d.fetch_dataset("swiss-land-cover"), opacity=0.8)
    viewer.set_orbit_camera(phi_deg=40, theta_deg=58, radius=9000)
    viewer.snapshot("swiss-land-cover.png")
```
