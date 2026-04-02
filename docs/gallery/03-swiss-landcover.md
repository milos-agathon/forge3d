# Swiss Land-Cover

![Swiss land-cover preview](images/03-swiss-landcover.png)

Swiss terrain with land-cover classification draped as a lit overlay,
rendered from an EPSG:2056 terrain grid with reduced vertical exaggeration, a
brighter relief-driven northwest hillshade, near-top-down framing, and a
snapshot layout that adds the map title, caption, and legend.

## Ingredients

- `forge3d.fetch_dem("swiss")`
- `forge3d.fetch_dataset("swiss-land-cover")`
- `ViewerHandle.load_overlay()`

## Sketch

```python
import forge3d as f3d

with f3d.open_viewer_async(terrain_path=f3d.fetch_dem("swiss")) as viewer:
    viewer.load_overlay(
        "land-cover",
        f3d.fetch_dataset("swiss-land-cover"),
        opacity=0.82,
        preserve_colors=True,
    )
    viewer.set_orbit_camera(phi_deg=90, theta_deg=10, radius=18000)
    viewer.set_sun(azimuth_deg=315, elevation_deg=17)
    viewer.snapshot("swiss-land-cover.png")
```
