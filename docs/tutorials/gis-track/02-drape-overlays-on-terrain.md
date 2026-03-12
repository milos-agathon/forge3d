# Drape Overlays On Terrain

Raster overlays stay in the main viewer path: load terrain, add a named layer,
then adjust opacity and draw order.

## Example: Swiss DEM plus land-cover overlay

```python
import forge3d as f3d

dem_path = f3d.fetch_dem("swiss")
overlay_path = f3d.fetch_dataset("swiss-land-cover")

with f3d.open_viewer_async(terrain_path=dem_path, width=1500, height=960) as viewer:
    viewer.load_overlay(
        name="land-cover",
        path=overlay_path,
        opacity=0.82,
        z_order=10,
    )
    viewer.set_orbit_camera(phi_deg=42, theta_deg=58, radius=9000)
    viewer.set_sun(azimuth_deg=312, elevation_deg=33)
    viewer.snapshot("swiss-land-cover.png")
```

## When you need vector overlays

The high-level viewer wrapper currently exposes raster drape loading directly.
For custom vector geometry, drop to raw IPC:

```python
viewer.send_ipc(
    {
        "cmd": "add_vector_overlay",
        "name": "ridge-line",
        "vertices": [
            [0.0, 0.0, 0.0, 0.95, 0.3, 0.2, 1.0],
            [50.0, 20.0, 0.0, 0.95, 0.3, 0.2, 1.0],
            [90.0, 35.0, 0.0, 0.95, 0.3, 0.2, 1.0],
        ],
        "indices": [0, 1, 1, 2],
        "primitive": "lines",
        "drape": True,
        "line_width": 3.0,
    }
)
```

That is the same IPC surface the lower-level helpers in `forge3d.viewer_ipc`
wrap for scripts.

Next: [](03-build-a-map-plate.md)

## Expected output

![Expected output for the overlay draping tutorial](../images/gis-02-overlays.png)
