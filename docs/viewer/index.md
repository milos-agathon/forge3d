# Viewer

The interactive viewer is the main live-rendering path in forge3d. Most of the
repo's examples use it directly or through `ViewerWidget`.

## Entry Points

### `forge3d.open_viewer_async()`

Preferred path for scripts and notebooks. It launches the installed
`interactive_viewer` executable, waits for the IPC endpoint, and gives you a
`ViewerHandle`.

```python
import forge3d as f3d

with f3d.open_viewer_async(terrain_path=f3d.fetch_dem("rainier")) as viewer:
    viewer.set_z_scale(0.1)
    viewer.set_orbit_camera(phi_deg=28, theta_deg=49, radius=5400)
    viewer.set_sun(azimuth_deg=302, elevation_deg=24)
    viewer.snapshot("rainier.png")
```

### `forge3d.open_viewer()`

Blocking wrapper for simpler desktop sessions where you do not need the async
handle pattern.

### `forge3d.ViewerWidget`

Notebook wrapper around the same viewer process. When the full viewer is not
available, it can fall back to an inline preview when you also supplied
previewable source data via `src=`.

## Common `ViewerHandle` Operations

| Task | Main API |
| --- | --- |
| Set orbit camera | `set_orbit_camera()` |
| Set look-at camera | `set_camera_lookat()` |
| Control sun | `set_sun()` |
| Load raster overlay | `load_overlay()` |
| Load point cloud | `load_point_cloud()` |
| Tune point-cloud display | `set_point_cloud_params()` |
| Save an image | `snapshot()` |
| Send unsupported or advanced commands | `send_ipc()` |

The examples in `terrain_viewer_interactive.py`, `swiss_terrain_landcover_viewer.py`,
`pointcloud_viewer_interactive.py`, `terrain_camera_rigs_demo.py`, and
`pnoa_river_showcase_video.py` are the fastest way to see those methods in real
use.

## When To Use Raw IPC

Use `send_ipc()` or the helpers in `forge3d.viewer_ipc` when you need commands
that do not have convenience wrappers yet, such as:

- custom vector overlays
- label placement and typography
- picking and lasso selection
- bundle and review-layer commands
- lower-level terrain effect toggles used by some examples

Example:

```python
with f3d.open_viewer_async(terrain_path=f3d.mini_dem_path()) as viewer:
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

## Viewer-Centric Example Map

- `terrain_viewer_interactive.py`: baseline interactive terrain viewer
- `swiss_terrain_landcover_viewer.py` and `bosnia_terrain_landcover_viewer.py`: raster overlays
- `luxembourg_rail_overlay.py`: vector-overlay IPC
- `fuji_labels_demo.py`: labels and decluttering
- `pointcloud_viewer_interactive.py`: point-cloud loading
- `picking_demo.py`: picking, lasso selection, and rich highlight flows
- `terrain_camera_rigs_demo.py` and `camera_animation_demo.py`: camera automation

## Related Pages

- [Quickstart](../start/quickstart.md)
- [Feature Map](../guides/feature_map.md)
- [Examples Catalog](../examples/index.md)
- [API Reference](../api/api_reference.rst)
