# Interactive Viewer

> **Status:** implemented

The interactive viewer is the primary live-rendering path for forge3d. Python
launches the Rust `interactive_viewer` binary, then drives it through TCP/NDJSON
IPC with `ViewerHandle`.

## Quick Start

### Desktop viewer

```bash
cargo build --release --bin interactive_viewer
```

```python
import forge3d as f3d

with f3d.open_viewer_async(
    terrain_path=f3d.fetch_dem("rainier"),
    width=1440,
    height=900,
) as viewer:
    viewer.set_orbit_camera(phi_deg=28, theta_deg=50, radius=5400, fov_deg=42)
    viewer.set_sun(azimuth_deg=300, elevation_deg=26)
    viewer.snapshot("rainier.png")
```

### Notebook widget

```python
import forge3d as f3d

widget = f3d.ViewerWidget(
    terrain_path=f3d.mini_dem_path(),
    src=f3d.mini_dem_path(),
    width=960,
    height=600,
)
widget
```

## Core controls

`ViewerHandle` exposes the high-frequency controls directly:

- `load_terrain()`
- `load_overlay()`
- `load_point_cloud()`
- `set_point_cloud_params()`
- `set_orbit_camera()`
- `set_camera_lookat()`
- `set_sun()`
- `snapshot()`
- `send_ipc()`

## Raw IPC example

Use raw IPC when you need commands that do not yet have convenience wrappers.

```python
with f3d.open_viewer_async(terrain_path=f3d.mini_dem_path()) as viewer:
    viewer.send_ipc({"cmd": "set_terrain", "phi": 45, "theta": 60, "zscale": 1.5})
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

## Common IPC commands

- `load_terrain`
- `set_terrain`
- `set_terrain_camera`
- `lit_sun`
- `load_overlay`
- `load_point_cloud`
- `set_point_cloud_params`
- `snapshot`
- `close`

For script-friendly wrappers around these commands, see `forge3d.viewer_ipc`.

## Platform support

- macOS: Metal backend
- Linux: Vulkan backend
- Windows: DX12 or Vulkan backend

## Related pages

- [Quickstart](../quickstart.md)
- [GIS Track](../tutorials/gis-track/index.md)
- [Python Track](../tutorials/python-track/index.md)
