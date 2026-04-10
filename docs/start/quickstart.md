# Quickstart

This page gets you onto the main public workflow as fast as possible:

1. install the package
2. open a terrain scene
3. add an overlay or notebook widget
4. jump to the example or guide that matches your real task

## Install

```bash
pip install forge3d
```

Common extras:

```bash
pip install "forge3d[jupyter]"
pip install "forge3d[datasets]"
pip install "forge3d[all]"
```

`pip install forge3d` installs the native extension and the `interactive_viewer`
launcher used by `open_viewer_async()`. In source checkouts, the Python wrapper
can still pick newer local binaries from `target/debug` or `target/release`
when they exist.

## First viewer session

Use the bundled mini DEM when you want a zero-download sanity check:

```python
import forge3d as f3d

with f3d.open_viewer_async(
    terrain_path=f3d.mini_dem_path(),
    width=1400,
    height=900,
) as viewer:
    viewer.set_orbit_camera(phi_deg=35, theta_deg=55, radius=1.8, fov_deg=45)
    viewer.set_sun(azimuth_deg=315, elevation_deg=32)
    viewer.snapshot("mini-dem.png", width=1600, height=900)
```

Use `fetch_dem()` when you want a larger sample dataset:

```python
import forge3d as f3d

dem_path = f3d.fetch_dem("rainier")

with f3d.open_viewer_async(terrain_path=dem_path, width=1440, height=900) as viewer:
    viewer.set_z_scale(0.1)
    viewer.set_orbit_camera(phi_deg=28, theta_deg=49, radius=5400, fov_deg=42)
    viewer.set_sun(azimuth_deg=302, elevation_deg=24)
    viewer.snapshot("rainier.png", width=1920, height=1080)
```

## First overlay

Raster overlays sit on the same viewer handle:

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
        preserve_colors=True,
    )
    viewer.set_orbit_camera(phi_deg=90, theta_deg=10, radius=18000, fov_deg=16)
    viewer.set_sun(azimuth_deg=315, elevation_deg=17)
    viewer.snapshot("swiss-land-cover.png")
```

For custom vector overlays, labels, and other lower-level scene commands, drop
to `viewer.send_ipc(...)` or the helper functions in `forge3d.viewer_ipc`.

## First notebook widget

Use `ViewerWidget` when you want the same viewer path in Jupyter:

```python
import forge3d as f3d

widget = f3d.ViewerWidget(
    terrain_path=f3d.mini_dem_path(),
    src=f3d.mini_dem(),
    width=960,
    height=600,
)
widget.set_camera(phi_deg=30, theta_deg=58, radius=1.7, fov_deg=42)
widget.set_sun(azimuth_deg=315, elevation_deg=30)
widget
```

When the full viewer cannot launch, the widget can fall back to an inline
preview if you also provided previewable source data through `src=`.

## When To Leave The Viewer Path

Stay with `open_viewer_async()` and `ViewerHandle` for most scene work.

Move to the lower-level native surface when you need deterministic offscreen
pipelines or explicit renderer control:

- `Scene` for compact offscreen rendering
- `Session` + `TerrainRenderer` + `TerrainRenderParams` for terrain PBR/POM
- `render_offline()` and `oidn_denoise()` for offline accumulation and denoise

## Next Steps

- Read the repo-wide [Feature Map](../guides/feature_map.md).
- Browse every runnable script in the [Examples Catalog](../examples/index.md).
- Follow the [GIS Track](../tutorials/gis-track/index.md) or [Python Track](../tutorials/python-track/index.md).
- Use the [API Reference](../api/api_reference.rst) when you already know the symbol you need.
