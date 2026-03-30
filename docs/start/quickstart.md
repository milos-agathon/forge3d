# Quickstart

This quickstart assumes you are working from a source checkout and want the full
Phase 2 developer workflow: viewer, datasets, widgets, and tutorial assets.

## Install

```bash
pip install -e .[all,jupyter,datasets]
cargo build --release --bin interactive_viewer
```

`open_viewer_async()` looks for `interactive_viewer` in `target/release`,
`target/debug`, or on `PATH`. If the binary is missing, `ViewerWidget` can
still show its inline fallback preview when you pass previewable terrain data
via `terrain_path=` or `src=`.

## First terrain

The smallest self-contained sample is the bundled `mini_dem.npy` dataset.

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

`mini_dem_path()` returns a `.npy` file. The Python wrapper converts it to a
temporary TIFF before launching the current viewer binary, so the same example
works with either packaged numpy DEMs or GeoTIFFs.

## Pull a real dataset

The larger sample registry resolves local repo assets first and downloads only
when needed.

```python
import forge3d as f3d

rainier = f3d.fetch_dem("rainier")

with f3d.open_viewer_async(terrain_path=rainier) as viewer:
    viewer.set_orbit_camera(phi_deg=28, theta_deg=48, radius=5500)
    viewer.set_sun(azimuth_deg=300, elevation_deg=24)
    viewer.snapshot("rainier.png")
```

Available dataset helpers:

```python
import forge3d as f3d

print(f3d.available_datasets())
print(f3d.dataset_info()["mt-st-helens"])
```

## Notebook workflow

Use `ViewerWidget` when you want the live viewer in Jupyter. Pass `src=` if you
also want the inline fallback preview to remain usable in headless sessions.

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

You can drive it from Python after display:

```python
widget.set_camera(phi_deg=20, theta_deg=60, radius=1.5)
widget.set_sun(azimuth_deg=250, elevation_deg=40)
widget.snapshot("widget-snapshot.png")
```

## Next steps

- Follow the GIS workflow in [GIS Track](../tutorials/gis-track/index.md).
- Follow the notebook-first workflow in [Python Track](../tutorials/python-track/index.md).
- Browse finished compositions in [Gallery](../gallery/index.md).
- See the surface area in [API Reference](../api/api_reference.rst).
