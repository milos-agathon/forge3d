# forge3d

**GPU-accelerated terrain and scene rendering for Python.**

![Hero render](assets/highres.png)

Built in Rust with WebGPU. Install the wheel, launch the viewer, and capture
publication-quality terrain snapshots without setting up a Rust toolchain.

## Install

```bash
pip install forge3d
```

Optional extras:

- `pip install "forge3d[jupyter]"` for `ViewerWidget`
- `pip install "forge3d[datasets]"` for on-demand tutorial datasets

## Quick Start

```python
import forge3d

with forge3d.open_viewer_async(
    terrain_path=forge3d.mini_dem_path(),
    width=1280,
    height=720,
) as viewer:
    viewer.set_orbit_camera(phi_deg=225.0, theta_deg=35.0, radius=1.2, fov_deg=45.0)
    viewer.set_sun(azimuth_deg=315.0, elevation_deg=32.0)
    viewer.snapshot("my_terrain.png", width=1920, height=1080)
```

## Open Core

- Interactive viewer with terrain loading, camera control, sun control, and high-res snapshots
- Bundled datasets plus fetch helpers for tutorial-scale DEM, CityJSON, and COPC assets
- Vector overlays, labels, point clouds, 3D Tiles, and notebook widgets
- CRS helpers, colormaps, PNG export, and camera animation preview

## Pro

- Map plate composition (`MapPlate`, legends, scale bars, north arrows)
- SVG/PDF export
- Scene bundle save/load
- GeoJSON/CityJSON/3D Tiles building import pipelines
- Mapbox Style Spec loading and style application
- [Get a Pro key](https://forge3d.dev/pro)

## Gallery

| Terrain | Buildings | Point Cloud |
| --- | --- | --- |
| ![Mount Rainier](docs/gallery/images/01-mount-rainier.png) | ![3D buildings](docs/gallery/images/05-3d-buildings.png) | ![Point cloud](docs/gallery/images/06-point-cloud.png) |

[See the full gallery](https://docs.forge3d.dev/gallery/)

## Tutorials

- [GIS track](https://docs.forge3d.dev/tutorials/gis-track/)
- [Python track](https://docs.forge3d.dev/tutorials/python-track/)
- [Architecture overview](https://docs.forge3d.dev/architecture/)

## Jupyter

```python
from forge3d.widgets import ViewerWidget

widget = ViewerWidget(terrain_path="dem.npy")
widget.set_camera(phi_deg=225.0, theta_deg=35.0, radius=1.2)
widget.set_sun(azimuth_deg=315.0, elevation_deg=32.0)
widget.snapshot()
```

## Links

- [Documentation](https://docs.forge3d.dev)
- [API reference](https://docs.forge3d.dev/api/)
- [GitHub](https://github.com/forge3d/forge3d)
- [PyPI](https://pypi.org/project/forge3d/)

## License

Open-source core: Apache-2.0 OR MIT. Pro-only workflows require a commercial
license key set with `forge3d.set_license_key(...)`.
