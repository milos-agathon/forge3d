<p align="center">
  <img src="https://raw.githubusercontent.com/milos-agathon/forge3d/main/docs/assets/logo/forge3d_dark.svg" alt="forge3d" width="320">
</p>

<p align="center">
  <strong>Python-first terrain and scene rendering on top of Rust + WebGPU.</strong><br>
  Interactive viewer, offscreen rendering, overlays, point clouds, notebooks, and production outputs.
</p>

<p align="center">
  <a href="https://pypi.org/project/forge3d/"><img src="https://img.shields.io/pypi/v/forge3d?color=EFA026&style=flat-square" alt="PyPI"></a>
  <a href="https://pypi.org/project/forge3d/"><img src="https://img.shields.io/pypi/pyversions/forge3d?color=D0C8BA&style=flat-square" alt="Python 3.10+"></a>
  <a href="https://github.com/milos-agathon/forge3d/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0%20%2F%20MIT-blue?style=flat-square" alt="License"></a>
  <a href="https://milos-agathon.github.io/forge3d/"><img src="https://img.shields.io/badge/docs-online-blue?style=flat-square" alt="Docs"></a>
</p>

---

<p align="center">
  <img src="https://raw.githubusercontent.com/milos-agathon/forge3d/main/docs/assets/highres.png" alt="forge3d hero render" width="720">
</p>

## Install

```bash
pip install forge3d
```

Optional extras:

```bash
pip install "forge3d[jupyter]"   # notebook widget support
pip install "forge3d[datasets]"  # on-demand sample datasets
pip install "forge3d[all]"       # everything
```

## Quick Start

```python
import forge3d as f3d

dem_path = f3d.fetch_dem("rainier")

with f3d.open_viewer_async(terrain_path=dem_path, width=1440, height=900) as viewer:
    viewer.set_z_scale(0.1)
    viewer.set_orbit_camera(phi_deg=28, theta_deg=49, radius=5400, fov_deg=42)
    viewer.set_sun(azimuth_deg=302, elevation_deg=24)
    viewer.snapshot("rainier.png", width=1920, height=1080)
```

## What forge3d Covers

Open-source workflow:

- Interactive terrain viewing through `open_viewer_async()` and `ViewerHandle`
- Terrain snapshots from GeoTIFF or `numpy` DEM inputs
- Raster overlays, vector overlays, labels, and camera automation
- LAZ/COPC/EPT point-cloud loading
- COG access, CRS helpers, datasets, presets, and notebook widgets
- Native/offscreen rendering with `Scene`, `Session`, `TerrainRenderer`, and `TerrainRenderParams`
- Geometry, mesh, vector, SDF, path-tracing, lighting, and terrain scatter helpers

Pro workflow:

- `MapPlate`, `Legend`, `ScaleBar`, and `NorthArrow` for cartographic composition
- SVG/PDF vector export
- Building import pipelines for GeoJSON, CityJSON, and 3D Tiles workflows
- Mapbox-style import and scene bundles

## Documentation Map

- [Quickstart](https://milos-agathon.github.io/forge3d/start/quickstart.html): install, first viewer session, first overlay, first notebook widget
- [Feature Map](https://milos-agathon.github.io/forge3d/guides/feature_map.html): repo-wide overview of the supported workflows
- [Examples Catalog](https://milos-agathon.github.io/forge3d/examples/index.html): every script and notebook in `examples/`
- [Tutorials](https://milos-agathon.github.io/forge3d/tutorials/index.html): guided GIS and Python tracks
- [Gallery](https://milos-agathon.github.io/forge3d/gallery/index.html): finished recipes and visuals
- [API Reference](https://milos-agathon.github.io/forge3d/api/api_reference.html): the full public Python surface

## Examples

The repo ships a broad set of runnable examples in [`examples/`](examples/), including:

- interactive terrain scenes and raster overlays
- Mapbox-style, labels, and picking demos
- point-cloud and building viewers
- camera animation and terrain camera rigs
- COG streaming and offscreen terrain rendering
- pure-Python composition examples and notebooks

Run any script with `python examples/<name>.py --help` when it exposes CLI options.

## License

The open-source core is released under Apache-2.0 OR MIT. Pro-gated features require a commercial license key set with `forge3d.set_license_key(...)`.
