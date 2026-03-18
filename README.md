# forge3d

<p align="center">
  <strong>GPU-accelerated 3D terrain rendering for Python.</strong><br>
  Built in Rust with WebGPU. Pre-built wheels — no Rust toolchain required.
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

**This image was generated with 5 lines of Python.** Load a DEM, launch the GPU viewer, position the camera, and capture a publication-quality snapshot. No shaders to write. No OpenGL boilerplate. Just `pip install forge3d`.

```python
import forge3d

with forge3d.open_viewer_async(terrain_path="rainier.tif") as viewer:
    viewer.set_orbit_camera(phi_deg=225, theta_deg=35, radius=1.2)
    viewer.set_sun(azimuth_deg=315, elevation_deg=32)
    viewer.snapshot("render.png", width=3840, height=2160)
```

## Install

```bash
pip install forge3d
```

Need Jupyter widgets or tutorial datasets?

```bash
pip install "forge3d[jupyter]"     # ViewerWidget for notebooks
pip install "forge3d[datasets]"    # on-demand sample DEMs, CityJSON, COPC
pip install "forge3d[all]"         # everything
```

## Gallery

<table>
  <tr>
    <td align="center"><img src="https://raw.githubusercontent.com/milos-agathon/forge3d/main/docs/gallery/images/01-mount-rainier.png" width="280"><br><sub>PBR terrain</sub></td>
    <td align="center"><img src="https://raw.githubusercontent.com/milos-agathon/forge3d/main/docs/gallery/images/03-swiss-landcover.png" width="280"><br><sub>Landcover overlay</sub></td>
    <td align="center"><img src="https://raw.githubusercontent.com/milos-agathon/forge3d/main/docs/gallery/images/06-point-cloud.png" width="280"><br><sub>LiDAR point cloud</sub></td>
  </tr>
  <tr>
    <td align="center"><img src="https://raw.githubusercontent.com/milos-agathon/forge3d/main/docs/gallery/images/09-shadow-comparison.png" width="280"><br><sub>Sun & shadow control</sub></td>
    <td align="center"><img src="https://raw.githubusercontent.com/milos-agathon/forge3d/main/docs/gallery/images/02-mount-fuji-labels.png" width="280"><br><sub>GeoPackage labels</sub></td>
    <td align="center"><img src="https://raw.githubusercontent.com/milos-agathon/forge3d/main/docs/gallery/images/05-3d-buildings.png" width="280"><br><sub>CityJSON buildings</sub></td>
  </tr>
  <tr>
    <td align="center"><img src="https://raw.githubusercontent.com/milos-agathon/forge3d/main/docs/gallery/images/04-luxembourg-rail-network.png" width="280"><br><sub>Vector overlays</sub></td>
    <td align="center"><img src="https://raw.githubusercontent.com/milos-agathon/forge3d/main/docs/gallery/images/07-camera-flyover.png" width="280"><br><sub>Camera animation</sub></td>
    <td align="center"><img src="https://raw.githubusercontent.com/milos-agathon/forge3d/main/docs/gallery/images/10-map-plate.png" width="280"><br><sub>Map plate compositor</sub></td>
  </tr>
</table>

<p align="center"><a href="https://milos-agathon.github.io/forge3d/gallery/"><strong>See the full gallery &rarr;</strong></a></p>

## What You Get

### Open Source (Apache-2.0 / MIT)

Everything you need to go from raw elevation data to a rendered 3D scene:

- **Interactive viewer** — real-time orbit, pan, zoom via a Rust/WebGPU subprocess controlled from Python over IPC
- **Terrain rendering** — load GeoTIFFs or numpy arrays, PBR materials, 100+ colormaps
- **Vector overlays** — GeoJSON/GeoPackage polygons, lines, and labels projected onto terrain
- **Point clouds** — COPC and LAZ files with millions of points, colored by elevation or classification
- **3D Tiles** — stream OGC 3D Tiles tilesets directly into the viewer
- **CRS reprojection** — automatic coordinate transforms via PROJ + pyproj
- **Camera animation** — keyframed flyover paths with frame-by-frame export
- **Jupyter integration** — `ViewerWidget` embeds the viewer inline in notebooks
- **High-res snapshots** — up to 8K PNG export from any camera angle

### Pro

Professional cartography and production workflows:

- **Map plate compositor** — legends, scale bars, north arrows, multi-panel layouts
- **SVG / PDF export** — publication-ready vector output
- **3D buildings** — GeoJSON, CityJSON, and 3D Tiles import with roof inference and PBR materials
- **Mapbox Style Spec** — load and apply Mapbox-compatible styles
- **Scene bundles** — save and share complete `.forge3d` scene packages
- **[Get a Pro key &rarr;](https://forge3d.dev/pro)**

## Tutorials

Two tracks depending on your background:

- **GIS professionals** — [Visualize your first DEM &rarr;](https://milos-agathon.github.io/forge3d/tutorials/gis-track/)
- **Python developers** — [Your first 3D terrain &rarr;](https://milos-agathon.github.io/forge3d/tutorials/python-track/)
- **Architecture overview** — [How forge3d works &rarr;](https://milos-agathon.github.io/forge3d/architecture.html)

## Jupyter

```python
from forge3d.widgets import ViewerWidget

widget = ViewerWidget(terrain_path="dem.npy")
widget.set_camera(phi_deg=225, theta_deg=35, radius=1.2)
widget.set_sun(azimuth_deg=315, elevation_deg=32)
widget.snapshot()  # renders inline
```

## Links

[Documentation](https://milos-agathon.github.io/forge3d/) &nbsp;&middot;&nbsp; [API Reference](https://milos-agathon.github.io/forge3d/api/api_reference.html) &nbsp;&middot;&nbsp; [GitHub](https://github.com/milos-agathon/forge3d) &nbsp;&middot;&nbsp; [PyPI](https://pypi.org/project/forge3d/)

## License

Open-source core released under Apache-2.0 OR MIT. Pro features require a [commercial license](https://forge3d.dev/pro) key set with `forge3d.set_license_key(...)`.
