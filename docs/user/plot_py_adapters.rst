plot_py Adapters (Matplotlib/Geo/Charts)
========================================

The repository does not currently ship a public ``forge3d.adapters`` module for
Matplotlib, GeoPandas, Rasterio, Cartopy, Plotly, or Seaborn conversion.

## Practical interop today

Use the standard image/array exchange points that already exist:

* render with forge3d to NumPy RGBA via ``render_offscreen_rgba()``
* display results with Matplotlib ``imshow(...)``
* save figures or rasters to PNG and use them as viewer overlays
* use ``forge3d.geometry`` / ``forge3d.buildings`` / ``forge3d.export`` for
  geometry and cartography workflows that are actually part of the package

For the currently supported plotting/image interop, see:

* ``docs/integration/matplotlib.md``
* ``docs/integration/cartopy.md``
* ``docs/user/datashader_interop.rst``
