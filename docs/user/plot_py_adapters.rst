===============================
plot_py Adapters (Matplotlib/Geo/Charts)
===============================

.. currentmodule:: forge3d.adapters

This page documents the plot_py adapters in Workstream M, which make it easy to bring figures, charts, rasters, and geospatial data into forge3d. All integrations are optional and provide clear guidance when a dependency is missing.

Overview
========

Workstream M includes:

- M1: Matplotlib Adapter (Image)
- M2: Matplotlib Adapter (Data)
- M3: GeoPandas Adapter
- M4: Rasterio/Xarray Adapter
- M5: Cartopy Integration
- M6: Seaborn/Plotly Convenience

Installation
============

Install only what you need. Examples:

.. code-block:: bash

   # Matplotlib (M1/M2) and PNG saving
   pip install matplotlib pillow

   # GeoPandas (M3)
   pip install geopandas shapely pyproj

   # Rasterio + Xarray (M4)
   pip install rasterio xarray rioxarray pillow

   # Cartopy (M5)
   pip install cartopy matplotlib pillow

   # Plotly + Seaborn (M6)
   pip install plotly kaleido seaborn matplotlib pillow

M1: Matplotlib Adapter (Image)
=============================

Rasterize a Matplotlib figure or axes to an RGBA NumPy array and optionally derive a heightmap from luminance.

.. code-block:: python

   from forge3d.adapters import mpl_rasterize_figure, mpl_rasterize_axes, mpl_height_from_luminance
   rgba = mpl_rasterize_figure(fig, dpi=150, facecolor='white')
   rgba_tight = mpl_rasterize_axes(ax, dpi=150, bbox_inches='tight', pad_inches=0.0)
   height = mpl_height_from_luminance(rgba)  # float32 [0,1]

See runnable script: ``examples/m1_mpl_image_demo.py``.

M2: Matplotlib Adapter (Data)
=============================

Extract polylines and polygons from Matplotlib artists and convert to meshes using forge3d geometry.

.. code-block:: python

   from forge3d.adapters import (
       extract_lines_from_axes, extract_polygons_from_axes,
       thicken_lines_to_meshes, extrude_polygons_to_meshes,
       line_width_world_from_pixels,
   )

   lines = extract_lines_from_axes(ax)
   polys = extract_polygons_from_axes(ax)

   # Convert to meshes (requires native geometry extension)
   world_w = line_width_world_from_pixels(6.0, z=5.0, fov_y_deg=45.0, height_px=600)
   line_meshes = thicken_lines_to_meshes(lines, width_world=world_w)
   poly_meshes = extrude_polygons_to_meshes(polys, height=0.2)

See runnable script: ``examples/m2_mpl_data_demo.py``.

M3: GeoPandas Adapter
=====================

Reproject a GeoSeries and extrude polygon geometries to meshes.

.. code-block:: python

   from forge3d.adapters import reproject_geoseries, geoseries_to_polygons, extrude_geometries_to_meshes

   gs_proj = reproject_geoseries(gs, 'EPSG:3857')
   meshes = extrude_geometries_to_meshes(gs_proj, height=10.0)

See runnable script: ``examples/m3_geopandas_demo.py``.

M4: Rasterio/Xarray Adapter
===========================

Convert a rasterio dataset or an xarray DataArray to an RGBA array with nodata→alpha handling. Use reproject helpers from ``forge3d.adapters.reproject`` as needed.

.. code-block:: python

   from forge3d.adapters import rasterio_to_rgba, dataarray_to_rgba

   # Rasterio
   with rasterio.open('input.tif') as ds:
       rgba = rasterio_to_rgba(ds, bands=(1,2,3))

   # Xarray DataArray (e.g., created via rioxarray)
   rgba2 = dataarray_to_rgba(da)

See runnable script: ``examples/m4_raster_xarray_demo.py``.

M5: Cartopy Integration
=======================

Rasterize a Cartopy GeoAxes to RGBA and query its CRS and extent.

.. code-block:: python

   from forge3d.adapters import rasterize_geoaxes, get_axes_crs, get_extent_in_crs

   rgba = rasterize_geoaxes(ax, dpi=150, facecolor='white')
   crs = get_axes_crs(ax)
   extent = get_extent_in_crs(ax, crs)

See runnable script: ``examples/m5_cartopy_demo.py``.

M6: Seaborn/Plotly Convenience
==============================

Render Plotly figures (via kaleido) or Seaborn/Matplotlib charts to RGBA.

.. code-block:: python

   from forge3d.adapters import render_chart_to_rgba

   # Plotly
   rgba = render_chart_to_rgba(plotly_fig, width=800, height=600)

   # Seaborn / Matplotlib
   rgba2 = render_chart_to_rgba(seaborn_obj, dpi=150)

See runnable script: ``examples/m6_charts_demo.py``.

Examples
========

All demo scripts are runnable from the repository root:

.. code-block:: bash

   python examples/m1_mpl_image_demo.py --out reports/m1_mpl_image.png
   python examples/m2_mpl_data_demo.py --outdir reports/m2_mpl_data_demo
   python examples/m3_geopandas_demo.py --outdir reports/m3_geopandas_demo --save-obj
   python examples/m4_raster_xarray_demo.py --synthetic --out reports/m4_raster_xarray.png
   python examples/m5_cartopy_demo.py --out reports/m5_cartopy.png
   python examples/m6_charts_demo.py --backend plotly --out reports/m6_charts.png

Notes
=====

- Optional dependencies are probed at import-time and functions raise helpful messages when unavailable.
- RGBA arrays are standard NumPy ``uint8`` with shape ``(H, W, 4)`` suitable for upload to GPU textures.
- For georeferenced rasters, validate alignment and transforms using the helpers in ``forge3d.adapters.reproject``.
