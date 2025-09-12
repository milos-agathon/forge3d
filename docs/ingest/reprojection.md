# docs/ingest/reprojection.md
# WarpedVRT-based reprojection helpers and CRS inspection.
# Exists to document S3 API surface and graceful fallbacks.
# RELEVANT FILES:python/forge3d/adapters/reproject.py,tests/test_reproject_window.py,examples/reproject_window_demo.py

Reprojection uses `WarpedVRTWrapper` to manage a `rasterio.vrt.WarpedVRT` and read windows in a target CRS.

- `WarpedVRTWrapper(dataset, dst_crs, resampling=...)` normalizes CRS and resampling.
- `reproject_window(src_dataset, dst_crs, window, ...)` reads a window reprojected to the destination CRS.
- `get_crs_info(crs)` returns details for a CRS via `pyproj`.

When optional deps are absent, functions raise clear `ImportError`. Tests patch these symbols to validate behavior without heavy installs.

