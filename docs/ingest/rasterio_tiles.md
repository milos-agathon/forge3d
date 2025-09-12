# docs/ingest/rasterio_tiles.md
# Overview of windowed reads and block iteration utilities.
# Exists to document S1 API and expected behavior with/without rasterio.
# RELEVANT FILES:python/forge3d/adapters/rasterio_tiles.py,tests/test_rasterio_adapter.py,examples/raster_window_demo.py

Windowed reading and block iteration provide efficient access to raster data without full materialization.

- `windowed_read(dataset, window, out_shape=None, resampling=None, indexes=None, dtype=None)` reads a subwindow optionally resampled.
- `block_iterator(dataset, blocksize=None)` yields windows covering the dataset, either using native blocks or a synthetic grid.

Behavior degrades gracefully when rasterio is not installed. The adapter exposes an `is_rasterio_available()` helper.

See `examples/raster_window_demo.py` for a headless demonstration.

