# docs/ingest/dask.md
# Dask-chunked raster ingestion and streaming materialization.
# Exists to document S5 API and memory guardrails.
# RELEVANT FILES:python/forge3d/ingest/dask_adapter.py,tests/test_dask_ingestion.py,examples/dask_ingest_demo.py

`ingest_dask_array(a, target_tile_size=(512,512), memory_limit_mb=512, progress_callback=None)` yields tiles without full materialization.

`materialize_dask_array_streaming(...)` assembles an array by streaming tiles, with output size guarding.

Planning respects memory budgets and generates a tile grid aligned with array shape.

