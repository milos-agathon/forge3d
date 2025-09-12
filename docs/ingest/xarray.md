# docs/ingest/xarray.md
# xarray/rioxarray DataArray ingestion preserving spatial metadata.
# Exists to document S4 adapter contract and performance notes.
# RELEVANT FILES:python/forge3d/ingest/xarray_adapter.py,tests/test_xarray_ingestion.py,examples/xarray_ingest_demo.py

`ingest_dataarray(da, preserve_dims=True, target_dtype=None, ensure_c_contiguous=True)` validates dims, preserves `(y,x[,band])` ordering, and returns a NumPy array with metadata.

The adapter prefers zero-copy for already C-contiguous inputs and converts dtype on request.

Optional rioxarray CRS/transform are passed through when available. Functions error clearly if extras are missing.

