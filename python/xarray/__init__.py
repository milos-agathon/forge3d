# python/xarray/__init__.py
# Minimal xarray stub to support tests that patch xarray.DataArray.
# Exists so `import xarray as xr` works without the real dependency.
# RELEVANT FILES:python/forge3d/ingest/xarray_adapter.py,tests/test_xarray_ingestion.py

"""
Stub for xarray package to enable patching.
"""

class DataArray:  # pragma: no cover - placeholder for patching in tests
    def __init__(self, *args, **kwargs):
        pass
