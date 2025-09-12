# python/rasterio/transform.py
# Minimal transform stub exposing from_bounds and xy for tests/demos to patch.
# Exists to make import targets resolvable during tests.
# RELEVANT FILES:python/forge3d/ingest/xarray_adapter.py,python/forge3d/adapters/reproject.py

"""
Stubbed transform utilities.
"""

class _Affine:
    def __init__(self, *args, **kwargs):
        self.params = (args, kwargs)

def from_bounds(left, bottom, right, top, width, height):  # pragma: no cover
    return _Affine(left, bottom, right, top, width, height)

def xy(transform, rows, cols, offset='center'):  # pragma: no cover
    # Return simple evenly spaced coordinates based on indices for testing.
    return rows, cols
