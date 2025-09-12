# python/dask/array/random.py
# Minimal submodule stub providing `random` for tests to patch.
# Exists so import path `dask.array.random.random` resolves.
# RELEVANT FILES:python/dask/array/__init__.py,python/forge3d/ingest/dask_adapter.py

"""
Random utilities placeholder for dask.array.
"""

import numpy as _np

def random(shape, chunks=None, dtype=None):
    return _np.empty(shape, dtype=dtype)
