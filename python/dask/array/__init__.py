# python/dask/array/__init__.py
# Minimal dask.array stub exposing functions and a random submodule for patching.
# Exists to satisfy tests without real dask installed.
# RELEVANT FILES:python/dask/array/random.py,python/forge3d/ingest/dask_adapter.py

"""
Minimal dask.array stub for local testing.
"""

import types
import numpy as _np

def linspace(start, stop, num, chunks=None, dtype=None):
    return _np.linspace(start, stop, int(num), dtype=dtype)

def meshgrid(x, y, indexing='xy'):
    return _np.meshgrid(x, y, indexing=indexing)

def broadcast_to(array, shape):
    return _np.broadcast_to(array, shape)

# Create a `random` submodule with a placeholder function.
random = types.ModuleType("dask.array.random")

def _random_random(shape, chunks=None, dtype=None):
    return _np.empty(shape, dtype=dtype)

random.random = _random_random
