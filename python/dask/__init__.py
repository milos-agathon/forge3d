# python/dask/__init__.py
# Minimal dask shim for environments where the real dependency is absent.
# RELEVANT FILES:python/dask/base.py,python/dask/array/__init__.py,examples/belgium_bivariate_climate_map.py

"""Lightweight repo-local ``dask`` stub for optional xarray compatibility."""

from __future__ import annotations

__forge3d_stub__ = True
__version__ = "0.0-stub"

from . import array, base
from .base import is_dask_collection

__all__ = ["array", "base", "is_dask_collection"]
