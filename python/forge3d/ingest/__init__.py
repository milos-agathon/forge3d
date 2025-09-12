"""
forge3d.ingest - Data ingestion adapters for raster/vector data

This package provides adapters for various geospatial data formats
with optional dependencies that gracefully degrade when unavailable.
"""

__all__ = []

# Check for optional dependencies and conditionally export functions
try:
    from .xarray_adapter import ingest_dataarray
    __all__.append("ingest_dataarray")
    _HAS_XARRAY = True
except ImportError:
    _HAS_XARRAY = False

try:
    from .dask_adapter import ingest_dask_array
    __all__.append("ingest_dask_array")
    _HAS_DASK = True
except ImportError:
    _HAS_DASK = False


def is_xarray_available():
    """Check if xarray and rioxarray are available."""
    return _HAS_XARRAY


def is_dask_available():
    """Check if dask is available."""
    return _HAS_DASK