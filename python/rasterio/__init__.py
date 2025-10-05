# python/rasterio/__init__.py
# Minimal rasterio stub package to satisfy tests and demos without installing rasterio.
# Exists so imports like rasterio.transform/from windows/enums resolve.
# RELEVANT FILES:python/rasterio/transform.py,python/forge3d/ingest/xarray_adapter.py

"""
Stub rasterio package (transform, windows, enums modules provided).
"""

# Try to import the real rasterio if available, otherwise provide stubs
import sys as _sys

# Store the path to this stub module
_stub_path = __file__

# Try to find and import the real rasterio
_real_rasterio = None
for _path in _sys.path[1:]:  # Skip the first path (this package's location)
    if _path and _path != _stub_path:
        try:
            import importlib.util as _util
            _spec = _util.find_spec('rasterio')
            if _spec and _spec.origin and _spec.origin != _stub_path:
                # Found real rasterio, import it
                import importlib as _importlib
                _real_rasterio = _importlib.import_module('rasterio')
                # Import key functions from real rasterio
                if hasattr(_real_rasterio, 'open'):
                    open = _real_rasterio.open
                if hasattr(_real_rasterio, '__version__'):
                    __version__ = _real_rasterio.__version__
                break
        except:
            continue

# If real rasterio not found, provide a helpful error message
if _real_rasterio is None:
    def open(*args, **kwargs):
        raise ImportError(
            "Real rasterio package is required for this operation. "
            "Install with: pip install rasterio"
        )

# Submodules provided via local files (e.g., transform).
