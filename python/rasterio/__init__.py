# python/rasterio/__init__.py
# Minimal rasterio stub package to satisfy tests and demos without installing rasterio.
# Exists so imports like rasterio.transform/from windows/enums resolve.
# RELEVANT FILES:python/rasterio/transform.py,python/forge3d/ingest/xarray_adapter.py

"""
Stub rasterio package (transform, windows, enums modules provided).
"""

# Try to import the real rasterio if available, otherwise provide stubs
import sys as _sys
import os as _os

# Store the directory containing this stub module
_stub_dir = _os.path.dirname(__file__)

# Try to find and import the real rasterio by temporarily removing stub from path
_real_rasterio = None
_original_path = _sys.path.copy()
try:
    # Remove any path entries that contain this stub (the parent of the rasterio dir)
    _stub_parent = _os.path.dirname(_stub_dir)
    _filtered_path = [p for p in _sys.path if p != _stub_parent]
    _sys.path = _filtered_path

    # Now try to import the real rasterio
    try:
        import importlib as _importlib
        import importlib.util as _util

        # Force reimport by removing from cache if present
        if 'rasterio' in _sys.modules and hasattr(_sys.modules['rasterio'], '__file__'):
            if _sys.modules['rasterio'].__file__ and _stub_dir in _sys.modules['rasterio'].__file__:
                # This is the stub, remove it
                del _sys.modules['rasterio']

        _spec = _util.find_spec('rasterio')
        if _spec and _spec.origin:
            _real_rasterio = _importlib.import_module('rasterio')
            # Import key functions from real rasterio
            if hasattr(_real_rasterio, 'open'):
                open = _real_rasterio.open
            if hasattr(_real_rasterio, '__version__'):
                __version__ = _real_rasterio.__version__
    except ImportError:
        pass
finally:
    # Restore original sys.path
    _sys.path = _original_path

# If real rasterio not found, provide a helpful error message
if _real_rasterio is None:
    def open(*args, **kwargs):
        raise ImportError(
            "Real rasterio package is required for this operation. "
            "Install with: pip install rasterio"
        )

# Submodules provided via local files (e.g., transform).
