# python/rasterio/__init__.py
# Minimal rasterio stub package to satisfy tests and demos without installing rasterio.
# Exists so imports like rasterio.transform/from windows/enums resolve.
# RELEVANT FILES:python/rasterio/transform.py,python/forge3d/ingest/xarray_adapter.py

"""
Stub rasterio package (transform, windows, enums modules provided).
When the real rasterio is installed, this stub will transparently defer to it
even if this stub appears earlier on sys.path (e.g., PYTHONPATH=python).
"""

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
    # Try to locate site-packages rasterio manually and load it in-place
    _candidates = []
    try:
        _candidates.extend(_site.getsitepackages())
    except Exception:
        pass
    try:
        _candidates.append(_site.getusersitepackages())
    except Exception:
        pass
    _real_init = None
    for _base in _candidates:
        _pkg_init = _Path(_base) / 'rasterio' / '__init__.py'
        if _pkg_init.is_file():
            _real_init = str(_pkg_init)
            _real_pkg_path = str(_pkg_init.parent)
            break
    if _real_init:
        try:
            # Remove the stub entry temporarily so subimports resolve under the real package
            _stub_root = _Path(__file__).resolve().parents[1]
            _orig_path = list(_sys.path)
            _sys.path = [p for p in _orig_path if _Path(p).resolve() != _stub_root]
            _sys.modules.pop('rasterio', None)
            # Create a package spec pointing at the real rasterio package location
            _spec = _util.spec_from_file_location(
                __name__, _real_init, submodule_search_locations=[_real_pkg_path]
            )
            if _spec and _spec.loader:
                _mod = _util.module_from_spec(_spec)
                # Install into sys.modules under the canonical name so that
                # 'import rasterio.submodule' in the real package works
                _sys.modules[__name__] = _mod
                _spec.loader.exec_module(_mod)  # type: ignore[attr-defined]
                _real_rasterio = _mod
            # Restore path
            _sys.path = _orig_path
        except Exception:
            _real_rasterio = None

if _real_rasterio is not None:
    # If the real module has been installed under our name in sys.modules,
    # nothing further to do; otherwise mirror public attributes.
    _this = _sys.modules.get(__name__)
    if _this is not _real_rasterio:
        for _name in dir(_real_rasterio):
            if _name.startswith('_'):
                continue
            try:
                setattr(_this, _name, getattr(_real_rasterio, _name))  # type: ignore[arg-type]
            except Exception:
                pass
else:
    # Real rasterio not available — provide a clear message on open().
    def open(*args, **kwargs):  # type: ignore
        raise ImportError(
            "Real rasterio package is required for this operation. "
            "Install with: pip install rasterio"
        )

# Note: local submodules (e.g., transform) are available when real rasterio
# is missing; when real rasterio is present, its own submodules are exposed.
