# python/rasterio/__init__.py
# Minimal rasterio shim for environments where the real dependency is absent.
# RELEVANT FILES:python/rasterio/transform.py,python/forge3d/cog.py,python/forge3d/io.py

"""Shim rasterio package.

If a real rasterio installation is available elsewhere on ``sys.path``, proxy to it.
Otherwise expose a lightweight stub so imports like ``rasterio.transform`` resolve.
"""

from __future__ import annotations

import importlib.machinery as _machinery
import os as _os
import sys as _sys

__forge3d_stub__ = True

_stub_dir = _os.path.dirname(__file__)
_stub_parent = _os.path.dirname(_stub_dir)
_stub_parent_norm = _os.path.normcase(_os.path.abspath(_stub_parent))


def _filtered_search_path() -> list[str]:
    paths: list[str] = []
    for entry in _sys.path:
        entry_norm = _os.path.normcase(_os.path.abspath(entry or _os.curdir))
        if entry_norm != _stub_parent_norm:
            paths.append(entry)
    return paths


def _load_real_rasterio() -> bool:
    spec = _machinery.PathFinder.find_spec(__name__, _filtered_search_path())
    if spec is None or spec.loader is None or not spec.origin:
        return False

    origin_norm = _os.path.normcase(_os.path.abspath(spec.origin))
    if origin_norm == _os.path.normcase(_os.path.abspath(__file__)):
        return False

    module = _sys.modules[__name__]
    module.__file__ = spec.origin
    module.__loader__ = spec.loader
    module.__package__ = __name__
    module.__spec__ = spec
    if spec.submodule_search_locations is not None:
        module.__path__ = list(spec.submodule_search_locations)

    spec.loader.exec_module(module)
    return True


try:
    __forge3d_stub__ = not _load_real_rasterio()
except Exception:
    __forge3d_stub__ = True


if __forge3d_stub__:
    __version__ = "0.0-stub"

    def open(*args, **kwargs):
        raise ImportError(
            "Real rasterio package is required for this operation. "
            "Install with: pip install rasterio"
        )
