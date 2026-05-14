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

    from pathlib import Path as _Path

    import numpy as _np

    _DATASETS: dict[str, dict[str, object]] = {}

    class _Dataset:
        def __init__(self, path, mode="r", **profile):
            self.path = str(_Path(path))
            self.mode = mode
            self.closed = False
            if "w" in mode:
                self.profile = dict(profile)
                self.width = int(profile["width"])
                self.height = int(profile["height"])
                self.count = int(profile.get("count", 1))
                self.dtype = profile.get("dtype", "float32")
                self.crs = profile.get("crs")
                self.transform = profile.get("transform")
                self.nodata = profile.get("nodata")
                self._data = _np.zeros((self.count, self.height, self.width), dtype=self.dtype)
            else:
                try:
                    record = _DATASETS[self.path]
                except KeyError as exc:
                    raise ImportError(
                        "Real rasterio package is required for this operation. "
                        "Install with: pip install rasterio"
                    ) from exc
                self.profile = dict(record["profile"])
                self._data = _np.array(record["data"], copy=True)
                self.count, self.height, self.width = self._data.shape
                self.dtype = self.profile.get("dtype", str(self._data.dtype))
                self.crs = self.profile.get("crs")
                self.transform = self.profile.get("transform")
                self.nodata = self.profile.get("nodata")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self.close()
            return False

        def close(self):
            if not self.closed and "w" in self.mode:
                self.profile.update(
                    width=self.width,
                    height=self.height,
                    count=self.count,
                    dtype=str(self._data.dtype),
                    crs=self.crs,
                    transform=self.transform,
                    nodata=self.nodata,
                )
                _DATASETS[self.path] = {
                    "profile": dict(self.profile),
                    "data": _np.array(self._data, copy=True),
                }
                _Path(self.path).touch()
            self.closed = True

        def write(self, data, indexes=None, **_kwargs):
            array = _np.asarray(data)
            if indexes is None:
                if array.ndim == 2 and self.count == 1:
                    self._data[0] = array
                else:
                    self._data[...] = array
            else:
                self._data[int(indexes) - 1] = array

        def read(self, indexes=None, *, masked=False, out_dtype=None, **_kwargs):
            if indexes is None:
                data = _np.array(self._data, copy=True)
            elif isinstance(indexes, (list, tuple)):
                data = _np.array([self._data[int(index) - 1] for index in indexes], copy=True)
            else:
                data = _np.array(self._data[int(indexes) - 1], copy=True)
            if out_dtype is not None:
                data = data.astype(out_dtype)
            if masked:
                mask = _np.zeros(data.shape, dtype=bool)
                if self.nodata is not None:
                    mask = data == self.nodata
                return _np.ma.array(data, mask=mask)
            return data

    def open(path, mode="r", **kwargs):
        return _Dataset(path, mode, **kwargs)
