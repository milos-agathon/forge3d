# python/forge3d/converters.py
# Conversion helpers for geometry pipelines (Workstream F)

from __future__ import annotations

from typing import List

import numpy as np

from .geometry import MeshBuffers, _mesh_from_py

try:
    from . import _forge3d
except ImportError:  # pragma: no cover
    _forge3d = None  # type: ignore


def _ensure_native() -> None:
    if _forge3d is None:  # pragma: no cover
        raise RuntimeError("forge3d native module is not available; build the extension first")


def multipolygonz_to_mesh(polygons: List[np.ndarray]) -> MeshBuffers:
    """Convert a list of (N,3) rings into a triangulated mesh.

    Each polygon is triangulated with a simple fan around the first vertex.
    """
    _ensure_native()
    py_list = []
    for ring in polygons:
        arr = np.ascontiguousarray(np.asarray(ring, dtype=np.float32))
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError("each polygon must be an array of shape (N, 3)")
        py_list.append(arr)
    result = _forge3d.converters_multipolygonz_to_obj_py(py_list)
    return _mesh_from_py(result)  # type: ignore[arg-type]
