# python/forge3d/geometry.py
# High-level geometry API exposing extrusion, primitives, validation, and welding
# Exists to provide a typed Python surface over Forge3D geometry core for Workstream F
# RELEVANT FILES:src/geometry/mod.rs,src/geometry/extrude.rs,src/geometry/primitives.rs,tests/test_f1_extrude.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    from . import _forge3d
except ImportError:  # pragma: no cover - handled in tests via skip
    _forge3d = None  # type: ignore[assignment]


@dataclass
class MeshBuffers:
    """Simple mesh container mirroring the Rust MeshBuffers struct."""

    positions: np.ndarray
    normals: np.ndarray
    uvs: np.ndarray
    indices: np.ndarray

    @property
    def vertex_count(self) -> int:
        return int(self.positions.shape[0])

    @property
    def triangle_count(self) -> int:
        if self.indices.ndim == 2:
            return int(self.indices.shape[0])
        return int(self.indices.size // 3)


def _ensure_native() -> None:
    if _forge3d is None:  # pragma: no cover - import guard
        raise RuntimeError("forge3d native module is not available; build the extension first")


def _to_float32(array: np.ndarray, expected_cols: int) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != expected_cols:
        raise ValueError(f"expected array with shape (N, {expected_cols})")
    return np.ascontiguousarray(arr)


def _to_uint32(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=np.uint32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 3)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("indices must have shape (M, 3) or flat multiple of three")
    return np.ascontiguousarray(arr)


def _mesh_from_py(obj: Dict[str, Any]) -> MeshBuffers:
    return MeshBuffers(
        positions=np.asarray(obj["positions"], dtype=np.float32),
        normals=np.asarray(obj["normals"], dtype=np.float32),
        uvs=np.asarray(obj["uvs"], dtype=np.float32),
        indices=np.asarray(obj["indices"], dtype=np.uint32).reshape(-1, 3),
    )


def _mesh_to_py(mesh: MeshBuffers) -> Dict[str, Any]:
    positions = _to_float32(np.asarray(mesh.positions, dtype=np.float32), 3)
    normals_arr = np.asarray(mesh.normals, dtype=np.float32)
    if normals_arr.size:
        normals = _to_float32(normals_arr, 3)
    else:
        normals = np.empty((0, 3), dtype=np.float32)

    uvs_arr = np.asarray(mesh.uvs, dtype=np.float32)
    if uvs_arr.size:
        uvs = _to_float32(uvs_arr, 2)
    else:
        uvs = np.empty((0, 2), dtype=np.float32)

    indices = _to_uint32(np.asarray(mesh.indices, dtype=np.uint32))

    return {
        "positions": positions,
        "normals": normals,
        "uvs": uvs,
        "indices": indices,
    }


def extrude_polygon(polygon: np.ndarray, height: float, cap_uv_scale: float = 1.0) -> MeshBuffers:
    """Extrude a planar polygon into a prism mesh."""

    _ensure_native()
    poly = _to_float32(np.asarray(polygon), 2)
    result = _forge3d.geometry_extrude_polygon_py(poly, float(height), float(cap_uv_scale))
    return _mesh_from_py(result)  # type: ignore[arg-type]


def primitive_mesh(kind: str, **kwargs: Any) -> MeshBuffers:
    """Generate a unit primitive mesh (plane, box, sphere, cylinder, cone, torus)."""

    _ensure_native()
    params: Dict[str, Any] = {}
    for key in (
        "resolution",
        "radial_segments",
        "rings",
        "height_segments",
        "tube_segments",
        "radius",
        "tube_radius",
        "include_caps",
    ):
        if key in kwargs and kwargs[key] is not None:
            params[key] = kwargs[key]
    result = _forge3d.geometry_generate_primitive_py(kind, params if params else None)
    return _mesh_from_py(result)  # type: ignore[arg-type]


def validate_mesh(positions: np.ndarray, indices: np.ndarray) -> Dict[str, Any]:
    """Run topology validation on a mesh."""

    _ensure_native()
    pos = _to_float32(positions, 3)
    idx = _to_uint32(indices)
    return _forge3d.geometry_validate_mesh_py(pos, idx)


def weld_mesh(
    positions: np.ndarray,
    indices: np.ndarray,
    uvs: Optional[np.ndarray] = None,
    *,
    position_epsilon: float = 1e-5,
    uv_epsilon: float = 1e-4,
) -> Tuple[MeshBuffers, np.ndarray, int]:
    """Weld vertices within the provided tolerances and recompute normals."""

    _ensure_native()
    pos = _to_float32(positions, 3)
    idx = _to_uint32(indices)
    uv_array = None
    if uvs is not None:
        uv_array = _to_float32(uvs, 2)
    result = _forge3d.geometry_weld_mesh_py(
        pos,
        idx,
        uv_array,
        {
            "position_epsilon": float(position_epsilon),
            "uv_epsilon": float(uv_epsilon),
        },
    )
    mesh = _mesh_from_py(result["mesh"])  # type: ignore[arg-type]
    remap = np.asarray(result["remap"], dtype=np.uint32)
    collapsed = int(result["collapsed"])
    return mesh, remap, collapsed


def center_mesh(
    mesh: MeshBuffers,
    target: Optional[Tuple[float, float, float]] = None,
) -> Tuple[MeshBuffers, np.ndarray]:
    """Translate the mesh so that its bounding-box center matches ``target``.

    Returns the transformed mesh and the original center.
    """

    _ensure_native()
    payload = _mesh_to_py(mesh)
    target_tuple = None if target is None else (float(target[0]), float(target[1]), float(target[2]))
    transformed_dict, previous_center = _forge3d.geometry_transform_center_py(payload, target_tuple)
    transformed_mesh = _mesh_from_py(transformed_dict)
    return transformed_mesh, np.asarray(previous_center, dtype=np.float32)


def scale_mesh(
    mesh: MeshBuffers,
    scale: Tuple[float, float, float],
    pivot: Optional[Tuple[float, float, float]] = None,
) -> Tuple[MeshBuffers, bool]:
    """Apply a non-uniform scale about the given pivot.

    Returns the transformed mesh and whether winding was flipped.
    """

    _ensure_native()
    payload = _mesh_to_py(mesh)
    scale_tuple = (float(scale[0]), float(scale[1]), float(scale[2]))
    pivot_tuple = None if pivot is None else (float(pivot[0]), float(pivot[1]), float(pivot[2]))
    transformed_dict, flipped = _forge3d.geometry_transform_scale_py(payload, scale_tuple, pivot_tuple)
    return _mesh_from_py(transformed_dict), bool(flipped)


def flip_mesh_axis(mesh: MeshBuffers, axis: int) -> Tuple[MeshBuffers, bool]:
    """Flip the mesh across the specified axis (0=X, 1=Y, 2=Z).

    Returns the transformed mesh and whether winding was flipped.
    """

    _ensure_native()
    payload = _mesh_to_py(mesh)
    transformed_dict, flipped = _forge3d.geometry_transform_flip_axis_py(payload, int(axis))
    return _mesh_from_py(transformed_dict), bool(flipped)


def swap_mesh_axes(mesh: MeshBuffers, axis_a: int, axis_b: int) -> Tuple[MeshBuffers, bool]:
    """Swap the specified coordinate axes on the mesh.

    Returns the transformed mesh and whether winding was flipped.
    """

    _ensure_native()
    payload = _mesh_to_py(mesh)
    transformed_dict, flipped = _forge3d.geometry_transform_swap_axes_py(
        payload,
        int(axis_a),
        int(axis_b),
    )
    return _mesh_from_py(transformed_dict), bool(flipped)


def mesh_bounds(mesh: MeshBuffers) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return the axis-aligned bounding box of the mesh as ``(min, max)`` arrays."""

    _ensure_native()
    payload = _mesh_to_py(mesh)
    bounds = _forge3d.geometry_transform_bounds_py(payload)
    if bounds is None:
        return None
    min_bounds, max_bounds = bounds
    return (
        np.asarray(min_bounds, dtype=np.float32),
        np.asarray(max_bounds, dtype=np.float32),
    )
