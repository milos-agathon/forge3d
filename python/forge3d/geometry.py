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
    tangents: np.ndarray | None = None

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
    tangents = obj.get("tangents")
    tarr = None
    if tangents is not None:
        tarr = np.asarray(tangents, dtype=np.float32)
        if tarr.size and (tarr.ndim != 2 or tarr.shape[1] != 4):
            raise ValueError("tangents must have shape (N, 4)")
    return MeshBuffers(
        positions=np.asarray(obj["positions"], dtype=np.float32),
        normals=np.asarray(obj["normals"], dtype=np.float32),
        uvs=np.asarray(obj["uvs"], dtype=np.float32),
        indices=np.asarray(obj["indices"], dtype=np.uint32).reshape(-1, 3),
        tangents=tarr,
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

    tangents_arr = None
    if getattr(mesh, "tangents", None) is not None:
        t = np.asarray(mesh.tangents, dtype=np.float32)
        if t.size:
            if t.ndim != 2 or t.shape[1] != 4:
                raise ValueError("tangents must have shape (N, 4)")
            tangents_arr = np.ascontiguousarray(t)
        else:
            tangents_arr = np.empty((0, 4), dtype=np.float32)

    return {
        "positions": positions,
        "normals": normals,
        "uvs": uvs,
        "indices": indices,
        "tangents": tangents_arr,
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


def unwrap_planar(mesh: MeshBuffers, axis: int = 2) -> MeshBuffers:
    """Planar UV unwrap dropping the specified axis (0=X, 1=Y, 2=Z).

    Generates UVs in [0,1] based on the mesh AABB projected onto the remaining axes.
    """
    _ensure_native()
    payload = _mesh_to_py(mesh)
    out = _forge3d.uv_planar_unwrap_py(payload, int(axis))
    return _mesh_from_py(out)  # type: ignore[arg-type]


def instance_mesh(mesh: MeshBuffers, transforms: np.ndarray) -> MeshBuffers:
    """Instance a base mesh by a set of 4x4 row-major transforms.

    Parameters
    ----------
    mesh: MeshBuffers
        Base geometry to instance.
    transforms: (N,16) float32
        Row-major 4x4 transforms; one per instance.
    """
    _ensure_native()
    payload = _mesh_to_py(mesh)
    arr = np.asarray(transforms, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 16:
        raise ValueError("transforms must have shape (N,16) row-major 4x4")
    out = _forge3d.geometry_instance_mesh_py(payload, arr)
    return _mesh_from_py(out)  # type: ignore[arg-type]


def gpu_instancing_available() -> bool:
    """Return True if the native GPU instancing path is available in this build.

    Notes
    -----
    - In CPU-only environments, this will typically return False.
    - When available, GPU instancing enables per-instance transforms with indirect draws.
    """
    _ensure_native()
    fn = getattr(_forge3d, "gpu_instancing_available_py", None)
    if fn is None:
        return False
    try:
        return bool(fn())
    except Exception:
        return False


def instance_mesh_gpu_render(
    mesh: MeshBuffers,
    transforms: np.ndarray,
    width: int = 512,
    height: int = 512,
) -> np.ndarray:
    """Render an instanced mesh via the native GPU instancing path, returning an RGBA8 image.

    Parameters
    ----------
    mesh : MeshBuffers
        Base mesh to instance.
    transforms : (N,16) float32
        Row-major 4x4 transforms, one per instance.
    width, height : int
        Output image dimensions.

    Returns
    -------
    np.ndarray
        Image array of shape (height, width, 4), dtype=uint8 (RGBA).

    Notes
    -----
    - Requires the crate to be built with Cargo feature ``enable-gpu-instancing``.
    - In environments without GPU support, prefer the CPU ``instance_mesh`` and external rendering.
    """
    _ensure_native()
    fn = getattr(_forge3d, "geometry_instance_mesh_gpu_render_py", None)
    if fn is None:
        raise RuntimeError(
            "GPU instanced renderer is not available. Build with Cargo feature 'enable-gpu-instancing'."
        )
    if int(width) <= 0 or int(height) <= 0:
        raise ValueError("width and height must be positive")
    arr = np.asarray(transforms, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 16:
        raise ValueError("transforms must have shape (N,16) row-major 4x4")
    payload = _mesh_to_py(mesh)
    return fn(int(width), int(height), payload, arr)


def generate_thick_polyline(
    path: np.ndarray,
    width_world: float,
    *,
    depth_offset: float = 0.0,
    join_style: str = "miter",
    miter_limit: float = 4.0,
) -> MeshBuffers:
    """Generate a thick 3D polyline as a ribbon with constant world-space width.

    Notes
    -----
    To achieve constant pixel width on screen for a camera at distance `z` and vertical FOV `fov_y`
    rendering to a height `H` in pixels, use:

    width_world ≈ pixel_width * (2*z*tan(fov_y/2) / H)
    """
    _ensure_native()
    arr = _to_float32(np.asarray(path, dtype=np.float32), 3)
    out = _forge3d.geometry_generate_thick_polyline_py(
        arr, float(width_world), float(depth_offset), str(join_style), float(miter_limit)
    )
    return _mesh_from_py(out)  # type: ignore[arg-type]


def unwrap_spherical(mesh: MeshBuffers) -> MeshBuffers:
    """Spherical UV unwrap around the AABB center.

    Projects vertices onto a sphere and computes UVs from (phi, theta).
    """
    _ensure_native()
    payload = _mesh_to_py(mesh)
    out = _forge3d.uv_spherical_unwrap_py(payload)
    return _mesh_from_py(out)  # type: ignore[arg-type]


# -----------------------------
# Phase 4: Subdivision (F11)
# -----------------------------

def subdivide_mesh(
    mesh: MeshBuffers,
    levels: int = 1,
    *,
    creases: Optional[np.ndarray] = None,
    preserve_boundary: bool = True,
) -> MeshBuffers:
    """Refine a triangle mesh with Loop subdivision and optional crease/boundary preservation.

    Notes
    -----
    - Each level splits triangles and applies Loop vertex smoothing.
    - Crease edges (and boundaries if ``preserve_boundary``) keep mid-edge points and edge-aware smoothing.
    - UVs are interpolated with the same weights; normals are recomputed.
    """
    _ensure_native()
    payload = _mesh_to_py(mesh)
    crease_arr = None
    if creases is not None:
        crease_arr = np.asarray(creases, dtype=np.uint32)
        if crease_arr.ndim != 2 or crease_arr.shape[1] != 2:
            raise ValueError("creases must have shape (K, 2)")
    out = _forge3d.geometry_subdivide_py(payload, int(levels), crease_arr, bool(preserve_boundary))
    return _mesh_from_py(out)  # type: ignore[arg-type]


# -----------------------------
# Phase 4: Displacement (F12)
# -----------------------------

def displace_heightmap(
    mesh: MeshBuffers,
    heightmap: np.ndarray,
    scale: float = 1.0,
    *,
    uv_space: bool = False,
) -> MeshBuffers:
    """Displace vertices along normals using a heightmap sampled in XY or UV space.

    Parameters
    ----------
    heightmap: np.ndarray
        2D float32 array. Values are sampled bilinearly and multiplied by `scale`.
    scale: float
        Displacement scale.
    uv_space: bool
        If True, sample the heightmap using vertex UVs; otherwise sample over the mesh XY bounds.
    """
    _ensure_native()
    payload = _mesh_to_py(mesh)
    hm = np.asarray(heightmap, dtype=np.float32)
    if hm.ndim != 2:
        raise ValueError("heightmap must be a 2D array")
    out = _forge3d.geometry_displace_heightmap_py(payload, hm, float(scale), bool(uv_space))
    return _mesh_from_py(out)  # type: ignore[arg-type]


def displace_procedural(mesh: MeshBuffers, amplitude: float = 1.0, frequency: float = 1.0) -> MeshBuffers:
    """Apply a simple procedural displacement along normals using sin/cos on XY.

    Parameters
    ----------
    amplitude: float
        Displacement amplitude.
    frequency: float
        Frequency in world units.
    """
    _ensure_native()
    payload = _mesh_to_py(mesh)
    out = _forge3d.geometry_displace_procedural_py(payload, float(amplitude), float(frequency))
    return _mesh_from_py(out)  # type: ignore[arg-type]


# -----------------------------
# Phase 4: Curves & Tubes (F17)
# -----------------------------

def generate_ribbon(
    path: np.ndarray,
    width_start: float,
    width_end: float,
    *,
    join_style: str = "miter",
    miter_limit: float = 4.0,
    join_styles: Optional[np.ndarray] = None,
) -> MeshBuffers:
    """Generate a ribbon mesh (two-sided strip) along a 3D path with configurable join styles.

    Parameters
    ----------
    path: (N,3) float32 array of points.
    width_start, width_end: start/end width for tapering.
    join_style: one of {"miter", "bevel", "round"}.
    miter_limit: clamps miter length at sharp angles.
    """
    _ensure_native()
    arr = _to_float32(np.asarray(path, dtype=np.float32), 3)
    js = None
    if join_styles is not None:
        js = np.asarray(join_styles, dtype=np.uint8)
    out = _forge3d.geometry_generate_ribbon_py(
        arr, float(width_start), float(width_end), str(join_style), float(miter_limit), js
    )
    return _mesh_from_py(out)  # type: ignore[arg-type]


def generate_tube(
    path: np.ndarray,
    radius_start: float,
    radius_end: float,
    *,
    radial_segments: int = 16,
    cap_ends: bool = True,
) -> MeshBuffers:
    """Generate a tube mesh along a 3D path.

    Parameters
    ----------
    path: (N,3) float32 array of points.
    radius_start, radius_end: start/end radius for tapering.
    radial_segments: number of segments around the tube (>=3).
    cap_ends: add end caps if True.
    """
    _ensure_native()
    arr = _to_float32(np.asarray(path, dtype=np.float32), 3)
    out = _forge3d.geometry_generate_tube_py(arr, float(radius_start), float(radius_end), int(radial_segments), bool(cap_ends))
    return _mesh_from_py(out)  # type: ignore[arg-type]


def generate_tangents(mesh: MeshBuffers) -> np.ndarray:
    """Generate per-vertex tangents as (N, 4) float32 array [tx, ty, tz, w].

    Requires valid UVs and normals for best results. `w` encodes handedness (+1/-1).
    """
    _ensure_native()
    payload = _mesh_to_py(mesh)
    return _forge3d.geometry_generate_tangents_py(payload)


def attach_tangents(mesh: MeshBuffers) -> MeshBuffers:
    """Generate and attach tangents to the mesh, returning a new mesh with `tangents` field set."""
    _ensure_native()
    payload = _mesh_to_py(mesh)
    out = _forge3d.geometry_attach_tangents_py(payload)
    return _mesh_from_py(out)  # type: ignore[arg-type]


def subdivide_adaptive(
    mesh: MeshBuffers,
    *,
    edge_length_limit: Optional[float] = None,
    curvature_threshold: Optional[float] = None,
    max_levels: int = 3,
    creases: Optional[np.ndarray] = None,
    preserve_boundary: bool = True,
) -> MeshBuffers:
    """Adaptive subdivision selecting levels based on edge length and/or curvature thresholds.

    Parameters
    ----------
    edge_length_limit: optional float
        Target max edge length in world units.
    curvature_threshold: optional float (radians)
        Max dihedral angle across edges; if exceeded, refinement level increases.
    max_levels: int
        Clamp on maximum subdivision levels.
    creases: Optional[(K,2) uint32]
        Edge list for creases (preserve as sharp).
    preserve_boundary: bool
        Treat boundary edges as creases.
    """
    _ensure_native()
    payload = _mesh_to_py(mesh)
    crease_arr = None
    if creases is not None:
        crease_arr = np.asarray(creases, dtype=np.uint32)
        if crease_arr.ndim != 2 or crease_arr.shape[1] != 2:
            raise ValueError("creases must have shape (K, 2)")
    out = _forge3d.geometry_subdivide_adaptive_py(
        payload,
        None if edge_length_limit is None else float(edge_length_limit),
        None if curvature_threshold is None else float(curvature_threshold),
        int(max_levels),
        crease_arr,
        bool(preserve_boundary),
    )
    return _mesh_from_py(out)  # type: ignore[arg-type]
