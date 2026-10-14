"""
Mesh generation, TBN (Tangent, Bitangent, Normal) utilities, and BVH construction

Provides functions for:
1. Generating mesh data and computing tangent space information for normal mapping and PBR rendering
2. Triangle mesh BVH construction for path tracing (Task A3)
3. GPU mesh upload for accelerated ray tracing
"""

from typing import Dict, List, Tuple, Any
import struct
import time
import numpy as np
import warnings

from ._native import get_native_symbol

# Packed layout of one CPU BvhNode (repr(C) + Pod): aabb_min f32 x3, left u32,
# aabb_max f32 x3, right u32, flags u32, pad u32. "<" keeps the struct module
# from adding alignment padding, so this is exactly 40 bytes.
_BVH_NODE_BYTES = 40
_BVH_NODE_STRUCT = struct.Struct("<3f I 3f I I I")


def __getattr__(name):
    if name == "_HAS_TBN":
        return has_tbn_support()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def generate_cube_tbn() -> Tuple[List[Dict[str, Any]], List[int], List[Dict[str, Any]]]:
    """Generate TBN data for a unit cube.
    
    Creates a unit cube with 6 faces, each face having proper tangent space
    information for normal mapping.
    
    Returns
    -------
    vertices : List[Dict[str, Any]]
        List of vertex dictionaries containing 'position', 'normal', and 'uv' keys
    indices : List[int]
        Triangle indices (36 indices for 12 triangles)
    tbn_data : List[Dict[str, Any]]
        List of TBN dictionaries containing 'tangent', 'bitangent', 'normal', and 'handedness' keys
        
    Examples
    --------
    >>> import forge3d.mesh as mesh
    >>> vertices, indices, tbn_data = mesh.generate_cube_tbn()
    >>> print(f"Generated cube with {len(vertices)} vertices")
    >>> 
    >>> # Check TBN validity
    >>> for i, tbn in enumerate(tbn_data):
    ...     t = tbn['tangent']
    ...     n = tbn['normal'] 
    ...     print(f"Vertex {i}: tangent={t}, normal={n}")
    
    Raises
    ------
    RuntimeError
        If TBN generation feature is not enabled
    """
    native_fn = get_native_symbol("mesh_generate_cube_tbn")
    if native_fn is not None:
        result = native_fn()
        return result['vertices'], result['indices'], result['tbn_data']
    # Pure-Python fallback: 6 faces, 4 vertices per face (no sharing)
    vertices: List[Dict[str, Any]] = []
    indices: List[int] = []
    tbn_data: List[Dict[str, Any]] = []
    # Define faces with normal and tangent/binormal axes
    faces = [
        # +X
        (np.array([1, 0, 0], np.float32), np.array([0, 0, -1], np.float32), np.array([0, 1, 0], np.float32),
         [np.array([1, -1, -1]), np.array([1, -1, 1]), np.array([1, 1, 1]), np.array([1, 1, -1])]),
        # -X
        (np.array([-1, 0, 0], np.float32), np.array([0, 0, 1], np.float32), np.array([0, 1, 0], np.float32),
         [np.array([-1, -1, 1]), np.array([-1, -1, -1]), np.array([-1, 1, -1]), np.array([-1, 1, 1])]),
        # +Y
        (np.array([0, 1, 0], np.float32), np.array([1, 0, 0], np.float32), np.array([0, 0, 1], np.float32),
         [np.array([-1, 1, -1]), np.array([1, 1, -1]), np.array([1, 1, 1]), np.array([-1, 1, 1])]),
        # -Y
        (np.array([0, -1, 0], np.float32), np.array([1, 0, 0], np.float32), np.array([0, 0, -1], np.float32),
         [np.array([-1, -1, 1]), np.array([1, -1, 1]), np.array([1, -1, -1]), np.array([-1, -1, -1])]),
        # +Z
        (np.array([0, 0, 1], np.float32), np.array([1, 0, 0], np.float32), np.array([0, 1, 0], np.float32),
         [np.array([-1, -1, 1]), np.array([-1, 1, 1]), np.array([1, 1, 1]), np.array([1, -1, 1])]),
        # -Z
        (np.array([0, 0, -1], np.float32), np.array([-1, 0, 0], np.float32), np.array([0, 1, 0], np.float32),
         [np.array([1, -1, -1]), np.array([1, 1, -1]), np.array([-1, 1, -1]), np.array([-1, -1, -1])]),
    ]
    for n, t, b, corners in faces:
        base = len(vertices)
        uvs = [np.array([0, 0]), np.array([1, 0]), np.array([1, 1]), np.array([0, 1])]
        for i in range(4):
            pos = corners[i].astype(np.float32) * 0.5
            vertices.append({'position': pos.tolist(), 'normal': n.tolist(), 'uv': uvs[i].tolist()})
            handedness = float(np.sign(np.dot(np.cross(t, b), n)))
            tbn_data.append({'tangent': t.tolist(), 'bitangent': b.tolist(), 'normal': n.tolist(), 'handedness': handedness})
        # Two triangles per face
        indices.extend([base + 0, base + 1, base + 2, base + 0, base + 2, base + 3])
    return vertices, indices, tbn_data


def generate_plane_tbn(width: int, height: int) -> Tuple[List[Dict[str, Any]], List[int], List[Dict[str, Any]]]:
    """Generate TBN data for a planar grid.
    
    Creates a flat plane subdivided into a grid with proper tangent space
    information for normal mapping.
    
    Parameters
    ---------- 
    width : int
        Number of vertices along X axis (must be >= 2)
    height : int
        Number of vertices along Z axis (must be >= 2)
        
    Returns
    -------
    vertices : List[Dict[str, Any]]
        List of vertex dictionaries containing 'position', 'normal', and 'uv' keys
    indices : List[int] 
        Triangle indices forming quads
    tbn_data : List[Dict[str, Any]]
        List of TBN dictionaries containing 'tangent', 'bitangent', 'normal', and 'handedness' keys
        
    Examples
    --------
    >>> vertices, indices, tbn_data = mesh.generate_plane_tbn(3, 3)
    >>> print(f"Generated {width}x{height} plane with {len(vertices)} vertices")
    >>>
    >>> # All normals should point up for a flat plane
    >>> for tbn in tbn_data:
    ...     n = tbn['normal']
    ...     assert abs(n[1] - 1.0) < 1e-3, "Normal should be Y-up"
    
    Raises
    ------
    ValueError
        If width or height is less than 2
    RuntimeError
        If TBN generation feature is not enabled
    """
    native_fn = get_native_symbol("mesh_generate_plane_tbn")
    if native_fn is not None:
        result = native_fn(width, height)
        return result['vertices'], result['indices'], result['tbn_data']
    if width < 2 or height < 2:
        raise ValueError("width and height must be >= 2")
    vertices: List[Dict[str, Any]] = []
    indices: List[int] = []
    tbn_data: List[Dict[str, Any]] = []
    # Grid in XZ plane, Y up
    for j in range(height):
        v = j / (height - 1)
        z = -0.5 + v
        for i in range(width):
            u = i / (width - 1)
            x = -0.5 + u
            pos = np.array([x, 0.0, z], np.float32)
            n = np.array([0.0, 1.0, 0.0], np.float32)
            t = np.array([1.0, 0.0, 0.0], np.float32)
            b = np.array([0.0, 0.0, 1.0], np.float32)
            vertices.append({'position': pos.tolist(), 'normal': n.tolist(), 'uv': [u, v]})
            handedness = float(np.sign(np.dot(np.cross(t, b), n)))
            tbn_data.append({'tangent': t.tolist(), 'bitangent': b.tolist(), 'normal': n.tolist(), 'handedness': handedness})
    # Indices (two triangles per cell)
    def idx(ii, jj):
        return jj * width + ii
    for j in range(height - 1):
        for i in range(width - 1):
            a = idx(i, j)
            b = idx(i + 1, j)
            c = idx(i, j + 1)
            d = idx(i + 1, j + 1)
            indices.extend([a, c, b, b, c, d])
    return vertices, indices, tbn_data


def validate_tbn_data(tbn_data: List[Dict[str, Any]], tolerance: float = 1e-3) -> Dict[str, Any]:
    """Validate TBN data for correctness.
    
    Checks that tangent, bitangent, and normal vectors satisfy the requirements
    for proper tangent space: unit length, orthogonality, and valid handedness.
    
    Parameters
    ----------
    tbn_data : List[Dict[str, Any]]
        TBN data from generate_cube_tbn() or generate_plane_tbn()
    tolerance : float, default 1e-3
        Numerical tolerance for validation checks
        
    Returns
    -------
    validation_results : Dict[str, Any]
        Dictionary containing validation results:
        - 'valid': bool - True if all checks pass
        - 'errors': List[str] - List of validation error messages
        - 'unit_length_ok': bool - True if all vectors are unit length
        - 'orthogonal_ok': bool - True if tangent/normal are orthogonal
        - 'handedness_ok': bool - True if handedness is consistent
        
    Examples
    --------
    >>> vertices, indices, tbn_data = mesh.generate_cube_tbn()
    >>> results = mesh.validate_tbn_data(tbn_data)
    >>> if results['valid']:
    ...     print("TBN data is valid")
    ... else:
    ...     for error in results['errors']:
    ...         print(f"Error: {error}")
    """
    errors = []
    unit_length_ok = True
    orthogonal_ok = True  
    handedness_ok = True
    
    for i, tbn in enumerate(tbn_data):
        # Extract vectors
        t = np.array(tbn['tangent'])
        b = np.array(tbn['bitangent'])
        n = np.array(tbn['normal'])
        h = tbn['handedness']
        
        # Check unit length
        t_len = np.linalg.norm(t)
        b_len = np.linalg.norm(b)
        n_len = np.linalg.norm(n)
        
        if abs(t_len - 1.0) > tolerance:
            errors.append(f"Vertex {i}: Tangent length {t_len:.6f} not unit")
            unit_length_ok = False
        if abs(b_len - 1.0) > tolerance:
            errors.append(f"Vertex {i}: Bitangent length {b_len:.6f} not unit")
            unit_length_ok = False
        if abs(n_len - 1.0) > tolerance:
            errors.append(f"Vertex {i}: Normal length {n_len:.6f} not unit")
            unit_length_ok = False
            
        # Check orthogonality  
        dot_tn = abs(np.dot(t, n))
        if dot_tn > tolerance:
            errors.append(f"Vertex {i}: Tangent-normal dot product {dot_tn:.6f} > tolerance")
            orthogonal_ok = False
            
        # Check handedness consistency
        cross = np.cross(t, b)
        computed_handedness = 1.0 if np.dot(cross, n) >= 0.0 else -1.0
        if abs(computed_handedness - h) > tolerance:
            errors.append(f"Vertex {i}: Handedness {h} inconsistent with cross product")
            handedness_ok = False
    
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'unit_length_ok': unit_length_ok,
        'orthogonal_ok': orthogonal_ok,
        'handedness_ok': handedness_ok,
    }


def compute_tbn_matrix(tbn: Dict[str, Any]) -> np.ndarray:
    """Compute the 3x3 TBN transformation matrix.
    
    Constructs the tangent-to-world space transformation matrix from
    tangent, bitangent, and normal vectors.
    
    Parameters
    ----------
    tbn : Dict[str, Any]
        TBN data containing 'tangent', 'bitangent', 'normal' keys
        
    Returns
    -------
    tbn_matrix : np.ndarray
        3x3 transformation matrix with columns [tangent, bitangent, normal]
        
    Examples
    --------
    >>> vertices, indices, tbn_data = mesh.generate_cube_tbn()
    >>> tbn_matrix = mesh.compute_tbn_matrix(tbn_data[0])
    >>> print(f"TBN matrix shape: {tbn_matrix.shape}")
    >>> print(f"Determinant: {np.linalg.det(tbn_matrix):.6f}")
    """
    t = np.array(tbn['tangent'])
    b = np.array(tbn['bitangent'])  
    n = np.array(tbn['normal'])
    
    return np.column_stack([t, b, n])


# Feature availability check
def has_tbn_support() -> bool:
    """Check if TBN generation features are available.

    Returns
    -------
    bool
        True if TBN generation functions are available
    """
    return get_native_symbol("mesh_generate_cube_tbn") is not None


# ============================================================================
# Triangle Mesh BVH Construction and Path Tracing (Task A3)
# ============================================================================

def make_mesh(vertices: np.ndarray, indices: np.ndarray) -> Dict[str, Any]:
    """
    Create a mesh from vertices and indices arrays for path tracing.

    Args:
        vertices: NumPy array of shape (N, 3) with dtype float32, vertex positions
        indices: NumPy array of shape (M, 3) with dtype uint32, triangle indices (CCW winding)

    Returns:
        Dictionary representing the mesh with keys:
        - 'vertices': validated vertex array
        - 'indices': validated index array
        - 'vertex_count': number of vertices
        - 'triangle_count': number of triangles

    Raises:
        ValueError: If input arrays have wrong shape, dtype, or are not C-contiguous
        RuntimeError: If indices reference invalid vertices
    """
    # Validate vertices
    if not isinstance(vertices, np.ndarray):
        raise ValueError("vertices must be a NumPy array")

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(
            f"vertices must have shape (N, 3); got {vertices.shape}. "
            f"Each vertex should have 3 coordinates (x, y, z)."
        )

    if vertices.dtype not in (np.float32, np.float64):
        raise ValueError(
            f"vertices must have dtype float32 or float64; got {vertices.dtype}. "
            f"Use vertices.astype(np.float32) to convert."
        )

    if not vertices.flags.c_contiguous:
        raise ValueError(
            "vertices must be C-contiguous; use np.ascontiguousarray(vertices) to fix."
        )

    # Convert to float32 if needed
    if vertices.dtype != np.float32:
        vertices = vertices.astype(np.float32)

    # Validate indices
    if not isinstance(indices, np.ndarray):
        raise ValueError("indices must be a NumPy array")

    if indices.ndim != 2 or indices.shape[1] != 3:
        raise ValueError(
            f"indices must have shape (M, 3); got {indices.shape}. "
            f"Each triangle should have 3 vertex indices."
        )

    if indices.dtype not in (np.uint32, np.int32, np.uint64, np.int64):
        raise ValueError(
            f"indices must have integer dtype; got {indices.dtype}. "
            f"Use indices.astype(np.uint32) to convert."
        )

    if not indices.flags.c_contiguous:
        raise ValueError(
            "indices must be C-contiguous; use np.ascontiguousarray(indices) to fix."
        )

    # Convert to uint32 if needed
    if indices.dtype != np.uint32:
        indices = indices.astype(np.uint32)

    # Validate index ranges
    vertex_count = len(vertices)
    max_index = np.max(indices) if len(indices) > 0 else 0

    if max_index >= vertex_count:
        raise RuntimeError(
            f"indices reference vertex {max_index} but only {vertex_count} vertices exist "
            f"(valid range: 0-{vertex_count-1})"
        )

    triangle_count = len(indices)

    # Check for degenerate triangles
    degenerate_count = 0
    for i in range(triangle_count):
        tri_indices = indices[i]
        if len(np.unique(tri_indices)) < 3:
            degenerate_count += 1

    if degenerate_count > 0:
        warnings.warn(f"Mesh contains {degenerate_count} degenerate triangles (repeated vertices)")

    return {
        'vertices': vertices,
        'indices': indices,
        'vertex_count': vertex_count,
        'triangle_count': triangle_count,
    }


def _python_build_bvh(
    vertices: np.ndarray,
    indices: np.ndarray,
    max_leaf_size: int,
) -> Dict[str, Any]:
    """Median-split BVH build used when the native extension is unavailable.

    Mirrors `accel::cpu_bvh::build_bvh_cpu`: split on the longest parent-AABB
    axis at the median centroid, partition by swapping past-the-median triangles
    to the end, and emit children before their parent. The packed node layout is
    identical, so a handle built here uploads through the same native path.
    """
    start_time = time.perf_counter()

    tri = np.asarray(indices, dtype=np.int64).reshape(-1, 3)
    points = np.asarray(vertices, dtype=np.float32)
    v0, v1, v2 = points[tri[:, 0]], points[tri[:, 1]], points[tri[:, 2]]
    tri_min = np.minimum(np.minimum(v0, v1), v2).tolist()
    tri_max = np.maximum(np.maximum(v0, v1), v2).tolist()
    centroids = ((v0 + v1 + v2) / np.float32(3.0)).tolist()

    triangle_count = int(tri.shape[0])
    order = list(range(triangle_count))
    nodes: List[Tuple[List[float], int, List[float], int, int]] = []
    leaf_count = 0
    max_depth = 0

    def compute_bounds(first: int, count: int) -> Tuple[List[float], List[float]]:
        low = [float("inf"), float("inf"), float("inf")]
        high = [float("-inf"), float("-inf"), float("-inf")]
        for position in range(first, first + count):
            triangle = order[position]
            amin = tri_min[triangle]
            amax = tri_max[triangle]
            for axis in range(3):
                if amin[axis] < low[axis]:
                    low[axis] = amin[axis]
                if amax[axis] > high[axis]:
                    high[axis] = amax[axis]
        return low, high

    def find_split(first: int, count: int, aabb):
        if count < 2:
            return None
        extent = [aabb[1][axis] - aabb[0][axis] for axis in range(3)]
        if extent[0] > extent[1] and extent[0] > extent[2]:
            axis = 0
        elif extent[1] > extent[2]:
            axis = 1
        else:
            axis = 2
        values = sorted(centroids[order[position]][axis] for position in range(first, first + count))
        return axis, values[len(values) // 2]

    def partition(first: int, count: int, axis: int, split_position: float) -> int:
        left = first
        right = first + count
        while left < right:
            if centroids[order[left]][axis] < split_position:
                left += 1
            else:
                right -= 1
                order[left], order[right] = order[right], order[left]
        return left

    def build(first: int, count: int, aabb, depth: int) -> int:
        nonlocal leaf_count, max_depth
        max_depth = max(max_depth, depth)
        split = None if count <= max_leaf_size or depth > 64 else find_split(first, count, aabb)
        if split is not None:
            axis, split_position = split
            split_index = partition(first, count, axis, split_position)
            left_count = split_index - first
            right_count = count - left_count
            if left_count == 0 or right_count == 0:
                split = None
        if split is None:
            leaf_count += 1
            index = len(nodes)
            nodes.append((aabb[0], first, aabb[1], count, 1))
            return index

        left_aabb = compute_bounds(first, left_count)
        right_aabb = compute_bounds(split_index, right_count)
        left_child = build(first, left_count, left_aabb, depth + 1)
        right_child = build(split_index, right_count, right_aabb, depth + 1)
        index = len(nodes)
        nodes.append((aabb[0], left_child, aabb[1], right_child, 0))
        return index

    root_aabb = compute_bounds(0, triangle_count)
    build(0, triangle_count, root_aabb, 0)

    packed = np.zeros((len(nodes), _BVH_NODE_BYTES), dtype=np.uint8)
    for index, node in enumerate(nodes):
        aabb_min, left, aabb_max, right, flags = node
        _BVH_NODE_STRUCT.pack_into(
            packed, index * _BVH_NODE_BYTES,
            aabb_min[0], aabb_min[1], aabb_min[2], left,
            aabb_max[0], aabb_max[1], aabb_max[2], right,
            flags, 0,
        )

    build_time_ms = (time.perf_counter() - start_time) * 1000.0
    return {
        'method': "median",
        'triangle_count': triangle_count,
        'node_count': len(nodes),
        'internal_count': len(nodes) - leaf_count,
        'max_depth': max_depth,
        'leaf_count': leaf_count,
        'avg_leaf_size': (triangle_count / leaf_count) if leaf_count else 0.0,
        'build_time_ms': build_time_ms,
        'memory_usage_bytes': len(nodes) * _BVH_NODE_BYTES + triangle_count * 4,
        'world_aabb_min': list(root_aabb[0]),
        'world_aabb_max': list(root_aabb[1]),
        'tri_indices': np.asarray(order, dtype=np.uint32),
        'nodes': packed.reshape(-1),
    }


def build_bvh_cpu(
    mesh: Dict[str, Any],
    method: str = "median",
    max_leaf_size: int = 4,
) -> Dict[str, Any]:
    """
    Build BVH acceleration structure on CPU using specified method.

    Args:
        mesh: Mesh dictionary from make_mesh()
        method: BVH construction method, currently only "median" supported
        max_leaf_size: Maximum triangles per leaf node

    Returns:
        Dictionary representing the BVH with keys:
        - 'method': construction method used
        - 'triangle_count': number of triangles
        - 'node_count': number of BVH nodes actually built
        - 'internal_count': number of internal nodes
        - 'max_depth': maximum tree depth
        - 'leaf_count': number of leaf nodes
        - 'avg_leaf_size': average triangles per leaf
        - 'build_time_ms': construction time in milliseconds
        - 'memory_usage_bytes': bytes held by the packed nodes and triangle order
        - 'world_aabb_min': world bounding box minimum
        - 'world_aabb_max': world bounding box maximum
        - 'tri_indices': uint32 array with the triangle order referenced by leaves
        - 'nodes': flat uint8 array of packed 40-byte BvhNode records

    The native Rust builder (`accel::cpu_bvh`) is used when the compiled
    extension is present; otherwise an equivalent median-split build runs in pure
    Python. Both paths return the same keys and the same packed node layout.

    Raises:
        ValueError: If mesh is invalid or method is unsupported
        RuntimeError: If BVH construction fails
    """
    if not isinstance(mesh, dict) or 'vertices' not in mesh or 'indices' not in mesh:
        raise ValueError("mesh must be a dictionary from make_mesh()")

    if method != "median":
        raise ValueError(f"Unsupported BVH method '{method}'; only 'median' is supported")

    if int(max_leaf_size) < 1:
        raise ValueError("max_leaf_size must be at least 1")

    triangle_count = mesh['triangle_count']
    if triangle_count == 0:
        raise ValueError("Cannot build BVH for mesh with no triangles")

    native_build = get_native_symbol("mesh_build_bvh_cpu")
    if callable(native_build):
        result = dict(
            native_build(
                mesh['vertices'],
                mesh['indices'],
                method,
                int(max_leaf_size),
            )
        )
        result['nodes'] = np.asarray(result['nodes'], dtype=np.uint8)
        result['tri_indices'] = np.asarray(result['tri_indices'], dtype=np.uint32)
        return result

    return _python_build_bvh(mesh['vertices'], mesh['indices'], int(max_leaf_size))


class MeshHandle:
    """
    Handle to an uploaded GPU mesh for use in path tracing.

    This is an opaque handle that represents mesh data uploaded to GPU memory.
    It should not be created directly - use upload_mesh() instead.
    """

    def __init__(self, mesh_data: Dict[str, Any], bvh_data: Dict[str, Any]):
        """Internal constructor - use upload_mesh() instead."""
        self._mesh = mesh_data
        self._bvh = bvh_data
        self._native = None
        self._gpu_memory_bytes = 0

    @property
    def uploaded(self) -> bool:
        """True once the mesh and BVH live in tracked GPU buffers."""
        return self._native is not None

    @property
    def gpu_memory_bytes(self) -> int:
        """Bytes of tracked GPU memory held by the uploaded mesh, index, BVH and descriptor buffers."""
        return int(self._gpu_memory_bytes)

    @property
    def triangle_count(self) -> int:
        """Number of triangles in the mesh."""
        return self._mesh['triangle_count']

    @property
    def vertex_count(self) -> int:
        """Number of vertices in the mesh."""
        return self._mesh['vertex_count']

    @property
    def node_count(self) -> int:
        """Number of BVH nodes."""
        return self._bvh['node_count']

    @property
    def build_stats(self) -> Dict[str, Any]:
        """BVH build statistics."""
        return {
            'triangle_count': self.triangle_count,
            'node_count': self.node_count,
            'max_depth': self._bvh['max_depth'],
            'leaf_count': self._bvh['leaf_count'],
            'avg_leaf_size': self._bvh['avg_leaf_size'],
            'build_time_ms': self._bvh['build_time_ms'],
            'memory_usage_bytes': self._bvh['memory_usage_bytes'],
        }

    @property
    def world_aabb(self) -> Tuple[np.ndarray, np.ndarray]:
        """World axis-aligned bounding box as (min, max) NumPy arrays."""
        min_bounds = np.array(self._bvh['world_aabb_min'], dtype=np.float32)
        max_bounds = np.array(self._bvh['world_aabb_max'], dtype=np.float32)
        return min_bounds, max_bounds

    def __repr__(self) -> str:
        return (f"MeshHandle(triangles={self.triangle_count}, "
                f"vertices={self.vertex_count}, nodes={self.node_count}, "
                f"uploaded={self.uploaded})")


def upload_mesh(mesh: Dict[str, Any], bvh: Dict[str, Any]) -> MeshHandle:
    """
    Upload mesh and BVH data to GPU for use in path tracing.

    Args:
        mesh: Mesh dictionary from make_mesh()
        bvh: BVH dictionary from build_bvh_cpu()

    Returns:
        MeshHandle that can be passed to path tracing functions

    Raises:
        ValueError: If mesh or BVH data is invalid
        RuntimeError: If GPU upload fails (e.g., out of memory)
    """
    if not isinstance(mesh, dict) or 'vertices' not in mesh or 'indices' not in mesh:
        raise ValueError("mesh must be a dictionary from make_mesh()")

    if not isinstance(bvh, dict) or 'triangle_count' not in bvh:
        raise ValueError("bvh must be a dictionary from build_bvh_cpu()")

    # Validate mesh and BVH are compatible
    if mesh['triangle_count'] != bvh['triangle_count']:
        raise ValueError(
            f"Mesh and BVH triangle counts don't match: "
            f"mesh={mesh['triangle_count']}, bvh={bvh['triangle_count']}"
        )

    tri_indices = bvh.get('tri_indices')
    nodes = bvh.get('nodes')
    if tri_indices is None or nodes is None:
        raise ValueError(
            "bvh is missing 'tri_indices'/'nodes'; rebuild it with build_bvh_cpu()"
        )

    native_upload = get_native_symbol("mesh_upload_gpu")
    if not callable(native_upload):
        raise RuntimeError(
            "GPU mesh upload requires the compiled forge3d extension; rebuild it "
            "with: maturin develop --release"
        )

    native_handle = native_upload(
        np.ascontiguousarray(mesh['vertices'], dtype=np.float32),
        np.ascontiguousarray(mesh['indices'], dtype=np.uint32),
        np.ascontiguousarray(np.asarray(tri_indices, dtype=np.uint32).reshape(-1)),
        np.ascontiguousarray(np.asarray(nodes, dtype=np.uint8).reshape(-1)),
    )

    handle = MeshHandle(mesh, bvh)
    handle._native = native_handle
    handle._gpu_memory_bytes = int(native_handle.gpu_memory_bytes)

    return handle


def create_triangle_mesh() -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a simple triangle mesh for testing.

    Returns:
        Tuple of (vertices, indices) arrays suitable for make_mesh()
    """
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
    ], dtype=np.float32)

    indices = np.array([
        [0, 1, 2],
    ], dtype=np.uint32)

    return vertices, indices


def create_cube_mesh() -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a unit cube mesh for testing (12 triangles, 8 vertices).

    Returns:
        Tuple of (vertices, indices) arrays suitable for make_mesh()
    """
    vertices = np.array([
        # Front face
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        # Back face
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
    ], dtype=np.float32)

    indices = np.array([
        # Front face
        [0, 1, 2], [0, 2, 3],
        # Right face
        [1, 5, 6], [1, 6, 2],
        # Back face
        [5, 4, 7], [5, 7, 6],
        # Left face
        [4, 0, 3], [4, 3, 7],
        # Top face
        [3, 2, 6], [3, 6, 7],
        # Bottom face
        [4, 5, 1], [4, 1, 0],
    ], dtype=np.uint32)

    return vertices, indices


def create_quad_mesh() -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a quad mesh for testing (2 triangles, 4 vertices).

    Returns:
        Tuple of (vertices, indices) arrays suitable for make_mesh()
    """
    vertices = np.array([
        [-1.0, -1.0, 0.0],
        [ 1.0, -1.0, 0.0],
        [ 1.0,  1.0, 0.0],
        [-1.0,  1.0, 0.0],
    ], dtype=np.float32)

    indices = np.array([
        [0, 1, 2],  # First triangle
        [0, 2, 3],  # Second triangle
    ], dtype=np.uint32)

    return vertices, indices


def validate_mesh_arrays(vertices: np.ndarray, indices: np.ndarray) -> None:
    """
    Validate vertex and index arrays before creating a mesh.

    Args:
        vertices: Vertex position array
        indices: Triangle index array

    Raises:
        ValueError: If arrays are invalid for mesh creation
    """
    try:
        # This will perform all the validation
        make_mesh(vertices, indices)
    except (ValueError, RuntimeError) as e:
        # Re-raise with more context
        raise ValueError(f"Mesh validation failed: {e}")


def mesh_info(handle: MeshHandle) -> Dict[str, Any]:
    """
    Get information about an uploaded mesh.

    Args:
        handle: MeshHandle from upload_mesh()

    Returns:
        Dictionary with mesh information including statistics and bounds
    """
    if not isinstance(handle, MeshHandle):
        raise ValueError("handle must be a MeshHandle from upload_mesh()")

    aabb_min, aabb_max = handle.world_aabb
    extent = aabb_max - aabb_min

    info = {
        'triangle_count': handle.triangle_count,
        'vertex_count': handle.vertex_count,
        'bvh_node_count': handle.node_count,
        'uploaded': handle.uploaded,
        'gpu_memory_bytes': handle.gpu_memory_bytes,
        'world_bounds': {
            'min': aabb_min.tolist(),
            'max': aabb_max.tolist(),
            'extent': extent.tolist(),
            'center': ((aabb_min + aabb_max) * 0.5).tolist(),
        },
        'build_stats': handle.build_stats,
    }

    return info
