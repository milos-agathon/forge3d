"""
Mesh generation, TBN (Tangent, Bitangent, Normal) utilities, and BVH construction

Provides functions for:
1. Generating mesh data and computing tangent space information for normal mapping and PBR rendering
2. Triangle mesh BVH construction for path tracing (Task A3)
3. GPU mesh upload for accelerated ray tracing
"""

from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import warnings

try:
    from . import _forge3d
    # Check if TBN functions are available (gated by feature flag)
    _HAS_TBN = hasattr(_forge3d, 'mesh_generate_cube_tbn')
except ImportError:
    _forge3d = None
    _HAS_TBN = False


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
    if _HAS_TBN:
        result = _forge3d.mesh_generate_cube_tbn()
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
    for face_idx, (n, t, b, corners) in enumerate(faces):
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
    if _HAS_TBN:
        result = _forge3d.mesh_generate_plane_tbn(width, height)
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
    return _HAS_TBN


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


def build_bvh_cpu(mesh: Dict[str, Any], method: str = "median") -> Dict[str, Any]:
    """
    Build BVH acceleration structure on CPU using specified method.

    Args:
        mesh: Mesh dictionary from make_mesh()
        method: BVH construction method, currently only "median" supported

    Returns:
        Dictionary representing the BVH with keys:
        - 'method': construction method used
        - 'triangle_count': number of triangles
        - 'node_count': number of BVH nodes
        - 'max_depth': maximum tree depth
        - 'leaf_count': number of leaf nodes
        - 'avg_leaf_size': average triangles per leaf
        - 'build_time_ms': construction time in milliseconds
        - 'memory_usage_bytes': estimated memory usage
        - 'world_aabb_min': world bounding box minimum
        - 'world_aabb_max': world bounding box maximum

    Raises:
        ValueError: If mesh is invalid or method is unsupported
        RuntimeError: If BVH construction fails
    """
    if not isinstance(mesh, dict) or 'vertices' not in mesh or 'indices' not in mesh:
        raise ValueError("mesh must be a dictionary from make_mesh()")

    if method != "median":
        raise ValueError(f"Unsupported BVH method '{method}'; only 'median' is supported")

    triangle_count = mesh['triangle_count']
    if triangle_count == 0:
        raise ValueError("Cannot build BVH for mesh with no triangles")

    # This would call into Rust via PyO3, but for now return mock data
    # In the real implementation, this would call the Rust BVH builder
    import time
    start_time = time.time()

    # Mock BVH statistics - in real implementation this comes from Rust
    build_time_ms = (time.time() - start_time) * 1000

    # Estimate nodes needed (binary tree with max 4 triangles per leaf)
    estimated_leaves = max(1, (triangle_count + 3) // 4)
    estimated_nodes = estimated_leaves * 2 - 1  # Binary tree property

    # Mock world AABB calculation
    vertices = mesh['vertices']
    world_aabb_min = np.min(vertices, axis=0)
    world_aabb_max = np.max(vertices, axis=0)

    bvh = {
        'method': method,
        'triangle_count': triangle_count,
        'node_count': estimated_nodes,
        'max_depth': max(1, int(np.log2(estimated_nodes)) + 1),
        'leaf_count': estimated_leaves,
        'avg_leaf_size': triangle_count / estimated_leaves,
        'build_time_ms': build_time_ms,
        'memory_usage_bytes': estimated_nodes * 32 + triangle_count * 4,  # Rough estimate
        'world_aabb_min': world_aabb_min.tolist(),
        'world_aabb_max': world_aabb_max.tolist(),
    }

    return bvh


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
        self._uploaded = True

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
                f"vertices={self.vertex_count}, nodes={self.node_count})")


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

    # In real implementation, this would call Rust code to upload to GPU
    # For now, just create a handle with the data
    handle = MeshHandle(mesh, bvh)

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
        'world_bounds': {
            'min': aabb_min.tolist(),
            'max': aabb_max.tolist(),
            'extent': extent.tolist(),
            'center': ((aabb_min + aabb_max) * 0.5).tolist(),
        },
        'build_stats': handle.build_stats,
    }

    return info
