"""
Mesh generation and TBN (Tangent, Bitangent, Normal) utilities

Provides functions for generating mesh data and computing tangent space
information required for normal mapping and PBR rendering.
"""

from typing import Dict, List, Tuple, Any
import numpy as np

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
    if not _HAS_TBN:
        raise RuntimeError("TBN generation not available - enable-tbn feature required")
    
    result = _forge3d.mesh_generate_cube_tbn()
    return result['vertices'], result['indices'], result['tbn_data']


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
    if not _HAS_TBN:
        raise RuntimeError("TBN generation not available - enable-tbn feature required")
        
    result = _forge3d.mesh_generate_plane_tbn(width, height)
    return result['vertices'], result['indices'], result['tbn_data']


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