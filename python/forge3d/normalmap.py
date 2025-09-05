"""
Normal mapping utilities for forge3d

Provides Python API for creating and working with normal maps, including
loading normal map textures, creating test patterns, and validating TBN data.
"""

from typing import Tuple, Optional, Union, Dict, Any
import numpy as np
from pathlib import Path

try:
    from . import _forge3d
    # Check if normal mapping functions are available
    _HAS_NORMAL_MAPPING = hasattr(_forge3d, 'create_checkerboard_normal_texture')
except ImportError:
    _forge3d = None
    _HAS_NORMAL_MAPPING = False


def has_normal_mapping_support() -> bool:
    """Check if normal mapping features are available.
    
    Returns
    -------
    bool
        True if normal mapping functions are available
    """
    return _HAS_NORMAL_MAPPING


def create_checkerboard_normal_map(size: int = 256) -> np.ndarray:
    """Create a checkerboard pattern normal map for testing.
    
    Creates a procedural normal map with alternating flat and slightly
    perturbed normals in a checkerboard pattern. Useful for validating
    normal mapping implementation and visual testing.
    
    Parameters
    ----------
    size : int, default 256
        Width and height of the normal map texture in pixels
        
    Returns
    -------
    np.ndarray
        RGBA normal map as (height, width, 4) uint8 array.
        RGB channels contain encoded normal vectors, alpha is 255.
        
    Examples
    --------
    >>> normal_map = create_checkerboard_normal_map(128)
    >>> print(f"Normal map shape: {normal_map.shape}")
    Normal map shape: (128, 128, 4)
    
    >>> # Check that normals are properly encoded
    >>> flat_normal = normal_map[0, 0, :3]  # Should be ~(128, 128, 255)
    >>> print(f"Flat normal (encoded): {flat_normal}")
    """
    if size <= 0:
        raise ValueError("Size must be positive")
        
    # Create checkerboard normal map
    normal_map = np.zeros((size, size, 4), dtype=np.uint8)
    
    for y in range(size):
        for x in range(size):
            checker = ((x // 8) + (y // 8)) % 2
            if checker == 0:
                # Flat normal (0, 0, 1) in tangent space -> (128, 128, 255) encoded
                normal_map[y, x] = [128, 128, 255, 255]
            else:
                # Create a properly normalized tilted normal and encode it
                # Tangent space normal (0.2, 0.2, sqrt(1 - 0.2^2 - 0.2^2))
                nx, ny = 0.2, 0.2
                nz = np.sqrt(max(0.0, 1.0 - nx*nx - ny*ny))
                normal_vec = np.array([nx, ny, nz])
                encoded = encode_normal_vector(normal_vec)
                normal_map[y, x] = [encoded[0], encoded[1], encoded[2], 255]
                
    return normal_map


def encode_normal_vector(normal: np.ndarray) -> np.ndarray:
    """Encode a normal vector from [-1,1] range to [0,255] texture format.
    
    Converts normalized normal vectors from the standard [-1,1] range used
    in calculations to the [0,255] uint8 range used in texture storage.
    
    Parameters
    ----------
    normal : np.ndarray
        Normal vector(s) in [-1,1] range. Can be:
        - Single normal: shape (3,) 
        - Multiple normals: shape (N, 3) or (H, W, 3)
        
    Returns
    -------
    np.ndarray
        Encoded normal(s) in [0,255] uint8 range, same shape as input
        
    Examples
    --------
    >>> # Encode a single upward-pointing normal
    >>> normal = np.array([0.0, 0.0, 1.0])
    >>> encoded = encode_normal_vector(normal)
    >>> print(encoded)  # Should be [128, 128, 255]
    
    >>> # Encode a tilted normal
    >>> tilted = np.array([0.3, 0.3, 0.9])  
    >>> encoded_tilted = encode_normal_vector(tilted)
    """
    normal = np.asarray(normal, dtype=np.float32)
    
    # Clamp to [-1, 1] range for safety
    normal = np.clip(normal, -1.0, 1.0)
    
    # Convert from [-1, 1] to [0, 255]
    encoded = ((normal + 1.0) * 127.5).astype(np.uint8)
    
    return encoded


def decode_normal_vector(encoded: np.ndarray) -> np.ndarray:
    """Decode a normal vector from [0,255] texture format to [-1,1] range.
    
    Converts encoded normal vectors from [0,255] uint8 texture format back
    to the [-1,1] floating-point range used in calculations.
    
    Parameters  
    ----------
    encoded : np.ndarray
        Encoded normal vector(s) in [0,255] uint8 range
        
    Returns
    -------
    np.ndarray
        Decoded normal(s) in [-1,1] float32 range, same shape as input
        
    Examples
    --------
    >>> # Decode a flat normal texture value
    >>> encoded = np.array([128, 128, 255], dtype=np.uint8)
    >>> decoded = decode_normal_vector(encoded)
    >>> print(f"Decoded: {decoded}")  # Should be ~[0, 0, 1]
    """
    encoded = np.asarray(encoded, dtype=np.uint8)
    
    # Convert from [0, 255] to [-1, 1]
    decoded = (encoded.astype(np.float32) / 127.5) - 1.0
    
    return decoded


def validate_normal_map(normal_map: np.ndarray, tolerance: float = 0.1) -> Dict[str, Any]:
    """Validate a normal map texture for correctness.
    
    Checks that a normal map texture contains valid encoded normal vectors
    and meets basic quality requirements for use in rendering.
    
    Parameters
    ----------
    normal_map : np.ndarray
        Normal map texture as (H, W, 3) or (H, W, 4) array
    tolerance : float, default 1e-2
        Numerical tolerance for validation checks
        
    Returns
    -------
    Dict[str, Any]
        Validation results containing:
        - 'valid': bool - True if all checks pass
        - 'errors': List[str] - List of validation error messages  
        - 'unit_length_ok': bool - True if decoded normals are unit length
        - 'range_ok': bool - True if values are in valid [0,255] range
        - 'z_positive_ok': bool - True if Z components are reasonable
        
    Examples
    --------
    >>> normal_map = create_checkerboard_normal_map(64)
    >>> results = validate_normal_map(normal_map)
    >>> if results['valid']:
    ...     print("Normal map is valid")
    ... else:
    ...     for error in results['errors']:
    ...         print(f"Error: {error}")
    """
    normal_map = np.asarray(normal_map)
    errors = []
    unit_length_ok = True
    range_ok = True
    z_positive_ok = True
    
    # Check shape
    if len(normal_map.shape) != 3:
        errors.append(f"Normal map must be 3D array, got shape {normal_map.shape}")
        return {
            'valid': False,
            'errors': errors,
            'unit_length_ok': False,
            'range_ok': False,
            'z_positive_ok': False,
        }
    
    if normal_map.shape[2] not in [3, 4]:
        errors.append(f"Normal map must have 3 or 4 channels, got {normal_map.shape[2]}")
    
    # Check data type and range
    if normal_map.dtype != np.uint8:
        errors.append(f"Normal map must be uint8, got {normal_map.dtype}")
    
    # Check range [0, 255] for uint8
    if np.any(normal_map < 0) or np.any(normal_map > 255):
        errors.append("Normal map values outside valid [0, 255] range")
        range_ok = False
    
    # Decode normals and check unit length
    rgb_channels = normal_map[:, :, :3]
    decoded_normals = decode_normal_vector(rgb_channels)
    
    # Check unit length (should be close to 1.0)
    lengths = np.linalg.norm(decoded_normals, axis=2)
    length_errors = np.abs(lengths - 1.0) > tolerance
    
    if np.any(length_errors):
        error_count = np.sum(length_errors)
        total_pixels = decoded_normals.shape[0] * decoded_normals.shape[1]
        errors.append(f"{error_count}/{total_pixels} pixels have non-unit length normals")
        unit_length_ok = False
    
    # Check Z component (should be mostly positive for tangent-space normals)
    z_components = decoded_normals[:, :, 2]
    negative_z_count = np.sum(z_components < 0)
    if negative_z_count > 0.1 * z_components.size:  # More than 10% negative
        errors.append(f"Too many negative Z components: {negative_z_count}/{z_components.size}")
        z_positive_ok = False
    
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'unit_length_ok': unit_length_ok,
        'range_ok': range_ok,
        'z_positive_ok': z_positive_ok,
    }


def compute_luminance_difference(image1: np.ndarray, image2: np.ndarray) -> float:
    """Compute luminance difference between two images.
    
    Calculates the relative difference in mean luminance between two images,
    useful for validating normal mapping effectiveness according to acceptance criteria.
    
    Parameters
    ----------
    image1 : np.ndarray
        First image as (H, W, 3) or (H, W, 4) array
    image2 : np.ndarray  
        Second image as (H, W, 3) or (H, W, 4) array
        
    Returns
    -------
    float
        Relative luminance difference as percentage (0.0 to 100.0+)
        
    Examples
    --------
    >>> # Test that normal mapping changes luminance significantly
    >>> flat_render = render_with_flat_normals()
    >>> normal_render = render_with_normal_map() 
    >>> diff = compute_luminance_difference(normal_render, flat_render)
    >>> assert diff >= 10.0  # AC requirement: ≥10% difference
    """
    image1 = np.asarray(image1, dtype=np.float32)
    image2 = np.asarray(image2, dtype=np.float32)
    
    # Convert to grayscale using standard RGB weights
    if len(image1.shape) == 3 and image1.shape[2] >= 3:
        lum1 = 0.299 * image1[:, :, 0] + 0.587 * image1[:, :, 1] + 0.114 * image1[:, :, 2]
    else:
        lum1 = np.mean(image1, axis=2) if len(image1.shape) == 3 else image1
        
    if len(image2.shape) == 3 and image2.shape[2] >= 3:
        lum2 = 0.299 * image2[:, :, 0] + 0.587 * image2[:, :, 1] + 0.114 * image2[:, :, 2]  
    else:
        lum2 = np.mean(image2, axis=2) if len(image2.shape) == 3 else image2
    
    # Compute mean luminance
    mean_lum1 = np.mean(lum1)
    mean_lum2 = np.mean(lum2)
    
    # Compute relative difference as percentage
    if mean_lum2 > 1e-8:  # Avoid division by zero
        diff_percent = abs(mean_lum1 - mean_lum2) / mean_lum2 * 100.0
    else:
        diff_percent = 100.0 if mean_lum1 > 1e-8 else 0.0
        
    return diff_percent


def load_normal_map(path: Union[str, Path]) -> np.ndarray:
    """Load a normal map from an image file.
    
    Loads a normal map texture from common image formats (PNG, JPEG, TGA, etc.)
    and ensures it's in the correct format for use with the normal mapping pipeline.
    
    Parameters
    ----------
    path : str or Path
        Path to the normal map image file
        
    Returns
    -------
    np.ndarray
        Normal map as (height, width, 4) uint8 array ready for GPU upload
        
    Raises
    ------
    FileNotFoundError
        If the specified file doesn't exist
    ValueError
        If the image format is not supported or invalid
        
    Examples
    --------
    >>> normal_map = load_normal_map("assets/brick_normal.png")
    >>> print(f"Loaded normal map: {normal_map.shape}")
    
    >>> # Validate the loaded normal map
    >>> results = validate_normal_map(normal_map)
    >>> assert results['valid'], "Invalid normal map"
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("PIL (Pillow) is required for loading normal maps")
    
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Normal map file not found: {path}")
    
    # Load image
    image = Image.open(path)
    
    # Convert to RGBA if needed
    if image.mode != 'RGBA':
        if image.mode == 'RGB':
            image = image.convert('RGBA')
        else:
            # Convert grayscale or other formats to RGB first, then RGBA
            image = image.convert('RGB').convert('RGBA')
    
    # Convert to numpy array
    normal_map = np.array(image, dtype=np.uint8)
    
    # Validate the loaded normal map
    validation = validate_normal_map(normal_map)
    if not validation['valid']:
        print(f"Warning: Normal map validation failed: {validation['errors']}")
    
    return normal_map


def save_normal_map(normal_map: np.ndarray, path: Union[str, Path]) -> None:
    """Save a normal map to an image file.
    
    Saves a normal map texture to disk in PNG format with proper encoding.
    
    Parameters
    ----------
    normal_map : np.ndarray
        Normal map as (height, width, 3) or (height, width, 4) uint8 array
    path : str or Path
        Output path for the saved normal map image
        
    Examples
    --------
    >>> normal_map = create_checkerboard_normal_map(128)
    >>> save_normal_map(normal_map, "test_normal.png")
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("PIL (Pillow) is required for saving normal maps")
    
    path = Path(path)
    normal_map = np.asarray(normal_map, dtype=np.uint8)
    
    # Ensure we have 4 channels (RGBA)
    if len(normal_map.shape) == 3 and normal_map.shape[2] == 3:
        # Add alpha channel  
        alpha = np.full((normal_map.shape[0], normal_map.shape[1], 1), 255, dtype=np.uint8)
        normal_map = np.concatenate([normal_map, alpha], axis=2)
    
    # Convert to PIL image and save
    image = Image.fromarray(normal_map, 'RGBA')
    image.save(path, 'PNG')