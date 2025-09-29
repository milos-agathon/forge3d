# python/forge3d/adapters/datashader_adapter.py
# Datashader Canvas/shade output adapter for forge3d overlays.
# This exists to convert Datashader aggregation arrays to forge3d RGBA overlays
# with zero-copy semantics and coordinate validation.
"""
Datashader integration adapter for forge3d.

This module provides seamless integration between Datashader's Canvas/shade
outputs and forge3d's overlay texture system, enabling large-scale point/line
rendering with minimal memory overhead.
"""

from typing import Union, Optional, Tuple, Any, Dict
import numpy as np
import warnings

# Optional datashader dependency
try:
    import datashader as ds
    import datashader.transfer_functions as tf
    from datashader.core import Canvas
    import xarray as xr
    _HAS_DATASHADER = True
except ImportError:
    _HAS_DATASHADER = False
    # Stub classes for type hints
    Canvas = Any
    xr = Any


def is_datashader_available() -> bool:
    """Check if datashader is available."""
    return _HAS_DATASHADER


def _require_datashader():
    """Raise ImportError with helpful message if datashader not available."""
    if not _HAS_DATASHADER:
        raise ImportError(
            "Datashader is required for datashader adapters. "
            "Install with: pip install datashader"
        )


class DatashaderAdapter:
    """
    Adapter for converting Datashader outputs to forge3d overlays.
    
    This class provides methods to convert Datashader Canvas aggregations
    and shaded images into RGBA arrays suitable for forge3d overlay textures,
    with coordinate alignment validation and zero-copy optimization.
    """
    
    def __init__(self):
        """Initialize the DatashaderAdapter."""
        _require_datashader()
        self._copy_count = 0  # Track copies for performance monitoring
    
    @property
    def copy_count(self) -> int:
        """Get the number of copies made (for performance monitoring)."""
        return self._copy_count
    
    def reset_copy_count(self):
        """Reset the copy counter."""
        self._copy_count = 0


def rgba_view_from_agg(agg_or_img: Union[Any, np.ndarray]) -> np.ndarray:
    """
    Convert Datashader aggregation or shaded image to RGBA view.
    
    Args:
        agg_or_img: Either a Datashader aggregation (xarray.DataArray) or
                   an already-shaded image (numpy array)
    
    Returns:
        RGBA array as (H, W, 4) uint8, zero-copy where possible
        
    Raises:
        ImportError: If datashader is not available
        ValueError: If input format is not supported
    """
    _require_datashader()
    
    if _HAS_DATASHADER and hasattr(agg_or_img, 'values'):
        # Convert aggregation to image using default colormap
        img = tf.shade(agg_or_img, how='linear')
        rgba = img.values  # Extract numpy array from xarray
    elif isinstance(agg_or_img, np.ndarray):
        rgba = agg_or_img
    else:
        raise ValueError(
            f"Expected xarray.DataArray or numpy.ndarray, got {type(agg_or_img)}"
        )
    
    # Ensure RGBA format (H, W, 4) uint8
    if rgba.ndim != 3 or rgba.shape[2] not in [3, 4]:
        raise ValueError(
            f"Expected (H,W,3) or (H,W,4) array, got shape {rgba.shape}"
        )
    
    # Convert RGB to RGBA if needed
    if rgba.shape[2] == 3:
        # Add alpha channel (requires copy)
        rgba_with_alpha = np.empty(
            (rgba.shape[0], rgba.shape[1], 4), 
            dtype=rgba.dtype
        )
        rgba_with_alpha[..., :3] = rgba
        rgba_with_alpha[..., 3] = 255  # Full opacity
        rgba = rgba_with_alpha
        
    # Ensure uint8 dtype
    if rgba.dtype != np.uint8:
        rgba = rgba.astype(np.uint8)
        
    # Ensure C-contiguous for efficient GPU upload
    if not rgba.flags.c_contiguous:
        rgba = np.ascontiguousarray(rgba)
    
    return rgba


def premultiply_rgba(rgba: np.ndarray) -> np.ndarray:
    """
    Return a premultiplied-alpha copy of the given RGBA uint8 array.

    - Input must be shape (H, W, 4), dtype=uint8, C-contiguous.
    - Returns a new array (shares_memory=False) with RGB multiplied by A/255.
    """
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError(f"Expected (H,W,4) RGBA array, got shape {rgba.shape}")
    if rgba.dtype != np.uint8:
        raise ValueError(f"Expected uint8 dtype, got {rgba.dtype}")
    if not rgba.flags.c_contiguous:
        raise ValueError("RGBA array must be C-contiguous for premultiplication")

    out = rgba.copy()
    alpha = out[..., 3].astype(np.float32) / 255.0
    if np.all(alpha == 1.0):
        # Already effectively premultiplied (opaque), return copy
        return out
    # Multiply channels by alpha
    for c in range(3):
        channel = out[..., c].astype(np.float32) * alpha
        out[..., c] = np.clip(channel + 0.5, 0.0, 255.0).astype(np.uint8)
    return out


def validate_alignment(extent: Tuple[float, float, float, float], 
                      transform: Optional[Any], 
                      width: int, 
                      height: int) -> Dict[str, Any]:
    """
    Validate coordinate alignment between datashader extent and forge3d transform.
    
    Args:
        extent: (xmin, ymin, xmax, ymax) in world coordinates
        transform: Optional affine transform (from rasterio/GDAL)
        width: Image width in pixels  
        height: Image height in pixels
        
    Returns:
        Dictionary with validation results and pixel error metrics
        
    Raises:
        ValueError: If alignment error exceeds 0.5 pixels or extent invalid
    """
    # Validation does not require datashader; keep this function usable without it
    
    xmin, ymin, xmax, ymax = extent
    
    # Calculate pixel size
    pixel_width = (xmax - xmin) / width
    pixel_height = (ymax - ymin) / height
    
    # Basic validation without transform (simple case)
    if transform is None:
        # For simple extent-based alignment, check if coordinates make sense
        if xmin >= xmax or ymin >= ymax:
            raise ValueError(f"Invalid extent: {extent}")
        
        return {
            'pixel_width': pixel_width,
            'pixel_height': pixel_height,
            'pixel_error_x': 0.0,
            'pixel_error_y': 0.0,
            'within_tolerance': True,
            'transform_provided': False
        }
    
    # Advanced validation with affine transform
    # This would integrate with rasterio transforms if available
    # For now, implement basic checks
    try:
        # Try to get transform parameters if it's a rasterio Affine
        if hasattr(transform, 'a') and hasattr(transform, 'e'):
            # Rasterio Affine transform
            transform_pixel_width = abs(transform.a)
            transform_pixel_height = abs(transform.e)
            
            # Calculate pixel center offset error
            error_x = abs(pixel_width - transform_pixel_width) / pixel_width
            error_y = abs(pixel_height - transform_pixel_height) / pixel_height
            
            # Convert to pixel units
            pixel_error_x = error_x * width
            pixel_error_y = error_y * height
            
            within_tolerance = pixel_error_x <= 0.5 and pixel_error_y <= 0.5
            
            if not within_tolerance:
                raise ValueError(
                    f"Alignment error exceeds 0.5px tolerance: "
                    f"x_error={pixel_error_x:.3f}px, y_error={pixel_error_y:.3f}px"
                )
            
            return {
                'pixel_width': pixel_width,
                'pixel_height': pixel_height, 
                'transform_pixel_width': transform_pixel_width,
                'transform_pixel_height': transform_pixel_height,
                'pixel_error_x': pixel_error_x,
                'pixel_error_y': pixel_error_y,
                'within_tolerance': within_tolerance,
                'transform_provided': True
            }
    except AttributeError:
        pass
    
    # Fallback for unknown transform types
    warnings.warn(
        f"Unknown transform type {type(transform)}, skipping detailed validation"
    )
    
    return {
        'pixel_width': pixel_width,
        'pixel_height': pixel_height,
        'pixel_error_x': 0.0,
        'pixel_error_y': 0.0,
        'within_tolerance': True,
        'transform_provided': True,
        'transform_type': str(type(transform))
    }


def to_overlay_texture(rgba: np.ndarray,
                      extent: Tuple[float, float, float, float],
                      *,
                      premultiply: bool = False,
                      transform: Optional[Any] = None) -> Dict[str, Any]:
    """
    Prepare RGBA array and extent for forge3d overlay texture.
    
    Args:
        rgba: RGBA array as (H, W, 4) uint8
        extent: (xmin, ymin, xmax, ymax) in world coordinates
        
    Returns:
        Dictionary with texture data and metadata for forge3d
        
    Raises:
        ImportError: If datashader is not available
        ValueError: If RGBA format is invalid
    """
    _require_datashader()
    
    # Validate RGBA format
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError(
            f"Expected (H,W,4) RGBA array, got shape {rgba.shape}"
        )
    
    if rgba.dtype != np.uint8:
        raise ValueError(
            f"Expected uint8 dtype, got {rgba.dtype}"
        )
    
    if not rgba.flags.c_contiguous:
        raise ValueError("RGBA array must be C-contiguous for GPU upload")

    # Premultiply alpha if requested (creates a copy)
    premultiplied = False
    prepared = rgba
    if premultiply:
        prepared = premultiply_rgba(rgba)
        premultiplied = True
    
    height, width = prepared.shape[:2]
    xmin, ymin, xmax, ymax = extent
    
    # Calculate texture parameters
    pixel_width = (xmax - xmin) / width
    pixel_height = (ymax - ymin) / height
    
    return {
        'rgba': prepared,
        'width': width,
        'height': height,
        'extent': extent,
        'pixel_width': pixel_width, 
        'pixel_height': pixel_height,
        'format': 'RGBA8',
        'bytes_per_pixel': 4,
        'total_bytes': prepared.nbytes,
        'is_contiguous': prepared.flags.c_contiguous,
        'shares_memory': np.shares_memory(prepared, rgba),
        'premultiplied': premultiplied,
        'transform': transform,
    }


# Convenience functions for common operations
def shade_to_overlay(agg: Any,
                    extent: Tuple[float, float, float, float],
                    cmap: str = 'viridis',
                    how: str = 'linear',
                    *,
                    premultiply: bool = False,
                    transform: Optional[Any] = None) -> Dict[str, Any]:
    """
    Convert Datashader aggregation directly to forge3d overlay.
    
    Args:
        agg: Datashader aggregation array
        extent: (xmin, ymin, xmax, ymax) coordinates
        cmap: Colormap name for shading
        how: Shading method ('linear', 'log', 'eq_hist')
        
    Returns:
        Dictionary ready for forge3d overlay texture
    """
    _require_datashader()
    
    # Shade the aggregation
    img = tf.shade(agg, cmap=cmap, how=how)
    
    # Convert to RGBA
    rgba = rgba_view_from_agg(img)
    
    # Validate alignment (basic check without transform)
    validate_alignment(extent, transform, rgba.shape[1], rgba.shape[0])
    
    # Prepare overlay texture
    return to_overlay_texture(rgba, extent, premultiply=premultiply, transform=transform)


def get_datashader_info() -> Dict[str, Any]:
    """
    Get information about datashader availability and configuration.
    
    Returns:
        Dictionary with datashader status and version info
    """
    if not _HAS_DATASHADER:
        return {
            'available': False,
            'version': None,
            'transfer_functions': [],
            'colormaps': []
        }
    
    try:
        import datashader
        version = getattr(datashader, '__version__', 'unknown')
    except AttributeError:
        version = 'unknown'
    
    # Get available colormaps from datashader
    try:
        import colorcet as cc
        colormaps = list(cc.palette.keys())
    except ImportError:
        colormaps = ['viridis', 'plasma', 'inferno', 'magma']  # Common defaults
    
    return {
        'available': True,
        'version': version,
        'transfer_functions': ['linear', 'log', 'eq_hist', 'cbrt'],
        'colormaps': colormaps[:10] if len(colormaps) > 10 else colormaps,
        'premultiply_supported': True,
        'premultiply_default': False,
        'total_colormaps': len(colormaps)
    }


# Export main classes and functions
__all__ = [
    'DatashaderAdapter',
    'is_datashader_available', 
    'rgba_view_from_agg',
    'premultiply_rgba',
    'validate_alignment',
    'to_overlay_texture',
    'shade_to_overlay',
    'get_datashader_info'
]
