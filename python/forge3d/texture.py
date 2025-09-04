"""Texture processing and mipmap generation utilities."""

import numpy as np
from typing import List, Tuple, Optional, Union, Literal
import warnings

# Import the Rust backend for CPU mipmap generation (this will need to be exposed from Rust)
try:
    from . import _forge3d as _f3d
except ImportError as e:
    warnings.warn(f"Could not import forge3d backend: {e}")
    _f3d = None


def generate_mipmaps(
    image: np.ndarray,
    method: Literal["box", "gpu", "weighted"] = "box",
    gamma_aware: bool = False,
    gamma: float = 2.2,
    max_levels: Optional[int] = None
) -> List[np.ndarray]:
    """Generate a complete mipmap chain from an input image.
    
    Parameters
    ----------
    image : np.ndarray
        Input image as float32 array of shape (H, W, 4) for RGBA.
        Values should be in [0, 1] range for best results.
    method : {"box", "gpu", "weighted"}, default "box"
        Filtering method to use:
        - "box": Simple box filtering (CPU)
        - "gpu": GPU compute shader (if available)  
        - "weighted": Weighted filtering with bilinear interpolation
    gamma_aware : bool, default False
        Whether to perform gamma-aware filtering (sRGB -> linear -> sRGB)
    gamma : float, default 2.2
        Gamma value for gamma correction
    max_levels : int, optional
        Maximum number of mip levels to generate. If None, generates complete chain.
    
    Returns
    -------
    List[np.ndarray]
        List of mipmap levels, where index 0 is the original image.
        Each level is a float32 array of shape (H_i, W_i, 4).
        
    Raises
    ------
    ValueError
        If image format is invalid or parameters are out of range
    RuntimeError
        If mipmap generation fails
        
    Examples
    --------
    >>> import numpy as np
    >>> from forge3d.texture import generate_mipmaps
    >>> 
    >>> # Create a test image
    >>> image = np.random.rand(256, 256, 4).astype(np.float32)
    >>> 
    >>> # Generate mipmaps with default settings
    >>> mipmaps = generate_mipmaps(image)
    >>> print(f"Generated {len(mipmaps)} mip levels")
    >>> print(f"Level 0: {mipmaps[0].shape}")
    >>> print(f"Level 1: {mipmaps[1].shape}")
    >>> 
    >>> # Generate with gamma-aware filtering
    >>> mipmaps_gamma = generate_mipmaps(image, gamma_aware=True)
    """
    # Validate input
    if not isinstance(image, np.ndarray):
        raise ValueError("image must be a numpy ndarray")
    
    if image.dtype != np.float32:
        raise ValueError("image must have dtype float32")
        
    if len(image.shape) != 3:
        raise ValueError("image must be 3D array of shape (H, W, 4)")
        
    if image.shape[2] != 4:
        raise ValueError("image must have 4 channels (RGBA)")
        
    height, width = image.shape[:2]
    if width == 0 or height == 0:
        raise ValueError("image dimensions must be > 0")
    
    # Ensure C-contiguous memory layout for efficient processing
    if not image.flags['C_CONTIGUOUS']:
        image = np.ascontiguousarray(image)
    
    # Validate parameters
    if method not in ["box", "gpu", "weighted"]:
        raise ValueError("method must be one of: 'box', 'gpu', 'weighted'")
        
    if not isinstance(gamma_aware, bool):
        raise ValueError("gamma_aware must be a boolean")
        
    if not isinstance(gamma, (int, float)) or gamma <= 0:
        raise ValueError("gamma must be a positive number")
        
    if max_levels is not None:
        if not isinstance(max_levels, int) or max_levels < 1:
            raise ValueError("max_levels must be a positive integer")
    
    # Calculate theoretical max levels
    theoretical_max = int(np.ceil(np.log2(max(width, height)))) + 1
    if max_levels is None:
        max_levels = theoretical_max
    else:
        max_levels = min(max_levels, theoretical_max)
    
    # For now, implement CPU-based mipmap generation
    # TODO: When GPU backend is ready, delegate to Rust
    if method == "gpu":
        warnings.warn("GPU method not yet implemented, falling back to box filtering")
        method = "box"
        
    if method == "weighted":
        warnings.warn("Weighted method not yet implemented, falling back to box filtering")
        method = "box"
    
    # Generate mipmaps using CPU box filtering
    return _generate_mipmaps_cpu_box(image, gamma_aware, gamma, max_levels)


def _generate_mipmaps_cpu_box(
    image: np.ndarray, 
    gamma_aware: bool, 
    gamma: float, 
    max_levels: int
) -> List[np.ndarray]:
    """Generate mipmaps using CPU box filtering."""
    mipmaps = [image.copy()]
    current = image.copy()
    
    level = 1
    while level < max_levels and (current.shape[0] > 1 or current.shape[1] > 1):
        # Calculate next level dimensions
        next_height = max(1, current.shape[0] // 2)
        next_width = max(1, current.shape[1] // 2)
        
        # Perform downsampling
        next_level = _downsample_box_filter_cpu(
            current, next_width, next_height, gamma_aware, gamma
        )
        
        mipmaps.append(next_level)
        current = next_level
        level += 1
    
    return mipmaps


def _downsample_box_filter_cpu(
    src: np.ndarray,
    dst_width: int,
    dst_height: int, 
    gamma_aware: bool,
    gamma: float
) -> np.ndarray:
    """Downsample an image using box filtering."""
    src_height, src_width = src.shape[:2]
    dst = np.zeros((dst_height, dst_width, 4), dtype=np.float32)
    
    x_ratio = src_width / dst_width
    y_ratio = src_height / dst_height
    
    for dst_y in range(dst_height):
        for dst_x in range(dst_width):
            # Calculate source region
            src_x_start = int(dst_x * x_ratio)
            src_y_start = int(dst_y * y_ratio)
            src_x_end = min(int(np.ceil((dst_x + 1) * x_ratio)), src_width)
            src_y_end = min(int(np.ceil((dst_y + 1) * y_ratio)), src_height)
            
            # Extract the region
            region = src[src_y_start:src_y_end, src_x_start:src_x_end, :]
            
            if region.size == 0:
                continue
                
            if gamma_aware:
                # Convert to linear space for RGB channels
                region_linear = region.copy()
                region_linear[:, :, :3] = _srgb_to_linear(region[:, :, :3], gamma)
                
                # Average in linear space
                avg_linear = np.mean(region_linear, axis=(0, 1))
                
                # Convert back to sRGB
                avg_linear[:3] = _linear_to_srgb(avg_linear[:3], gamma)
                dst[dst_y, dst_x, :] = avg_linear
            else:
                # Simple linear average
                dst[dst_y, dst_x, :] = np.mean(region, axis=(0, 1))
    
    return dst


def _srgb_to_linear(srgb: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    """Convert sRGB to linear space."""
    # Use the standard sRGB to linear conversion
    linear = np.where(
        srgb <= 0.04045,
        srgb / 12.92,
        np.power((srgb + 0.055) / 1.055, 2.4)
    )
    return linear


def _linear_to_srgb(linear: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    """Convert linear to sRGB space."""
    # Use the standard linear to sRGB conversion
    srgb = np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * np.power(linear, 1.0 / 2.4) - 0.055
    )
    return srgb


def calculate_mip_levels(width: int, height: int) -> int:
    """Calculate the number of mip levels for given dimensions.
    
    Parameters
    ----------
    width : int
        Image width
    height : int
        Image height
        
    Returns
    -------
    int
        Number of mip levels (including the base level)
    """
    if width <= 0 or height <= 0:
        return 0
    return int(np.ceil(np.log2(max(width, height)))) + 1


def mipmap_memory_usage(width: int, height: int, channels: int = 4, dtype_size: int = 4) -> int:
    """Calculate total memory usage for a complete mipmap chain.
    
    Parameters
    ----------
    width : int
        Base level width
    height : int  
        Base level height
    channels : int, default 4
        Number of channels (e.g., 4 for RGBA)
    dtype_size : int, default 4
        Size of data type in bytes (e.g., 4 for float32)
        
    Returns
    -------
    int
        Total memory usage in bytes for the complete mipmap chain
    """
    total_bytes = 0
    current_width = width
    current_height = height
    
    while current_width > 0 and current_height > 0:
        level_bytes = current_width * current_height * channels * dtype_size
        total_bytes += level_bytes
        
        if current_width == 1 and current_height == 1:
            break
            
        current_width = max(1, current_width // 2)
        current_height = max(1, current_height // 2)
    
    return total_bytes


def save_mipmap_pyramid(mipmaps: List[np.ndarray], base_path: str) -> List[str]:
    """Save a mipmap pyramid to PNG files.
    
    Parameters
    ----------
    mipmaps : List[np.ndarray]
        List of mipmap levels
    base_path : str
        Base filename (without extension)
        
    Returns
    -------
    List[str]
        List of filenames that were created
        
    Examples
    --------
    >>> mipmaps = generate_mipmaps(image)
    >>> files = save_mipmap_pyramid(mipmaps, "output/texture")
    >>> # Creates: output/texture_mip0.png, output/texture_mip1.png, etc.
    """
    from . import numpy_to_png
    import os
    
    filenames = []
    for i, mip in enumerate(mipmaps):
        # Convert float32 [0,1] to uint8 [0,255]
        mip_u8 = (np.clip(mip, 0, 1) * 255).astype(np.uint8)
        
        # Create filename
        dirname = os.path.dirname(base_path)
        basename = os.path.basename(base_path)
        filename = os.path.join(dirname, f"{basename}_mip{i}.png")
        
        # Save using forge3d's PNG writer
        numpy_to_png(filename, mip_u8)
        filenames.append(filename)
    
    return filenames
