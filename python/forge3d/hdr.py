"""HDR (Radiance) image loading and processing utilities."""

import numpy as np
from typing import Union, Tuple, Optional
from pathlib import Path
import warnings

# For now, we'll implement a pure Python HDR loader
# In the future, this could call into the Rust backend for better performance


def load_hdr(path: Union[str, Path]) -> np.ndarray:
    """Load a Radiance HDR file and return as float32 NumPy array.
    
    Parameters
    ----------
    path : str or Path
        Path to the HDR file
        
    Returns  
    -------
    np.ndarray
        HDR image data as float32 array of shape (H, W, 3) for RGB or (H, W, 4) for RGBA.
        Values are in linear color space and can exceed 1.0 for high dynamic range.
        
    Raises
    ------
    FileNotFoundError
        If the HDR file cannot be found
    ValueError
        If the file is not a valid HDR format or cannot be parsed
    RuntimeError
        If there's an error during HDR decoding
        
    Examples
    --------
    >>> import numpy as np
    >>> from forge3d.hdr import load_hdr
    >>>
    >>> # Load HDR image
    >>> hdr_image = load_hdr("environment.hdr")
    >>> print(f"HDR image shape: {hdr_image.shape}")
    >>> print(f"Value range: {hdr_image.min():.3f} to {hdr_image.max():.3f}")
    >>>
    >>> # Convert to LDR for display (simple tone mapping)
    >>> ldr_image = np.clip(hdr_image**(1/2.2), 0, 1)
    >>> 
    >>> # Save as PNG using forge3d
    >>> from forge3d import numpy_to_png
    >>> ldr_u8 = (ldr_image * 255).astype(np.uint8)
    >>> numpy_to_png("output.png", ldr_u8)
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"HDR file not found: {path}")
    
    try:
        return _load_hdr_impl(path)
    except Exception as e:
        raise RuntimeError(f"Failed to load HDR file {path}: {e}") from e


def _load_hdr_impl(path: Path) -> np.ndarray:
    """Internal HDR loading implementation."""
    with open(path, 'rb') as f:
        # Parse header
        width, height = _parse_hdr_header(f)
        
        # Read scanlines
        rgbe_data = _read_hdr_scanlines(f, width, height)
        
        # Convert RGBe to linear RGB
        rgb_data = _convert_rgbe_to_rgb(rgbe_data, width, height)
        
    return rgb_data


def _parse_hdr_header(f) -> Tuple[int, int]:
    """Parse HDR file header and return (width, height)."""
    # Read magic line
    magic = f.readline().decode('ascii').strip()
    if not (magic.startswith('#?RADIANCE') or magic.startswith('#?RGBE')):
        raise ValueError(f"Invalid HDR magic: {magic}")
    
    # Read header lines until empty line
    format_found = False
    while True:
        line = f.readline().decode('ascii').strip()
        if not line:
            break  # Empty line marks end of header
            
        if line.startswith('FORMAT='):
            if line not in ['FORMAT=32-bit_rle_rgbe', 'FORMAT=32-bit_rle_xyze']:
                raise ValueError(f"Unsupported HDR format: {line}")
            format_found = True
    
    if not format_found:
        raise ValueError("HDR file missing FORMAT specification")
    
    # Read resolution line
    resolution_line = f.readline().decode('ascii').strip()
    
    # Parse resolution (format: "-Y height +X width" or "+Y height +X width")
    parts = resolution_line.split()
    if len(parts) != 4:
        raise ValueError(f"Invalid HDR resolution line: {resolution_line}")
    
    try:
        height = int(parts[1])
        width = int(parts[3])
    except ValueError:
        raise ValueError(f"Invalid HDR dimensions in: {resolution_line}")
    
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid HDR dimensions: {width}x{height}")
    
    return width, height


def _read_hdr_scanlines(f, width: int, height: int) -> np.ndarray:
    """Read HDR scanlines with RLE decompression."""
    rgbe_data = np.zeros((height, width, 4), dtype=np.uint8)
    
    for y in range(height):
        scanline = _read_hdr_scanline(f, width, y)
        rgbe_data[y] = scanline
    
    return rgbe_data


def _read_hdr_scanline(f, width: int, y: int) -> np.ndarray:
    """Read a single HDR scanline."""
    # Read scanline header
    header = f.read(4)
    if len(header) != 4:
        raise ValueError(f"Unexpected end of file at scanline {y}")
    
    header_bytes = list(header)
    
    # Check if this is a new-style RLE scanline
    if (header_bytes[0] == 2 and header_bytes[1] == 2 and 
        header_bytes[2] == ((width >> 8) & 0xFF) and 
        header_bytes[3] == (width & 0xFF)):
        # New-style RLE
        return _read_rle_scanline(f, width)
    else:
        # Old-style or uncompressed
        scanline = np.zeros((width, 4), dtype=np.uint8)
        scanline[0] = header_bytes  # First pixel is in the header
        
        # Read remaining pixels
        for x in range(1, width):
            pixel_data = f.read(4)
            if len(pixel_data) != 4:
                raise ValueError(f"Unexpected end of file reading pixel at ({x}, {y})")
            scanline[x] = list(pixel_data)
        
        return scanline


def _read_rle_scanline(f, width: int) -> np.ndarray:
    """Read RLE-compressed scanline."""
    scanline = np.zeros((width, 4), dtype=np.uint8)
    
    # Read each component (RGBE) separately
    for component in range(4):
        pos = 0
        
        while pos < width:
            run_info = f.read(1)
            if len(run_info) != 1:
                raise ValueError("Unexpected end of file in RLE data")
            
            run_length = run_info[0]
            
            if run_length > 128:
                # RLE run: repeat next value
                repeat_count = run_length - 128
                if pos + repeat_count > width:
                    raise ValueError("RLE run exceeds scanline width")
                
                value = f.read(1)
                if len(value) != 1:
                    raise ValueError("Unexpected end of file in RLE repeat value")
                
                for i in range(repeat_count):
                    scanline[pos + i, component] = value[0]
                
                pos += repeat_count
            else:
                # Literal run: copy next values
                copy_count = run_length
                if pos + copy_count > width:
                    raise ValueError("RLE literal run exceeds scanline width")
                
                for i in range(copy_count):
                    value = f.read(1)
                    if len(value) != 1:
                        raise ValueError("Unexpected end of file in RLE literal value")
                    scanline[pos + i, component] = value[0]
                
                pos += copy_count
    
    return scanline


def _convert_rgbe_to_rgb(rgbe_data: np.ndarray, width: int, height: int) -> np.ndarray:
    """Convert RGBe data to linear RGB."""
    # Extract components
    r = rgbe_data[:, :, 0].astype(np.float32)
    g = rgbe_data[:, :, 1].astype(np.float32)
    b = rgbe_data[:, :, 2].astype(np.float32)
    e = rgbe_data[:, :, 3].astype(np.int32)
    
    # Convert using shared exponent
    # Formula: RGB = (r,g,b) * 2^(e-128-8)
    # Handle e=0 case (all zeros)
    valid_mask = e != 0
    
    # Calculate multiplier
    multiplier = np.where(valid_mask, 2.0 ** (e - 128 - 8), 0.0).astype(np.float32)
    
    # Apply to RGB channels
    rgb_data = np.zeros((height, width, 3), dtype=np.float32)
    rgb_data[:, :, 0] = r * multiplier
    rgb_data[:, :, 1] = g * multiplier  
    rgb_data[:, :, 2] = b * multiplier
    
    return rgb_data


def hdr_to_ldr(hdr_image: np.ndarray, method: str = "reinhard", exposure: float = 1.0) -> np.ndarray:
    """Convert HDR image to LDR using tone mapping.
    
    Parameters
    ----------
    hdr_image : np.ndarray
        HDR image data as float32 array of shape (H, W, 3)
    method : str, default "reinhard"
        Tone mapping method: "reinhard", "gamma", "clamp", or "aces"
    exposure : float, default 1.0
        Exposure adjustment factor
        
    Returns
    -------
    np.ndarray
        LDR image data as float32 array in [0, 1] range
        
    Examples
    --------
    >>> hdr = load_hdr("environment.hdr")
    >>> ldr = hdr_to_ldr(hdr, method="reinhard", exposure=0.8)
    >>> ldr_u8 = (ldr * 255).astype(np.uint8)
    """
    if hdr_image.ndim != 3 or hdr_image.shape[2] != 3:
        raise ValueError("HDR image must have shape (H, W, 3)")
    
    if hdr_image.dtype != np.float32:
        hdr_image = hdr_image.astype(np.float32)
    
    # Apply exposure
    exposed = hdr_image * exposure
    
    if method == "reinhard":
        # Reinhard tone mapping: x / (1 + x)
        return exposed / (1.0 + exposed)
    
    elif method == "gamma":
        # Simple gamma correction
        return np.power(np.clip(exposed, 0, 1), 1.0 / 2.2)
    
    elif method == "clamp":
        # Simple clamping
        return np.clip(exposed, 0, 1)
    
    elif method == "aces":
        # ACES filmic tone mapping approximation
        a = 2.51
        b = 0.03
        c = 2.43  
        d = 0.59
        e = 0.14
        return np.clip((exposed * (a * exposed + b)) / (exposed * (c * exposed + d) + e), 0, 1)
    
    else:
        raise ValueError(f"Unknown tone mapping method: {method}")


def save_hdr_as_ldr(hdr_path: Union[str, Path], output_path: Union[str, Path], 
                   method: str = "reinhard", exposure: float = 1.0) -> None:
    """Load HDR file and save as LDR PNG.
    
    Parameters
    ----------
    hdr_path : str or Path
        Path to input HDR file
    output_path : str or Path  
        Path for output PNG file
    method : str, default "reinhard"
        Tone mapping method
    exposure : float, default 1.0
        Exposure adjustment
        
    Examples
    --------
    >>> save_hdr_as_ldr("env.hdr", "env_ldr.png", exposure=0.5)
    """
    from . import numpy_to_png
    
    # Load HDR
    hdr_image = load_hdr(hdr_path)
    
    # Convert to LDR
    ldr_image = hdr_to_ldr(hdr_image, method=method, exposure=exposure)
    
    # Convert to uint8 and save
    ldr_u8 = (ldr_image * 255).astype(np.uint8)
    numpy_to_png(str(output_path), ldr_u8)


def get_hdr_info(path: Union[str, Path]) -> dict:
    """Get information about an HDR file without loading the full image data.
    
    Parameters
    ----------
    path : str or Path
        Path to HDR file
        
    Returns
    -------
    dict
        Dictionary with HDR file information:
        - width: int
        - height: int  
        - pixel_count: int
        - estimated_size_mb: float
        
    Examples
    --------
    >>> info = get_hdr_info("large_env.hdr")
    >>> print(f"HDR size: {info['width']}x{info['height']}")
    >>> print(f"Estimated memory: {info['estimated_size_mb']:.1f} MB")
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"HDR file not found: {path}")
    
    with open(path, 'rb') as f:
        width, height = _parse_hdr_header(f)
    
    pixel_count = width * height
    # Estimate size: 3 channels × 4 bytes per float32
    estimated_size_mb = (pixel_count * 3 * 4) / (1024 * 1024)
    
    return {
        'width': width,
        'height': height,
        'pixel_count': pixel_count,
        'estimated_size_mb': estimated_size_mb,
    }