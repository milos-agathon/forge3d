"""HDR (Radiance) image loading, processing, and off-screen rendering utilities."""

import numpy as np
from typing import Union, Tuple, Optional, Dict, Any, List
from pathlib import Path
import warnings
from enum import Enum
import logging

logger = logging.getLogger(__name__)

# For now, we'll implement a pure Python HDR loader
# In the future, this could call into the Rust backend for better performance


class ToneMappingOperator(Enum):
    """Advanced tone mapping operators for HDR to LDR conversion."""
    REINHARD = "reinhard"
    REINHARD_EXTENDED = "reinhard_extended"
    ACES = "aces"
    UNCHARTED2 = "uncharted2"  
    EXPOSURE = "exposure"
    GAMMA = "gamma"  # Simple gamma correction (existing)
    CLAMP = "clamp"  # Simple clamping (existing)


class HdrConfig:
    """Configuration for HDR rendering and tone mapping."""
    
    def __init__(
        self,
        width: int = 1920,
        height: int = 1080,
        hdr_format: str = "rgba16float",
        tone_mapping: ToneMappingOperator = ToneMappingOperator.REINHARD,
        exposure: float = 1.0,
        white_point: float = 4.0,
        gamma: float = 2.2
    ):
        """
        Initialize HDR configuration.
        
        Args:
            width: Render target width
            height: Render target height
            hdr_format: HDR texture format ('rgba16float' or 'rgba32float')
            tone_mapping: Tone mapping operator to use
            exposure: Exposure adjustment factor
            white_point: White point for extended tone mapping
            gamma: Gamma correction value
        """
        self.width = int(width)
        self.height = int(height)
        self.hdr_format = str(hdr_format)
        self.tone_mapping = tone_mapping
        self.exposure = float(exposure)
        self.white_point = float(white_point)
        self.gamma = float(gamma)
        
        # Validate parameters
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Invalid dimensions: {self.width}x{self.height}")
        
        if self.hdr_format not in ["rgba16float", "rgba32float"]:
            raise ValueError(f"Unsupported HDR format: {self.hdr_format}")
            
        if self.exposure <= 0:
            raise ValueError(f"Exposure must be positive: {self.exposure}")
            
        if self.white_point <= 0:
            raise ValueError(f"White point must be positive: {self.white_point}")
            
        if self.gamma <= 0:
            raise ValueError(f"Gamma must be positive: {self.gamma}")


class HdrRenderer:
    """HDR off-screen renderer with tone mapping capabilities."""
    
    def __init__(self, config: HdrConfig):
        """
        Initialize HDR renderer.
        
        Args:
            config: HDR configuration
        """
        self.config = config
        self._hdr_data = None
        self._ldr_data = None
        
        logger.info(f"HDR renderer initialized: {config.width}x{config.height} {config.hdr_format}")
    
    def render_hdr_scene(self, scene_data: Dict[str, Any]) -> np.ndarray:
        """
        Render scene to HDR buffer.
        
        In a complete implementation, this would use GPU rendering.
        For now, we create synthetic HDR data for testing.
        
        Args:
            scene_data: Scene description dictionary
            
        Returns:
            HDR image data as (height, width, 4) float32 array
        """
        # Create synthetic HDR scene
        hdr_image = self._create_synthetic_hdr_scene(scene_data)
        self._hdr_data = hdr_image
        
        logger.debug(f"Rendered HDR scene: shape={hdr_image.shape}, range=[{np.min(hdr_image):.3f}, {np.max(hdr_image):.3f}]")
        return hdr_image
    
    def _create_synthetic_hdr_scene(self, scene_data: Dict[str, Any]) -> np.ndarray:
        """Create synthetic HDR scene for testing."""
        hdr_image = np.zeros((self.config.height, self.config.width, 4), dtype=np.float32)
        
        # Get scene parameters
        sun_intensity = scene_data.get('sun_intensity', 50.0)
        sky_intensity = scene_data.get('sky_intensity', 2.0)
        ground_intensity = scene_data.get('ground_intensity', 0.3)
        
        center_x = self.config.width // 2
        center_y = self.config.height // 2
        
        for y in range(self.config.height):
            for x in range(self.config.width):
                # Distance from center for sun simulation
                dx = (x - center_x) / self.config.width
                dy = (y - center_y) / self.config.height
                dist = np.sqrt(dx*dx + dy*dy)
                
                # Create HDR lighting scenario
                if dist < 0.1:
                    # Sun - very bright
                    intensity = sun_intensity * np.exp(-dist * 50)
                    color = [intensity * 1.2, intensity * 1.1, intensity * 0.9]  # Warm sun
                elif y < self.config.height * 0.6:
                    # Sky - moderate brightness with gradient
                    sky_factor = 1.0 - (y / (self.config.height * 0.6))
                    intensity = sky_intensity * sky_factor
                    color = [intensity * 0.7, intensity * 0.8, intensity * 1.0]  # Blue sky
                else:
                    # Ground - lower intensity
                    ground_factor = ((y - self.config.height * 0.6) / (self.config.height * 0.4))
                    intensity = ground_intensity * (1 - ground_factor * 0.5)
                    color = [intensity * 0.8, intensity * 0.6, intensity * 0.4]  # Brown ground
                
                hdr_image[y, x, :3] = color
                hdr_image[y, x, 3] = 1.0  # Alpha
        
        return hdr_image
    
    def apply_tone_mapping(self, hdr_data: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply tone mapping to convert HDR to LDR.
        
        Args:
            hdr_data: HDR data to tone map. If None, uses last rendered data.
            
        Returns:
            LDR image data as (height, width, 4) uint8 array
        """
        if hdr_data is None:
            if self._hdr_data is None:
                raise ValueError("No HDR data available. Call render_hdr_scene first.")
            hdr_data = self._hdr_data
        
        if not isinstance(hdr_data, np.ndarray):
            raise TypeError("hdr_data must be numpy ndarray")
        
        if hdr_data.shape[:2] != (self.config.height, self.config.width):
            raise ValueError(f"HDR data shape {hdr_data.shape} doesn't match config {self.config.height}x{self.config.width}")
        
        # Apply exposure
        exposed = hdr_data * self.config.exposure
        
        # Apply tone mapping
        if self.config.tone_mapping == ToneMappingOperator.REINHARD:
            tone_mapped = self._reinhard_tonemap(exposed)
        elif self.config.tone_mapping == ToneMappingOperator.REINHARD_EXTENDED:
            tone_mapped = self._reinhard_extended_tonemap(exposed)
        elif self.config.tone_mapping == ToneMappingOperator.ACES:
            tone_mapped = self._aces_tonemap(exposed)
        elif self.config.tone_mapping == ToneMappingOperator.UNCHARTED2:
            tone_mapped = self._uncharted2_tonemap(exposed)
        elif self.config.tone_mapping == ToneMappingOperator.EXPOSURE:
            tone_mapped = self._exposure_tonemap(exposed)
        elif self.config.tone_mapping == ToneMappingOperator.GAMMA:
            tone_mapped = self._gamma_tonemap(exposed)
        elif self.config.tone_mapping == ToneMappingOperator.CLAMP:
            tone_mapped = self._clamp_tonemap(exposed)
        else:
            tone_mapped = self._reinhard_tonemap(exposed)  # Default
        
        # Apply gamma correction (except for gamma and clamp modes which handle it internally)
        if self.config.tone_mapping not in [ToneMappingOperator.GAMMA, ToneMappingOperator.CLAMP]:
            gamma_corrected = np.power(np.clip(tone_mapped, 0, 1), 1.0 / self.config.gamma)
        else:
            gamma_corrected = tone_mapped
        
        # Convert to 8-bit
        ldr_data = (gamma_corrected * 255).astype(np.uint8)
        self._ldr_data = ldr_data
        
        logger.debug(f"Applied tone mapping: {self.config.tone_mapping.value}, exposure={self.config.exposure}")
        return ldr_data
    
    def _reinhard_tonemap(self, hdr: np.ndarray) -> np.ndarray:
        """Apply Reinhard tone mapping: color / (color + 1)"""
        return hdr / (hdr + 1.0)
    
    def _reinhard_extended_tonemap(self, hdr: np.ndarray) -> np.ndarray:
        """Apply extended Reinhard tone mapping with white point."""
        white_sq = self.config.white_point * self.config.white_point
        return hdr * (1.0 + hdr / white_sq) / (1.0 + hdr)
    
    def _aces_tonemap(self, hdr: np.ndarray) -> np.ndarray:
        """Apply ACES filmic tone mapping."""
        a = 2.51
        b = 0.03
        c = 2.43
        d = 0.59
        e = 0.14
        
        return np.clip((hdr * (hdr * a + b)) / (hdr * (hdr * c + d) + e), 0, 1)
    
    def _uncharted2_tonemap(self, hdr: np.ndarray) -> np.ndarray:
        """Apply Uncharted 2 filmic tone mapping."""
        def uncharted2_tonemap_partial(x):
            a = 0.15
            b = 0.50
            c = 0.10
            d = 0.20
            e = 0.02
            f = 0.30
            return ((x * (x * a + c * b) + d * e) / (x * (x * a + b) + d * f)) - e / f
        
        curr = uncharted2_tonemap_partial(hdr)
        white_scale = 1.0 / uncharted2_tonemap_partial(self.config.white_point)
        return curr * white_scale
    
    def _exposure_tonemap(self, hdr: np.ndarray) -> np.ndarray:
        """Apply simple exposure-based tone mapping."""
        return 1.0 - np.exp(-hdr)
    
    def _gamma_tonemap(self, hdr: np.ndarray) -> np.ndarray:
        """Apply gamma correction tone mapping (existing method)."""
        return np.power(np.clip(hdr, 0, 1), 1.0 / self.config.gamma)
    
    def _clamp_tonemap(self, hdr: np.ndarray) -> np.ndarray:
        """Apply simple clamping tone mapping (existing method)."""
        return np.clip(hdr, 0, 1)
    
    def get_hdr_statistics(self, hdr_data: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Get statistics about HDR data.
        
        Args:
            hdr_data: HDR data to analyze. If None, uses last rendered data.
            
        Returns:
            Dictionary with HDR statistics
        """
        if hdr_data is None:
            if self._hdr_data is None:
                raise ValueError("No HDR data available")
            hdr_data = self._hdr_data
        
        # Compute luminance
        luminance = 0.299 * hdr_data[:, :, 0] + 0.587 * hdr_data[:, :, 1] + 0.114 * hdr_data[:, :, 2]
        
        return {
            'min_luminance': float(np.min(luminance)),
            'max_luminance': float(np.max(luminance)), 
            'mean_luminance': float(np.mean(luminance)),
            'median_luminance': float(np.median(luminance)),
            'std_luminance': float(np.std(luminance)),
            'dynamic_range': float(np.max(luminance) / max(np.min(luminance), 1e-6)),
            'pixels_above_1': int(np.sum(luminance > 1.0)),
            'pixels_above_10': int(np.sum(luminance > 10.0)),
            'pixels_above_100': int(np.sum(luminance > 100.0)),
        }
    
    def save_hdr_data(self, filepath: Union[str, Path], format: str = "exr") -> None:
        """
        Save HDR data to file.
        
        Args:
            filepath: Output file path
            format: Output format ('exr', 'hdr', or 'npy')
        """
        if self._hdr_data is None:
            raise ValueError("No HDR data to save")
        
        filepath = Path(filepath)
        
        if format.lower() == "npy":
            # Save as numpy array
            np.save(filepath, self._hdr_data)
            logger.info(f"Saved HDR data as numpy array: {filepath}")
        else:
            # For now, save as numpy until HDR formats are implemented
            logger.warning(f"HDR format '{format}' not implemented, saving as numpy array")
            np.save(filepath.with_suffix('.npy'), self._hdr_data)
    
    def save_ldr_data(self, filepath: Union[str, Path]) -> None:
        """
        Save tone-mapped LDR data as PNG.
        
        Args:
            filepath: Output file path
        """
        if self._ldr_data is None:
            raise ValueError("No LDR data to save. Call apply_tone_mapping first.")
        
        filepath = Path(filepath)
        
        try:
            # Try using forge3d PNG utilities
            from . import numpy_to_png
            numpy_to_png(str(filepath), self._ldr_data[:, :, :3])
            logger.info(f"Saved LDR image: {filepath}")
        except Exception as e:
            logger.warning(f"Could not save PNG: {e}")
            # Fallback to numpy save
            np.save(filepath.with_suffix('.npy'), self._ldr_data)
    
    def update_config(self, **kwargs) -> None:
        """
        Update HDR configuration parameters.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.debug(f"Updated HDR config: {key} = {value}")
            else:
                raise ValueError(f"Unknown config parameter: {key}")


def create_hdr_test_scene(
    width: int = 512,
    height: int = 512,
    sun_intensity: float = 50.0,
    sky_intensity: float = 2.0
) -> Dict[str, Any]:
    """
    Create test scene data for HDR rendering.
    
    Args:
        width: Scene width
        height: Scene height
        sun_intensity: Sun brightness (HDR units)
        sky_intensity: Sky brightness (HDR units)
        
    Returns:
        Scene data dictionary
    """
    return {
        'width': width,
        'height': height,
        'sun_intensity': sun_intensity,
        'sky_intensity': sky_intensity,
        'ground_intensity': 0.3,
        'description': 'Synthetic HDR test scene with sun and sky'
    }


def compare_tone_mapping_operators(
    hdr_data: np.ndarray,
    operators: List[ToneMappingOperator],
    exposure: float = 1.0
) -> Dict[str, Dict[str, Any]]:
    """
    Compare different tone mapping operators on the same HDR data.
    
    Args:
        hdr_data: HDR image data
        operators: List of tone mapping operators to test
        exposure: Exposure value to use
        
    Returns:
        Dictionary with results for each operator
    """
    results = {}
    
    height, width = hdr_data.shape[:2]
    
    for operator in operators:
        config = HdrConfig(
            width=width,
            height=height,
            tone_mapping=operator,
            exposure=exposure
        )
        
        renderer = HdrRenderer(config)
        renderer._hdr_data = hdr_data  # Set HDR data directly
        
        ldr_data = renderer.apply_tone_mapping()
        hdr_stats = renderer.get_hdr_statistics()
        
        # Compute LDR statistics
        ldr_luminance = 0.299 * ldr_data[:, :, 0] + 0.587 * ldr_data[:, :, 1] + 0.114 * ldr_data[:, :, 2]
        
        results[operator.value] = {
            'ldr_data': ldr_data,
            'hdr_stats': hdr_stats,
            'ldr_mean': float(np.mean(ldr_luminance)),
            'ldr_std': float(np.std(ldr_luminance)),
            'contrast_ratio': float(np.max(ldr_luminance) / max(np.min(ldr_luminance), 1)),
        }
    
    return results


def has_hdr_support() -> bool:
    """
    Check if HDR rendering features are available.
    
    Returns:
        True if HDR rendering is supported
    """
    try:
        # Try creating a simple HDR configuration
        config = HdrConfig(width=32, height=32)
        renderer = HdrRenderer(config)
        return True
    except Exception as e:
        logger.debug(f"HDR rendering not available: {e}")
        return False


# Legacy function mapping to new advanced tone mapping
def advanced_hdr_to_ldr(hdr_image: np.ndarray, method: Union[str, ToneMappingOperator] = "reinhard", 
                       exposure: float = 1.0, white_point: float = 4.0, gamma: float = 2.2) -> np.ndarray:
    """Advanced HDR to LDR conversion with extended tone mapping options.
    
    Args:
        hdr_image: HDR image data as float32 array of shape (H, W, 3)
        method: Tone mapping method (string or ToneMappingOperator enum)
        exposure: Exposure adjustment factor
        white_point: White point for extended tone mapping
        gamma: Gamma correction value
        
    Returns:
        LDR image data as float32 array in [0, 1] range
    """
    if isinstance(method, str):
        try:
            method = ToneMappingOperator(method)
        except ValueError:
            # Fall back to legacy method for compatibility
            return hdr_to_ldr(hdr_image, method, exposure)
    
    height, width = hdr_image.shape[:2]
    
    # Add alpha channel if missing
    if hdr_image.shape[2] == 3:
        hdr_rgba = np.zeros((height, width, 4), dtype=hdr_image.dtype)
        hdr_rgba[:, :, :3] = hdr_image
        hdr_rgba[:, :, 3] = 1.0
    else:
        hdr_rgba = hdr_image
    
    config = HdrConfig(
        width=width,
        height=height,
        tone_mapping=method,
        exposure=exposure,
        white_point=white_point,
        gamma=gamma
    )
    
    renderer = HdrRenderer(config)
    renderer._hdr_data = hdr_rgba
    
    ldr_data = renderer.apply_tone_mapping()
    
    # Return as float32 in [0,1] range
    return ldr_data.astype(np.float32) / 255.0


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