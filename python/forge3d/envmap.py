"""
Environment mapping and image-based lighting (IBL) utilities.

Provides CPU-side environment map loading, validation, and GPU texture management
for realistic environment-based lighting using cubemap textures and roughness-based 
mip sampling for physically-based rendering.
"""

import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional, Union
import struct

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

logger = logging.getLogger(__name__)


class EnvironmentMap:
    """Environment map data structure for CPU-side processing and validation."""
    
    def __init__(self, width: int, height: int, data: np.ndarray, mip_levels: int = 1):
        """
        Create environment map from HDR data.
        
        Args:
            width: Environment map width in pixels
            height: Environment map height in pixels  
            data: RGB float data as (height, width, 3) array
            mip_levels: Number of mip levels for GPU upload
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("data must be numpy ndarray")
        
        if data.shape != (height, width, 3):
            raise ValueError(f"data shape {data.shape} does not match expected ({height}, {width}, 3)")
            
        if data.dtype not in [np.float32, np.float64]:
            raise TypeError(f"data must be float32 or float64, got {data.dtype}")
            
        self.width = int(width)
        self.height = int(height)
        self.data = data.astype(np.float32)  # Ensure float32 for GPU
        self.mip_levels = int(mip_levels)
    
    @classmethod
    def create_test_envmap(cls, size: int) -> 'EnvironmentMap':
        """
        Create a synthetic environment map for testing.
        
        Args:
            size: Environment map size (square)
            
        Returns:
            Synthetic environment map with gradient pattern
        """
        data = np.zeros((size, size, 3), dtype=np.float32)
        
        for y in range(size):
            for x in range(size):
                # Convert to spherical coordinates
                u = x / size
                v = y / size
                
                phi = u * 2.0 * np.pi  # Longitude
                theta = v * np.pi      # Latitude
                
                # Create gradient-based environment
                r = (theta / np.pi) ** 0.5
                g = abs(np.sin((phi + theta) / (3.0 * np.pi)))
                b = max(1.0 - r * g, 0.1)
                
                data[y, x] = [r, g, b]
        
        mip_levels = int(np.log2(max(size, size))) + 1
        return cls(size, size, data, mip_levels)
    
    def sample_direction(self, direction: np.ndarray) -> np.ndarray:
        """
        Sample environment map using direction vector.
        
        Args:
            direction: 3D direction vector (x, y, z)
            
        Returns:
            RGB color as (3,) array
        """
        direction = np.asarray(direction, dtype=np.float32)
        if direction.shape != (3,):
            raise ValueError(f"direction must be (3,) array, got {direction.shape}")
            
        # Normalize direction
        direction = direction / np.linalg.norm(direction)
        
        # Convert to spherical coordinates
        phi = np.arctan2(direction[2], direction[0])
        theta = np.arccos(np.clip(direction[1], -1.0, 1.0))
        
        # Convert to UV coordinates
        u = (phi / (2.0 * np.pi) + 0.5) % 1.0
        v = theta / np.pi
        
        # Sample with bilinear interpolation
        x = u * (self.width - 1)
        y = v * (self.height - 1)
        
        x0, x1 = int(x), min(int(x) + 1, self.width - 1)
        y0, y1 = int(y), min(int(y) + 1, self.height - 1)
        
        fx = x - x0
        fy = y - y0
        
        # Bilinear interpolation
        c00 = self.data[y0, x0]
        c01 = self.data[y1, x0]
        c10 = self.data[y0, x1]
        c11 = self.data[y1, x1]
        
        c0 = c00 * (1 - fy) + c01 * fy
        c1 = c10 * (1 - fy) + c11 * fy
        
        return c0 * (1 - fx) + c1 * fx


def load_environment_map(filepath: Union[str, Path]) -> EnvironmentMap:
    """
    Load environment map from file.
    
    Currently supports simple test pattern generation.
    In production, this would load HDR formats like .exr or .hdr files.
    
    Args:
        filepath: Path to environment map file
        
    Returns:
        Loaded environment map
    """
    filepath = Path(filepath)
    
    # For now, create test pattern
    logger.warning(f"HDR loading not implemented, creating test pattern for {filepath}")
    return EnvironmentMap.create_test_envmap(256)


def validate_environment_map(envmap: EnvironmentMap) -> Dict[str, Any]:
    """
    Validate environment map data for GPU usage.
    
    Args:
        envmap: Environment map to validate
        
    Returns:
        Validation results dictionary
    """
    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'statistics': {}
    }
    
    # Check dimensions
    if envmap.width <= 0 or envmap.height <= 0:
        results['valid'] = False
        results['errors'].append(f"Invalid dimensions: {envmap.width}x{envmap.height}")
    
    if envmap.width > 4096 or envmap.height > 4096:
        results['warnings'].append(f"Large dimensions: {envmap.width}x{envmap.height} may impact performance")
    
    # Check data range
    data_min = np.min(envmap.data)
    data_max = np.max(envmap.data)
    data_mean = np.mean(envmap.data)
    
    results['statistics']['min_value'] = float(data_min)
    results['statistics']['max_value'] = float(data_max)
    results['statistics']['mean_value'] = float(data_mean)
    
    if data_min < 0.0:
        results['errors'].append(f"Negative values found: min={data_min}")
        results['valid'] = False
    
    if data_max > 100.0:
        results['warnings'].append(f"Very high HDR values: max={data_max}")
    
    # Check for NaN or infinite values
    if not np.isfinite(envmap.data).all():
        results['valid'] = False
        results['errors'].append("NaN or infinite values found in environment map")
    
    # Check memory requirements
    memory_mb = envmap.data.nbytes / (1024 * 1024)
    results['statistics']['memory_mb'] = float(memory_mb)
    
    if memory_mb > 64:
        results['warnings'].append(f"Large memory usage: {memory_mb:.1f} MB")
    
    logger.debug(f"Environment map validation: {results}")
    return results


def _halton_sequence(index: int, base: int) -> float:
    """Generate Halton low-discrepancy sequence value."""
    result = 0.0
    f = 1.0 / base
    i = index
    while i > 0:
        result += f * (i % base)
        i //= base
        f /= base
    return result


def _generate_hammersley_directions(count: int, seed: int = 42) -> List[np.ndarray]:
    """
    Generate deterministic low-discrepancy sample directions using Hammersley sequence.
    
    Args:
        count: Number of directions to generate  
        seed: Random seed for deterministic generation
        
    Returns:
        List of normalized 3D direction vectors
    """
    np.random.seed(seed)  # Fixed seed for deterministic sampling
    directions = []
    
    for i in range(count):
        # Use Hammersley sequence for better distribution
        u1 = (i + 0.5) / count  # Stratified sampling
        u2 = _halton_sequence(i + 1, 3)  # Halton base-3
        
        # Convert to spherical coordinates (uniform distribution on sphere)
        theta = np.arccos(1.0 - 2.0 * u1)  # Elevation
        phi = 2.0 * np.pi * u2            # Azimuth
        
        # Convert to Cartesian coordinates
        direction = np.array([
            np.sin(theta) * np.cos(phi),
            np.cos(theta), 
            np.sin(theta) * np.sin(phi)
        ], dtype=np.float32)
        
        directions.append(direction)
    
    return directions


def compute_roughness_luminance_series(envmap: EnvironmentMap, 
                                     roughness_values: List[float]) -> List[float]:
    """
    Compute luminance values for different roughness levels using deterministic sampling.
    
    This demonstrates the effect of roughness on environment lighting.
    Higher roughness should produce lower luminance due to increased scattering.
    Uses low-discrepancy Hammersley sequence for consistent, high-quality sampling.
    
    Args:
        envmap: Environment map to sample
        roughness_values: List of roughness values to test
        
    Returns:
        List of luminance values corresponding to each roughness
    """
    if not roughness_values:
        return []
    
    # Use fixed seed for deterministic results and ≥256 samples for 5% tolerance target
    base_sample_count = 256
    
    luminances = []
    for roughness in roughness_values:
        # Increase sample count for higher roughness to maintain quality
        sample_count = max(base_sample_count, int(base_sample_count * (1.0 + roughness)))
        
        # Generate deterministic sample directions using Hammersley sequence
        directions = _generate_hammersley_directions(sample_count, seed=42)
        
        # Sample environment with roughness-based perturbation
        samples = []
        for base_direction in directions:
            if roughness < 0.01:
                # Very low roughness - direct sampling
                sampled_dir = base_direction
            else:
                # Apply roughness-based perturbation using cosine-weighted distribution
                # Generate perturbation in local coordinate system
                
                # Create local coordinate system around base direction
                up = np.array([0.0, 1.0, 0.0])
                if abs(np.dot(base_direction, up)) > 0.9:
                    up = np.array([1.0, 0.0, 0.0])  # Use different up if parallel
                
                tangent = np.cross(base_direction, up)
                tangent = tangent / np.linalg.norm(tangent)
                bitangent = np.cross(base_direction, tangent)
                
                # Generate cosine-weighted sample within roughness cone
                # Use deterministic sequence based on direction index
                dir_seed = hash(tuple(base_direction)) % 10000
                np.random.seed(dir_seed)
                
                # Cosine-weighted hemisphere sampling with roughness scaling
                xi1 = np.random.uniform(0, 1)
                xi2 = np.random.uniform(0, 1)
                
                # Scale sampling cone by roughness
                cos_theta = np.sqrt((1.0 - xi1 * roughness))
                sin_theta = np.sqrt(1.0 - cos_theta * cos_theta)
                phi = 2.0 * np.pi * xi2
                
                # Convert to local coordinates
                local_dir = np.array([
                    sin_theta * np.cos(phi),
                    sin_theta * np.sin(phi), 
                    cos_theta
                ])
                
                # Transform to world coordinates
                sampled_dir = (local_dir[0] * tangent + 
                              local_dir[1] * bitangent + 
                              local_dir[2] * base_direction)
                sampled_dir = sampled_dir / np.linalg.norm(sampled_dir)
            
            # Sample environment
            color = envmap.sample_direction(sampled_dir)
            samples.append(color)
        
        # Average the samples
        avg_color = np.mean(samples, axis=0)
        
        # Compute luminance using standard weights
        luminance = 0.299 * avg_color[0] + 0.587 * avg_color[1] + 0.114 * avg_color[2]
        luminances.append(float(luminance))
    
    return luminances


def save_environment_map(envmap: EnvironmentMap, filepath: Union[str, Path]) -> None:
    """
    Save environment map to file.
    
    Currently saves as PNG with tone mapping.
    In production, this would support HDR formats.
    
    Args:
        envmap: Environment map to save
        filepath: Output file path
    """
    filepath = Path(filepath)
    
    if not _PIL_AVAILABLE:
        # Fallback to numpy save
        np.save(filepath.with_suffix('.npy'), envmap.data)
        logger.warning(f"PIL not available, saved as numpy array: {filepath.with_suffix('.npy')}")
        return
    
    # Tone map HDR data to LDR for PNG saving
    tone_mapped = envmap.data / (envmap.data + 1.0)  # Simple Reinhard tone mapping
    gamma_corrected = tone_mapped ** (1.0 / 2.2)     # Gamma correction
    
    # Convert to 8-bit
    rgb_8bit = np.clip(gamma_corrected * 255, 0, 255).astype(np.uint8)
    
    # Save as PNG
    image = Image.fromarray(rgb_8bit, mode='RGB')
    image.save(filepath)
    logger.info(f"Saved tone-mapped environment map: {filepath}")


def compute_luminance_difference(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    Compute mean luminance difference between two images.
    
    Args:
        image1: First image as (H, W, C) array
        image2: Second image as (H, W, C) array  
        
    Returns:
        Mean luminance difference as percentage
    """
    if image1.shape != image2.shape:
        raise ValueError(f"Image shapes must match: {image1.shape} vs {image2.shape}")
    
    # Compute luminance for each image
    def to_luminance(img):
        if len(img.shape) == 3 and img.shape[2] >= 3:
            # RGB to luminance
            return 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        else:
            # Grayscale or single channel
            return np.mean(img, axis=2) if len(img.shape) == 3 else img
    
    lum1 = to_luminance(image1.astype(np.float32))
    lum2 = to_luminance(image2.astype(np.float32))
    
    # Compute mean luminance difference
    mean1 = np.mean(lum1)
    mean2 = np.mean(lum2)
    
    if mean1 == 0 and mean2 == 0:
        return 0.0
    
    # Return percentage difference
    diff = abs(mean2 - mean1) / max(mean1, mean2, 1e-6) * 100
    return float(diff)


def has_envmap_support() -> bool:
    """
    Check if environment mapping features are available.
    
    Returns:
        True if environment mapping is supported
    """
    try:
        # Try creating a test environment map
        envmap = EnvironmentMap.create_test_envmap(32)
        validation = validate_environment_map(envmap)
        return validation['valid']
    except Exception as e:
        logger.debug(f"Environment mapping not available: {e}")
        return False


# Export main classes and functions
__all__ = [
    'EnvironmentMap',
    'load_environment_map', 
    'validate_environment_map',
    'compute_roughness_luminance_series',
    'save_environment_map',
    'compute_luminance_difference',
    'has_envmap_support'
]