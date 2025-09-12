# python/forge3d/adapters/mpl_cmap.py
# Matplotlib colormap and normalization adapters for forge3d.
# This exists to map Matplotlib names/objects and norms to forge3d-friendly forms.
# RELEVANT FILES:python/forge3d/adapters/__init__.py,tests/test_mpl_cmap.py,tests/test_mpl_norms.py
"""
Matplotlib colormap and normalization adapters for forge3d.

This module provides seamless integration between matplotlib's colormap
and normalization systems and forge3d's internal representations.
"""

from typing import Union, Optional, Tuple, List, Any, Dict
import numpy as np
import warnings

# Optional matplotlib dependency
try:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.colors import Colormap, Normalize, LogNorm, PowerNorm, BoundaryNorm
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False
    # Stub classes for type hints
    Colormap = Any
    Normalize = Any
    LogNorm = Any
    PowerNorm = Any
    BoundaryNorm = Any


def is_matplotlib_available() -> bool:
    """Check if matplotlib is available."""
    return _HAS_MATPLOTLIB


def _require_matplotlib():
    """Raise ImportError with helpful message if matplotlib not available."""
    if not _HAS_MATPLOTLIB:
        raise ImportError(
            "Matplotlib is required for colormap adapters. "
            "Install with: pip install matplotlib"
        )


def get_matplotlib_colormap_names() -> List[str]:
    """
    Get list of available matplotlib colormap names.
    
    Returns:
        List of colormap names available in matplotlib
        
    Raises:
        ImportError: If matplotlib is not available
    """
    _require_matplotlib()
    return sorted(plt.colormaps())


def matplotlib_to_forge3d_colormap(
    cmap: Union[str, Colormap],
    n_colors: int = 256
) -> np.ndarray:
    """
    Convert matplotlib colormap to forge3d RGBA LUT format.
    
    Args:
        cmap: Matplotlib colormap name or Colormap object
        n_colors: Number of colors in output LUT (default: 256)
        
    Returns:
        RGBA array of shape (n_colors, 4) with dtype uint8, 
        suitable for forge3d colormap usage
        
    Raises:
        ImportError: If matplotlib is not available
        ValueError: If colormap name is invalid or n_colors is invalid
        
    Example:
        >>> lut = matplotlib_to_forge3d_colormap('viridis')
        >>> lut.shape
        (256, 4)
        >>> lut.dtype
        dtype('uint8')
    """
    _require_matplotlib()
    
    if n_colors < 1 or n_colors > 8192:
        raise ValueError(f"n_colors must be in range [1, 8192], got {n_colors}")
    
    # Handle colormap input
    if isinstance(cmap, str):
        try:
            cmap_obj = plt.get_cmap(cmap)
        except ValueError as e:
            available = get_matplotlib_colormap_names()
            raise ValueError(
                f"Unknown colormap name: '{cmap}'. "
                f"Available colormaps: {available[:10]}... (and {len(available)-10} more)"
            ) from e
    elif hasattr(cmap, '__call__') and hasattr(cmap, 'N'):
        # Looks like a matplotlib Colormap object
        cmap_obj = cmap
    else:
        raise ValueError(
            f"cmap must be colormap name (str) or matplotlib Colormap object, "
            f"got {type(cmap)}"
        )
    
    # Generate colors using matplotlib
    colors = cmap_obj(np.linspace(0.0, 1.0, n_colors))
    
    # Convert to uint8 RGBA format expected by forge3d
    if colors.shape[1] == 3:  # RGB
        # Add alpha channel
        rgba = np.ones((n_colors, 4), dtype=np.float64)
        rgba[:, :3] = colors
    elif colors.shape[1] == 4:  # RGBA
        rgba = colors.astype(np.float64)
    else:
        raise ValueError(f"Unexpected colormap output shape: {colors.shape}")
    
    # Clamp to [0, 1] range and convert to uint8
    rgba = np.clip(rgba, 0.0, 1.0)
    rgba_uint8 = (rgba * 255).round().astype(np.uint8)
    
    return rgba_uint8


def matplotlib_normalize(
    data: np.ndarray,
    norm: Optional[Union[str, Normalize]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    clip: bool = False
) -> np.ndarray:
    """
    Normalize data using matplotlib normalization.
    
    Args:
        data: Input data array to normalize
        norm: Matplotlib normalization object or preset name
              ('linear', 'log', 'symlog'). If None, uses linear normalization.
        vmin: Minimum value for normalization. If None, uses data.min()
        vmax: Maximum value for normalization. If None, uses data.max()
        clip: Whether to clip normalized values to [0, 1]
        
    Returns:
        Normalized data array with values in [0, 1] (approximately)
        
    Raises:
        ImportError: If matplotlib is not available
        ValueError: If normalization parameters are invalid
        
    Example:
        >>> data = np.array([[1, 10, 100], [1000, 10000, 100000]])
        >>> normalized = matplotlib_normalize(data, 'log')
        >>> normalized.min(), normalized.max()
        (0.0, 1.0)
    """
    _require_matplotlib()
    
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)
    
    # Determine vmin/vmax if not provided
    if vmin is None:
        vmin = float(np.nanmin(data))
    if vmax is None:
        vmax = float(np.nanmax(data))
        
    if vmin >= vmax:
        raise ValueError(f"vmin ({vmin}) must be < vmax ({vmax})")
    
    # Handle normalization object
    if norm is None or norm == 'linear':
        norm_obj = Normalize(vmin=vmin, vmax=vmax, clip=clip)
    elif norm == 'log':
        if vmin <= 0:
            vmin = np.finfo(data.dtype).eps
            warnings.warn(
                f"LogNorm requires vmin > 0, adjusting vmin to {vmin}",
                UserWarning
            )
        norm_obj = LogNorm(vmin=vmin, vmax=vmax, clip=clip)
    elif norm == 'symlog':
        norm_obj = mcolors.SymLogNorm(vmin=vmin, vmax=vmax, linthresh=1.0, clip=clip)
    elif hasattr(norm, '__call__') and hasattr(norm, 'vmin'):
        # Looks like a matplotlib normalization object
        norm_obj = norm
        if norm_obj.vmin is None:
            norm_obj.vmin = vmin
        if norm_obj.vmax is None:
            norm_obj.vmax = vmax
    else:
        raise ValueError(
            f"norm must be None, 'linear', 'log', 'symlog', or matplotlib "
            f"Normalize object, got {type(norm)}: {norm}"
        )
    
    # Apply normalization
    try:
        normalized = norm_obj(data)
    except ValueError as e:
        if "log" in str(norm).lower() and np.any(data <= 0):
            raise ValueError(
                "LogNorm cannot handle non-positive values. "
                "Consider filtering data or using different normalization."
            ) from e
        raise
    
    # Ensure finite values
    if not np.all(np.isfinite(normalized)):
        n_invalid = np.sum(~np.isfinite(normalized))
        warnings.warn(
            f"Normalization produced {n_invalid} non-finite values. "
            f"Consider adjusting vmin/vmax or using clipping.",
            UserWarning
        )
        
        if clip:
            normalized = np.clip(normalized, 0.0, 1.0)
        
    return normalized


class LogNormAdapter:
    """
    Adapter for matplotlib LogNorm with forge3d compatibility.
    
    Provides the same interface as matplotlib.colors.LogNorm but
    ensures compatibility with forge3d's expected behavior.
    """
    
    def __init__(self, vmin: Optional[float] = None, vmax: Optional[float] = None, 
                 clip: bool = False):
        """
        Initialize LogNorm adapter.
        
        Args:
            vmin: Minimum value (must be > 0)
            vmax: Maximum value  
            clip: Whether to clip output to [0, 1]
        """
        _require_matplotlib()
        self.vmin = vmin
        self.vmax = vmax
        self.clip = clip
    
    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Normalize data using logarithmic scaling."""
        return matplotlib_normalize(data, 'log', self.vmin, self.vmax, self.clip)
    
    def inverse(self, normalized: np.ndarray) -> np.ndarray:
        """Apply inverse logarithmic normalization."""
        _require_matplotlib()
        if self.vmin is None or self.vmax is None:
            raise ValueError("vmin and vmax must be set for inverse normalization")
        
        # Use matplotlib's LogNorm for inverse
        norm_obj = LogNorm(vmin=self.vmin, vmax=self.vmax, clip=self.clip)
        return norm_obj.inverse(normalized)


class PowerNormAdapter:
    """
    Adapter for matplotlib PowerNorm with forge3d compatibility.
    """
    
    def __init__(self, gamma: float, vmin: Optional[float] = None, 
                 vmax: Optional[float] = None, clip: bool = False):
        """
        Initialize PowerNorm adapter.
        
        Args:
            gamma: Power law exponent
            vmin: Minimum value
            vmax: Maximum value
            clip: Whether to clip output to [0, 1]
        """
        _require_matplotlib()
        self.gamma = gamma
        self.vmin = vmin
        self.vmax = vmax
        self.clip = clip
        
        if gamma <= 0:
            raise ValueError(f"gamma must be > 0, got {gamma}")
    
    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Normalize data using power-law scaling."""
        _require_matplotlib()
        
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        
        # Determine vmin/vmax if not provided
        vmin = self.vmin if self.vmin is not None else float(np.nanmin(data))
        vmax = self.vmax if self.vmax is not None else float(np.nanmax(data))
        
        if vmin >= vmax:
            raise ValueError(f"vmin ({vmin}) must be < vmax ({vmax})")
        
        # Create matplotlib PowerNorm and apply
        norm_obj = PowerNorm(gamma=self.gamma, vmin=vmin, vmax=vmax, clip=self.clip)
        return norm_obj(data)
    
    def inverse(self, normalized: np.ndarray) -> np.ndarray:
        """Apply inverse power-law normalization."""
        _require_matplotlib()
        if self.vmin is None or self.vmax is None:
            raise ValueError("vmin and vmax must be set for inverse normalization")
        
        # Use matplotlib's PowerNorm for inverse
        norm_obj = PowerNorm(gamma=self.gamma, vmin=self.vmin, vmax=self.vmax, clip=self.clip)
        return norm_obj.inverse(normalized)


class BoundaryNormAdapter:
    """
    Adapter for matplotlib BoundaryNorm with forge3d compatibility.
    """
    
    def __init__(self, boundaries: List[float], ncolors: int, clip: bool = False, 
                 extend: str = 'neither'):
        """
        Initialize BoundaryNorm adapter.
        
        Args:
            boundaries: Monotonically increasing sequence of boundaries
            ncolors: Number of colors in the colormap
            clip: Whether to clip values outside boundaries
            extend: How to handle values outside boundaries ('neither', 'both', 'min', 'max')
        """
        _require_matplotlib()
        self.boundaries = np.asarray(boundaries, dtype=float)
        self.ncolors = ncolors
        self.clip = clip
        self.extend = extend
        
        if len(self.boundaries) < 2:
            raise ValueError("boundaries must have at least 2 values")
        
        if not np.all(np.diff(self.boundaries) > 0):
            raise ValueError("boundaries must be monotonically increasing")
        
        if ncolors < 1:
            raise ValueError(f"ncolors must be >= 1, got {ncolors}")
    
    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Normalize data using boundary mapping."""
        _require_matplotlib()
        
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        
        # Create matplotlib BoundaryNorm and apply
        norm_obj = BoundaryNorm(
            boundaries=self.boundaries,
            ncolors=self.ncolors, 
            clip=self.clip,
            extend=self.extend
        )
        return norm_obj(data)


# Convenience function for all normalization types
def create_matplotlib_normalizer(
    norm_type: str,
    **kwargs
) -> Union[LogNormAdapter, PowerNormAdapter, BoundaryNormAdapter]:
    """
    Create a matplotlib normalization adapter.
    
    Args:
        norm_type: Type of normalization ('log', 'power', 'boundary')
        **kwargs: Parameters for the specific normalizer
        
    Returns:
        Normalization adapter instance
        
    Raises:
        ValueError: If norm_type is unknown
        ImportError: If matplotlib is not available
    """
    _require_matplotlib()
    
    if norm_type == 'log':
        return LogNormAdapter(**kwargs)
    elif norm_type == 'power':
        return PowerNormAdapter(**kwargs) 
    elif norm_type == 'boundary':
        return BoundaryNormAdapter(**kwargs)
    else:
        raise ValueError(
            f"Unknown norm_type: '{norm_type}'. "
            f"Valid types: 'log', 'power', 'boundary'"
        )
