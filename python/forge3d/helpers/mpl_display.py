# python/forge3d/helpers/mpl_display.py
# Display helpers for showing forge3d RGBA arrays in Matplotlib.
# This exists to provide a thin, optional Matplotlib bridge with zero-copy paths for uint8 data.
# RELEVANT FILES:python/forge3d/helpers/__init__.py,python/forge3d/adapters/mpl_cmap.py,examples/mpl_imshow_demo.py
"""
Matplotlib display helpers for forge3d RGBA buffers.

This module provides functions to display forge3d RGBA output arrays
in matplotlib figures with proper aspect ratio, extent handling, and
DPI awareness while avoiding unnecessary memory copies.
"""

from typing import Optional, Tuple, Union, Any
import numpy as np
import warnings

# Optional matplotlib dependency
try:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.axes as mpl_axes
    from matplotlib.image import AxesImage
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False
    # Stub classes for type hints - create a mock module-like object
    class MockAxes:
        Axes = Any
    mpl_axes = MockAxes()
    AxesImage = Any


def is_matplotlib_display_available() -> bool:
    """Check if matplotlib is available for display operations."""
    return _HAS_MATPLOTLIB


def _require_matplotlib():
    """Raise ImportError with helpful message if matplotlib not available."""
    if not _HAS_MATPLOTLIB:
        raise ImportError(
            "Matplotlib is required for display helpers. "
            "Install with: pip install matplotlib"
        )


def validate_rgba_array(
    rgba: np.ndarray,
    name: str = "rgba"
) -> np.ndarray:
    """
    Validate RGBA array format and properties.
    
    Args:
        rgba: RGBA array to validate
        name: Parameter name for error messages
        
    Returns:
        Validated RGBA array
        
    Raises:
        TypeError: If array is not numpy ndarray
        ValueError: If array has wrong shape, dtype, or properties
    """
    if not isinstance(rgba, np.ndarray):
        raise TypeError(
            f"{name} must be numpy.ndarray, got {type(rgba).__name__}"
        )
    
    if rgba.ndim != 3:
        raise ValueError(
            f"{name} must be 3D array (H, W, C), got {rgba.ndim}D shape {rgba.shape}"
        )
    
    height, width, channels = rgba.shape
    
    if channels not in (3, 4):
        raise ValueError(
            f"{name} must have 3 (RGB) or 4 (RGBA) channels, got {channels}"
        )
    
    if not rgba.flags['C_CONTIGUOUS']:
        warnings.warn(
            f"{name} is not C-contiguous, which may cause performance issues. "
            f"Consider using np.ascontiguousarray() if this array will be reused.",
            UserWarning
        )
    
    # Check data range based on dtype
    if rgba.dtype == np.uint8:
        if np.any(rgba > 255) or np.any(rgba < 0):
            raise ValueError(
                f"{name} uint8 values must be in range [0, 255], "
                f"got range [{rgba.min()}, {rgba.max()}]"
            )
    elif rgba.dtype in (np.float32, np.float64):
        if np.any(rgba > 1.0) or np.any(rgba < 0.0):
            warnings.warn(
                f"{name} float values are outside [0, 1] range: "
                f"[{rgba.min():.3f}, {rgba.max():.3f}]. "
                f"Values will be clipped.",
                UserWarning
            )
    else:
        raise ValueError(
            f"{name} must have dtype uint8 or float32/float64, got {rgba.dtype}"
        )
    
    if height == 0 or width == 0:
        raise ValueError(f"{name} cannot have zero dimensions: {rgba.shape}")
    
    return rgba


def setup_matplotlib_backend(backend: Optional[str] = None, force: bool = False) -> str:
    """
    Setup matplotlib backend for optimal display performance.
    
    Args:
        backend: Desired backend ('Agg', 'TkAgg', etc.). If None, keeps current.
        force: Whether to force backend change even if already set
        
    Returns:
        Current backend name after setup
        
    Raises:
        ImportError: If matplotlib is not available
        ValueError: If backend is not available
    """
    _require_matplotlib()
    
    current_backend = plt.get_backend()
    
    if backend is None:
        return current_backend
    
    if backend == current_backend and not force:
        return current_backend
    
    try:
        plt.switch_backend(backend)
        new_backend = plt.get_backend()
        return new_backend
    except ImportError as e:
        raise ValueError(f"Backend '{backend}' is not available: {e}")


def imshow_rgba(
    ax: mpl_axes.Axes,
    rgba: np.ndarray,
    extent: Optional[Tuple[float, float, float, float]] = None,
    dpi: Optional[float] = None,
    interpolation: str = 'nearest',
    aspect: Union[str, float] = 'equal',
    alpha: Optional[float] = None,
    **kwargs
) -> AxesImage:
    """
    Display forge3d RGBA buffer on matplotlib axes with proper formatting.
    
    This function displays RGBA arrays from forge3d with correct orientation,
    aspect ratio, and extent handling. It avoids unnecessary copies when
    possible and handles both uint8 and float32 input formats.
    
    Args:
        ax: Matplotlib axes to display on
        rgba: RGBA array from forge3d (H, W, 3|4) 
        extent: Image extent as (left, right, bottom, top) in data coordinates.
                If None, uses pixel coordinates.
        dpi: DPI for display. If None, uses axes figure DPI.
        interpolation: Interpolation method ('nearest', 'bilinear', etc.)
        aspect: Aspect ratio ('equal', 'auto', or numeric value)
        alpha: Overall alpha for the image (overrides alpha channel)
        **kwargs: Additional arguments passed to ax.imshow()
        
    Returns:
        AxesImage object from matplotlib
        
    Raises:
        ImportError: If matplotlib is not available
        TypeError: If ax is not a matplotlib axes
        ValueError: If rgba array is invalid
        
    Example:
        >>> fig, ax = plt.subplots()
        >>> rgba = renderer.render_rgba()
        >>> img = imshow_rgba(ax, rgba, extent=(0, 10, 0, 10))
        >>> plt.show()
    """
    _require_matplotlib()
    
    # Validate inputs
    if not hasattr(ax, 'imshow'):
        raise TypeError("ax must be a matplotlib Axes object")
    
    rgba = validate_rgba_array(rgba, "rgba")
    height, width, channels = rgba.shape
    
    # Handle extent validation
    if extent is not None:
        if not isinstance(extent, (tuple, list)) or len(extent) != 4:
            raise ValueError(
                "extent must be tuple/list of 4 values (left, right, bottom, top)"
            )
        left, right, bottom, top = extent
        if left >= right:
            raise ValueError(f"extent left ({left}) must be < right ({right})")
        if bottom >= top:
            raise ValueError(f"extent bottom ({bottom}) must be < top ({top})")
    
    # Handle DPI
    if dpi is not None and dpi <= 0:
        raise ValueError(f"dpi must be positive, got {dpi}")
    
    # Process RGBA data for matplotlib
    # Goal: zero-copy path for C-contiguous uint8 input.
    # - If uint8 and C-contiguous: pass through as-is (Matplotlib supports uint8 RGB/RGBA).
    # - If float: clip to [0,1] (may allocate if not already in range).
    # - If non-contiguous: make a single contiguous copy.
    
    if rgba.flags['C_CONTIGUOUS']:
        display_data = rgba
    else:
        display_data = np.ascontiguousarray(rgba)

    if display_data.dtype in (np.float32, np.float64):
        display_data = np.clip(display_data, 0.0, 1.0)
    elif display_data.dtype != np.uint8:
        # Unsupported type; do a minimal, explicit conversion.
        display_data = display_data.astype(np.uint8, copy=True)

    # Set up display parameters
    display_kwargs = {
        'interpolation': interpolation,
        'aspect': aspect,
        'origin': 'upper',  # forge3d uses top-left origin
        **kwargs  # Allow override of defaults
    }
    
    if extent is not None:
        display_kwargs['extent'] = extent
    
    # Display image
    try:
        im = ax.imshow(display_data, **display_kwargs)
        
        # Apply DPI if specified
        if dpi is not None:
            try:
                ax.figure.dpi = dpi
            except AttributeError:
                warnings.warn("Could not set DPI on figure", UserWarning)
        
        # Apply overall alpha if specified
        if alpha is not None:
            im.set_alpha(alpha)
        
        return im
        
    except Exception as e:
        # Provide helpful error context
        raise ValueError(
            f"Failed to display RGBA array: {e}. "
            f"Array shape: {rgba.shape}, dtype: {rgba.dtype}, "
            f"extent: {extent}, kwargs: {display_kwargs}"
        ) from e


def imshow_rgba_subplots(
    rgba_arrays: list,
    titles: Optional[list] = None,
    figsize: Optional[Tuple[float, float]] = None,
    extent: Optional[Tuple[float, float, float, float]] = None,
    dpi: Optional[float] = None,
    ncols: Optional[int] = None,
    **kwargs
) -> Tuple[Any, list]:
    """
    Display multiple RGBA arrays in subplot layout.
    
    Args:
        rgba_arrays: List of RGBA arrays to display
        titles: Optional list of subplot titles
        figsize: Figure size (width, height) in inches
        extent: Common extent for all subplots
        dpi: Figure DPI
        ncols: Number of columns (if None, uses square layout)
        **kwargs: Additional arguments passed to imshow_rgba()
        
    Returns:
        Tuple of (figure, list of AxesImage objects)
        
    Raises:
        ImportError: If matplotlib is not available
        ValueError: If inputs are invalid
    """
    _require_matplotlib()
    
    if not rgba_arrays:
        raise ValueError("rgba_arrays cannot be empty")
    
    n_arrays = len(rgba_arrays)
    
    if titles is not None and len(titles) != n_arrays:
        raise ValueError(
            f"Number of titles ({len(titles)}) must match "
            f"number of arrays ({n_arrays})"
        )
    
    # Determine layout
    if ncols is None:
        ncols = int(np.ceil(np.sqrt(n_arrays)))
    nrows = int(np.ceil(n_arrays / ncols))
    
    # Create figure
    if figsize is None:
        figsize = (4 * ncols, 4 * nrows)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi)
    if n_arrays == 1:
        axes = [axes]
    elif nrows == 1 or ncols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    # Display arrays
    images = []
    for i, rgba in enumerate(rgba_arrays):
        if i < len(axes):
            im = imshow_rgba(axes[i], rgba, extent=extent, **kwargs)
            images.append(im)
            
            if titles is not None:
                axes[i].set_title(titles[i])
        
    # Hide unused subplots
    for i in range(n_arrays, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig, images


def save_rgba_comparison(
    rgba_arrays: list,
    output_path: str,
    titles: Optional[list] = None,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: float = 150,
    **kwargs
) -> None:
    """
    Save comparison of multiple RGBA arrays to file.
    
    Args:
        rgba_arrays: List of RGBA arrays to compare
        output_path: Output file path
        titles: Optional subplot titles
        figsize: Figure size (width, height) in inches
        dpi: Output DPI
        **kwargs: Additional arguments passed to imshow_rgba()
    """
    _require_matplotlib()
    
    fig, _ = imshow_rgba_subplots(
        rgba_arrays, 
        titles=titles, 
        figsize=figsize, 
        dpi=dpi,
        **kwargs
    )
    
    try:
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        plt.close(fig)
        raise ValueError(f"Failed to save comparison to {output_path}: {e}") from e


# Convenience functions for common use cases
def quick_show(rgba: np.ndarray, title: str = "forge3d Output", **kwargs) -> None:
    """
    Quick display of single RGBA array with minimal setup.
    
    Args:
        rgba: RGBA array to display
        title: Figure title
        **kwargs: Additional arguments passed to imshow_rgba()
    """
    _require_matplotlib()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    imshow_rgba(ax, rgba, **kwargs)
    ax.set_title(title)
    plt.show()


def rgba_to_pil(rgba: np.ndarray) -> Any:
    """
    Convert forge3d RGBA array to PIL Image for additional processing.
    
    Args:
        rgba: RGBA array from forge3d
        
    Returns:
        PIL Image object
        
    Raises:
        ImportError: If PIL is not available
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("PIL/Pillow is required for PIL conversion")
    
    rgba = validate_rgba_array(rgba, "rgba")
    
    # Convert to uint8 if necessary
    if rgba.dtype != np.uint8:
        if rgba.dtype in (np.float32, np.float64):
            rgba = (np.clip(rgba, 0, 1) * 255).astype(np.uint8)
        else:
            raise ValueError(f"Cannot convert {rgba.dtype} to PIL Image")
    
    # Handle RGB vs RGBA
    if rgba.shape[2] == 3:
        mode = 'RGB'
    else:
        mode = 'RGBA'
    
    return Image.fromarray(rgba, mode=mode)
