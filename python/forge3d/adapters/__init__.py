# python/forge3d/adapters/__init__.py
# Adapter entry points bridging external libs (e.g., Matplotlib) to forge3d.
# This exists to expose optional integration with clear fallbacks when deps are missing.
# RELEVANT FILES:python/forge3d/adapters/mpl_cmap.py,python/forge3d/helpers/mpl_display.py,docs/integration/matplotlib.md
"""
Adapters for external library integration.

This module provides adapters for integrating forge3d with popular Python
scientific libraries like matplotlib, maintaining optional dependencies
and graceful degradation when libraries are not available.
"""

from typing import Optional, Any
import warnings


# Optional matplotlib integration
try:
    from .mpl_cmap import (
        matplotlib_to_forge3d_colormap,
        matplotlib_normalize,
        get_matplotlib_colormap_names,
        is_matplotlib_available
    )
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False
    
    def _matplotlib_unavailable(*args, **kwargs):
        raise ImportError(
            "Matplotlib is required for colormap adapters. "
            "Install with: pip install matplotlib"
        )
    
    # Stub functions that provide helpful error messages
    matplotlib_to_forge3d_colormap = _matplotlib_unavailable
    matplotlib_normalize = _matplotlib_unavailable  
    get_matplotlib_colormap_names = _matplotlib_unavailable
    is_matplotlib_available = lambda: False


# Optional datashader integration
try:
    from .datashader_adapter import (
        DatashaderAdapter,
        is_datashader_available,
        rgba_view_from_agg,
        premultiply_rgba,
        validate_alignment,
        to_overlay_texture,
        shade_to_overlay,
        get_datashader_info
    )
    # Determine availability by probing the adapter, not just import success
    _HAS_DATASHADER = bool(is_datashader_available())
except ImportError:
    _HAS_DATASHADER = False
    
    def _datashader_unavailable(*args, **kwargs):
        raise ImportError(
            "Datashader is required for datashader adapters. "
            "Install with: pip install datashader"
        )
    
    # Stub functions that provide helpful error messages  
    DatashaderAdapter = _datashader_unavailable
    rgba_view_from_agg = _datashader_unavailable
    premultiply_rgba = _datashader_unavailable
    validate_alignment = _datashader_unavailable
    to_overlay_texture = _datashader_unavailable
    shade_to_overlay = _datashader_unavailable
    get_datashader_info = _datashader_unavailable
    is_datashader_available = lambda: False


__all__ = [
    # Matplotlib integration
    'matplotlib_to_forge3d_colormap',
    'matplotlib_normalize', 
    'get_matplotlib_colormap_names',
    'is_matplotlib_available',
    # Datashader integration
    'DatashaderAdapter',
    'is_datashader_available',
    'rgba_view_from_agg',
    'premultiply_rgba',
    'validate_alignment', 
    'to_overlay_texture',
    'shade_to_overlay',
    'get_datashader_info'
]


def check_optional_dependency(library: str) -> bool:
    """
    Check if an optional dependency is available.
    
    Args:
        library: Library name ('matplotlib', 'datashader', etc.)
        
    Returns:
        True if library is available, False otherwise
    """
    if library == 'matplotlib':
        return _HAS_MATPLOTLIB
    elif library == 'datashader':
        return _HAS_DATASHADER
    else:
        raise ValueError(f"Unknown library: {library}")


def get_adapter_info() -> dict:
    """
    Get information about available adapters.
    
    Returns:
        Dictionary with adapter availability and versions
    """
    info = {
        'matplotlib': {
            'available': _HAS_MATPLOTLIB,
            'version': None,
            'adapters': ['colormap', 'normalize']
        },
        'datashader': {
            'available': _HAS_DATASHADER,
            'version': None,
            'adapters': ['rgba_overlay', 'aggregation', 'shading']
        }
    }
    
    if _HAS_MATPLOTLIB:
        try:
            import matplotlib
            info['matplotlib']['version'] = matplotlib.__version__
        except AttributeError:
            info['matplotlib']['version'] = 'unknown'
    
    if _HAS_DATASHADER:
        try:
            import datashader
            info['datashader']['version'] = getattr(datashader, '__version__', 'unknown')
        except (ImportError, AttributeError):
            info['datashader']['version'] = 'unknown'
    
    return info
