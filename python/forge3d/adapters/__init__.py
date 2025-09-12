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


__all__ = [
    'matplotlib_to_forge3d_colormap',
    'matplotlib_normalize', 
    'get_matplotlib_colormap_names',
    'is_matplotlib_available'
]


def check_optional_dependency(library: str) -> bool:
    """
    Check if an optional dependency is available.
    
    Args:
        library: Library name ('matplotlib', etc.)
        
    Returns:
        True if library is available, False otherwise
    """
    if library == 'matplotlib':
        return _HAS_MATPLOTLIB
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
        }
    }
    
    if _HAS_MATPLOTLIB:
        try:
            import matplotlib
            info['matplotlib']['version'] = matplotlib.__version__
        except AttributeError:
            info['matplotlib']['version'] = 'unknown'
    
    return info
