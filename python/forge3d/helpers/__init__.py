# python/forge3d/helpers/__init__.py
# Helper entry points for visualization backends (Matplotlib).
# This exists to expose optional display utilities with clear fallbacks.
# RELEVANT FILES:python/forge3d/helpers/mpl_display.py,tests/test_mpl_display.py,examples/mpl_imshow_demo.py
"""
Display and visualization helpers for forge3d.

This module provides convenience functions for displaying forge3d output
in various contexts, including matplotlib integration for visualization
and analysis workflows.
"""

from typing import Optional, Any
import warnings


# Optional matplotlib display helpers
try:
    from .mpl_display import (
        imshow_rgba,
        setup_matplotlib_backend,
        validate_rgba_array,
        is_matplotlib_display_available
    )
    _HAS_MATPLOTLIB_DISPLAY = True
except ImportError:
    _HAS_MATPLOTLIB_DISPLAY = False
    
    def _matplotlib_display_unavailable(*args, **kwargs):
        raise ImportError(
            "Matplotlib is required for display helpers. "
            "Install with: pip install matplotlib"
        )
    
    # Stub functions that provide helpful error messages
    imshow_rgba = _matplotlib_display_unavailable
    setup_matplotlib_backend = _matplotlib_display_unavailable
    validate_rgba_array = _matplotlib_display_unavailable
    is_matplotlib_display_available = lambda: False


__all__ = [
    'imshow_rgba',
    'setup_matplotlib_backend', 
    'validate_rgba_array',
    'is_matplotlib_display_available'
]


def check_display_support(backend: str = 'matplotlib') -> bool:
    """
    Check if a display backend is available.
    
    Args:
        backend: Display backend ('matplotlib', etc.)
        
    Returns:
        True if backend is available, False otherwise
    """
    if backend == 'matplotlib':
        return _HAS_MATPLOTLIB_DISPLAY
    else:
        raise ValueError(f"Unknown backend: {backend}")


def get_display_info() -> dict:
    """
    Get information about available display backends.
    
    Returns:
        Dictionary with backend availability and versions
    """
    info = {
        'matplotlib': {
            'available': _HAS_MATPLOTLIB_DISPLAY,
            'version': None,
            'backends': []
        }
    }
    
    if _HAS_MATPLOTLIB_DISPLAY:
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            info['matplotlib']['version'] = matplotlib.__version__
            info['matplotlib']['backends'] = [plt.get_backend()]
        except (AttributeError, ImportError):
            info['matplotlib']['version'] = 'unknown'
            info['matplotlib']['backends'] = []
    
    return info
