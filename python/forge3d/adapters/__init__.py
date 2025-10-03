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


# Optional matplotlib integration (colormaps/norms)
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


# Optional matplotlib image rasterization (M1)
try:
    from .mpl_image import (
        rasterize_figure as mpl_rasterize_figure,
        rasterize_axes as mpl_rasterize_axes,
        height_from_luminance as mpl_height_from_luminance,
        is_matplotlib_available as is_matplotlib_available_image,
    )
    _HAS_MATPLOTLIB_IMAGE = True
except Exception:
    _HAS_MATPLOTLIB_IMAGE = False
    def _mpl_image_unavailable(*args, **kwargs):
        raise ImportError("Matplotlib is required for rasterization adapters. Install with: pip install matplotlib")
    mpl_rasterize_figure = _mpl_image_unavailable
    mpl_rasterize_axes = _mpl_image_unavailable
    mpl_height_from_luminance = _mpl_image_unavailable
    is_matplotlib_available_image = lambda: False

# Optional matplotlib data adapters (M2)
try:
    from .mpl_data import (
        extract_lines_from_axes,
        extract_polygons_from_axes,
        text_to_polygons,
        extrude_polygons_to_meshes,
        thicken_lines_to_meshes,
        line_width_world_from_pixels,
        is_matplotlib_available as is_matplotlib_available_data,
    )
    _HAS_MATPLOTLIB_DATA = True
except Exception:
    _HAS_MATPLOTLIB_DATA = False
    def _mpl_data_unavailable(*args, **kwargs):
        raise ImportError("Matplotlib is required for data adapters. Install with: pip install matplotlib")
    extract_lines_from_axes = _mpl_data_unavailable
    extract_polygons_from_axes = _mpl_data_unavailable
    text_to_polygons = _mpl_data_unavailable
    extrude_polygons_to_meshes = _mpl_data_unavailable
    thicken_lines_to_meshes = _mpl_data_unavailable
    line_width_world_from_pixels = _mpl_data_unavailable
    is_matplotlib_available_data = lambda: False

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

# Optional GeoPandas adapter (M3)
try:
    from .geopandas_adapter import (
        is_geopandas_available,
        reproject_geoseries,
        geoseries_to_polygons,
        extrude_geometries_to_meshes,
    )
    _HAS_GPD = True
except Exception:
    _HAS_GPD = False
    def _gpd_unavailable(*args, **kwargs):
        raise ImportError("GeoPandas/Shapely are required. Install with: pip install geopandas shapely pyproj")
    is_geopandas_available = lambda: False
    reproject_geoseries = _gpd_unavailable
    geoseries_to_polygons = _gpd_unavailable
    extrude_geometries_to_meshes = _gpd_unavailable

# Optional Cartopy integration (M5)
try:
    from .cartopy_adapter import (
        is_cartopy_available,
        rasterize_geoaxes,
        get_axes_crs,
        get_extent_in_crs,
    )
    _HAS_CARTOPY = True
except Exception:
    _HAS_CARTOPY = False
    def _cartopy_unavailable(*args, **kwargs):
        raise ImportError("Cartopy is required. Install with: pip install cartopy")
    is_cartopy_available = lambda: False
    rasterize_geoaxes = _cartopy_unavailable
    get_axes_crs = _cartopy_unavailable
    get_extent_in_crs = _cartopy_unavailable

# Optional Seaborn/Plotly convenience (M6)
try:
    from .charts import (
        render_chart_to_rgba,
        is_plotly_available,
        is_seaborn_available,
    )
    _HAS_CHARTS = True
except Exception:
    _HAS_CHARTS = False
    def _charts_unavailable(*args, **kwargs):
        raise ImportError("Plotly/Seaborn integration requires optional deps. Install with: pip install plotly kaleido seaborn")
    render_chart_to_rgba = _charts_unavailable
    is_plotly_available = lambda: False
    is_seaborn_available = lambda: False

# Optional Rasterio/Xarray adapter (M4)
try:
    from .raster_xarray_adapter import (
        rasterio_to_rgba,
        dataarray_to_rgba,
        is_rasterio_available as is_rasterio_available_m4,
        is_xarray_available as is_xarray_available_m4,
    )
    _HAS_RASTER_XARRAY = True
except Exception:
    _HAS_RASTER_XARRAY = False
    def _rxa_unavailable(*args, **kwargs):
        raise ImportError("Rasterio/Xarray adapters require optional deps. Install with: pip install rasterio xarray rioxarray")
    rasterio_to_rgba = _rxa_unavailable
    dataarray_to_rgba = _rxa_unavailable
    is_rasterio_available_m4 = lambda: False
    is_xarray_available_m4 = lambda: False


__all__ = [
    # Matplotlib integration
    'matplotlib_to_forge3d_colormap',
    'matplotlib_normalize', 
    'get_matplotlib_colormap_names',
    'is_matplotlib_available',
    'mpl_rasterize_figure', 'mpl_rasterize_axes', 'mpl_height_from_luminance', 'is_matplotlib_available_image',
    'extract_lines_from_axes', 'extract_polygons_from_axes', 'text_to_polygons', 'extrude_polygons_to_meshes', 'thicken_lines_to_meshes', 'line_width_world_from_pixels', 'is_matplotlib_available_data',
    # Datashader integration
    'DatashaderAdapter',
    'is_datashader_available',
    'rgba_view_from_agg',
    'premultiply_rgba',
    'validate_alignment', 
    'to_overlay_texture',
    'shade_to_overlay',
    'get_datashader_info'
    ,
    # Rasterio/Xarray (M4)
    'rasterio_to_rgba', 'dataarray_to_rgba', 'is_rasterio_available_m4', 'is_xarray_available_m4',
    # GeoPandas (M3)
    'is_geopandas_available', 'reproject_geoseries', 'geoseries_to_polygons', 'extrude_geometries_to_meshes',
    # Cartopy (M5)
    'is_cartopy_available', 'rasterize_geoaxes', 'get_axes_crs', 'get_extent_in_crs',
    # Charts (M6)
    'render_chart_to_rgba', 'is_plotly_available', 'is_seaborn_available'
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
