"""
python/forge3d/ingest/xarray_adapter.py
Ingestion utilities for xarray DataArray with optional rioxarray metadata handling.
Exists to support tests and runtime that may lack xarray/rioxarray by using duck typing.
RELEVANT FILES:python/forge3d/ingest/__init__.py,python/forge3d/ingest/dask_adapter.py
"""

from typing import Optional, Tuple, Union, Dict, Any
import warnings
import numpy as np

try:
    import xarray as xr
    import rioxarray as rio  # Adds .rio accessor to xarray
    _HAS_XARRAY = True
except ImportError:
    _HAS_XARRAY = False
    xr = None


def _require_xarray():
    """Raise ImportError if xarray/rioxarray are not available."""
    if not _HAS_XARRAY:
        raise ImportError(
            "xarray and rioxarray are required for DataArray ingestion. "
            "Install with: pip install 'forge3d[raster]' or pip install 'xarray[complete]' rioxarray"
        )


def is_xarray_available() -> bool:
    """Check if xarray and rioxarray are available."""
    return _HAS_XARRAY


def validate_dataarray(da: "xr.DataArray") -> dict:
    """
    Validate a DataArray for ingestion and return metadata.
    
    Args:
        da: xarray DataArray to validate
        
    Returns:
        Dictionary with validation results and metadata
        
    Raises:
        ImportError: If xarray/rioxarray are not available
        ValueError: If DataArray validation fails
    """
    _require_xarray()
    
    # Avoid strict isinstance to support tests that mock DataArray when xarray isn't installed.
    # Require basic attributes typical of a DataArray.
    # Basic presence check for dims to align with test expectations
    if not hasattr(da, 'dims'):
        raise ValueError("Expected xarray.DataArray, missing 'dims'")
    
    # Check dimensions
    dims = da.dims
    if len(dims) < 2:
        raise ValueError(f"DataArray must have at least 2 dimensions, got {len(dims)}: {dims}")
    
    # Identify spatial dimensions (common patterns)
    spatial_dims = []
    for dim in dims:
        if dim.lower() in ['y', 'lat', 'latitude', 'north']:
            spatial_dims.append(('y', dim))
        elif dim.lower() in ['x', 'lon', 'longitude', 'east']:
            spatial_dims.append(('x', dim))
    
    if len(spatial_dims) < 2:
        # Fall back to last two dimensions as y, x
        if len(dims) >= 2:
            spatial_dims = [('y', dims[-2]), ('x', dims[-1])]
        else:
            raise ValueError("Could not identify spatial dimensions")
    
    # Check for band/channel dimension
    band_dim = None
    for dim in dims:
        if dim.lower() in ['band', 'bands', 'channel', 'channels', 'time']:
            band_dim = dim
            break
    
    # Get rio metadata if available
    rio_info = {}
    has_rio_accessor = hasattr(da, 'rio')
    if has_rio_accessor:
        try:
            rio_info = {
                'crs': da.rio.crs,
                'transform': da.rio.transform(),
                'width': da.rio.width,
                'height': da.rio.height,
                'resolution': da.rio.resolution(),
                'bounds': da.rio.bounds(),
                'nodata': da.rio.nodata
            }
        except Exception as e:
            warnings.warn(f"Could not extract rio metadata: {e}")
            has_rio_accessor = False
    
    coords_obj = getattr(da, 'coords', {})
    coords_dict = {}
    try:
        coords_dict = {name: coord.values for name, coord in coords_obj.items()}
    except Exception:
        coords_dict = {}

    values = getattr(da, 'values', None)
    is_c_contig = None
    if values is not None:
        flags = getattr(values, 'flags', None)
        if flags is not None:
            is_c_contig = getattr(flags, 'c_contiguous', None)

    return {
        'dims': dims,
        'shape': getattr(da, 'shape', tuple()),
        'dtype': getattr(da, 'dtype', None),
        'spatial_dims': dict(spatial_dims),
        'band_dim': band_dim,
        'has_rio_accessor': has_rio_accessor,
        'rio_info': rio_info,
        'attrs': dict(getattr(da, 'attrs', {})),
        'coords': coords_dict,
        'is_c_contiguous': is_c_contig
    }


def ingest_dataarray(
    da: "xr.DataArray", 
    preserve_dims: bool = True,
    target_dtype: Optional[Union[str, np.dtype]] = None,
    ensure_c_contiguous: bool = True
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Ingest xarray DataArray preserving CRS/transform and coordinates.
    
    Args:
        da: xarray DataArray with spatial data
        preserve_dims: Whether to preserve original dimension order
        target_dtype: Target data type for output array
        ensure_c_contiguous: Whether to ensure output is C-contiguous
        
    Returns:
        Tuple of (array, metadata) where:
        - array: NumPy array with data in standard (y, x[, bands]) order
        - metadata: Dictionary with preserved CRS, transform, and coordinate info
        
    Raises:
        ImportError: If xarray/rioxarray are not available
        ValueError: If DataArray is invalid or missing required metadata
    """
    _require_xarray()
    
    # Validate input
    validation_info = validate_dataarray(da)
    
    # Get spatial dimensions mapping
    spatial_dims = validation_info['spatial_dims']
    y_dim = spatial_dims.get('y')
    x_dim = spatial_dims.get('x')
    
    if not y_dim or not x_dim:
        raise ValueError("Could not identify y and x spatial dimensions")
    
    # Prepare dimension order for output
    if preserve_dims:
        # Keep original order but ensure y, x are last two
        dim_order = list(validation_info['dims'])
        if y_dim in dim_order:
            dim_order.remove(y_dim)
        if x_dim in dim_order:
            dim_order.remove(x_dim)
        dim_order.extend([y_dim, x_dim])
    else:
        # Standard order: [bands,] y, x
        dim_order = []
        for dim in validation_info['dims']:
            if dim not in [y_dim, x_dim]:
                dim_order.append(dim)
        dim_order.extend([y_dim, x_dim])
    
    # Transpose to desired order
    da_transposed = da.transpose(*dim_order)
    
    # Extract array data
    data = da_transposed.values
    
    # Convert dtype if requested
    if target_dtype is not None:
        target_dtype = np.dtype(target_dtype)
        if data.dtype != target_dtype:
            if target_dtype == np.uint8 and np.issubdtype(data.dtype, np.floating):
                # Special handling for float to uint8 conversion
                data = np.clip(data * 255, 0, 255).astype(np.uint8)
            else:
                data = data.astype(target_dtype)
    
    # Ensure C-contiguous if requested
    if ensure_c_contiguous and not data.flags.c_contiguous:
        data = np.ascontiguousarray(data)
    
    # Build comprehensive metadata
    metadata = {
        'original_dims': validation_info['dims'],
        'output_dims': dim_order,
        'spatial_dims': spatial_dims,
        'band_dim': validation_info['band_dim'],
        'shape': data.shape,
        'dtype': np.dtype(data.dtype).name,
        'is_c_contiguous': data.flags.c_contiguous,
        'has_rio_metadata': validation_info['has_rio_accessor'],
        'rio_info': validation_info['rio_info'],
        'attrs': validation_info['attrs'],
        'coords': validation_info['coords']
    }
    
    # Add coordinate arrays for spatial dimensions
    coords_obj = getattr(da, 'coords', None)
    try:
        if isinstance(coords_obj, dict) and y_dim in coords_obj:
            metadata['y_coords'] = coords_obj[y_dim].values
        if isinstance(coords_obj, dict) and x_dim in coords_obj:
            metadata['x_coords'] = coords_obj[x_dim].values
    except Exception:
        # Coordinates not available or not indexable; skip optional coords
        pass
    
    return data, metadata


def create_synthetic_dataarray(
    shape: Tuple[int, ...],
    crs: str = "EPSG:4326",
    bounds: Tuple[float, float, float, float] = (-180, -90, 180, 90),
    dims: Optional[Tuple[str, ...]] = None,
    seed: Optional[int] = None
) -> "xr.DataArray":
    """
    Create a synthetic DataArray for testing purposes.
    
    Args:
        shape: Shape of the array (e.g., (height, width) or (bands, height, width))
        crs: Coordinate reference system
        bounds: Spatial bounds as (left, bottom, right, top)
        dims: Dimension names (default: inferred from shape)
        seed: Random seed for reproducible data
        
    Returns:
        xarray DataArray with synthetic data and spatial metadata
        
    Raises:
        ImportError: If xarray/rioxarray are not available
    """
    _require_xarray()
    
    if seed is not None:
        np.random.seed(seed)
    
    # Generate synthetic data
    data = np.random.rand(*shape).astype(np.float32)
    
    # Infer dimensions
    if dims is None:
        if len(shape) == 2:
            dims = ('y', 'x')
        elif len(shape) == 3:
            dims = ('band', 'y', 'x')
        else:
            dims = tuple(f'dim_{i}' for i in range(len(shape)))
    
    # Create coordinate arrays
    coords = {}
    
    # Find spatial dimensions
    height_idx = width_idx = None
    for i, dim in enumerate(dims):
        if dim in ['y', 'lat', 'latitude']:
            height_idx = i
        elif dim in ['x', 'lon', 'longitude']:
            width_idx = i
    
    if height_idx is None or width_idx is None:
        # Assume last two dimensions are spatial
        if len(dims) >= 2:
            height_idx = len(dims) - 2
            width_idx = len(dims) - 1
    
    if height_idx is not None and width_idx is not None:
        height = shape[height_idx]
        width = shape[width_idx]
        
        # Create spatial coordinates
        left, bottom, right, top = bounds
        x_coords = np.linspace(left, right, width)
        y_coords = np.linspace(top, bottom, height)  # Top to bottom
        
        coords[dims[height_idx]] = y_coords
        coords[dims[width_idx]] = x_coords
    
    # Create other coordinate dimensions
    for i, dim in enumerate(dims):
        if dim not in coords:
            coords[dim] = np.arange(shape[i])
    
    # Create DataArray (import xarray here so tests can patch xr.DataArray)
    import xarray as xr  # local import to honor test monkeypatching
    da = xr.DataArray(
        data,
        dims=dims,
        coords=coords,
        attrs={
            'description': 'Synthetic data for testing',
            'units': 'unitless',
            'created_by': 'forge3d.ingest.xarray_adapter'
        }
    )
    
    # Add rio metadata if spatial dimensions are present
    if height_idx is not None and width_idx is not None:
        try:
            # Set CRS and spatial metadata using rioxarray
            da = da.rio.write_crs(crs)
            
            # Calculate transform
            left, bottom, right, top = bounds
            pixel_width = (right - left) / width
            pixel_height = (top - bottom) / height
            
            # Create affine transform (GDAL-style)
            from rasterio.transform import from_bounds
            transform = from_bounds(left, bottom, right, top, width, height)
            da = da.rio.write_transform(transform)
            
        except Exception as e:
            warnings.warn(f"Could not set rio metadata: {e}")
    
    return da


def dataarray_to_raster_info(da: "xr.DataArray") -> dict:
    """
    Extract raster-like information from a DataArray for integration with other adapters.
    
    Args:
        da: xarray DataArray with spatial data
        
    Returns:
        Dictionary with raster metadata compatible with rasterio-style operations
        
    Raises:
        ImportError: If xarray/rioxarray are not available
    """
    _require_xarray()
    
    validation_info = validate_dataarray(da)
    
    # Build raster info structure
    spatial_dims = validation_info['spatial_dims']
    y_dim = spatial_dims.get('y')
    x_dim = spatial_dims.get('x')
    
    # Get array shape for spatial dimensions
    height = da.sizes.get(y_dim, 0)
    width = da.sizes.get(x_dim, 0)
    
    # Count bands
    count = 1
    for dim in da.dims:
        if dim not in [y_dim, x_dim]:
            count *= da.sizes[dim]
    
    raster_info = {
        'width': width,
        'height': height,
        'count': count,
        # Normalize to canonical dtype name (e.g., 'float32') even when given a type object.
        'dtype': np.dtype(da.dtype).name,
        'dims': da.dims,
        'spatial_dims': spatial_dims,
        'shape': da.shape
    }
    
    # Add rio info if available
    if validation_info['has_rio_accessor']:
        raster_info.update(validation_info['rio_info'])
    
    return raster_info


# Export public API
__all__ = [
    'validate_dataarray',
    'ingest_dataarray',
    'create_synthetic_dataarray',
    'dataarray_to_raster_info',
    'is_xarray_available'
]
