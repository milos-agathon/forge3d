"""
forge3d.adapters.reproject - CRS normalization and reprojection using WarpedVRT

This module provides functionality to reproject raster data between coordinate
reference systems using rasterio's WarpedVRT and pyproj for coordinate transformations.
"""

from typing import Optional, Tuple, Union, Any
import warnings
import numpy as np

try:
    import rasterio
    from rasterio.vrt import WarpedVRT
    from rasterio.windows import Window
    from rasterio.enums import Resampling
    from rasterio.warp import calculate_default_transform, reproject
    import pyproj
    _HAS_REPROJECT_DEPS = True
except ImportError:
    _HAS_REPROJECT_DEPS = False
    WarpedVRT = Any
    Window = Any
    Resampling = Any


def _require_reproject_deps():
    """Raise ImportError if required dependencies are not available."""
    if not _HAS_REPROJECT_DEPS:
        raise ImportError(
            "rasterio and pyproj are required for reprojection operations. "
            "Install with: pip install 'forge3d[raster]' or pip install rasterio pyproj"
        )


def is_reproject_available() -> bool:
    """Check if reprojection dependencies are available."""
    return _HAS_REPROJECT_DEPS


class WarpedVRTWrapper:
    """
    Wrapper around rasterio WarpedVRT for easier reprojection operations.
    
    This class manages the lifecycle of a WarpedVRT and provides convenient
    methods for reading reprojected data.
    """
    
    def __init__(
        self,
        dataset,
        dst_crs: Union[str, Any],
        resampling: Optional[Union[str, "Resampling"]] = None,
        dst_transform: Optional[Any] = None,
        dst_width: Optional[int] = None,
        dst_height: Optional[int] = None
    ):
        """
        Initialize WarpedVRT wrapper.
        
        Args:
            dataset: Source rasterio dataset
            dst_crs: Target CRS as string (e.g., 'EPSG:4326') or Any
            resampling: Resampling method (default: bilinear)
            dst_transform: Optional destination transform (calculated if None)
            dst_width: Optional destination width (calculated if None)
            dst_height: Optional destination height (calculated if None)
        """
        _require_reproject_deps()
        
        self.src_dataset = dataset
        self.dst_crs = str(dst_crs) if not isinstance(dst_crs, str) else dst_crs
        
        if resampling is None:
            resampling = Resampling.bilinear
        elif isinstance(resampling, str):
            resampling = getattr(Resampling, resampling.lower(), Resampling.bilinear)
        self.resampling = resampling
        
        # Calculate default transform if not provided
        if dst_transform is None or dst_width is None or dst_height is None:
            transform, width, height = calculate_default_transform(
                dataset.crs,
                self.dst_crs,
                dataset.width,
                dataset.height,
                *dataset.bounds
            )
            
            self.dst_transform = dst_transform or transform
            self.dst_width = dst_width or width
            self.dst_height = dst_height or height
        else:
            self.dst_transform = dst_transform
            self.dst_width = dst_width
            self.dst_height = dst_height
        
        # Create the WarpedVRT
        self._vrt = WarpedVRT(
            dataset,
            crs=self.dst_crs,
            transform=self.dst_transform,
            width=self.dst_width,
            height=self.dst_height,
            resampling=self.resampling
        )
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def close(self):
        """Close the WarpedVRT."""
        if hasattr(self, '_vrt') and self._vrt is not None:
            self._vrt.close()
            self._vrt = None
    
    @property
    def vrt(self):
        """Get the underlying WarpedVRT object."""
        return self._vrt
    
    def read(
        self,
        indexes: Optional[Union[int, list]] = None,
        window: Optional[Window] = None,
        out_shape: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Read reprojected data from the WarpedVRT.
        
        Args:
            indexes: Band indexes to read (default: all bands)
            window: Window to read from the reprojected data
            out_shape: Output shape for resampling
            
        Returns:
            NumPy array with reprojected data
        """
        return self._vrt.read(
            indexes=indexes,
            window=window,
            out_shape=out_shape
        )
    
    def get_metadata(self) -> dict:
        """Get metadata for the reprojected dataset."""
        return {
            'width': self.dst_width,
            'height': self.dst_height,
            'count': self._vrt.count,
            'dtype': str(self._vrt.dtype),
            'crs': self.dst_crs,
            'transform': list(self.dst_transform),
            'bounds': list(self._vrt.bounds),
            'resampling': str(self.resampling)
        }


def reproject_window(
    src_dataset,
    dst_crs: Union[str, Any],
    window: Window,
    out_shape: Optional[Tuple[int, int]] = None,
    resampling: Optional[Union[str, "Resampling"]] = None,
    indexes: Optional[Union[int, list]] = None
) -> Tuple[np.ndarray, Any]:
    """
    Reproject a windowed portion of a dataset to a target CRS.
    
    Args:
        src_dataset: Source rasterio dataset
        dst_crs: Target CRS as string or Any
        window: Window to reproject from source dataset
        out_shape: Optional output shape (height, width)
        resampling: Resampling method (default: bilinear)
        indexes: Band indexes to read (default: all bands)
        
    Returns:
        Tuple of (reprojected_array, transform) where:
        - reprojected_array: NumPy array with reprojected data
        - transform: Affine transform for the reprojected window
        
    Raises:
        ImportError: If required dependencies are not installed
    """
    _require_reproject_deps()
    
    # Create WarpedVRT for the entire dataset
    with WarpedVRTWrapper(src_dataset, dst_crs, resampling) as vrt_wrapper:
        # Calculate window bounds in destination CRS
        # For simplicity, we'll read the requested window from the VRT
        data = vrt_wrapper.read(indexes=indexes, window=window, out_shape=out_shape)
        
        # Calculate the transform for this specific window
        if window is not None:
            # Adjust transform for the window offset
            transform = vrt_wrapper.dst_transform
            window_transform = rasterio.windows.transform(window, transform)
        else:
            window_transform = vrt_wrapper.dst_transform
            
        return data, window_transform


def transform_bounds(
    src_bounds: Tuple[float, float, float, float],
    src_crs: Union[str, Any],
    dst_crs: Union[str, Any]
) -> Tuple[float, float, float, float]:
    """
    Transform bounds from one CRS to another.
    
    Args:
        src_bounds: Source bounds as (left, bottom, right, top)
        src_crs: Source CRS
        dst_crs: Destination CRS
        
    Returns:
        Transformed bounds as (left, bottom, right, top)
        
    Raises:
        ImportError: If pyproj is not available
    """
    _require_reproject_deps()
    
    transformer = pyproj.Transformer.from_crs(
        str(src_crs), str(dst_crs), always_xy=True
    )
    
    left, bottom, right, top = src_bounds
    
    # Transform corners
    corners_x = [left, left, right, right]
    corners_y = [bottom, top, bottom, top]
    
    transformed_x, transformed_y = transformer.transform(corners_x, corners_y)
    
    # Return new bounds
    return (
        min(transformed_x),
        min(transformed_y), 
        max(transformed_x),
        max(transformed_y)
    )


def get_crs_info(crs: Union[str, Any]) -> dict:
    """
    Get information about a coordinate reference system.
    
    Args:
        crs: CRS as string or Any object
        
    Returns:
        Dictionary with CRS information
        
    Raises:
        ImportError: If pyproj is not available
    """
    _require_reproject_deps()
    
    if isinstance(crs, str):
        crs = Any.from_string(crs)
        
    return {
        'name': crs.name,
        'authority': f"{crs.to_authority()[0]}:{crs.to_authority()[1]}" if crs.to_authority() else None,
        'proj_string': crs.to_proj4(),
        'wkt': crs.to_wkt(),
        'is_geographic': crs.is_geographic,
        'is_projected': crs.is_projected,
        'axis_info': [
            {'name': axis.name, 'abbreviation': axis.abbreviation, 'direction': axis.direction}
            for axis in crs.axis_info
        ] if crs.axis_info else [],
        'area_of_use': {
            'bounds': crs.area_of_use.bounds if crs.area_of_use else None,
            'name': crs.area_of_use.name if crs.area_of_use else None
        } if crs.area_of_use else None
    }


def estimate_reproject_error(
    src_dataset,
    dst_crs: Union[str, Any],
    sample_points: int = 100
) -> dict:
    """
    Estimate reprojection accuracy by sampling points across the dataset.
    
    Args:
        src_dataset: Source rasterio dataset
        dst_crs: Target CRS
        sample_points: Number of points to sample for error estimation
        
    Returns:
        Dictionary with error statistics
        
    Raises:
        ImportError: If required dependencies are not available
    """
    _require_reproject_deps()
    
    # Sample points across the dataset
    width, height = src_dataset.width, src_dataset.height
    
    # Create regular grid of sample points
    cols = np.linspace(0, width-1, int(np.sqrt(sample_points)))
    rows = np.linspace(0, height-1, int(np.sqrt(sample_points)))
    
    col_grid, row_grid = np.meshgrid(cols, rows)
    sample_cols = col_grid.flatten()
    sample_rows = row_grid.flatten()
    
    # Convert pixel coordinates to geographic coordinates
    src_transform = src_dataset.transform
    src_x, src_y = rasterio.transform.xy(src_transform, sample_rows, sample_cols)
    
    # Transform to destination CRS
    transformer = pyproj.Transformer.from_crs(
        src_dataset.crs, str(dst_crs), always_xy=True
    )
    dst_x, dst_y = transformer.transform(src_x, src_y)
    
    # Transform back to source CRS
    inverse_transformer = pyproj.Transformer.from_crs(
        str(dst_crs), src_dataset.crs, always_xy=True
    )
    back_x, back_y = inverse_transformer.transform(dst_x, dst_y)
    
    # Calculate errors
    x_errors = np.array(back_x) - np.array(src_x)
    y_errors = np.array(back_y) - np.array(src_y)
    
    return {
        'sample_count': len(x_errors),
        'x_error_stats': {
            'mean': float(np.mean(x_errors)),
            'std': float(np.std(x_errors)),
            'max_abs': float(np.max(np.abs(x_errors))),
            'rms': float(np.sqrt(np.mean(x_errors**2)))
        },
        'y_error_stats': {
            'mean': float(np.mean(y_errors)),
            'std': float(np.std(y_errors)),
            'max_abs': float(np.max(np.abs(y_errors))),
            'rms': float(np.sqrt(np.mean(y_errors**2)))
        },
        'combined_rms_error': float(np.sqrt(np.mean(x_errors**2 + y_errors**2)))
    }


# Export public API
__all__ = [
    'WarpedVRTWrapper',
    'reproject_window',
    'transform_bounds',
    'get_crs_info',
    'estimate_reproject_error',
    'is_reproject_available'
]