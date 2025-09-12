"""
forge3d.adapters.rasterio_tiles - Windowed reading and block iteration for rasterio datasets

This module provides efficient windowed reading functionality for large raster datasets,
supporting block-aligned iteration and various resampling modes.
"""

from typing import Optional, Tuple, Iterator, Union, Any
import warnings

import numpy as np

try:
    import rasterio
    from rasterio.windows import Window
    from rasterio.enums import Resampling
    _HAS_RASTERIO = True
except ImportError:
    _HAS_RASTERIO = False
    Window = Any
    Resampling = Any


def _require_rasterio():
    """Raise ImportError if rasterio is not available."""
    if not _HAS_RASTERIO:
        raise ImportError(
            "rasterio is required for raster I/O operations. "
            "Install with: pip install 'forge3d[raster]' or pip install rasterio"
        )


def is_rasterio_available() -> bool:
    """Check if rasterio is available."""
    return _HAS_RASTERIO


def windowed_read(
    dataset,
    window: Window,
    out_shape: Optional[Tuple[int, int]] = None,
    resampling: Optional[Union[str, "Resampling"]] = None,
    indexes: Optional[Union[int, list]] = None,
    dtype: Optional[str] = None
) -> np.ndarray:
    """
    Read a windowed portion of a rasterio dataset with optional resampling.
    
    Args:
        dataset: Rasterio dataset object
        window: Rasterio Window specifying the region to read
        out_shape: Optional (height, width) tuple for resampling output
        resampling: Resampling method (default: nearest)  
        indexes: Band indexes to read (default: all bands)
        dtype: Output data type (default: dataset dtype)
        
    Returns:
        NumPy array with shape (bands, height, width) or (height, width) for single band
        
    Raises:
        ImportError: If rasterio is not installed
        ValueError: If window or parameters are invalid
    """
    _require_rasterio()
    
    if resampling is None:
        resampling = Resampling.nearest
    elif isinstance(resampling, str):
        resampling = getattr(Resampling, resampling.lower(), Resampling.nearest)
    
    if indexes is None:
        indexes = list(range(1, dataset.count + 1))
    elif isinstance(indexes, int):
        indexes = [indexes]
    
    try:
        # Read with optional resampling
        data = dataset.read(
            indexes=indexes,
            window=window,
            out_shape=out_shape,
            resampling=resampling,
            dtype=dtype
        )
        
        # Squeeze single band dimension if only one band
        if len(indexes) == 1:
            data = data.squeeze(0)
            
        return data
        
    except Exception as e:
        raise ValueError(f"Failed to read window {window}: {e}")


def block_iterator(
    dataset, 
    blocksize: Optional[Union[int, Tuple[int, int]]] = None
) -> Iterator[Tuple[Window, np.ndarray]]:
    """
    Iterate over blocks of a rasterio dataset yielding windows and data.
    
    Args:
        dataset: Rasterio dataset object
        blocksize: Optional block size as int or (height, width) tuple.
                  If None, uses dataset's natural block structure.
                  
    Yields:
        Tuples of (window, array) where:
        - window: rasterio.Window for the block
        - array: NumPy array with the block data
        
    Raises:
        ImportError: If rasterio is not installed
    """
    _require_rasterio()
    
    if blocksize is None:
        # Use natural block structure from dataset
        block_windows = list(dataset.block_windows())
    else:
        # Create custom block structure
        if isinstance(blocksize, int):
            blocksize = (blocksize, blocksize)
            
        height, width = dataset.height, dataset.width
        block_height, block_width = blocksize
        
        block_windows = []
        for row_start in range(0, height, block_height):
            for col_start in range(0, width, block_width):
                # Calculate actual block size (may be smaller at edges)
                actual_height = min(block_height, height - row_start)
                actual_width = min(block_width, width - col_start)
                
                window = Window(
                    col_off=col_start,
                    row_off=row_start, 
                    width=actual_width,
                    height=actual_height
                )
                block_windows.append((0, window))  # (block_id, window)
    
    # Yield windows and corresponding data
    for block_id, window in block_windows:
        try:
            # Read all bands for this block
            data = windowed_read(dataset, window)
            yield window, data
        except Exception as e:
            warnings.warn(f"Failed to read block {window}: {e}")
            continue


def get_dataset_info(dataset) -> dict:
    """
    Get summary information about a rasterio dataset.
    
    Args:
        dataset: Rasterio dataset object
        
    Returns:
        Dictionary with dataset metadata
        
    Raises:
        ImportError: If rasterio is not installed
    """
    _require_rasterio()
    
    return {
        'width': dataset.width,
        'height': dataset.height,
        'count': dataset.count,
        'dtype': str(dataset.dtype),
        'crs': str(dataset.crs) if dataset.crs else None,
        'transform': list(dataset.transform) if dataset.transform else None,
        'bounds': list(dataset.bounds),
        'nodata': dataset.nodata,
        'block_shapes': dataset.block_shapes,
        'overviews': [dataset.overviews(i) for i in range(1, dataset.count + 1)],
        'colorinterp': [str(ci) for ci in dataset.colorinterp] if dataset.colorinterp else None,
    }


def validate_window(dataset, window: Window) -> bool:
    """
    Validate that a window is within dataset bounds.
    
    Args:
        dataset: Rasterio dataset object
        window: Window to validate
        
    Returns:
        True if window is valid, False otherwise
        
    Raises:
        ImportError: If rasterio is not available
    """
    _require_rasterio()
    
    if window.col_off < 0 or window.row_off < 0:
        return False
        
    if (window.col_off + window.width) > dataset.width:
        return False
        
    if (window.row_off + window.height) > dataset.height:
        return False
        
    return True


def extract_masks(dataset, window: Optional[Window] = None) -> np.ndarray:
    """
    Extract mask data from a rasterio dataset.
    
    Args:
        dataset: Rasterio dataset object
        window: Optional window to read masks for (default: full dataset)
        
    Returns:
        NumPy array with mask data where 0=nodata, 255=valid data
        Shape is (bands, height, width) or (height, width) for single band
        
    Raises:
        ImportError: If rasterio is not installed
    """
    _require_rasterio()
    
    try:
        # Try dataset-level mask first
        if hasattr(dataset, 'dataset_mask'):
            if window is not None:
                mask = dataset.dataset_mask(window=window)
            else:
                mask = dataset.dataset_mask()
            return mask
            
        # Fall back to per-band masks
        if window is not None:
            masks = dataset.read_masks(window=window)
        else:
            masks = dataset.read_masks()
            
        # If single band, squeeze the band dimension
        if dataset.count == 1:
            masks = masks.squeeze(0)
            
        return masks
        
    except Exception as e:
        # Create default mask (all valid) if mask reading fails
        if window is not None:
            shape = (window.height, window.width)
        else:
            shape = (dataset.height, dataset.width)
            
        if dataset.count > 1:
            shape = (dataset.count,) + shape
            
        warnings.warn(f"Could not read masks, using default valid mask: {e}")
        return np.full(shape, 255, dtype=np.uint8)


def windowed_read_with_alpha(
    dataset,
    window: Window,
    out_shape: Optional[Tuple[int, int]] = None,
    resampling: Optional[Union[str, "Resampling"]] = None,
    indexes: Optional[Union[int, list]] = None,
    dtype: Optional[str] = None,
    add_alpha: bool = True
) -> np.ndarray:
    """
    Read windowed data with alpha channel derived from nodata/mask.
    
    Args:
        dataset: Rasterio dataset object
        window: Rasterio Window specifying the region to read
        out_shape: Optional (height, width) tuple for resampling output
        resampling: Resampling method (default: nearest)
        indexes: Band indexes to read (default: all bands)
        dtype: Output data type (default: uint8)
        add_alpha: Whether to add alpha channel from masks (default: True)
        
    Returns:
        NumPy array with RGBA data where alpha=0 for nodata areas
        Shape is (height, width, channels) with channels=3 or 4
        
    Raises:
        ImportError: If rasterio is not installed
    """
    _require_rasterio()
    
    # Read the color data
    data = windowed_read(dataset, window, out_shape, resampling, indexes, dtype)
    
    # Ensure we have the right shape and type for RGBA processing
    if data.ndim == 2:
        # Single band - convert to (H, W, 1)
        data = data[:, :, np.newaxis]
    elif data.ndim == 3 and data.shape[0] <= 4:
        # Band-first format (bands, H, W) -> (H, W, bands)
        data = np.transpose(data, (1, 2, 0))
    
    # Convert to uint8 if needed
    if dtype is None or dtype == 'uint8':
        if data.dtype != np.uint8:
            # Scale to 0-255 range
            if np.issubdtype(data.dtype, np.floating):
                data = np.clip(data * 255, 0, 255).astype(np.uint8)
            else:
                data = data.astype(np.uint8)
    
    if not add_alpha:
        return data
        
    # Get mask data for alpha channel
    try:
        mask = extract_masks(dataset, window)
        
        # Resample mask if needed
        if out_shape is not None and mask.shape[-2:] != out_shape:
            from scipy.ndimage import zoom
            if mask.ndim == 3:
                # Multiple band masks
                zoom_factors = (1.0, out_shape[0]/mask.shape[1], out_shape[1]/mask.shape[2])
            else:
                # Single mask
                zoom_factors = (out_shape[0]/mask.shape[0], out_shape[1]/mask.shape[1])
            mask = zoom(mask, zoom_factors, order=0, prefilter=False)
            mask = mask.astype(np.uint8)
        
        # Create alpha channel
        if mask.ndim == 3:
            # Multi-band mask - combine using AND operation
            alpha = np.all(mask > 0, axis=0).astype(np.uint8) * 255
        else:
            # Single mask
            alpha = mask
            
        # Ensure alpha has the right shape
        if alpha.shape != data.shape[:2]:
            alpha = np.resize(alpha, data.shape[:2])
            
        alpha = alpha[:, :, np.newaxis]
        
    except Exception as e:
        warnings.warn(f"Failed to create alpha channel, using opaque: {e}")
        # Create fully opaque alpha
        alpha = np.full((*data.shape[:2], 1), 255, dtype=np.uint8)
    
    # Handle different numbers of color channels
    if data.shape[2] == 1:
        # Grayscale -> RGBA (duplicate to RGB)
        rgb = np.repeat(data, 3, axis=2)
        rgba = np.concatenate([rgb, alpha], axis=2)
    elif data.shape[2] == 3:
        # RGB -> RGBA
        rgba = np.concatenate([data, alpha], axis=2)
    elif data.shape[2] >= 4:
        # Already has alpha channel, replace it
        rgba = np.concatenate([data[:, :, :3], alpha], axis=2)
    else:
        raise ValueError(f"Unexpected number of channels: {data.shape[2]}")
    
    return rgba


def create_synthetic_nodata_mask(
    shape: Tuple[int, int],
    nodata_fraction: float = 0.1,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Create a synthetic mask for testing purposes.
    
    Args:
        shape: (height, width) of mask to create
        nodata_fraction: Fraction of pixels to mark as nodata (0-1)
        seed: Random seed for reproducible masks
        
    Returns:
        Mask array where 0=nodata, 255=valid
    """
    if seed is not None:
        np.random.seed(seed)
        
    mask = np.full(shape, 255, dtype=np.uint8)
    if nodata_fraction > 0:
        n_nodata = int(shape[0] * shape[1] * nodata_fraction)
        flat_mask = mask.flatten()
        nodata_indices = np.random.choice(len(flat_mask), n_nodata, replace=False)
        flat_mask[nodata_indices] = 0
        mask = flat_mask.reshape(shape)
    
    return mask


def select_overview_level(
    dataset,
    target_resolution: Union[float, Tuple[float, float]],
    band: int = 1
) -> Tuple[int, dict]:
    """
    Select the most appropriate overview level for a target resolution.
    
    Args:
        dataset: Rasterio dataset object
        target_resolution: Target resolution in dataset units, either as float
                          for square pixels or (x_res, y_res) tuple
        band: Band number to check for overviews (1-indexed, default: 1)
        
    Returns:
        Tuple of (overview_index, overview_info) where:
        - overview_index: Index of selected overview (-1 for full resolution)
        - overview_info: Dictionary with overview metadata
        
    Raises:
        ImportError: If rasterio is not available
    """
    _require_rasterio()
    
    if isinstance(target_resolution, (int, float)):
        target_x_res = target_y_res = float(target_resolution)
    else:
        target_x_res, target_y_res = target_resolution
    
    # Get dataset resolution
    transform = dataset.transform
    dataset_x_res = abs(transform.a)  # x pixel size
    dataset_y_res = abs(transform.e)  # y pixel size
    
    # Get available overviews for the specified band
    overview_factors = dataset.overviews(band)
    
    if not overview_factors:
        # No overviews available
        return -1, {
            'overview_index': -1,
            'overview_factor': 1,
            'resolution_x': dataset_x_res,
            'resolution_y': dataset_y_res,
            'width': dataset.width,
            'height': dataset.height,
            'bytes_reduction': 0.0
        }
    
    # Calculate resolution for each overview level
    overview_info = []
    for i, factor in enumerate(overview_factors):
        overview_x_res = dataset_x_res * factor
        overview_y_res = dataset_y_res * factor
        overview_width = dataset.width // factor
        overview_height = dataset.height // factor
        
        # Calculate how well this overview matches the target resolution
        # Use maximum of x/y resolution ratio as the matching score
        x_ratio = max(target_x_res / overview_x_res, overview_x_res / target_x_res)
        y_ratio = max(target_y_res / overview_y_res, overview_y_res / target_y_res)
        mismatch_score = max(x_ratio, y_ratio)
        
        bytes_reduction = 1.0 - (1.0 / (factor * factor))
        
        overview_info.append({
            'overview_index': i,
            'overview_factor': factor,
            'resolution_x': overview_x_res,
            'resolution_y': overview_y_res,
            'width': overview_width,
            'height': overview_height,
            'mismatch_score': mismatch_score,
            'bytes_reduction': bytes_reduction
        })
    
    # Add full resolution option
    full_res_x_ratio = max(target_x_res / dataset_x_res, dataset_x_res / target_x_res)
    full_res_y_ratio = max(target_y_res / dataset_y_res, dataset_y_res / target_y_res)
    full_res_mismatch = max(full_res_x_ratio, full_res_y_ratio)
    
    overview_info.append({
        'overview_index': -1,
        'overview_factor': 1,
        'resolution_x': dataset_x_res,
        'resolution_y': dataset_y_res,
        'width': dataset.width,
        'height': dataset.height,
        'mismatch_score': full_res_mismatch,
        'bytes_reduction': 0.0
    })
    
    # Select overview with minimum mismatch score
    best_overview = min(overview_info, key=lambda x: x['mismatch_score'])
    
    return best_overview['overview_index'], best_overview


def windowed_read_with_overview(
    dataset,
    window: Window,
    target_resolution: Optional[Union[float, Tuple[float, float]]] = None,
    out_shape: Optional[Tuple[int, int]] = None,
    resampling: Optional[Union[str, "Resampling"]] = None,
    indexes: Optional[Union[int, list]] = None,
    dtype: Optional[str] = None,
    band_for_overview: int = 1
) -> Tuple[np.ndarray, dict]:
    """
    Read windowed data using the best overview level for target resolution.
    
    Args:
        dataset: Rasterio dataset object
        window: Window to read from dataset
        target_resolution: Target resolution to select overview level
        out_shape: Optional output shape
        resampling: Resampling method
        indexes: Band indexes to read
        dtype: Output data type
        band_for_overview: Band to use for overview selection
        
    Returns:
        Tuple of (data, overview_info) where:
        - data: NumPy array with windowed data
        - overview_info: Dictionary with selected overview metadata
        
    Raises:
        ImportError: If rasterio is not available
    """
    _require_rasterio()
    
    # Select best overview level
    if target_resolution is not None:
        overview_index, overview_info = select_overview_level(
            dataset, target_resolution, band_for_overview
        )
    else:
        overview_index = -1
        overview_info = {
            'overview_index': -1,
            'overview_factor': 1,
            'resolution_x': abs(dataset.transform.a),
            'resolution_y': abs(dataset.transform.e),
            'width': dataset.width,
            'height': dataset.height,
            'bytes_reduction': 0.0
        }
    
    if overview_index == -1:
        # Use full resolution
        data = windowed_read(dataset, window, out_shape, resampling, indexes, dtype)
    else:
        # Use overview
        if indexes is None:
            indexes = list(range(1, dataset.count + 1))
        elif isinstance(indexes, int):
            indexes = [indexes]
        
        # Adjust window for overview resolution
        overview_factor = overview_info['overview_factor']
        overview_window = Window(
            col_off=window.col_off // overview_factor,
            row_off=window.row_off // overview_factor,
            width=max(1, window.width // overview_factor),
            height=max(1, window.height // overview_factor)
        )
        
        # Read from overview
        data_list = []
        for band_idx in indexes:
            with dataset.overviews(band_idx)[overview_index] as overview_dataset:
                band_data = overview_dataset.read(
                    window=overview_window,
                    out_shape=out_shape,
                    resampling=resampling or Resampling.nearest
                )
                data_list.append(band_data)
        
        if len(data_list) == 1:
            data = data_list[0]
        else:
            data = np.stack(data_list, axis=0)
    
    return data, overview_info


def calculate_overview_savings(
    dataset,
    target_resolutions: list,
    band: int = 1
) -> dict:
    """
    Calculate potential data savings from using overviews at different resolutions.
    
    Args:
        dataset: Rasterio dataset object
        target_resolutions: List of target resolutions to analyze
        band: Band number for overview analysis
        
    Returns:
        Dictionary with savings analysis for each target resolution
        
    Raises:
        ImportError: If rasterio is not available
    """
    _require_rasterio()
    
    results = {
        'dataset_info': {
            'width': dataset.width,
            'height': dataset.height,
            'full_res_pixels': dataset.width * dataset.height,
            'available_overviews': dataset.overviews(band)
        },
        'resolution_analysis': []
    }
    
    for target_res in target_resolutions:
        overview_index, overview_info = select_overview_level(dataset, target_res, band)
        
        # Calculate actual bytes that would be read
        if overview_index == -1:
            pixels_read = dataset.width * dataset.height
        else:
            pixels_read = overview_info['width'] * overview_info['height']
        
        full_res_pixels = dataset.width * dataset.height
        byte_reduction = 1.0 - (pixels_read / full_res_pixels)
        
        results['resolution_analysis'].append({
            'target_resolution': target_res,
            'selected_overview': overview_info,
            'pixels_read': pixels_read,
            'full_res_pixels': full_res_pixels,
            'byte_reduction_actual': byte_reduction,
            'meets_60_percent_savings': byte_reduction >= 0.6
        })
    
    return results


# Export availability check for other modules
__all__ = [
    'windowed_read',
    'block_iterator', 
    'get_dataset_info',
    'validate_window',
    'extract_masks',
    'windowed_read_with_alpha',
    'create_synthetic_nodata_mask',
    'select_overview_level',
    'windowed_read_with_overview',
    'calculate_overview_savings',
    'is_rasterio_available'
]