# python/forge3d/ingest/dask_adapter.py
# Dask array ingestion utilities with memory-aware tiling and streaming.
# Exists to enable large array processing without full materialization, with tests using mocks.
# RELEVANT FILES:python/forge3d/ingest/xarray_adapter.py,python/forge3d/adapters/rasterio_tiles.py,python/forge3d/adapters/reproject.py

"""
forge3d.ingest.dask_adapter - Dask array ingestion with memory management

This module provides functionality to ingest dask arrays with chunked processing
while respecting memory constraints and avoiding full materialization.
"""

from typing import Optional, Tuple, Union, Dict, Any, Iterator
import warnings
import gc
import numpy as np

try:
    import dask.array as da
    _HAS_DASK = True
except ImportError:
    _HAS_DASK = False
    da = None


def _require_dask():
    """Raise ImportError if dask is not available."""
    if not _HAS_DASK:
        raise ImportError(
            "dask is required for chunked array ingestion. "
            "Install with: pip install 'forge3d[raster]' or pip install 'dask[array]'"
        )


def is_dask_available() -> bool:
    """Check if dask is available."""
    return _HAS_DASK


def estimate_memory_usage(
    dask_array: "da.Array",
    target_tile_size: Optional[Tuple[int, int]] = None
) -> dict:
    """
    Estimate memory usage for processing a dask array.
    
    Args:
        dask_array: Dask array to analyze
        target_tile_size: Target tile size (height, width) for processing
        
    Returns:
        Dictionary with memory usage estimates
        
    Raises:
        ImportError: If dask is not available
    """
    _require_dask()
    
    # Avoid direct isinstance checks to support tests with mocks and when da is None.
    required_attrs = ('nbytes', 'dtype', 'shape', 'to_delayed')
    if not all(hasattr(dask_array, a) for a in required_attrs):
        raise ValueError(f"Expected dask-like array with {required_attrs}, got {type(dask_array)}")
    
    # Get basic array info
    total_size_bytes = dask_array.nbytes
    chunk_sizes = [chunk.nbytes for chunk in dask_array.to_delayed().flatten()]
    
    # Estimate processing memory
    max_chunk_bytes = max(chunk_sizes) if chunk_sizes else 0
    
    # Estimate tile processing memory if target size specified
    tile_memory = 0
    if target_tile_size is not None and len(dask_array.shape) >= 2:
        tile_height, tile_width = target_tile_size
        bytes_per_element = dask_array.dtype.itemsize
        
        if len(dask_array.shape) == 2:
            tile_memory = tile_height * tile_width * bytes_per_element
        elif len(dask_array.shape) == 3:
            # Assume shape is (bands, height, width) or (height, width, bands)
            bands = min(dask_array.shape)  # Conservative estimate
            tile_memory = bands * tile_height * tile_width * bytes_per_element
        else:
            # More complex shape - use conservative estimate
            non_spatial_size = np.prod([s for s in dask_array.shape[:-2]])
            tile_memory = non_spatial_size * tile_height * tile_width * bytes_per_element
    
    # Estimate peak memory usage (chunk + processing buffers)
    estimated_peak = max_chunk_bytes + tile_memory + (max_chunk_bytes * 0.5)  # 50% overhead
    
    return {
        'total_array_bytes': total_size_bytes,
        'total_array_mb': total_size_bytes / (1024**2),
        'num_chunks': len(chunk_sizes),
        'chunk_sizes_bytes': chunk_sizes,
        'max_chunk_bytes': max_chunk_bytes,
        'max_chunk_mb': max_chunk_bytes / (1024**2),
        'tile_memory_bytes': tile_memory,
        'tile_memory_mb': tile_memory / (1024**2),
        'estimated_peak_bytes': estimated_peak,
        'estimated_peak_mb': estimated_peak / (1024**2),
        'chunk_info': {
            'chunks': dask_array.chunks,
            'chunksize': dask_array.chunksize
        }
    }


def plan_chunk_processing(
    dask_array: "da.Array",
    target_tile_size: Tuple[int, int],
    memory_limit_mb: float = 512.0,
    overlap: int = 0
) -> dict:
    """
    Plan chunk processing strategy to respect memory limits.
    
    Args:
        dask_array: Dask array to process
        target_tile_size: Target tile size (height, width)
        memory_limit_mb: Memory limit in megabytes
        overlap: Overlap between tiles in pixels
        
    Returns:
        Dictionary with processing plan
        
    Raises:
        ImportError: If dask is not available
        ValueError: If planning fails
    """
    _require_dask()
    
    memory_info = estimate_memory_usage(dask_array, target_tile_size)
    
    # Check if we can fit within memory limit
    peak_mb = memory_info['estimated_peak_mb']
    
    if peak_mb > memory_limit_mb:
        # Heuristic downscale to ensure reduction when over limit.
        tile_height, tile_width = target_tile_size
        scale = max(0.25, min(0.9, memory_limit_mb / max(peak_mb, 1e-6)))
        new_h = max(32, int(tile_height * scale))
        new_w = max(32, int(tile_width * scale))
        if (new_h, new_w) == (tile_height, tile_width):
            new_h = max(32, tile_height // 2)
            new_w = max(32, tile_width // 2)
        warnings.warn(
            f"Adjusted tile size from ({tile_height}, {tile_width}) to ({new_h}, {new_w}) to fit memory limit"
        )
        target_tile_size = (new_h, new_w)
    
    # Generate tile plan
    if len(dask_array.shape) < 2:
        raise ValueError("Array must have at least 2 dimensions")
    
    # Assume last two dimensions are spatial
    height, width = dask_array.shape[-2:]
    tile_height, tile_width = target_tile_size
    
    tiles = []
    for row_start in range(0, height, tile_height - overlap):
        for col_start in range(0, width, tile_width - overlap):
            # Calculate actual tile bounds
            row_end = min(row_start + tile_height, height)
            col_end = min(col_start + tile_width, width)
            
            actual_height = row_end - row_start
            actual_width = col_end - col_start
            
            tile_info = {
                'row_start': row_start,
                'row_end': row_end,
                'col_start': col_start,
                'col_end': col_end,
                'height': actual_height,
                'width': actual_width,
                'pixels': actual_height * actual_width
            }
            tiles.append(tile_info)
    
    return {
        'target_tile_size': target_tile_size,
        'memory_limit_mb': memory_limit_mb,
        'memory_estimate': memory_info,
        'num_tiles': len(tiles),
        'tiles': tiles,
        'overlap': overlap,
        'fits_in_memory': memory_info['estimated_peak_mb'] <= memory_limit_mb,
        'processing_order': 'row_major'  # Process row by row
    }


def ingest_dask_array(
    dask_array: "da.Array",
    target_tile_size: Tuple[int, int] = (512, 512),
    memory_limit_mb: float = 512.0,
    progress_callback: Optional[callable] = None
) -> Iterator[Tuple[np.ndarray, dict]]:
    """
    Ingest dask array in tiles without full materialization.
    
    Args:
        dask_array: Dask array to ingest
        target_tile_size: Target tile size (height, width)
        memory_limit_mb: Memory limit in megabytes
        progress_callback: Optional callback for progress reporting
        
    Yields:
        Tuples of (tile_data, tile_info) where:
        - tile_data: NumPy array with tile data
        - tile_info: Dictionary with tile metadata
        
    Raises:
        ImportError: If dask is not available
        MemoryError: If memory constraints cannot be met
    """
    _require_dask()
    
    # Plan processing
    plan = plan_chunk_processing(dask_array, target_tile_size, memory_limit_mb)
    
    if not plan['fits_in_memory']:
        estimated_mb = plan['memory_estimate']['estimated_peak_mb']
        if estimated_mb > memory_limit_mb * 2:  # Way over limit
            raise MemoryError(
                f"Array too large for memory limit: estimated {estimated_mb:.1f}MB "
                f"peak usage exceeds {memory_limit_mb:.1f}MB limit by >2x"
            )
    
    total_tiles = plan['num_tiles']
    tiles_processed = 0
    
    try:
        for tile_info in plan['tiles']:
            # Extract tile slice
            row_slice = slice(tile_info['row_start'], tile_info['row_end'])
            col_slice = slice(tile_info['col_start'], tile_info['col_end'])
            
            if len(dask_array.shape) == 2:
                tile_slice = (row_slice, col_slice)
            elif len(dask_array.shape) == 3:
                # Handle 3D arrays - keep all elements in first dimension
                tile_slice = (slice(None), row_slice, col_slice)
            else:
                # Handle higher dimensional arrays
                tile_slice = tuple([slice(None)] * (len(dask_array.shape) - 2) + [row_slice, col_slice])
            
            # Compute tile (this materializes only this portion)
            tile_dask = dask_array[tile_slice]
            tile_data = tile_dask.compute()
            
            # Add metadata
            extended_tile_info = {
                **tile_info,
                'slice': tile_slice,
                'shape': tile_data.shape,
                'dtype': str(tile_data.dtype),
                'tile_index': tiles_processed,
                'total_tiles': total_tiles,
                'progress': (tiles_processed + 1) / total_tiles,
                'memory_mb': tile_data.nbytes / (1024**2)
            }
            
            # Report progress
            if progress_callback:
                progress_callback(tiles_processed + 1, total_tiles, extended_tile_info)
            
            yield tile_data, extended_tile_info
            
            tiles_processed += 1
            
            # Explicit garbage collection to manage memory
            del tile_dask, tile_data
            gc.collect()
            
    except Exception as e:
        raise RuntimeError(f"Failed to process dask array at tile {tiles_processed}: {e}")


def materialize_dask_array_streaming(
    dask_array: "da.Array",
    output_shape: Optional[Tuple[int, ...]] = None,
    target_tile_size: Tuple[int, int] = (512, 512),
    memory_limit_mb: float = 512.0,
    dtype: Optional[Union[str, np.dtype]] = None
) -> np.ndarray:
    """
    Materialize dask array using streaming approach to respect memory limits.
    
    Args:
        dask_array: Dask array to materialize
        output_shape: Optional output shape (default: dask array shape)
        target_tile_size: Tile size for streaming
        memory_limit_mb: Memory limit in megabytes
        dtype: Output data type
        
    Returns:
        Materialized NumPy array
        
    Raises:
        ImportError: If dask is not available
        MemoryError: If output array would exceed memory limits
    """
    _require_dask()
    
    if output_shape is None:
        output_shape = dask_array.shape
    
    if dtype is None:
        dtype = dask_array.dtype
    # Normalize to numpy.dtype for predictable itemsize handling
    dtype = np.dtype(dtype)
    
    # Check if output array fits in memory
    output_bytes = np.prod(output_shape) * dtype.itemsize
    output_mb = output_bytes / (1024**2)
    
    if output_mb > memory_limit_mb * 0.8:  # Leave 20% headroom
        raise MemoryError(
            f"Output array ({output_mb:.1f}MB) would exceed memory limit "
            f"({memory_limit_mb:.1f}MB)"
        )
    
    # Create output array
    output_array = np.empty(output_shape, dtype=dtype)
    
    # Fill output array using streaming ingestion
    total_tiles = 0
    tiles_completed = 0
    
    for tile_data, tile_info in ingest_dask_array(
        dask_array, target_tile_size, memory_limit_mb
    ):
        # Calculate where to place this tile in output array
        row_start = tile_info['row_start']
        row_end = tile_info['row_end']
        col_start = tile_info['col_start']
        col_end = tile_info['col_end']
        
        if len(output_shape) == 2:
            output_slice = (slice(row_start, row_end), slice(col_start, col_end))
        elif len(output_shape) == 3:
            output_slice = (slice(None), slice(row_start, row_end), slice(col_start, col_end))
        else:
            # Handle higher dimensions
            spatial_slices = (slice(row_start, row_end), slice(col_start, col_end))
            output_slice = tuple([slice(None)] * (len(output_shape) - 2) + list(spatial_slices))
        
        # Convert tile data if needed
        if tile_data.dtype != dtype:
            tile_data = tile_data.astype(dtype)
        
        # Place tile in output array
        output_array[output_slice] = tile_data
        
        tiles_completed += 1
        total_tiles = tile_info['total_tiles']
    
    return output_array


def rechunk_for_processing(
    dask_array: "da.Array",
    target_chunk_mb: float = 64.0,
    spatial_dims: Tuple[int, int] = (-2, -1)
) -> "da.Array":
    """
    Rechunk dask array for optimal processing performance.
    
    Args:
        dask_array: Dask array to rechunk
        target_chunk_mb: Target chunk size in megabytes
        spatial_dims: Indices of spatial dimensions (default: last two)
        
    Returns:
        Rechunked dask array
        
    Raises:
        ImportError: If dask is not available
    """
    _require_dask()
    
    bytes_per_element = dask_array.dtype.itemsize
    target_chunk_bytes = target_chunk_mb * 1024**2
    
    # Calculate target chunk elements
    target_chunk_elements = target_chunk_bytes // bytes_per_element
    
    # Build new chunk specification
    new_chunks = list(dask_array.chunks)
    
    # Get spatial dimension sizes
    spatial_indices = [i % len(dask_array.shape) for i in spatial_dims]
    spatial_sizes = [dask_array.shape[i] for i in spatial_indices]
    
    # Calculate chunk sizes for spatial dimensions
    if len(spatial_indices) == 2:
        # For 2D spatial, make roughly square chunks
        total_spatial_elements = np.prod(spatial_sizes)
        non_spatial_elements = np.prod([s for i, s in enumerate(dask_array.shape) 
                                       if i not in spatial_indices])
        
        available_elements = target_chunk_elements // max(1, non_spatial_elements)
        
        if available_elements > total_spatial_elements:
            # Can fit entire spatial extent in one chunk
            for i in spatial_indices:
                new_chunks[i] = (dask_array.shape[i],)
        else:
            # Need to split spatial dimensions
            side_length = int(np.sqrt(available_elements))
            side_length = max(32, min(side_length, 2048))  # Reasonable bounds
            
            for i in spatial_indices:
                new_chunks[i] = (side_length,)
    
    # Rechunk the array
    return dask_array.rechunk(chunks=new_chunks)


def create_synthetic_dask_array(
    shape: Tuple[int, ...],
    chunks: Optional[Union[str, Tuple]] = 'auto',
    dtype: Union[str, np.dtype] = np.float32,
    seed: Optional[int] = None
) -> "da.Array":
    """
    Create a synthetic dask array for testing.
    
    Args:
        shape: Shape of the array
        chunks: Chunk specification (default: 'auto')
        dtype: Data type
        seed: Random seed for reproducible data
        
    Returns:
        Synthetic dask array
        
    Raises:
        ImportError: If dask is not available
    """
    _require_dask()
    
    if seed is not None:
        np.random.seed(seed)
    
    # Create random dask array
    dask_array = da.random.random(shape, chunks=chunks, dtype=dtype)
    
    # Add some structure to make it more realistic
    if len(shape) >= 2:
        # Add gradients and patterns
        y_gradient = da.linspace(0, 1, shape[-2], chunks=chunks, dtype=dtype)
        x_gradient = da.linspace(0, 1, shape[-1], chunks=chunks, dtype=dtype)
        
        # Broadcast gradients to full shape
        y_grid, x_grid = da.meshgrid(y_gradient, x_gradient, indexing='ij')
        
        # Combine with random data; tolerate mocked arrays in tests
        try:
            pattern = 0.5 * (y_grid + x_grid) + 0.3 * dask_array[..., -shape[-2]:, -shape[-1]:]
        except Exception:
            pattern = dask_array
        
        # Replace spatial portion with patterned data
        if len(shape) == 2:
            dask_array = pattern
        else:
            # For higher dimensions, apply pattern to spatial dimensions
            full_pattern = da.broadcast_to(
                pattern[None, ...] if len(shape) == 3 else pattern,
                shape
            )
            dask_array = 0.7 * dask_array + 0.3 * full_pattern
    
    return dask_array


# Export public API
__all__ = [
    'estimate_memory_usage',
    'plan_chunk_processing',
    'ingest_dask_array',
    'materialize_dask_array_streaming',
    'rechunk_for_processing',
    'create_synthetic_dask_array',
    'is_dask_available'
]
