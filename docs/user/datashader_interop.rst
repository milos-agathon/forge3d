====================
Datashader Integration
====================

.. currentmodule:: forge3d.adapters

forge3d provides seamless integration with `Datashader <https://datashader.org/>`_ 
for large-scale point and line visualization. This integration enables efficient 
rendering of millions of data points with minimal memory overhead through 
zero-copy optimizations and coordinate alignment validation.

Overview
========

Datashader excels at aggregating large datasets into rasterized representations,
while forge3d provides GPU-accelerated rendering and overlay compositing. Together,
they enable interactive visualization of massive point datasets that would be 
impossible to render directly.

The integration supports:

- **Zero-copy RGBA conversion** from Datashader outputs
- **Coordinate alignment validation** for precise overlay positioning
- **Memory-efficient processing** within forge3d's 512 MB budget
- **Performance monitoring** and optimization guidance

Quick Start
===========

.. code-block:: python

   import pandas as pd
   import datashader as ds
   from forge3d.adapters import shade_to_overlay
   
   # Create or load your point data
   df = pd.DataFrame({
       'x': [/* longitude/easting values */],
       'y': [/* latitude/northing values */], 
       'value': [/* data values to visualize */]
   })
   
   # Define the geographic extent
   extent = (-120.0, -90.0, -60.0, 45.0)  # (xmin, ymin, xmax, ymax)
   
   # Create Datashader aggregation
   canvas = ds.Canvas(plot_width=800, plot_height=600,
                      x_range=(extent[0], extent[2]),
                      y_range=(extent[1], extent[3]))
   agg = canvas.points(df, 'x', 'y', ds.mean('value'))
   
   # Convert to forge3d overlay with zero-copy optimization
   overlay = shade_to_overlay(agg, extent, cmap='viridis', how='linear', premultiply=True)
   
   # Use overlay in forge3d rendering pipeline
   rgba_array = overlay['rgba']  # Ready for GPU upload

Installation
============

Datashader integration requires the optional datashader dependency:

.. code-block:: bash

   pip install datashader

The integration will gracefully degrade if datashader is not available,
providing helpful error messages when datashader-specific functions are called.

Verify availability:

.. code-block:: python

   from forge3d.adapters import is_datashader_available
   
   if is_datashader_available():
       print("Datashader integration ready")
   else:
       print("Install datashader: pip install datashader")

API Reference
=============

Availability Check
------------------

.. autofunction:: is_datashader_available

Core Functions
--------------

.. autofunction:: rgba_view_from_agg

   Convert Datashader aggregation or shaded image to RGBA array.
   
   **Zero-copy behavior**: When the input is already in RGBA uint8 format
   and C-contiguous, this function returns a view with no memory copying.
   
   **Parameters**:

   - ``premultiply`` (bool, default False): if True, returns a premultiplied-alpha
     copy of the input (RGB pre-multiplied by A/255). This creates a copy.
   - ``transform`` (optional): a transform object (e.g., rasterio Affine) used
     to validate alignment and returned in the overlay metadata for downstream use.

   **Returned metadata** additions:

   - ``premultiplied``: whether RGB has been premultiplied by alpha.
   - ``transform``: the transform object (if provided).

   **Example**:
   
   .. code-block:: python
   
      import datashader.transfer_functions as tf
      
      # Shade aggregation
      img = tf.shade(agg, cmap='plasma')
      
      # Convert with zero-copy where possible
      rgba = rgba_view_from_agg(img)
      
      # Verify zero-copy success
      if np.shares_memory(rgba, img.values):
          print("Zero-copy achieved!")

.. autofunction:: validate_alignment

   Validate coordinate alignment between datashader extent and transforms.
   
   **Alignment tolerance**: Pixel center offset must be ≤ 0.5 pixels to pass validation.
   
   **Example**:
   
   .. code-block:: python
   
      from rasterio.transform import from_bounds
      
      extent = (-100, 40, -90, 50)
      width, height = 1024, 768
      
      # Create a rasterio-style transform
      transform = from_bounds(*extent, width, height)
      
      # Validate alignment
      result = validate_alignment(extent, transform, width, height)
      print(f"Pixel error: {result['pixel_error_x']:.3f}px")

.. autofunction:: to_overlay_texture

   Prepare RGBA array and extent for forge3d overlay texture.
   
   **Format requirements**:
   
   - Shape: (height, width, 4) 
   - Dtype: uint8
   - Memory layout: C-contiguous
   
   **Example**:
   
   .. code-block:: python
   
      overlay = to_overlay_texture(rgba, extent)
      
      print(f"Overlay size: {overlay['width']}x{overlay['height']}")
      print(f"Memory: {overlay['total_bytes'] / 1024**2:.1f} MB")
      print(f"Zero-copy: {overlay['shares_memory']}")

Convenience Functions
---------------------

.. autofunction:: shade_to_overlay

   End-to-end conversion from Datashader aggregation to forge3d overlay.
   
   **Parameters**:
   
   - ``agg``: Datashader aggregation array
   - ``extent``: Geographic bounds (xmin, ymin, xmax, ymax)
   - ``cmap``: Colormap name (viridis, plasma, inferno, etc.)
   - ``how``: Shading method (linear, log, eq_hist, cbrt)
   - ``premultiply`` (bool): if True, premultiply the shaded RGBA before returning.
   - ``transform`` (optional): pass-through transform used for alignment validation
     and returned in overlay metadata.
   
   **Example**:
   
   .. code-block:: python
   
      # One-line conversion with validation
      overlay = shade_to_overlay(agg, extent, cmap='magma', how='log', premultiply=True)

Premultiplied Alpha
===================

Some render paths and compositors prefer premultiplied-alpha textures. You can
either request premultiplication from ``shade_to_overlay`` or apply it manually:

.. code-block:: python

   from forge3d.adapters import premultiply_rgba, to_overlay_texture

   rgba = rgba_view_from_agg(tf.shade(agg, cmap='viridis'))
   rgba_premult = premultiply_rgba(rgba)   # returns a new array
   overlay = to_overlay_texture(rgba_premult, extent, premultiply=False)
   assert overlay['premultiplied'] is False  # already premultiplied in data

   # Alternatively, let to_overlay_texture do it for you
   overlay2 = to_overlay_texture(rgba, extent, premultiply=True)
   assert overlay2['premultiplied'] is True

Coordinate Transforms
=====================

For precise geospatial alignment, you can pass a transform (e.g., rasterio
``Affine``) into ``shade_to_overlay`` and ``to_overlay_texture``. The transform
is used by ``validate_alignment`` and included in the returned metadata:

.. code-block:: python

   from rasterio.transform import from_bounds

   transform = from_bounds(*extent, width=800, height=600)
   overlay = shade_to_overlay(agg, extent, transform=transform)
   assert overlay['transform'] is transform

Adapter Class
-------------

.. autoclass:: DatashaderAdapter

   Advanced adapter with copy counting and performance monitoring.
   
   .. automethod:: __init__
   .. autoattribute:: copy_count
   .. automethod:: reset_copy_count

Information Functions  
--------------------

.. autofunction:: get_datashader_info

   Get comprehensive information about datashader availability and configuration.

Memory Management
=================

forge3d operates under a **512 MB host-visible memory budget**. The datashader
integration is designed to respect this constraint through several mechanisms:

Memory Optimization Strategies
------------------------------

**Zero-copy conversions**: When datashader outputs are already in the required
RGBA uint8 format, the adapter creates views rather than copies:

.. code-block:: python

   rgba = rgba_view_from_agg(datashader_image)
   
   # Check if zero-copy was achieved
   if np.shares_memory(rgba, original_data):
       print("Zero-copy successful - no additional memory used")

**Contiguous arrays**: GPU upload requires C-contiguous memory layout. The
adapter ensures this efficiently:

.. code-block:: python

   # Automatic layout optimization
   overlay = to_overlay_texture(rgba, extent)
   
   assert overlay['is_contiguous'] == True
   assert overlay['rgba'].flags.c_contiguous == True

**Memory monitoring**: Track memory usage during processing:

.. code-block:: python

   import psutil
   
   process = psutil.Process()
   mem_before = process.memory_info().rss / 1024**2
   
   # Process large dataset
   overlay = shade_to_overlay(large_agg, extent)
   
   mem_after = process.memory_info().rss / 1024**2
   print(f"Memory delta: {mem_after - mem_before:.1f} MB")

Memory Budget Guidelines
------------------------

For datasets approaching the memory budget:

**Tile-based processing**: Break large extents into smaller tiles:

.. code-block:: python

   def process_in_tiles(df, full_extent, tile_size=1024):
       """Process large dataset in memory-efficient tiles."""
       tiles = []
       
       # Calculate tile grid
       xmin, ymin, xmax, ymax = full_extent
       x_tiles = int(np.ceil((xmax - xmin) / tile_size))
       y_tiles = int(np.ceil((ymax - ymin) / tile_size))
       
       for i in range(x_tiles):
           for j in range(y_tiles):
               # Calculate tile extent
               tile_xmin = xmin + i * tile_size
               tile_ymin = ymin + j * tile_size
               tile_extent = (tile_xmin, tile_ymin, 
                            min(tile_xmin + tile_size, xmax),
                            min(tile_ymin + tile_size, ymax))
               
               # Filter points to tile
               tile_df = df[(df.x >= tile_extent[0]) & (df.x < tile_extent[2]) &
                           (df.y >= tile_extent[1]) & (df.y < tile_extent[3])]
               
               if len(tile_df) > 0:
                   overlay = process_single_tile(tile_df, tile_extent)
                   tiles.append(overlay)
       
       return tiles

**Dask integration**: For extremely large datasets, use dask-backed arrays:

.. code-block:: python

   import dask.dataframe as dd
   
   # Load large dataset with dask
   df = dd.read_parquet('massive_points.parquet')
   
   # Process in chunks that fit memory budget
   chunk_size = 100_000  # Adjust based on available memory
   for chunk in df.to_delayed():
       chunk_df = chunk.compute()
       overlay = shade_to_overlay(process_chunk(chunk_df), extent)
       # Process overlay...

Performance Considerations
==========================

The datashader integration is optimized for performance while respecting
memory constraints. Understanding the performance characteristics helps
optimize your visualization pipeline.

Zoom Level Performance
----------------------

Different zoom levels have distinct performance profiles:

.. list-table:: Typical Performance by Zoom Level
   :header-rows: 1
   :widths: 10 20 25 25 20

   * - Zoom
     - Extent Scale
     - Aggregation Time
     - Memory Usage
     - Frame Rate Target
   * - Z0
     - World (1:1)
     - Fast
     - Low
     - >30 FPS
   * - Z4
     - Continental (1:16)
     - Moderate
     - Medium
     - >20 FPS
   * - Z8
     - Regional (1:256)
     - Slower
     - Higher
     - >15 FPS
   * - Z12
     - City (1:4096)
     - Slowest
     - Highest
     - >10 FPS

**Optimization strategies**:

- **Pre-aggregate** common zoom levels during data preprocessing
- **Cache** aggregations for frequently accessed extents  
- **Downsample** point datasets for lower zoom levels
- **Progressive** loading for interactive applications

Performance Monitoring
----------------------

The integration provides built-in performance monitoring:

.. code-block:: python

   # Enable performance tracking
   adapter = DatashaderAdapter()
   
   # Process data
   result = render_with_adapter(adapter, dataset)
   
   # Check performance metrics
   print(f"Copies made: {adapter.copy_count}")
   print(f"Frame time: {result['performance']['total_time_s']:.3f}s")
   print(f"Points/sec: {result['performance']['points_per_second']:.0f}")

Coordinate Systems
==================

Proper coordinate alignment is critical for accurate overlay positioning.
The datashader integration provides tools to validate and ensure alignment.

Supported Coordinate Systems
----------------------------

The integration works with any projected coordinate system, including:

- **Geographic** (WGS84, EPSG:4326): longitude/latitude in degrees
- **Web Mercator** (EPSG:3857): meters, commonly used in web maps
- **UTM zones** (EPSG:326xx/327xx): meters, local accuracy
- **State Plane** (US): feet or meters, local projections
- **Custom projections**: any planar coordinate system

.. note::
   
   The integration assumes **planar coordinates**. Geographic coordinates
   should be projected before processing for accurate distance calculations.

Alignment Validation
--------------------

Use alignment validation to ensure pixel-perfect overlay positioning:

.. code-block:: python

   # Example: Web Mercator coordinates
   extent = (-8238310, 4969803, -8238210, 4969903)  # 100m x 100m area
   width, height = 1000, 1000  # 0.1m/pixel resolution
   
   # Validate alignment
   result = validate_alignment(extent, None, width, height)
   
   print(f"Pixel size: {result['pixel_width']:.3f}m x {result['pixel_height']:.3f}m")
   print(f"Alignment error: {result['pixel_error_x']:.6f} pixels")

For applications requiring sub-pixel accuracy, provide transform objects:

.. code-block:: python

   from rasterio.transform import from_bounds
   
   # Create precise transform
   transform = from_bounds(*extent, width, height)
   
   # Validate with transform
   result = validate_alignment(extent, transform, width, height)
   
   if not result['within_tolerance']:
       raise ValueError("Alignment exceeds 0.5 pixel tolerance")

Best Practices
==============

Follow these best practices for optimal performance and reliability:

Data Preparation
----------------

**Use appropriate data types**:

.. code-block:: python

   # Optimize DataFrame dtypes for memory and performance
   df = df.astype({
       'x': 'float32',      # Sufficient precision for most coordinates
       'y': 'float32', 
       'value': 'float32'   # Match your value range
   })

**Pre-filter large datasets**:

.. code-block:: python

   def filter_to_extent(df, extent, buffer=0.1):
       """Filter points to extent with small buffer."""
       xmin, ymin, xmax, ymax = extent
       return df[
           (df.x >= xmin - buffer) & (df.x <= xmax + buffer) &
           (df.y >= ymin - buffer) & (df.y <= ymax + buffer)
       ]

Canvas Configuration
--------------------

**Choose appropriate canvas sizes**:

.. code-block:: python

   # Balance resolution vs performance
   canvas_configs = {
       'preview': (400, 300),    # Fast preview
       'standard': (800, 600),   # Good balance  
       'high_res': (1600, 1200), # Detailed output
       'print': (3200, 2400)     # Print quality
   }

**Match canvas aspect ratio to extent**:

.. code-block:: python

   def calculate_canvas_size(extent, target_width):
       """Calculate height to match extent aspect ratio."""
       xmin, ymin, xmax, ymax = extent
       aspect_ratio = (ymax - ymin) / (xmax - xmin)
       height = int(target_width * aspect_ratio)
       return target_width, height
   
   width, height = calculate_canvas_size(extent, 800)

Error Handling
--------------

**Graceful degradation**:

.. code-block:: python

   from forge3d.adapters import is_datashader_available
   
   if not is_datashader_available():
       # Fallback to alternative visualization
       print("Datashader not available - using alternative renderer")
       return render_with_matplotlib(data)

**Validate inputs**:

.. code-block:: python

   def safe_datashader_render(df, extent):
       """Render with input validation and error handling."""
       
       # Validate DataFrame
       required_cols = ['x', 'y', 'value']
       if not all(col in df.columns for col in required_cols):
           raise ValueError(f"DataFrame must have columns: {required_cols}")
       
       # Validate extent
       xmin, ymin, xmax, ymax = extent
       if xmin >= xmax or ymin >= ymax:
           raise ValueError("Invalid extent: min values must be < max values")
       
       # Check for empty result
       filtered_df = filter_to_extent(df, extent)
       if len(filtered_df) == 0:
           raise ValueError("No data points in specified extent")
       
       return shade_to_overlay(create_aggregation(filtered_df), extent)

Troubleshooting
===============

Common Issues and Solutions
---------------------------

**"Array must be C-contiguous" error**:

.. code-block:: python

   # Fix non-contiguous arrays
   if not rgba.flags.c_contiguous:
       rgba = np.ascontiguousarray(rgba)

**Memory budget exceeded**:

.. code-block:: python

   # Monitor and reduce memory usage
   def check_memory_usage():
       import psutil
       process = psutil.Process()
       memory_mb = process.memory_info().rss / 1024**2
       if memory_mb > 400:  # Warning threshold
           print(f"Warning: Memory usage {memory_mb:.1f}MB approaching budget")
       return memory_mb

**Alignment validation failures**:

.. code-block:: python

   # Adjust extent or canvas size for better alignment
   def align_extent_to_pixels(extent, target_pixel_size):
       """Snap extent to pixel boundaries."""
       xmin, ymin, xmax, ymax = extent
       
       # Snap to pixel grid
       xmin = np.floor(xmin / target_pixel_size) * target_pixel_size
       ymin = np.floor(ymin / target_pixel_size) * target_pixel_size  
       xmax = np.ceil(xmax / target_pixel_size) * target_pixel_size
       ymax = np.ceil(ymax / target_pixel_size) * target_pixel_size
       
       return (xmin, ymin, xmax, ymax)

**Poor performance at high zoom levels**:

.. code-block:: python

   # Implement adaptive point density
   def adaptive_sampling(df, zoom_level, max_points=500_000):
       """Reduce point density at high zoom levels."""
       if zoom_level <= 8 or len(df) <= max_points:
           return df
       
       # Sample based on zoom level and point density
       sample_fraction = max_points / len(df)
       return df.sample(frac=sample_fraction, random_state=42)

Debugging Tools
---------------

**Enable verbose logging**:

.. code-block:: python

   import logging
   logging.basicConfig(level=logging.DEBUG)
   
   # Datashader operations will show detailed timing

**Performance profiling**:

.. code-block:: python

   import cProfile
   import pstats
   
   def profile_datashader_render(df, extent):
       """Profile datashader rendering pipeline."""
       profiler = cProfile.Profile()
       
       profiler.enable()
       result = shade_to_overlay(create_agg(df), extent)
       profiler.disable()
       
       stats = pstats.Stats(profiler)
       stats.sort_stats('cumulative')
       stats.print_stats(10)  # Top 10 functions
       
       return result

**Memory tracking**:

.. code-block:: python

   from memory_profiler import profile
   
   @profile
   def memory_tracked_render(df, extent):
       """Track memory usage line by line."""
       overlay = shade_to_overlay(create_agg(df), extent)
       return overlay

Examples
========

The forge3d repository includes a complete example demonstrating datashader
integration:

.. code-block:: bash

   # Run the datashader overlay demo
   python examples/datashader_overlay_demo.py --points 1000000

This example demonstrates:

- Synthetic point data generation
- Multi-scale aggregation 
- Zero-copy RGBA conversion
- Coordinate alignment validation
- Performance monitoring
- Memory budget compliance

See Also
========

- :doc:`../integration/matplotlib` - Matplotlib colormap integration
- :doc:`../gpu_memory_guide` - GPU memory management guide  
- :doc:`../examples_guide` - General examples and usage patterns
- `Datashader documentation <https://datashader.org/>`_ - Official datashader docs
- `Holoviews <https://holoviews.org/>`_ - High-level datashader integration

Benchmarks & Reproducing Performance
====================================

Workstream G2 provides a reproducible performance playbook. The repository
includes performance tests and utilities to generate goldens and collect
timing/memory metrics across zoom levels.

Generate Goldens
----------------

.. code-block:: bash

   # Generate golden reference images for SSIM comparisons
   pytest tests/perf/test_datashader_zoom.py::TestDatashaderZoomPerformance --maxfail=1 -q

Run Performance Tests
---------------------

.. code-block:: bash

   # Run the zoom-level performance tests (requires datashader + skimage)
   pytest tests/perf/test_datashader_zoom.py -q

Plot Simple Time/Memory Charts
------------------------------

You can visualize the collected metrics by lightly editing the test to emit
JSON or by calling the helper directly. A minimal plotting example:

.. code-block:: python

   import pandas as pd
   from pathlib import Path
   from forge3d.tests.perf.test_datashader_zoom import (
       generate_deterministic_dataset, render_zoom_level
   )

   df = generate_deterministic_dataset(200_000, seed=123)
   rows = []
   for z in [0, 4, 8, 12]:
       res = render_zoom_level(df, z)
       m = res['metrics']
       rows.append({
           'zoom': z,
           'frame_ms': m['total_frame_time_ms'],
           'mem_mb': m['memory_peak_mb'],
           'pps': m['points_per_second'],
       })
   pdf = pd.DataFrame(rows)
   print(pdf)
   ax = pdf.plot(x='zoom', y=['frame_ms', 'mem_mb'])
   ax.figure.savefig('datashader_perf_summary.png', dpi=150)

The above script writes a basic time/memory chart to
``datashader_perf_summary.png`` which can be included in reports.