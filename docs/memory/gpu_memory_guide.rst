GPU Memory Management Guide
============================

This guide provides comprehensive information about GPU memory management in forge3d, including best practices, troubleshooting, and optimization strategies.

Overview
--------

forge3d implements a sophisticated memory management system to ensure reliable GPU operations across different hardware configurations. The system tracks memory usage, enforces budgets, and provides detailed metrics for optimization.

**Key Features:**

- **Automatic tracking** of all GPU allocations
- **512 MiB budget** for host-visible memory
- **Real-time monitoring** with detailed metrics  
- **Cross-platform compatibility** with different GPU vendors
- **Resource lifecycle management** with RAII cleanup

Memory Architecture
------------------

GPU Memory Types
~~~~~~~~~~~~~~~

**Device-Local Memory**
  High-performance memory on the GPU. Used for:
  
  - Vertex and index buffers
  - Textures and render targets
  - Compute shader storage
  - Internal GPU operations

**Host-Visible Memory**
  Memory accessible from both CPU and GPU. Used for:
  
  - Uniform buffers (shader parameters)
  - Readback buffers (GPU→CPU data transfer)
  - Upload staging buffers
  - Dynamic data that changes frequently

**Budget Constraints**

forge3d enforces a **512 MiB limit** on host-visible memory allocations:

.. code-block:: python

   from forge3d.core.memory_tracker import global_tracker
   
   # Check current usage
   metrics = global_tracker().get_metrics()
   print(f"Host-visible: {metrics.host_visible_bytes / (1024*1024):.1f} MiB / 512 MiB")
   print(f"Within budget: {metrics.within_budget}")

Memory Tracking System
---------------------

Automatic Tracking
~~~~~~~~~~~~~~~~~~

All GPU allocations are automatically tracked:

.. code-block:: python

   # These operations are tracked automatically
   renderer = forge3d.Renderer(1024, 1024)  # Tracks render target
   scene = forge3d.Scene(512, 512)          # Tracks terrain buffers
   rgba_data = renderer.render_triangle_rgba()  # Tracks readback buffer

Manual Tracking
~~~~~~~~~~~~~~~

For custom allocations, use the resource tracker:

.. code-block:: python

   from forge3d.core.resource_tracker import register_buffer, register_texture
   import wgpu
   
   # Track custom buffer
   buffer_handle = register_buffer(size=1024*1024, is_host_visible=True)
   
   # Track custom texture
   texture_handle = register_texture(
       width=512, 
       height=512, 
       format=wgpu.TextureFormat.Rgba8Unorm
   )
   
   # Resources are automatically freed when handles go out of scope

Detailed Metrics
~~~~~~~~~~~~~~~

Get comprehensive memory usage information:

.. code-block:: python

   metrics = global_tracker().get_metrics()
   
   print(f"Buffers: {metrics.buffer_count} ({metrics.buffer_bytes / 1024 / 1024:.1f} MiB)")
   print(f"Textures: {metrics.texture_count} ({metrics.texture_bytes / 1024 / 1024:.1f} MiB)")
   print(f"Host-visible: {metrics.host_visible_bytes / 1024 / 1024:.1f} MiB")
   print(f"Peak usage: {metrics.peak_bytes / 1024 / 1024:.1f} MiB")
   print(f"Budget status: {'✓' if metrics.within_budget else '✗'}")

Memory Budget Management
-----------------------

Understanding the 512 MiB Limit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The 512 MiB budget applies specifically to **host-visible** allocations:

**What counts toward the budget:**

- Uniform buffers (shader parameters)
- Readback buffers (GPU→CPU transfers)  
- Upload staging buffers
- Dynamic buffers with MAP_READ/MAP_WRITE usage

**What doesn't count:**

- Textures (stored in device-local memory)
- Vertex/index buffers (device-local)
- Render targets (device-local)
- Static GPU resources

**Why 512 MiB?**

This limit ensures compatibility with:

- Integrated GPUs (Intel, AMD APUs)
- Mobile GPUs with shared memory
- Older discrete GPUs with limited host-visible heaps
- Multi-process scenarios with memory sharing

Budget Validation
~~~~~~~~~~~~~~~~

Check allocations before they happen:

.. code-block:: python

   try:
       # Check if 100 MiB allocation would succeed
       global_tracker().check_budget(100 * 1024 * 1024)
       print("Allocation would succeed")
   except Exception as e:
       print(f"Budget would be exceeded: {e}")

Shadow Memory Validation
~~~~~~~~~~~~~~~~~~~~~~~

Shadow mapping has specific memory constraints:

.. code-block:: python

   from forge3d.shadows import get_preset_config, validate_shadow_memory_constraint
   
   # Check shadow configuration
   config = get_preset_config("high_quality")
   
   try:
       validate_shadow_memory_constraint(config)
       print(f"Shadow config OK: {config.calculate_memory_usage() / 1024 / 1024:.1f} MiB")
   except ValueError as e:
       print(f"Shadow config exceeds budget: {e}")

Texture Memory Accounting
-------------------------

Comprehensive Format Support
~~~~~~~~~~~~~~~~~~~~~~~~~~~

forge3d accurately calculates memory usage for all WebGPU texture formats:

**Uncompressed Formats:**

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Format
     - Bytes/Pixel
     - Use Case
   * - ``R8Unorm``
     - 1
     - Single-channel data (masks, height)
   * - ``Rg8Unorm``
     - 2
     - Two-channel data (normal maps XY)
   * - ``Rgba8Unorm``
     - 4
     - Standard color textures
   * - ``R16Float``
     - 2
     - High-precision single channel
   * - ``Rgba16Float``
     - 8
     - HDR color textures
   * - ``R32Float``
     - 4
     - Height fields, data textures
   * - ``Rgba32Float``
     - 16
     - Maximum precision color

**Compressed Formats:**

.. list-table::
   :header-rows: 1
   :widths: 30 20 20 30

   * - Format
     - Compression
     - Block Size
     - Use Case
   * - ``BC1`` (DXT1)
     - 4:1
     - 4×4 = 8 bytes
     - RGB + 1-bit alpha
   * - ``BC3`` (DXT5) 
     - 2:1
     - 4×4 = 16 bytes
     - RGBA textures
   * - ``BC5``
     - 2:1
     - 4×4 = 16 bytes
     - Normal maps
   * - ``BC6H``
     - 2:1
     - 4×4 = 16 bytes
     - HDR textures

**Memory Calculation Example:**

.. code-block:: python

   # 1024×1024 RGBA8 texture
   uncompressed = 1024 * 1024 * 4  # 4 MiB
   
   # 1024×1024 BC3 compressed texture
   blocks_x = (1024 + 3) // 4      # 256 blocks
   blocks_y = (1024 + 3) // 4      # 256 blocks
   compressed = blocks_x * blocks_y * 16  # 1 MiB (4:1 compression)

Optimization Strategies
----------------------

Texture Optimization
~~~~~~~~~~~~~~~~~~~

**Choose Appropriate Formats:**

.. code-block:: python

   # Instead of RGBA8 for single-channel data
   height_data = np.random.random((512, 512)).astype(np.float32)
   # Use R32Float: 512×512×4 = 1 MiB instead of 4 MiB

   # Instead of RGB32Float for color
   color_data = np.random.random((512, 512, 3)).astype(np.float16) 
   # Use RGB16Float: 512×512×6 = 1.5 MiB instead of 3 MiB

**Use Compression When Possible:**

.. code-block:: python

   # For textures that support compression
   # BC3: 512×512 RGBA → 128×128×16 bytes = 256 KiB instead of 1 MiB
   # 4:1 compression ratio

Buffer Optimization
~~~~~~~~~~~~~~~~~

**Reuse Buffers:**

.. code-block:: python

   class OptimizedRenderer:
       def __init__(self, width, height):
           # Create renderer once
           self.renderer = forge3d.Renderer(width, height)
           self._readback_buffer = None
           
       def render_many(self, count):
           results = []
           for i in range(count):
               # Reuses internal buffers automatically
               rgba = self.renderer.render_triangle_rgba()
               results.append(rgba)
           return results

**Batch Operations:**

.. code-block:: python

   # Instead of multiple small operations
   for i in range(100):
       small_result = render_small_triangle(i)
   
   # Batch into fewer larger operations  
   batch_results = render_triangle_batch(range(100))

Memory Pooling
~~~~~~~~~~~~~

Use the async readback system for automatic buffer pooling:

.. code-block:: python

   from forge3d.async_readback import AsyncReadbackConfig, AsyncRenderer
   
   # Configure buffer pooling
   config = AsyncReadbackConfig(
       pre_allocate=True,    # Pre-allocate buffers
       max_pending_ops=4     # Pool size
   )
   
   renderer = AsyncRenderer(512, 512, config)

Platform-Specific Considerations
-------------------------------

NVIDIA GPUs
~~~~~~~~~~

**Characteristics:**
- Large host-visible heaps (1-2 GiB typical)
- Efficient unified memory architecture
- Good performance with frequent readbacks

**Optimization:**
- Can use higher concurrent operations
- Async readback particularly beneficial
- Less concern about host-visible budget

AMD GPUs
~~~~~~~~

**Characteristics:**  
- Moderate host-visible heaps (256-512 MiB typical)
- May use system RAM for host-visible allocations
- Good compression support

**Optimization:**
- Stay closer to 512 MiB budget
- Use compressed textures aggressively  
- Prefer device-local allocations

Intel GPUs (Integrated)
~~~~~~~~~~~~~~~~~~~~~~

**Characteristics:**
- Shared system memory architecture
- Limited host-visible budget (128-256 MiB)
- Variable performance characteristics

**Optimization:**
- Conservative memory usage essential
- Avoid large readback operations
- Use lower quality settings by default

Mobile GPUs
~~~~~~~~~~

**Characteristics:**
- Tiled rendering architectures  
- Very limited memory budgets
- Power/thermal constraints

**Optimization:**
- Minimize memory footprint
- Use lower resolution textures
- Avoid frequent readbacks

Troubleshooting
--------------

Common Memory Issues
~~~~~~~~~~~~~~~~~~

**Out of Memory Errors**

.. code-block:: python

   # Symptom: Allocation failures
   try:
       large_texture = create_8k_texture()
   except MemoryError as e:
       print(f"Memory allocation failed: {e}")
       
       # Solution: Check available memory
       metrics = global_tracker().get_metrics()
       available = 512 * 1024 * 1024 - metrics.host_visible_bytes
       print(f"Available budget: {available / 1024 / 1024:.1f} MiB")
       
       # Use smaller texture or cleanup existing allocations
       cleanup_unused_resources()
       smaller_texture = create_4k_texture()

**Budget Exceeded Warnings**

.. code-block:: python

   metrics = global_tracker().get_metrics()
   if not metrics.within_budget:
       print("WARNING: Memory budget exceeded!")
       print(f"Usage: {metrics.host_visible_bytes / 1024 / 1024:.1f} MiB")
       
       # Find largest allocations
       print(f"Buffers: {metrics.buffer_bytes / 1024 / 1024:.1f} MiB")
       print(f"Textures: {metrics.texture_bytes / 1024 / 1024:.1f} MiB")
       
       # Consider:
       # 1. Reducing texture sizes
       # 2. Using compression
       # 3. Freeing unused resources
       # 4. Using async operations with smaller buffers

**Memory Leaks**

.. code-block:: python

   # Monitor memory over time
   import time
   
   baseline = global_tracker().get_metrics().total_bytes
   
   for i in range(100):
       do_rendering_operation()
       
       if i % 10 == 0:
           current = global_tracker().get_metrics().total_bytes
           growth = current - baseline
           print(f"Iteration {i}: +{growth / 1024:.1f} KiB")
           
           if growth > 10 * 1024 * 1024:  # 10 MiB growth
               print("WARNING: Possible memory leak detected!")
               break

Performance Profiling
~~~~~~~~~~~~~~~~~~~~

**Memory Usage Profiling:**

.. code-block:: python

   def profile_memory_usage(func, *args, **kwargs):
       # Get baseline
       start_metrics = global_tracker().get_metrics()
       
       # Run function
       import time
       start_time = time.time()
       result = func(*args, **kwargs)
       end_time = time.time()
       
       # Get final metrics  
       end_metrics = global_tracker().get_metrics()
       
       # Calculate differences
       buffer_growth = end_metrics.buffer_bytes - start_metrics.buffer_bytes
       texture_growth = end_metrics.texture_bytes - start_metrics.texture_bytes
       duration = end_time - start_time
       
       print(f"Function: {func.__name__}")
       print(f"Duration: {duration:.3f}s")
       print(f"Buffer growth: {buffer_growth / 1024:.1f} KiB")
       print(f"Texture growth: {texture_growth / 1024:.1f} KiB") 
       print(f"Peak memory: {end_metrics.peak_bytes / 1024 / 1024:.1f} MiB")
       
       return result
   
   # Usage
   result = profile_memory_usage(renderer.render_triangle_rgba)

**Memory Benchmark:**

.. code-block:: python

   def benchmark_memory_patterns():
       """Compare different memory usage patterns"""
       
       # Pattern 1: Create/destroy each time
       start_time = time.time()
       for i in range(10):
           r = forge3d.Renderer(512, 512)
           rgba = r.render_triangle_rgba()
           del r  # Explicit cleanup
       pattern1_time = time.time() - start_time
       
       # Pattern 2: Reuse renderer
       start_time = time.time()  
       r = forge3d.Renderer(512, 512)
       for i in range(10):
           rgba = r.render_triangle_rgba()
       pattern2_time = time.time() - start_time
       
       print(f"Create/destroy: {pattern1_time:.3f}s")
       print(f"Reuse renderer: {pattern2_time:.3f}s")
       print(f"Speedup: {pattern1_time / pattern2_time:.1f}x")

Best Practices
--------------

Memory-Efficient Coding
~~~~~~~~~~~~~~~~~~~~~~

**1. Resource Lifecycle Management**

.. code-block:: python

   # Good: Use context managers or RAII
   class TerrainRenderer:
       def __init__(self, width, height):
           self.scene = forge3d.Scene(width, height)
           self._height_data = None
           
       def load_terrain(self, heights):
           # Cleanup previous data
           if self._height_data is not None:
               del self._height_data
           
           self._height_data = heights
           self.scene.set_height_data(heights)
           
       def __del__(self):
           # Automatic cleanup
           if hasattr(self, 'scene'):
               del self.scene

**2. Lazy Allocation**

.. code-block:: python

   class LazyRenderer:
       def __init__(self, width, height):
           self.width = width
           self.height = height
           self._renderer = None
           
       @property
       def renderer(self):
           # Create only when needed
           if self._renderer is None:
               self._renderer = forge3d.Renderer(self.width, self.height)
           return self._renderer
           
       def render(self):
           return self.renderer.render_triangle_rgba()

**3. Batch Processing**

.. code-block:: python

   def process_heightmaps_efficiently(heightmap_list):
       # Create renderer once
       scene = forge3d.Scene(512, 512)
       results = []
       
       for heightmap in heightmap_list:
           # Reuse scene, just update data
           scene.set_height_data(heightmap)
           result = scene.render_terrain_rgba()
           results.append(result)
           
       return results

**4. Memory Budget Awareness**

.. code-block:: python

   def adaptive_quality_rendering(width, height, target_quality='high'):
       """Adapt quality based on available memory"""
       
       metrics = global_tracker().get_metrics()
       available_mb = (512 * 1024 * 1024 - metrics.host_visible_bytes) / 1024 / 1024
       
       if available_mb < 50:
           # Low memory: reduce quality
           quality = 'low'
           actual_width = width // 2  
           actual_height = height // 2
       elif available_mb < 200:
           # Medium memory: medium quality
           quality = 'medium'
           actual_width = width
           actual_height = height
       else:
           # High memory: full quality
           quality = target_quality
           actual_width = width
           actual_height = height
           
       print(f"Using {quality} quality: {actual_width}×{actual_height}")
       print(f"Available memory: {available_mb:.1f} MiB")
       
       return forge3d.Renderer(actual_width, actual_height)

Monitoring and Alerting
~~~~~~~~~~~~~~~~~~~~~~

**Set up memory monitoring:**

.. code-block:: python

   import warnings
   
   def check_memory_health():
       """Check memory usage and issue warnings"""
       metrics = global_tracker().get_metrics()
       
       usage_pct = (metrics.host_visible_bytes / (512 * 1024 * 1024)) * 100
       
       if usage_pct > 90:
           warnings.warn(f"Memory usage critical: {usage_pct:.1f}%")
       elif usage_pct > 75:  
           warnings.warn(f"Memory usage high: {usage_pct:.1f}%")
       
       return metrics
   
   # Call periodically
   metrics = check_memory_health()

Configuration Guidelines
-----------------------

Development vs Production
~~~~~~~~~~~~~~~~~~~~~~~

**Development Configuration:**

.. code-block:: python

   # Enable detailed logging
   import os
   os.environ['RUST_LOG'] = 'DEBUG'
   
   # Use smaller textures for faster iteration
   DEV_WIDTH = 256
   DEV_HEIGHT = 256
   
   # Enable memory tracking
   def dev_render():
       metrics_before = global_tracker().get_metrics()
       result = renderer.render_triangle_rgba()
       metrics_after = global_tracker().get_metrics()
       
       growth = metrics_after.total_bytes - metrics_before.total_bytes
       print(f"Memory growth: {growth / 1024:.1f} KiB")
       
       return result

**Production Configuration:**

.. code-block:: python

   # Minimal logging
   os.environ['RUST_LOG'] = 'WARN'
   
   # Use appropriate quality settings
   PROD_WIDTH = 1024
   PROD_HEIGHT = 1024
   
   # Monitor for memory issues
   def prod_render():
       try:
           return renderer.render_triangle_rgba()
       except MemoryError as e:
           # Log error and attempt recovery
           logging.error(f"Memory allocation failed: {e}")
           
           # Try with smaller size
           fallback_renderer = forge3d.Renderer(512, 512)
           return fallback_renderer.render_triangle_rgba()

Future Improvements
------------------

Planned Features
~~~~~~~~~~~~~~

**Dynamic Budget Adjustment**
  Automatically adjust memory budget based on available system memory and GPU capabilities.

**Memory Compression**
  Transparent compression of infrequently accessed GPU resources.

**Multi-GPU Support**  
  Load balancing and memory management across multiple GPUs.

**Advanced Profiling**
  Integration with platform-specific GPU profiling tools (NVIDIA Nsight, AMD Radeon GPU Profiler).

**Automatic Optimization**
  Machine learning-based memory usage optimization and quality adaptation.

Contributing
-----------

When contributing to forge3d's memory management system:

1. **Test on Multiple Platforms**: Ensure changes work on NVIDIA, AMD, and Intel GPUs
2. **Validate Memory Accounting**: Verify that new allocations are properly tracked  
3. **Respect Budget Constraints**: Don't break the 512 MiB host-visible limit
4. **Add Memory Tests**: Include tests that verify memory usage patterns
5. **Update Documentation**: Keep this guide current with any changes

**Example Test:**

.. code-block:: python

   def test_memory_budget_compliance():
       """Ensure operations stay within memory budget"""
       
       # Get baseline
       baseline = global_tracker().get_metrics()
       
       # Perform memory-intensive operation
       renderer = forge3d.Renderer(2048, 2048)
       for i in range(10):
           result = renderer.render_triangle_rgba()
       
       # Check final state
       final = global_tracker().get_metrics()
       
       # Verify budget compliance
       assert final.within_budget, f"Budget exceeded: {final.host_visible_bytes / 1024 / 1024:.1f} MiB"
       
       # Verify cleanup
       del renderer
       post_cleanup = global_tracker().get_metrics()
       assert post_cleanup.total_bytes <= baseline.total_bytes + 1024  # Allow small overhead

This comprehensive memory management system ensures reliable GPU operations across different hardware configurations while providing the tools needed for optimization and troubleshooting.