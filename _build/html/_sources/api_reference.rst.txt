API Reference
=============

This section provides complete API reference documentation for forge3d.

Core Classes
------------

Renderer
~~~~~~~~

.. autoclass:: forge3d.Renderer
   :members:
   :undoc-members:
   :show-inheritance:

The main rendering class that provides headless GPU rendering capabilities.

**Example Usage:**

.. code-block:: python

   import forge3d as f3d
   
   # Create a renderer
   renderer = f3d.Renderer(512, 512)
   
   # Render a triangle
   rgba_data = renderer.render_triangle_rgba()
   
   # Save as PNG
   renderer.render_triangle_png("output.png")

Scene
~~~~~

.. autoclass:: forge3d.Scene
   :members:
   :undoc-members:
   :show-inheritance:

High-level scene management for complex rendering scenarios.

**Example Usage:**

.. code-block:: python

   import numpy as np
   
   # Create height data
   heights = np.random.random((256, 256)).astype(np.float32)
   
   # Create scene
   scene = f3d.Scene(512, 512)
   scene.set_height_data(heights, spacing=10.0, exaggeration=2.0)
   
   # Render terrain
   terrain_image = scene.render_terrain_rgba()

Vector Graphics
---------------

Polygons
~~~~~~~~

.. autofunction:: forge3d.add_polygons

Add polygon geometries to the vector layer.

**Parameters:**

- **coords** (*array_like*): Polygon coordinates as [N, M, 2] array
- **colors** (*array_like*, optional): Colors as [N, 4] RGBA array
- **stroke_widths** (*array_like*, optional): Stroke widths as [N] array

**Example:**

.. code-block:: python

   # Create triangle polygon
   triangle = np.array([[[0, 0], [100, 0], [50, 100]]])
   colors = np.array([[1.0, 0.0, 0.0, 1.0]])  # Red
   
   f3d.add_polygons(triangle, colors=colors)

Lines
~~~~~

.. autofunction:: forge3d.add_lines

Add line geometries with optional styling.

**Parameters:**

- **coords** (*array_like*): Line coordinates as [N, M, 2] array
- **colors** (*array_like*, optional): Colors as [N, 4] RGBA array  
- **widths** (*array_like*, optional): Line widths as [N] array
- **caps** (*str*, optional): Line cap style ('round', 'square', 'butt')
- **joins** (*str*, optional): Line join style ('round', 'bevel', 'miter')

Points
~~~~~~

.. autofunction:: forge3d.add_points

Add point geometries with customizable sprites.

**Parameters:**

- **coords** (*array_like*): Point coordinates as [N, 2] array
- **sizes** (*array_like*, optional): Point sizes in pixels as [N] array
- **colors** (*array_like*, optional): Colors as [N, 4] RGBA array
- **shapes** (*str*, optional): Point shape ('circle', 'square', 'diamond')

Terrain Rendering
-----------------

Height Data Processing
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: forge3d.make_grid

Generate coordinate grids for terrain rendering.

**Returns:**
- **x_coords** (*ndarray*): X coordinate array
- **y_coords** (*ndarray*): Y coordinate array  
- **indices** (*ndarray*): Triangle indices for meshes

.. autofunction:: forge3d.add_terrain

Add terrain from height field data.

**Parameters:**

- **heights** (*array_like*): Height values as [H, W] array
- **spacing** (*float*): Grid spacing in world units
- **exaggeration** (*float*): Height scaling factor
- **colormap** (*str*): Color mapping ('viridis', 'magma', 'terrain')

Memory Management
-----------------

GPU Memory Tracking
~~~~~~~~~~~~~~~~~~~

.. autoclass:: forge3d.core.memory_tracker.ResourceRegistry
   :members:
   :undoc-members:

Resource tracking for GPU memory management.

**Example:**

.. code-block:: python

   from forge3d.core.memory_tracker import global_tracker
   
   # Get current memory usage
   metrics = global_tracker().get_metrics()
   print(f"Buffer memory: {metrics.buffer_bytes / 1024 / 1024:.1f} MiB")
   print(f"Texture memory: {metrics.texture_bytes / 1024 / 1024:.1f} MiB")
   print(f"Within budget: {metrics.within_budget}")

Budget Management
~~~~~~~~~~~~~~~~~

The system enforces a 512 MiB budget for host-visible GPU memory:

.. code-block:: python

   # Check if allocation would exceed budget
   try:
       global_tracker().check_budget(100 * 1024 * 1024)  # 100 MiB
       print("Allocation within budget")
   except Exception as e:
       print(f"Budget exceeded: {e}")

Advanced Features
-----------------

Shadow Mapping
~~~~~~~~~~~~~~

.. automodule:: forge3d.shadows
   :members:
   :undoc-members:

Cascaded shadow mapping with configurable quality presets.

**Available Presets:**

- **low_quality**: 512×512 shadow maps, 2 cascades (2 MiB)
- **medium_quality**: 1024×1024 shadow maps, 3 cascades (12 MiB)  
- **high_quality**: 2048×2048 shadow maps, 4 cascades (64 MiB)
- **ultra_quality**: 4096×4096 shadow maps, 4 cascades (256 MiB)

**Example:**

.. code-block:: python

   from forge3d.shadows import get_preset_config
   
   # Get shadow configuration
   config = get_preset_config("medium_quality")
   print(f"Shadow map size: {config.shadow_map_size}")
   print(f"Cascade count: {config.cascade_count}")
   
   # Check memory usage
   memory_mb = config.calculate_memory_usage() / (1024 * 1024)
   print(f"Memory usage: {memory_mb:.1f} MiB")

PBR Materials
~~~~~~~~~~~~~

.. automodule:: forge3d.core.material
   :members:
   :undoc-members:

Physically-based rendering materials following the metallic-roughness workflow.

**Material Presets:**

.. code-block:: python

   from forge3d.core.material import presets
   
   # Create material presets
   gold_mat = presets.gold()
   silver_mat = presets.silver()
   plastic_mat = presets.plastic([0.8, 0.2, 0.2])  # Red plastic
   glass_mat = presets.glass([0.9, 0.9, 1.0])      # Blue-tinted glass

Async Readback
~~~~~~~~~~~~~~

.. automodule:: forge3d.async_readback
   :members:
   :undoc-members:

Asynchronous texture readback for improved performance.

**Example:**

.. code-block:: python

   import asyncio
   from forge3d.async_readback import AsyncRenderer
   
   async def render_multiple():
       renderer = AsyncRenderer(512, 512)
       
       # Start multiple async operations
       handles = []
       for i in range(5):
           handle = await renderer.render_async()
           handles.append(handle)
       
       # Wait for all to complete
       results = await asyncio.gather(*[h.wait() for h in handles])
       return results
   
   # Run async rendering
   results = asyncio.run(render_multiple())

Utilities
---------

Device Information
~~~~~~~~~~~~~~~~~~

.. autofunction:: forge3d.device_diagnostics

Get information about available GPU devices.

**Returns:**
- Device adapter information
- Feature support details
- Memory limits and capabilities

Environment Reporting
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: forge3d.report_environment

Generate environment report for debugging.

**Returns:**
- System information
- GPU details
- Driver versions
- forge3d configuration

Color Management
~~~~~~~~~~~~~~~~

.. automodule:: forge3d.colormap
   :members:
   :undoc-members:

Color mapping and palette utilities.

**Available Colormaps:**

- ``viridis`` - Blue to yellow gradient
- ``magma`` - Black to white through purple/pink
- ``terrain`` - Blue (water) to green (land) to brown (mountains)
- ``plasma`` - Purple to pink to yellow
- ``inferno`` - Black to yellow through red

Configuration
-------------

Environment Variables
~~~~~~~~~~~~~~~~~~~~

forge3d behavior can be controlled through environment variables:

**WGPU_BACKENDS**
  Comma-separated list of preferred GPU backends.
  
  - ``VULKAN`` - Vulkan API (preferred on Linux/Windows)
  - ``METAL`` - Metal API (macOS only)
  - ``DX12`` - DirectX 12 (Windows only) 
  - ``GL`` - OpenGL (fallback)

  Example: ``WGPU_BACKENDS=VULKAN,DX12``

**VF_ENABLE_TERRAIN_TESTS**
  Enable terrain-specific test suite.
  
  Set to ``1`` to enable terrain tests that may require additional GPU memory.

**RUST_LOG**  
  Control logging level for debugging.
  
  - ``ERROR`` - Only errors
  - ``WARN`` - Warnings and errors
  - ``INFO`` - General information
  - ``DEBUG`` - Detailed debugging
  - ``TRACE`` - Maximum verbosity

GPU Backend Selection
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import os
   
   # Force Vulkan backend
   os.environ['WGPU_BACKENDS'] = 'VULKAN'
   
   # Enable debug logging
   os.environ['RUST_LOG'] = 'INFO'
   
   import forge3d as f3d

Error Handling
--------------

Common Exceptions
~~~~~~~~~~~~~~~~

**RenderError**
  Base exception for rendering errors.

**DeviceError**
  GPU device initialization or operation errors.

**ValidationError**
  Input validation errors (invalid parameters, formats, etc.).

**MemoryError**
  GPU memory allocation or budget exceeded errors.

**Example Error Handling:**

.. code-block:: python

   try:
       renderer = f3d.Renderer(8192, 8192)  # Very large texture
       result = renderer.render_triangle_rgba()
   except f3d.MemoryError as e:
       print(f"GPU memory error: {e}")
       # Fall back to smaller size
       renderer = f3d.Renderer(2048, 2048)
       result = renderer.render_triangle_rgba()
   except f3d.DeviceError as e:
       print(f"GPU device error: {e}")
       # Try software fallback
       os.environ['WGPU_BACKENDS'] = 'GL'
       renderer = f3d.Renderer(2048, 2048)
       result = renderer.render_triangle_rgba()

Debugging Tips
~~~~~~~~~~~~~~

1. **Enable Logging:** Set ``RUST_LOG=INFO`` for detailed operation logs
2. **Check Memory Usage:** Use memory tracker to monitor GPU allocations
3. **Validate Input:** Ensure arrays are contiguous and correct dtype
4. **Backend Issues:** Try different GPU backends if rendering fails  
5. **Size Limits:** Start with smaller textures and scale up as needed

Performance Tips
~~~~~~~~~~~~~~~

1. **Reuse Renderers:** Create renderer once, call render methods multiple times
2. **Batch Operations:** Group multiple render calls together
3. **Memory Budget:** Stay within 512 MiB host-visible memory limit
4. **Async Readback:** Use async operations for better throughput
5. **Appropriate Formats:** Choose efficient texture formats (compressed when possible)

Type Annotations
----------------

forge3d provides comprehensive type hints for better IDE support:

.. code-block:: python

   from typing import Tuple, Optional
   import numpy as np
   from numpy.typing import NDArray
   
   # Type-annotated function
   def process_heightmap(
       heights: NDArray[np.float32],
       spacing: float = 1.0,
       exaggeration: float = 1.0
   ) -> NDArray[np.uint8]:
       scene = f3d.Scene(512, 512)
       scene.set_height_data(heights, spacing, exaggeration) 
       return scene.render_terrain_rgba()

For more examples and detailed guides, see the :doc:`examples` section.