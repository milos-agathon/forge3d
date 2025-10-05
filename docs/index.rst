forge3d Documentation
=====================

Welcome to the forge3d documentation. This is a high-performance Rust-first WebGPU/wgpu renderer with comprehensive Python API for 3D visualization, terrain rendering, and GPU-accelerated graphics.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started:

   quickstart
   installation
   examples_guide

.. toctree::
   :maxdepth: 2
   :caption: Core Documentation:

   api_reference
   api/io
   api/uv
   api/converters
   api/subdivision
   api/displacement
   api/curves
   color_management
   interop_zero_copy
   memory_budget
   gpu_memory_guide
   memory/index
   api/path_tracing

.. toctree::
   :maxdepth: 2
   :caption: Advanced Features:

   interactive_viewer
   postfx/index
   pbr_materials
   shadow_mapping
   user/path_tracing
   user/vector_picking_oit
   vector_graphics
   terrain_rendering
   async_operations
   misc_topics

.. toctree::
   :maxdepth: 1
   :caption: Basemaps & Tiles:

   tiles/xyz_wmts
   integration/cartopy

.. toctree::
   :maxdepth: 2
   :caption: Library Integrations:

   integration/matplotlib
   integration/external_images
   user/datashader_interop
   user/plot_py_adapters

.. toctree::
   :maxdepth: 1
   :caption: Examples:

   examples/f3_thick_polyline
   examples/f2_city_demo
   examples/f16_instancing
   examples/f18_gltf_import

.. toctree::
   :maxdepth: 1
   :caption: CI & Testing:

   ci/notebook_integration

Overview
--------

forge3d provides:

**Core Rendering**

* Headless deterministic rendering with PNG/NumPy interoperability
* Cross-platform support (Windows, Linux, macOS) with GPU acceleration
* WebGPU/WGSL shaders with Vulkan 1.2, DirectX 12, and Metal compatibility
* Memory-efficient operations with 512 MiB host-visible GPU budget

**Advanced Features**

* **Post-Processing Effects**: GPU compute-based pipeline with bloom, tone mapping, FXAA, and temporal anti-aliasing
* **Terrain Rendering**: Height field visualization with DEM processing, LOD, and colormaps
* **Vector Graphics**: Anti-aliased polygons, lines, and points with Order-Independent Transparency (OIT)
* **PBR Materials**: Physically-Based Rendering with metallic-roughness workflow and texture support
* **Shadow Mapping**: Cascaded Shadow Maps (CSM) with Percentage-Closer Filtering (PCF)
* **GPU Profiling**: Comprehensive timing markers and timestamp queries for performance analysis
* **Async Operations**: Double-buffered readback and multi-threaded command recording
* **Memory Management**: Advanced memory systems including staging rings, GPU memory pools, compressed textures, and virtual texture streaming

**Developer Experience**

* Comprehensive Python API with robust input validation and clear error messages
* 10+ advanced examples showcasing real-world usage patterns
* Complete Sphinx documentation with API reference
* Automated testing and CI/CD with multi-platform wheel distribution

Quick Start
-----------

Install forge3d::

    pip install forge3d

Basic rendering::

    import forge3d as f3d
    
    # Basic triangle rendering
    renderer = f3d.Renderer(512, 512)
    image = renderer.render_triangle_rgba()
    f3d.numpy_to_png("triangle.png", image)

Terrain visualization::

    import numpy as np
    import forge3d as f3d
    
    # Generate height data
    height_data = np.random.rand(128, 128).astype(np.float32)
    
    # Create terrain scene
    scene = f3d.Scene(800, 600)
    scene.set_height_data(height_data, spacing=10.0, exaggeration=50.0)
    terrain_image = scene.render_terrain_rgba()
    f3d.numpy_to_png("terrain.png", terrain_image)

PBR materials::

    import forge3d.pbr as pbr
    
    # Create metallic material
    material = pbr.PbrMaterial(
        base_color=(1.0, 0.86, 0.57, 1.0),  # Gold
        metallic=1.0,
        roughness=0.1
    )

Vector graphics::

    import forge3d as f3d
    import numpy as np
    
    # Clear and add vector elements
    f3d.clear_vectors_py()
    
    # Add colored points
    points = np.array([[[100, 100], [200, 200]]], dtype=np.float32)
    colors = np.array([[1.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    f3d.add_points_py(points, colors=colors, sizes=np.array([10.0]))

Memory management::

    import forge3d.memory as memory
    import forge3d.streaming as streaming
    
    # Initialize advanced memory systems
    memory.init_memory_system(
        staging_memory_mb=64,    # O1: Staging buffer rings
        pool_memory_mb=128,      # O2: GPU memory pools
        compressed_textures=True, # O3: Compressed texture support
        virtual_textures=True    # O4: Virtual texture streaming
    )
    
    # Create virtual texture system for large textures
    device = f3d.get_device()
    vt_system = streaming.VirtualTextureSystem(device, max_memory_mb=256)
    
    # Load and stream large texture
    texture = vt_system.load_texture("large_world_texture.ktx2")
    result = vt_system.update_streaming(camera_pos=(1000, 2000, 500))

For comprehensive examples, see the ``examples/`` directory including:

* ``terrain_single_tile.py`` - Basic terrain rendering
* ``advanced_terrain_shadows_pbr.py`` - Full-featured terrain with PBR and shadows
* ``device_capability_probe.py`` - GPU capability detection and analysis
* ``staging_rings_demo.py`` - O1 staging buffer ring operations
* ``memory_pools_demo.py`` - O2 GPU memory pool management
* ``compressed_texture_demo.py`` - O3 compressed texture pipeline
* ``virtual_texture_demo.py`` - O4 virtual texture streaming

API Reference
=============

**Core Modules**

* :doc:`forge3d <api_reference>` - Main rendering classes and utilities
* :doc:`forge3d.pbr <pbr_materials>` - PBR materials system
* :doc:`forge3d.shadows <shadow_mapping>` - Shadow mapping functionality
* :doc:`forge3d.materials <pbr_materials>` - Legacy compatibility shim
* :doc:`forge3d.memory <memory/staging_rings>` - Memory management systems (O1-O2)
* :doc:`forge3d.compressed <memory/compressed_textures>` - Compressed texture pipeline (O3)
* :doc:`forge3d.streaming <memory/virtual_texturing>` - Virtual texture streaming (O4)

**Submodule Import Pattern**

Advanced functionality requires explicit imports::

    # Core functionality (always available)
    import forge3d as f3d
    
    # Advanced features (explicit imports)
    import forge3d.pbr as pbr
    import forge3d.shadows as shadows
    import forge3d.materials as mat  # Legacy compatibility
    
    # Memory management systems (Workstream O)
    import forge3d.memory as memory        # O1-O2: Staging rings & memory pools
    import forge3d.compressed as compressed # O3: Compressed texture pipeline
    import forge3d.streaming as streaming   # O4: Virtual texture streaming

**Feature Detection**

::

    import forge3d as f3d
    
    # Check GPU availability
    if f3d.has_gpu():
        print("GPU acceleration available")
    
    # Check for advanced features
    try:
        import forge3d.shadows as shadows
        if shadows.has_shadows_support():
            print("Shadow mapping available")
    except ImportError:
        print("Shadow mapping not available")
    
    # Check for memory management features (Workstream O)
    try:
        import forge3d.memory as memory
        result = memory.init_memory_system()
        if result['success']:
            print("Advanced memory management available")
    except ImportError:
        print("Memory management systems not available")
    
    try:
        import forge3d.streaming as streaming
        if hasattr(f3d, 'create_virtual_texture_system'):
            print("Virtual texture streaming available")
    except ImportError:
        print("Virtual texture streaming not available")

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
