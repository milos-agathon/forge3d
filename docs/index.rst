forge3d Documentation
====================

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
   color_management
   interop_zero_copy
   memory_budget
   gpu_memory_management

.. toctree::
   :maxdepth: 2
   :caption: Advanced Features:

   pbr_materials
   shadow_mapping
   vector_graphics
   terrain_rendering
   async_operations

Overview
--------

forge3d provides:

**Core Rendering**

* Headless deterministic rendering with PNG/NumPy interoperability
* Cross-platform support (Windows, Linux, macOS) with GPU acceleration
* WebGPU/WGSL shaders with Vulkan 1.2, DirectX 12, and Metal compatibility
* Memory-efficient operations with 512 MiB host-visible GPU budget

**Advanced Features**

* **Terrain Rendering**: Height field visualization with DEM processing, LOD, and colormaps
* **Vector Graphics**: Anti-aliased polygons, lines, and points with Order-Independent Transparency (OIT)
* **PBR Materials**: Physically-Based Rendering with metallic-roughness workflow and texture support
* **Shadow Mapping**: Cascaded Shadow Maps (CSM) with Percentage-Closer Filtering (PCF)
* **Async Operations**: Double-buffered readback and multi-threaded command recording

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

For comprehensive examples, see the ``examples/`` directory including:

* ``terrain_single_tile.py`` - Basic terrain rendering
* ``advanced_terrain_shadows_pbr.py`` - Full-featured terrain with PBR and shadows
* ``device_capability_probe.py`` - GPU capability detection and analysis

API Reference
=============

**Core Modules**

* :doc:`forge3d <api_reference>` - Main rendering classes and utilities
* :doc:`forge3d.pbr <pbr_materials>` - PBR materials system
* :doc:`forge3d.shadows <shadow_mapping>` - Shadow mapping functionality
* :doc:`forge3d.materials <pbr_materials>` - Legacy compatibility shim

**Submodule Import Pattern**

Advanced functionality requires explicit imports::

    # Core functionality (always available)
    import forge3d as f3d
    
    # Advanced features (explicit imports)
    import forge3d.pbr as pbr
    import forge3d.shadows as shadows
    import forge3d.materials as mat  # Legacy compatibility

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

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`