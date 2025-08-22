forge3d Documentation
====================

Welcome to the forge3d documentation. This is a Rust-first WebGPU/wgpu renderer with a thin Python API for 3D visualization.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   color_management

Overview
--------

forge3d provides:

* Headless deterministic rendering
* Terrain visualization with height mapping
* Cross-platform support (Windows, Linux, macOS)
* WebGPU/WGSL shaders with Vulkan 1.2 compatibility

Getting Started
--------------

Install forge3d::

    pip install forge3d

Basic usage::

    import forge3d as f3d
    
    # Create a renderer
    r = f3d.Renderer(512, 512)
    
    # Render a triangle
    r.render_triangle_png("triangle.png")

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`