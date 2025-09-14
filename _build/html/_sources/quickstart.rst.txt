Quick Start Guide
=================

This guide will get you up and running with forge3d in under 10 minutes.

Installation
------------

Install forge3d from PyPI:

.. code-block:: bash

    pip install forge3d

Or install from source:

.. code-block:: bash

    git clone https://github.com/anthropics/forge3d.git
    cd forge3d
    pip install maturin[patchelf]
    maturin develop --release

Basic Triangle Rendering
-------------------------

Your first forge3d program:

.. code-block:: python

    import forge3d as f3d
    
    # Create a renderer
    renderer = f3d.Renderer(512, 512)
    
    # Render a triangle to RGBA array
    image = renderer.render_triangle_rgba()
    
    # Save as PNG
    f3d.numpy_to_png("my_first_triangle.png", image)
    
    print("Triangle saved to my_first_triangle.png")

Terrain Visualization
---------------------

Create a simple terrain visualization:

.. code-block:: python

    import numpy as np
    import forge3d as f3d
    
    # Generate synthetic height data
    size = 128
    x = np.linspace(-2, 2, size)
    y = np.linspace(-2, 2, size)
    X, Y = np.meshgrid(x, y)
    
    # Create hills and valleys
    height_data = (
        np.sin(X * 2) * np.cos(Y * 1.5) +
        0.5 * np.sin(X * 4) * np.cos(Y * 3) +
        0.1 * np.random.random((size, size))
    )
    
    # Normalize to [0, 1]
    height_data = (height_data - height_data.min()) / (height_data.max() - height_data.min())
    height_data = height_data.astype(np.float32)
    
    # Create terrain scene
    scene = f3d.Scene(800, 600)
    scene.set_height_data(height_data, spacing=10.0, exaggeration=50.0)
    
    # Set camera for good view
    scene.set_camera(
        position=(size * 8, size * 2, size * 8),
        target=(size * 5, 0, size * 5),
        up=(0.0, 1.0, 0.0)
    )
    
    # Render terrain
    terrain_image = scene.render_terrain_rgba()
    f3d.numpy_to_png("terrain.png", terrain_image)
    
    print("Terrain saved to terrain.png")

Vector Graphics
---------------

Add vector overlays to your renders:

.. code-block:: python

    import numpy as np
    import forge3d as f3d
    
    # Create renderer
    renderer = f3d.Renderer(400, 300)
    
    # Clear any existing vectors
    f3d.clear_vectors_py()
    
    # Add colored points
    points = np.array([
        [[100, 100], [150, 120], [200, 200]],  # First batch
    ], dtype=np.float32)
    colors = np.array([
        [1.0, 0.0, 0.0, 1.0],  # Red
        [0.0, 1.0, 0.0, 1.0],  # Green  
        [0.0, 0.0, 1.0, 1.0],  # Blue
    ], dtype=np.float32)
    sizes = np.array([8.0, 10.0, 12.0], dtype=np.float32)
    
    f3d.add_points_py(points, colors=colors, sizes=sizes)
    
    # Add lines
    lines = np.array([
        [[50, 50], [100, 100], [150, 50]],  # Triangle
    ], dtype=np.float32)
    line_colors = np.array([[0.5, 0.5, 0.5, 1.0]], dtype=np.float32)
    line_widths = np.array([3.0], dtype=np.float32)
    
    f3d.add_lines_py(lines, colors=line_colors, widths=line_widths)
    
    # Render with vectors
    image = renderer.render_triangle_rgba()
    f3d.numpy_to_png("vectors.png", image)
    
    print("Vector graphics saved to vectors.png")

Device Detection
----------------

Check GPU capabilities:

.. code-block:: python

    import forge3d as f3d
    
    # Check if GPU is available
    if f3d.has_gpu():
        print("✓ GPU acceleration available")
        
        # Get adapter information
        adapters = f3d.enumerate_adapters()
        print(f"Available adapters: {len(adapters)}")
        for i, adapter in enumerate(adapters):
            print(f"  {i}: {adapter}")
    else:
        print("⚠ No GPU acceleration available")
    
    # Run device diagnostics
    try:
        diagnostics = f3d.device_probe()
        print("Device diagnostics:", diagnostics)
    except Exception as e:
        print("Device probe failed:", e)

What's Next?
------------

* :doc:`examples_guide` - Explore comprehensive examples
* :doc:`pbr_materials` - Learn about PBR materials
* :doc:`shadow_mapping` - Add realistic shadows  
* :doc:`api_reference` - Complete API documentation

**Example Scripts**

Check out the ``examples/`` directory for complete working examples:

* ``triangle_png.py`` - Basic triangle rendering
* ``terrain_single_tile.py`` - Terrain with colormaps
* ``advanced_terrain_shadows_pbr.py`` - Full-featured terrain rendering
* ``device_capability_probe.py`` - Comprehensive GPU analysis

**Getting Help**

* Check the :doc:`api_reference` for detailed function documentation
* Review the examples in the ``examples/`` directory
* Consult the :doc:`memory_budget` guide for performance optimization