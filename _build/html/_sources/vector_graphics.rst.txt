Vector Graphics
===============

forge3d provides a comprehensive vector graphics system for adding 2D overlays to 3D rendered scenes. The system supports anti-aliased polygons, lines, and points with Order-Independent Transparency (OIT).

Quick Start
-----------

Basic vector graphics usage:

.. code-block:: python

    import forge3d as f3d
    import numpy as np
    
    # Clear any existing vectors
    f3d.clear_vectors_py()
    
    # Add colored points
    points = np.array([[[100, 100], [200, 200]]], dtype=np.float32)
    colors = np.array([[1.0, 0.0, 0.0, 1.0]], dtype=np.float32)  # Red
    sizes = np.array([10.0], dtype=np.float32)
    
    f3d.add_points_py(points, colors=colors, sizes=sizes)
    
    # Render scene with vector overlay
    renderer = f3d.Renderer(400, 300)
    image = renderer.render_triangle_rgba()  # Includes vector overlay
    f3d.numpy_to_png("vectors.png", image)

Vector Types
------------

**Points**

Render circular points with variable sizes:

.. code-block:: python

    # Point coordinates: shape (batch, points, 2)
    points = np.array([
        [[50, 50], [100, 100], [150, 150]],  # First batch
        [[200, 50], [250, 100], [300, 150]], # Second batch
    ], dtype=np.float32)
    
    # Colors: shape (total_points, 4) RGBA
    colors = np.array([
        [1.0, 0.0, 0.0, 1.0],  # Red
        [0.0, 1.0, 0.0, 1.0],  # Green
        [0.0, 0.0, 1.0, 1.0],  # Blue
        [1.0, 1.0, 0.0, 1.0],  # Yellow
        [1.0, 0.0, 1.0, 1.0],  # Magenta
        [0.0, 1.0, 1.0, 1.0],  # Cyan
    ], dtype=np.float32)
    
    # Sizes: shape (total_points,)
    sizes = np.array([5, 8, 12, 6, 10, 15], dtype=np.float32)
    
    f3d.add_points_py(points, colors=colors, sizes=sizes)

**Lines**

Render anti-aliased lines with customizable width:

.. code-block:: python

    # Line coordinates: shape (batch, vertices, 2)
    lines = np.array([
        [[0, 0], [100, 50], [200, 0]],      # Curved line
        [[50, 100], [150, 150], [250, 100]] # Another curved line
    ], dtype=np.float32)
    
    # Colors: shape (batch, 4) RGBA per line
    line_colors = np.array([
        [1.0, 0.5, 0.0, 0.8],  # Orange, semi-transparent
        [0.0, 0.5, 1.0, 0.8],  # Blue, semi-transparent
    ], dtype=np.float32)
    
    # Widths: shape (batch,)
    widths = np.array([3.0, 5.0], dtype=np.float32)
    
    f3d.add_lines_py(lines, colors=line_colors, widths=widths)

**Polygons**

Render filled polygons with optional outlines:

.. code-block:: python

    # Polygon coordinates: shape (batch, vertices, 2)
    polygons = np.array([
        # Triangle
        [[100, 100], [150, 50], [200, 100], [100, 100]],  # Closed polygon
        # Rectangle  
        [[250, 50], [350, 50], [350, 150], [250, 150], [250, 50]],
    ], dtype=np.float32)
    
    # Fill colors: shape (batch, 4) RGBA
    fill_colors = np.array([
        [0.8, 0.2, 0.2, 0.6],  # Red, semi-transparent
        [0.2, 0.8, 0.2, 0.6],  # Green, semi-transparent
    ], dtype=np.float32)
    
    f3d.add_polygons_py(polygons, colors=fill_colors)

Advanced Features
-----------------

**Order-Independent Transparency (OIT)**

Vector graphics support proper transparency blending:

.. code-block:: python

    # Multiple overlapping transparent elements
    f3d.clear_vectors_py()
    
    # Background polygon
    bg_poly = np.array([[[50, 50], [250, 50], [250, 250], [50, 250], [50, 50]]], dtype=np.float32)
    bg_color = np.array([[0.0, 0.0, 1.0, 0.3]], dtype=np.float32)  # Blue, very transparent
    f3d.add_polygons_py(bg_poly, colors=bg_color)
    
    # Overlapping circles (as points)
    circles = np.array([[[100, 100], [150, 150], [200, 100]]], dtype=np.float32)
    circle_colors = np.array([
        [1.0, 0.0, 0.0, 0.5],  # Red, 50% transparent
        [0.0, 1.0, 0.0, 0.5],  # Green, 50% transparent
        [1.0, 1.0, 0.0, 0.5],  # Yellow, 50% transparent
    ], dtype=np.float32)
    circle_sizes = np.array([30, 30, 30], dtype=np.float32)
    f3d.add_points_py(circles, colors=circle_colors, sizes=circle_sizes)

**Vector State Management**

Control vector graphics state:

.. code-block:: python

    # Check current vector counts
    counts = f3d.get_vector_counts_py()
    print(f"Points: {counts.get('points', 0)}")
    print(f"Lines: {counts.get('lines', 0)}")
    print(f"Polygons: {counts.get('polygons', 0)}")
    
    # Clear specific vector types or all
    f3d.clear_vectors_py()  # Clears all vectors
    
    # Add vectors incrementally
    f3d.add_points_py(points1, colors=colors1, sizes=sizes1)
    f3d.add_points_py(points2, colors=colors2, sizes=sizes2)  # Accumulates

Data Format Requirements
------------------------

**Array Shapes**

Vector graphics functions have specific shape requirements:

.. code-block:: python

    # Points: (batch_count, points_per_batch, 2)
    points_shape = (3, 5, 2)  # 3 batches of 5 points each
    
    # Lines: (batch_count, vertices_per_line, 2) 
    lines_shape = (2, 10, 2)  # 2 lines with 10 vertices each
    
    # Polygons: (batch_count, vertices_per_polygon, 2)
    polygons_shape = (1, 6, 2)  # 1 hexagon with 6 vertices

**Data Types**

All coordinate and parameter arrays must use specific dtypes:

.. code-block:: python

    # Coordinates: float32
    coords = np.array([[[0, 0], [100, 100]]], dtype=np.float32)
    
    # Colors: float32, RGBA in range [0, 1]
    colors = np.array([[1.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    
    # Sizes/Widths: float32
    sizes = np.array([10.0], dtype=np.float32)

**Array Contiguity**

Arrays must be C-contiguous:

.. code-block:: python

    # Ensure contiguity if needed
    if not coords.flags.c_contiguous:
        coords = np.ascontiguousarray(coords)

Performance Considerations
--------------------------

**Batching**

Group similar elements for better performance:

.. code-block:: python

    # Good: Single batch with many points
    many_points = np.random.rand(1, 1000, 2).astype(np.float32) * 800
    many_colors = np.random.rand(1000, 4).astype(np.float32)
    many_sizes = np.random.rand(1000).astype(np.float32) * 10 + 2
    
    f3d.add_points_py(many_points, colors=many_colors, sizes=many_sizes)

**Memory Management**

Be mindful of vector graphics memory usage:

.. code-block:: python

    # Clear vectors when done
    f3d.clear_vectors_py()
    
    # Check vector counts periodically
    counts = f3d.get_vector_counts_py()
    total_elements = sum(counts.values())
    if total_elements > 10000:
        print("⚠ Large number of vector elements may impact performance")

**Coordinate Systems**

Vector graphics use screen-space coordinates:

.. code-block:: python

    # Renderer size determines coordinate range
    renderer = f3d.Renderer(800, 600)
    
    # Valid coordinates: x=[0, 800), y=[0, 600)
    # Origin (0,0) is top-left corner
    # +X is right, +Y is down

Integration with 3D Rendering
------------------------------

**Overlay Rendering**

Vector graphics are rendered as overlays on 3D content:

.. code-block:: python

    import forge3d as f3d
    import numpy as np
    
    # Create 3D scene
    scene = f3d.Scene(800, 600)
    terrain = np.random.rand(128, 128).astype(np.float32)
    scene.set_height_data(terrain, spacing=5.0, exaggeration=20.0)
    
    # Add vector annotations
    f3d.clear_vectors_py()
    
    # Mark important locations
    markers = np.array([[[100, 100], [400, 300], [700, 200]]], dtype=np.float32)
    marker_colors = np.array([
        [1.0, 0.0, 0.0, 1.0],  # Red marker
        [0.0, 1.0, 0.0, 1.0],  # Green marker
        [0.0, 0.0, 1.0, 1.0],  # Blue marker
    ], dtype=np.float32)
    marker_sizes = np.array([8.0, 8.0, 8.0], dtype=np.float32)
    
    f3d.add_points_py(markers, colors=marker_colors, sizes=marker_sizes)
    
    # Add connecting lines
    connections = np.array([
        [[100, 100], [400, 300]],  # Line 1
        [[400, 300], [700, 200]],  # Line 2
    ], dtype=np.float32)
    line_colors = np.array([
        [1.0, 1.0, 1.0, 0.8],  # White, semi-transparent
        [1.0, 1.0, 1.0, 0.8],
    ], dtype=np.float32)
    line_widths = np.array([2.0, 2.0], dtype=np.float32)
    
    f3d.add_lines_py(connections, colors=line_colors, widths=line_widths)
    
    # Render terrain with vector overlay
    final_image = scene.render_terrain_rgba()
    f3d.numpy_to_png("terrain_with_vectors.png", final_image)

Use Cases
---------

**Geographic Annotations**

.. code-block:: python

    # Mark cities on a terrain map
    cities = np.array([[[200, 150], [400, 300], [600, 250]]], dtype=np.float32)
    city_colors = np.array([[1.0, 1.0, 0.0, 1.0]] * 3, dtype=np.float32)  # Yellow
    city_sizes = np.array([6.0, 8.0, 5.0], dtype=np.float32)  # Size by population
    
    f3d.add_points_py(cities, colors=city_colors, sizes=city_sizes)

**Data Visualization**

.. code-block:: python

    # Plot data points on rendered background  
    data_points = np.random.rand(1, 50, 2).astype(np.float32) * [800, 600]
    values = np.random.rand(50)  # Data values [0, 1]
    
    # Color by value (red to blue)
    data_colors = np.zeros((50, 4), dtype=np.float32)
    data_colors[:, 0] = 1.0 - values  # Red component
    data_colors[:, 2] = values        # Blue component  
    data_colors[:, 3] = 0.8           # Alpha
    
    data_sizes = values * 15 + 3      # Size by value
    
    f3d.add_points_py(data_points, colors=data_colors, sizes=data_sizes)

**UI Elements**

.. code-block:: python

    # Add simple UI overlay
    # Scale bar
    scale_line = np.array([[[50, 550], [150, 550]]], dtype=np.float32)
    scale_color = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)  # Black
    scale_width = np.array([2.0], dtype=np.float32)
    
    f3d.add_lines_py(scale_line, colors=scale_color, widths=scale_width)
    
    # Scale markers
    scale_marks = np.array([[[50, 545], [50, 555], [150, 545], [150, 555]]], dtype=np.float32)
    mark_colors = np.array([[0.0, 0.0, 0.0, 1.0]] * 4, dtype=np.float32)
    mark_sizes = np.array([2.0] * 4, dtype=np.float32)
    
    f3d.add_points_py(scale_marks, colors=mark_colors, sizes=mark_sizes)

Troubleshooting
---------------

**Common Issues**

1. **Array Shape Errors**
   
   .. code-block:: python
   
       # Wrong: Points not in batches
       points = np.array([[100, 100], [200, 200]], dtype=np.float32)  # Shape: (2, 2)
       
       # Correct: Points in batches  
       points = np.array([[[100, 100], [200, 200]]], dtype=np.float32)  # Shape: (1, 2, 2)

2. **Color Format Errors**

   .. code-block:: python
   
       # Wrong: RGB colors
       colors = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)  # Missing alpha
       
       # Correct: RGBA colors
       colors = np.array([[1.0, 0.0, 0.0, 1.0]], dtype=np.float32)

3. **Data Type Errors**

   .. code-block:: python
   
       # Wrong: Integer coordinates
       coords = np.array([[[100, 100]]], dtype=int)
       
       # Correct: Float32 coordinates  
       coords = np.array([[[100, 100]]], dtype=np.float32)

**Performance Issues**

- Clear vectors regularly: ``f3d.clear_vectors_py()``
- Use appropriate batch sizes (100-1000 elements per batch)
- Avoid excessive transparency (alpha < 0.1) which can impact performance
- Monitor vector counts with ``f3d.get_vector_counts_py()``

**Visual Issues**

- Ensure colors are in [0, 1] range
- Check coordinate ranges match renderer size  
- Use appropriate sizes for points and line widths
- Consider z-ordering for complex overlays

Advanced Examples
-----------------

See these examples for comprehensive vector graphics usage:

- ``examples/vector_oit_layering.py`` - Order-Independent Transparency demonstration
- ``examples/contour_overlay_demo.py`` - Vector overlays on terrain
- ``examples/multithreaded_command_recording.py`` - Performance optimization techniques