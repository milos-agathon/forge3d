Terrain Rendering
=================

forge3d provides comprehensive terrain visualization capabilities with height field processing, Digital Elevation Model (DEM) support, Level-of-Detail (LOD), and colormap-based visualization.

Quick Start
-----------

Basic terrain rendering:

.. code-block:: python

    import numpy as np
    import forge3d as f3d
    
    # Generate height data
    size = 128
    x = np.linspace(-2, 2, size)
    y = np.linspace(-2, 2, size)
    X, Y = np.meshgrid(x, y)
    
    # Create hills using sine waves
    height_data = np.sin(X * 2) * np.cos(Y * 1.5)
    height_data = height_data.astype(np.float32)
    
    # Create terrain scene
    scene = f3d.Scene(800, 600)
    scene.set_height_data(height_data, spacing=10.0, exaggeration=20.0)
    
    # Render terrain
    image = scene.render_terrain_rgba()
    f3d.numpy_to_png("terrain.png", image)

Height Data Processing
----------------------

**Data Format Requirements**

Height data must be 2D arrays with specific requirements:

.. code-block:: python

    # Shape: (height, width) - row-major ordering
    height_data = np.random.rand(256, 256).astype(np.float32)
    
    # Data type: float32 (preferred) or float64
    height_data = height_data.astype(np.float32)
    
    # Array must be C-contiguous
    if not height_data.flags.c_contiguous:
        height_data = np.ascontiguousarray(height_data)

**DEM Utilities**

Process Digital Elevation Models:

.. code-block:: python

    # Get statistics
    min_elev, max_elev, mean_elev, std_elev = f3d.dem_stats(height_data)
    print(f"Elevation range: {min_elev:.2f} to {max_elev:.2f}")
    print(f"Mean elevation: {mean_elev:.2f} ± {std_elev:.2f}")
    
    # Normalize elevation data
    normalized = f3d.dem_normalize(
        height_data,
        vmin=None,                    # Auto-detect minimum
        vmax=None,                    # Auto-detect maximum  
        out_range=(0.0, 1.0),         # Target range
        dtype=np.float32              # Output type
    )
    
    # Custom normalization range
    scaled = f3d.dem_normalize(
        height_data,
        vmin=0.0,                     # Clip minimum
        vmax=1000.0,                  # Clip maximum
        out_range=(-50.0, 50.0),      # Custom range
        dtype=np.float32
    )

**Procedural Generation**

Create synthetic terrain:

.. code-block:: python

    def generate_mountainous_terrain(width: int, height: int) -> np.ndarray:
        """Generate realistic mountainous terrain."""
        x = np.linspace(-3, 3, width)
        y = np.linspace(-3, 3, height)
        X, Y = np.meshgrid(x, y)
        
        # Large-scale mountains
        terrain = 0.8 * np.exp(-(X**2 + Y**2) / 2)
        
        # Medium-scale hills  
        terrain += 0.4 * np.sin(X * 2 + 1) * np.cos(Y * 1.5 + 0.5)
        
        # Fine-scale detail
        terrain += 0.2 * np.sin(X * 8) * np.cos(Y * 6)
        
        # Add noise for realism
        terrain += 0.1 * np.random.random((height, width))
        
        return terrain.astype(np.float32)

Scene Configuration
-------------------

**Camera Setup**

Configure camera for optimal terrain viewing:

.. code-block:: python

    scene = f3d.Scene(800, 600)
    scene.set_height_data(height_data, spacing=5.0, exaggeration=15.0)
    
    # Set camera position and target
    terrain_size = height_data.shape[0]
    scene.set_camera(
        position=(terrain_size * 4, terrain_size * 1, terrain_size * 4),  # Eye position
        target=(terrain_size * 2, 0.0, terrain_size * 2),                 # Look at center
        up=(0.0, 1.0, 0.0)                                                # Up vector
    )

**Terrain Parameters**

Control terrain appearance:

.. code-block:: python

    scene.set_height_data(
        height_data,
        spacing=1.0,         # Distance between height samples (world units)
        exaggeration=10.0    # Vertical scale multiplier
    )
    
    # Large terrain with detailed features
    scene.set_height_data(terrain_large, spacing=25.0, exaggeration=5.0)
    
    # Small terrain with dramatic relief
    scene.set_height_data(terrain_small, spacing=1.0, exaggeration=50.0)

Colormap System
---------------

**Built-in Colormaps**

Use predefined color schemes:

.. code-block:: python

    # List available palettes
    palettes = f3d.list_palettes()
    for palette in palettes:
        print(f"{palette['index']}: {palette['name']} - {palette['description']}")
    
    # Set active colormap
    f3d.set_palette('viridis')    # Perceptually uniform
    f3d.set_palette('magma')      # Purple to yellow
    f3d.set_palette('terrain')    # Geographic colors
    
    # Get current palette
    current = f3d.get_current_palette()
    print(f"Active palette: {current['name']}")

**Colormap Switching**

Change colormaps dynamically:

.. code-block:: python

    # Render with different colormaps
    colormaps = ['viridis', 'magma', 'terrain']
    
    for cmap in colormaps:
        f3d.set_palette(cmap)
        image = scene.render_terrain_rgba()
        f3d.numpy_to_png(f"terrain_{cmap}.png", image)
    
    # Check colormap support
    if f3d.colormap_supported('custom_map'):
        f3d.set_palette('custom_map')

Level of Detail (LOD)
---------------------

**Automatic LOD**

The terrain system automatically applies LOD based on distance and screen-space size:

.. code-block:: python

    # LOD is automatically applied during rendering
    # Closer areas get higher detail, distant areas get lower detail
    
    # Control LOD behavior with camera distance
    # Closer camera = more detail visible
    scene.set_camera(
        position=(terrain_size * 2, terrain_size * 0.5, terrain_size * 2),  # Close view
        target=(terrain_size * 1, 0.0, terrain_size * 1),
        up=(0.0, 1.0, 0.0)
    )

**Performance Considerations**

Balance quality vs. performance:

.. code-block:: python

    # For performance-critical applications:
    # - Use smaller terrain grids (128x128 vs 512x512)  
    # - Reduce exaggeration to minimize overdraw
    # - Position camera for optimal LOD utilization
    
    # High performance setup
    scene.set_height_data(small_terrain, spacing=10.0, exaggeration=3.0)
    
    # High quality setup (slower)
    scene.set_height_data(large_terrain, spacing=2.0, exaggeration=20.0)

Terrain Analysis
----------------

**Surface Properties**

Analyze terrain characteristics:

.. code-block:: python

    # Calculate slopes (requires custom implementation)
    def calculate_slopes(height_data: np.ndarray, spacing: float) -> np.ndarray:
        """Calculate slope in degrees from height data."""
        dy, dx = np.gradient(height_data)
        slope_radians = np.arctan(np.sqrt(dx**2 + dy**2) / spacing)
        slope_degrees = np.degrees(slope_radians)
        return slope_degrees.astype(np.float32)
    
    # Calculate aspects (compass direction of steepest descent)
    def calculate_aspects(height_data: np.ndarray) -> np.ndarray:
        """Calculate aspect in degrees from height data."""
        dy, dx = np.gradient(height_data)
        aspect_radians = np.arctan2(-dx, dy)  # Note: -dx for correct orientation
        aspect_degrees = np.degrees(aspect_radians) % 360
        return aspect_degrees.astype(np.float32)
    
    # Apply analysis
    slopes = calculate_slopes(height_data, spacing=5.0)
    aspects = calculate_aspects(height_data)
    
    print(f"Slope range: {slopes.min():.1f}° to {slopes.max():.1f}°")
    print(f"Mean slope: {slopes.mean():.1f}°")

**Terrain Statistics**

.. code-block:: python

    # Comprehensive terrain analysis
    def analyze_terrain(height_data: np.ndarray, spacing: float) -> dict:
        """Comprehensive terrain analysis."""
        min_elev, max_elev, mean_elev, std_elev = f3d.dem_stats(height_data)
        
        slopes = calculate_slopes(height_data, spacing)
        aspects = calculate_aspects(height_data)
        
        # Calculate roughness (standard deviation of slopes in local neighborhood)
        from scipy.ndimage import uniform_filter
        slope_mean = uniform_filter(slopes, size=3)
        slope_sq_mean = uniform_filter(slopes**2, size=3)
        roughness = np.sqrt(np.maximum(0, slope_sq_mean - slope_mean**2))
        
        return {
            'elevation': {
                'min': min_elev, 'max': max_elev,
                'mean': mean_elev, 'std': std_elev,
                'range': max_elev - min_elev
            },
            'slopes': {
                'min': slopes.min(), 'max': slopes.max(),
                'mean': slopes.mean(), 'std': slopes.std()
            },
            'roughness': {
                'min': roughness.min(), 'max': roughness.max(),
                'mean': roughness.mean()
            },
            'grid_info': {
                'size': height_data.shape,
                'spacing': spacing,
                'area': height_data.shape[0] * height_data.shape[1] * spacing**2
            }
        }

Advanced Features
-----------------

**Multi-Scale Terrain**

Render terrain at different scales:

.. code-block:: python

    # Regional view (large area, low detail per unit)
    scene.set_height_data(height_data, spacing=100.0, exaggeration=2.0)
    regional_view = scene.render_terrain_rgba()
    
    # Local view (small area, high detail per unit)
    scene.set_height_data(height_data, spacing=1.0, exaggeration=20.0)
    local_view = scene.render_terrain_rgba()

**Terrain Blending**

Combine multiple height sources:

.. code-block:: python

    # Blend terrain sources
    def blend_terrains(terrain1: np.ndarray, terrain2: np.ndarray, alpha: float) -> np.ndarray:
        """Blend two terrain sources with specified weight."""
        return ((1 - alpha) * terrain1 + alpha * terrain2).astype(np.float32)
    
    # Combine base terrain with detail layer
    base_terrain = generate_mountainous_terrain(256, 256)
    detail_terrain = np.random.rand(256, 256).astype(np.float32) * 0.1
    
    combined = blend_terrains(base_terrain, detail_terrain, alpha=0.2)
    scene.set_height_data(combined, spacing=5.0, exaggeration=15.0)

**Terrain Tiling**

Handle large terrain datasets:

.. code-block:: python

    # Create terrain tiles for large datasets
    def create_terrain_tile(full_terrain: np.ndarray, 
                           tile_x: int, tile_y: int, 
                           tile_size: int) -> np.ndarray:
        """Extract tile from larger terrain."""
        x_start = tile_x * tile_size
        x_end = x_start + tile_size
        y_start = tile_y * tile_size  
        y_end = y_start + tile_size
        
        # Handle edge cases
        x_end = min(x_end, full_terrain.shape[1])
        y_end = min(y_end, full_terrain.shape[0])
        
        return full_terrain[y_start:y_end, x_start:x_end].copy()
    
    # Render specific tile
    large_terrain = np.random.rand(1024, 1024).astype(np.float32)
    tile = create_terrain_tile(large_terrain, tile_x=2, tile_y=1, tile_size=256)
    
    scene.set_height_data(tile, spacing=2.0, exaggeration=10.0)

Integration with Other Systems
------------------------------

**Terrain + Vector Graphics**

Combine terrain with vector overlays:

.. code-block:: python

    # Render terrain
    scene.set_height_data(height_data, spacing=5.0, exaggeration=15.0)
    
    # Add vector annotations
    f3d.clear_vectors_py()
    
    # Mark peaks and valleys
    peak_locations = np.array([[[200, 150], [400, 350]]], dtype=np.float32)
    peak_colors = np.array([[1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    peak_sizes = np.array([8.0, 8.0], dtype=np.float32)
    
    f3d.add_points_py(peak_locations, colors=peak_colors, sizes=peak_sizes)
    
    # Render combined result
    combined_image = scene.render_terrain_rgba()

**Terrain + Shadows**

Add realistic lighting:

.. code-block:: python

    try:
        import forge3d.shadows as shadows
        
        if shadows.has_shadows_support():
            # Configure shadows for terrain
            shadow_config = shadows.get_preset_config('medium_quality')
            sun_light = shadows.DirectionalLight(
                direction=(-0.3, -0.8, -0.5),  # Afternoon sun
                color=(1.0, 0.9, 0.7),         # Warm light
                intensity=2.5
            )
            
            # Render with shadows
            renderer = shadows.ShadowRenderer(800, 600, shadow_config)
            renderer.set_light(sun_light)
            
            scene_data = {'terrain': height_data}
            shadowed_terrain = renderer.render_with_shadows(scene_data)
        else:
            # Fallback to standard rendering
            shadowed_terrain = scene.render_terrain_rgba()
            
    except ImportError:
        # Shadows not available
        shadowed_terrain = scene.render_terrain_rgba()

Performance Optimization
------------------------

**Memory Management**

Optimize terrain memory usage:

.. code-block:: python

    # Choose appropriate terrain size for your needs
    # Large terrain: high memory usage, high detail
    large_terrain = np.random.rand(512, 512).astype(np.float32)  # ~1 MiB
    
    # Medium terrain: balanced memory/detail
    medium_terrain = np.random.rand(256, 256).astype(np.float32)  # ~256 KiB
    
    # Small terrain: low memory, good for testing
    small_terrain = np.random.rand(128, 128).astype(np.float32)   # ~64 KiB
    
    # Monitor memory usage
    print(f"Terrain memory: {terrain.nbytes / 1024:.1f} KiB")

**Rendering Performance**

Optimize rendering speed:

.. code-block:: python

    # Performance tips:
    # 1. Use appropriate exaggeration (avoid extreme values)
    scene.set_height_data(terrain, spacing=5.0, exaggeration=10.0)  # Good
    # scene.set_height_data(terrain, spacing=5.0, exaggeration=100.0)  # May cause issues
    
    # 2. Position camera efficiently (avoid extreme angles)
    scene.set_camera(
        position=(terrain_size * 1.5, terrain_size * 0.5, terrain_size * 1.5),
        target=(terrain_size * 0.5, 0.0, terrain_size * 0.5),
        up=(0.0, 1.0, 0.0)
    )
    
    # 3. Use smaller render targets for preview
    preview_scene = f3d.Scene(400, 300)  # Half resolution
    preview_scene.set_height_data(terrain, spacing=5.0, exaggeration=10.0)

Real-World Applications
-----------------------

**Geographic Visualization**

.. code-block:: python

    # Visualize real DEM data
    def visualize_dem_file(dem_file: str, output_file: str):
        """Visualize DEM data from file."""
        # Load DEM (implementation depends on format: GeoTIFF, etc.)
        # height_data = load_dem(dem_file)  # Custom loader required
        
        # Normalize for display
        height_norm = f3d.dem_normalize(height_data, out_range=(0.0, 1.0))
        
        # Create visualization
        scene = f3d.Scene(1200, 900)  # High resolution
        scene.set_height_data(height_norm, spacing=30.0, exaggeration=2.0)  # 30m resolution
        
        # Use terrain colormap
        f3d.set_palette('terrain')
        
        # Render and save
        image = scene.render_terrain_rgba()
        f3d.numpy_to_png(output_file, image)

**Scientific Visualization**

.. code-block:: python

    # Analyze and visualize surface processes
    def visualize_surface_analysis(height_data: np.ndarray):
        """Create multi-panel terrain analysis."""
        
        # Calculate derived products
        slopes = calculate_slopes(height_data, spacing=1.0)
        aspects = calculate_aspects(height_data)
        
        # Render elevation
        f3d.set_palette('terrain')
        scene = f3d.Scene(400, 300)
        scene.set_height_data(height_data, spacing=1.0, exaggeration=10.0)
        elevation_img = scene.render_terrain_rgba()
        
        # Visualize slopes
        f3d.set_palette('viridis')
        scene.set_height_data(slopes / slopes.max(), spacing=1.0, exaggeration=10.0)
        slope_img = scene.render_terrain_rgba()
        
        # Save results
        f3d.numpy_to_png("elevation.png", elevation_img)
        f3d.numpy_to_png("slopes.png", slope_img)

Troubleshooting
---------------

**Common Issues**

1. **Memory Errors with Large Terrain**
   
   .. code-block:: python
   
       # Problem: Terrain too large
       huge_terrain = np.random.rand(2048, 2048).astype(np.float32)  # ~16 MiB
       
       # Solution: Reduce size or use tiling
       reasonable_terrain = huge_terrain[::4, ::4]  # Downsample by 4x

2. **Poor Visual Quality**

   .. code-block:: python
   
       # Problem: Insufficient exaggeration
       scene.set_height_data(terrain, spacing=1.0, exaggeration=1.0)  # Too flat
       
       # Solution: Increase exaggeration
       scene.set_height_data(terrain, spacing=1.0, exaggeration=15.0)

3. **Array Format Errors**

   .. code-block:: python
   
       # Problem: Wrong data type or shape
       height_data = np.random.randint(0, 100, (128, 128))  # Integer data
       
       # Solution: Convert to float32
       height_data = height_data.astype(np.float32)

**Performance Issues**

- Use smaller terrain sizes for testing
- Reduce exaggeration if rendering is slow
- Check GPU memory usage with large terrains
- Consider LOD when rendering large datasets

**Visual Artifacts**

- Ensure height data is properly normalized
- Check for NaN or infinite values in data
- Verify camera positioning and field of view
- Use appropriate spacing and exaggeration values

Example Applications
--------------------

See these comprehensive examples:

- ``examples/terrain_single_tile.py`` - Basic terrain rendering
- ``examples/advanced_terrain_shadows_pbr.py`` - Full-featured terrain with lighting
- ``examples/contour_overlay_demo.py`` - Terrain with topographic overlays
- ``examples/normal_mapping_terrain.py`` - Advanced surface detail techniques