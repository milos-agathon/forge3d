Shadow Mapping
==============

forge3d provides high-quality Cascaded Shadow Maps (CSM) with Percentage-Closer Filtering (PCF) for realistic directional lighting and shadows.

.. note::
   This functionality is available via the ``forge3d.shadows`` module. Import it explicitly for access to shadow mapping features.

Quick Start
-----------

Basic shadow mapping setup:

.. code-block:: python

    import forge3d.shadows as shadows
    
    # Check if shadows are supported
    if shadows.has_shadows_support():
        # Get a high-quality shadow configuration
        config = shadows.get_preset_config('high_quality')
        print(f"Shadow config: {config}")
    else:
        print("Shadow mapping not available")

Memory Management
-----------------

**Critical Memory Constraint**

All shadow configurations are validated against a **256 MiB** host-visible GPU memory budget:

.. code-block:: python

    # Memory calculation formula:
    # memory = cascade_count × shadow_map_size² × 4 bytes
    
    cascade_count = 4
    shadow_map_size = 2048
    
    memory_bytes = shadows.calculate_shadow_atlas_memory(cascade_count, shadow_map_size)
    print(f"Atlas memory: {memory_bytes / (1024*1024):.1f} MiB")
    
    # Check if configuration fits within budget
    MAX_MEMORY = 256 * 1024 * 1024  # 256 MiB
    if memory_bytes <= MAX_MEMORY:
        print("✓ Configuration fits within memory budget")
    else:
        print("✗ Configuration exceeds memory budget")

**Memory Validation**

The system automatically validates all configurations:

.. code-block:: python

    try:
        # This will raise an error if memory budget is exceeded
        config = shadows.CsmConfig(
            cascade_count=8,        # Too many cascades
            shadow_map_size=4096    # Too large
        )
    except ValueError as e:
        print(f"Configuration rejected: {e}")

Shadow Configuration
--------------------

**Preset Configurations**

Use predefined quality presets:

.. code-block:: python

    # Available presets
    presets = ['low_quality', 'medium_quality', 'high_quality', 'ultra_quality']
    
    for preset in presets:
        try:
            config = shadows.get_preset_config(preset)
            memory_mb = shadows.calculate_shadow_atlas_memory(
                config.cascade_count, config.shadow_map_size
            ) / (1024 * 1024)
            
            print(f"{preset}:")
            print(f"  Cascades: {config.cascade_count}")
            print(f"  Size: {config.shadow_map_size}x{config.shadow_map_size}")
            print(f"  Memory: {memory_mb:.1f} MiB")
            
        except Exception as e:
            print(f"{preset}: Not available - {e}")

**Custom Configuration**

Create custom shadow configurations:

.. code-block:: python

    from forge3d.shadows import CsmConfig
    
    # Create custom configuration (memory-validated)
    custom_config = CsmConfig(
        cascade_count=3,                    # Number of shadow cascades
        shadow_map_size=1024,               # Size of each shadow map
        cascade_split_lambda=0.8,           # Cascade distribution
        max_shadow_distance=100.0,          # Maximum shadow range
        bias=0.001,                         # Depth bias for shadow acne
        normal_bias=0.01,                   # Normal-based bias
        pcf_radius=2,                       # PCF filter radius
        enable_pcf=True                     # Enable soft shadows
    )

Directional Lighting
--------------------

Configure directional lights (e.g., sun):

.. code-block:: python

    # Create directional light
    sun_light = shadows.DirectionalLight(
        direction=(-0.4, -0.7, -0.6),      # Light direction (toward light)
        color=(1.0, 0.95, 0.8),            # Warm sunlight color
        intensity=3.2,                      # Light intensity
        cast_shadows=True                   # Enable shadow casting
    )

Shadow Rendering
----------------

**Shadow Renderer Setup**

.. code-block:: python

    # Create shadow renderer
    width, height = 800, 600
    renderer = shadows.ShadowRenderer(width, height, config)
    renderer.set_light(sun_light)
    
    # Prepare scene data
    scene_data = {
        'terrain': terrain_height_data,
        'material': material_properties,
        'bounds': {
            'min': (0, 0, 0),
            'max': (terrain_size, max_height, terrain_size)
        }
    }
    
    # Render with shadows
    image = renderer.render_with_shadows(scene_data)

**Memory-Safe Rendering**

The shadow system enforces memory constraints at runtime:

.. code-block:: python

    try:
        # Attempt to render with shadows
        shadow_image = renderer.render_with_shadows(scene_data)
        print("✓ Shadow rendering successful")
        
    except MemoryError as e:
        print(f"✗ Shadow rendering failed: {e}")
        print("Consider using a lower quality preset")
        
        # Fallback to shadowless rendering
        fallback_image = scene.render_terrain_rgba()

Quality vs Performance
----------------------

**Quality Presets Comparison**

================================= ========= ============= ============
Preset                           Cascades  Size          Memory (MiB)
================================= ========= ============= ============
``low_quality``                  2         512           2.0
``medium_quality``               3         1024          12.0  
``high_quality``                 4         1024          16.0
``ultra_quality``                4         2048          64.0
================================= ========= ============= ============

**Memory Budget Guidelines**

- **Mobile/Integrated Graphics**: Use ``low_quality`` (2-4 MiB)
- **Desktop Graphics**: Use ``medium_quality`` or ``high_quality`` (12-16 MiB)
- **High-end Hardware**: Consider ``ultra_quality`` (64 MiB) if budget allows
- **Memory-constrained**: Custom configurations with fewer cascades

PCF Filtering
-------------

**Percentage-Closer Filtering Configuration**

.. code-block:: python

    config = shadows.CsmConfig(
        cascade_count=3,
        shadow_map_size=1024,
        enable_pcf=True,        # Enable soft shadows
        pcf_radius=2,           # Filter radius (larger = softer)
        pcf_samples=9           # Number of samples (quality vs performance)
    )

**PCF Quality Settings**

- **pcf_radius=1**: Sharp shadows, best performance
- **pcf_radius=2**: Balanced quality/performance
- **pcf_radius=3**: Soft shadows, higher cost
- **pcf_samples**: 4 (fast), 9 (balanced), 16 (high quality)

Advanced Features
-----------------

**Cascade Distribution**

Control shadow resolution distribution:

.. code-block:: python

    config = shadows.CsmConfig(
        cascade_count=4,
        cascade_split_lambda=0.5,   # Uniform distribution
        # cascade_split_lambda=0.8,   # More detail near camera
        # cascade_split_lambda=0.2,   # More detail far from camera
        max_shadow_distance=200.0
    )

**Shadow Bias Configuration**

Prevent shadow artifacts:

.. code-block:: python

    config = shadows.CsmConfig(
        bias=0.001,           # Depth bias (shadow acne prevention)
        normal_bias=0.01,     # Normal bias (peter panning prevention)
        slope_bias=0.001      # Slope-dependent bias
    )

**Debug Visualization**

Visualize shadow cascades:

.. code-block:: python

    # Enable cascade visualization
    debug_config = config.copy()
    debug_config.debug_cascades = True
    debug_config.cascade_colors = [
        (1.0, 0.0, 0.0),  # Red for cascade 0
        (0.0, 1.0, 0.0),  # Green for cascade 1
        (0.0, 0.0, 1.0),  # Blue for cascade 2
        (1.0, 1.0, 0.0),  # Yellow for cascade 3
    ]

Integration Examples
--------------------

**Terrain with Shadows**

.. code-block:: python

    import forge3d as f3d
    import forge3d.shadows as shadows
    import numpy as np
    
    # Generate terrain
    terrain = np.random.rand(256, 256).astype(np.float32)
    
    # Set up shadows
    if shadows.has_shadows_support():
        shadow_config = shadows.get_preset_config('medium_quality')
        sun = shadows.DirectionalLight(
            direction=(-0.3, -0.8, -0.5),
            color=(1.0, 0.9, 0.7),
            intensity=2.5
        )
        
        # Render with shadows
        renderer = shadows.ShadowRenderer(800, 600, shadow_config)
        renderer.set_light(sun)
        
        scene_data = {'terrain': terrain}
        image = renderer.render_with_shadows(scene_data)
    else:
        # Fallback rendering
        scene = f3d.Scene(800, 600)
        scene.set_height_data(terrain, spacing=10.0, exaggeration=20.0)
        image = scene.render_terrain_rgba()
    
    f3d.numpy_to_png("terrain_shadows.png", image)

Performance Monitoring
----------------------

**Memory Usage Tracking**

.. code-block:: python

    # Monitor shadow atlas memory
    config = shadows.get_preset_config('high_quality')
    memory_used = shadows.calculate_shadow_atlas_memory(
        config.cascade_count, 
        config.shadow_map_size
    )
    
    memory_budget = 256 * 1024 * 1024  # 256 MiB
    utilization = memory_used / memory_budget
    
    print(f"Shadow memory utilization: {utilization:.1%}")
    
    if utilization > 0.8:
        print("⚠ High memory usage - consider lower quality preset")

**Performance Profiling**

.. code-block:: python

    import time
    
    # Time shadow rendering
    start_time = time.perf_counter()
    shadow_image = renderer.render_with_shadows(scene_data)
    render_time = time.perf_counter() - start_time
    
    print(f"Shadow rendering time: {render_time*1000:.1f}ms")

Troubleshooting
---------------

**"Shadow mapping not available"**

- Ensure GPU supports required features
- Check that forge3d was built with shadow support
- Verify graphics drivers are up to date

**Memory budget exceeded errors**

.. code-block:: python

    # Reduce cascade count or shadow map size
    config = shadows.CsmConfig(
        cascade_count=2,        # Reduced from 4
        shadow_map_size=512     # Reduced from 1024
    )

**Shadow artifacts**

- Adjust bias parameters to reduce shadow acne
- Increase PCF radius for smoother shadows  
- Check light direction and scene bounds
- Ensure terrain normals are correct

**Performance issues**

- Use lower quality presets on slower hardware
- Reduce PCF samples for better performance
- Consider dynamic quality adjustment based on scene complexity

Legacy Support
--------------

The shadow system maintains compatibility with older configurations while providing improved memory validation and error handling.

For detailed implementation examples, see:
- ``examples/advanced_terrain_shadows_pbr.py`` - Complete terrain + shadows + PBR
- ``examples/shadow_demo.py`` - Basic shadow mapping demonstration