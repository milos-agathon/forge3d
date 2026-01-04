Post-Processing Effects
=======================

The post-processing system in forge3d provides a flexible compute-based pipeline for applying visual effects to rendered images. This system supports effect chaining, parameter control, and GPU timing integration.

Overview
--------

The post-processing pipeline consists of:

* **Effect Chain Management**: Automatically handles effect ordering and resource allocation
* **Compute-Based Effects**: GPU compute shaders for optimal performance
* **Temporal Effects**: Support for effects that require previous frame data
* **Parameter Control**: Real-time adjustment of effect parameters
* **GPU Timing Integration**: Performance monitoring for each effect

Quick Start
-----------

Basic post-processing usage::

    import forge3d.postfx as postfx
    
    # Enable post-processing effects
    postfx.enable("bloom", threshold=1.2, strength=0.8)
    postfx.enable("tonemap", exposure=1.0, gamma=2.2)
    
    # Check enabled effects (returns in execution order)
    enabled = postfx.list()
    print(f"Enabled effects: {enabled}")
    
    # Adjust parameters
    postfx.set_parameter("bloom", "strength", 1.0)
    
    # Apply a preset configuration
    postfx.apply_preset("cinematic")

Available Effects
-----------------

Bloom Effect
~~~~~~~~~~~~

The bloom effect creates a luminous glow around bright areas of the image using a three-pass pipeline:

1. **Bright-pass extraction**: Identifies pixels above a brightness threshold
2. **Horizontal blur**: Applies Gaussian blur horizontally
3. **Vertical blur**: Applies Gaussian blur vertically for the final bloom

Parameters:
* ``threshold`` (0.0 - 5.0): Brightness threshold for bloom extraction
* ``strength`` (0.0 - 2.0): Intensity of the bloom effect
* ``radius`` (0.1 - 3.0): Blur radius for the bloom

Example::

    postfx.enable("bloom", 
                  threshold=1.0,  # Only pixels brighter than 1.0 bloom
                  strength=0.6,   # Moderate bloom intensity
                  radius=1.5)     # Medium blur radius

Tone Mapping
~~~~~~~~~~~~

Tone mapping converts HDR (High Dynamic Range) images to displayable LDR (Low Dynamic Range) with gamma correction.

Parameters:
* ``exposure`` (0.1 - 10.0): Exposure multiplier for tone mapping
* ``gamma`` (1.0 - 4.0): Gamma correction value

Example::

    postfx.enable("tonemap",
                  exposure=1.2,  # Slightly brighter exposure
                  gamma=2.2)     # Standard sRGB gamma

Other Effects
~~~~~~~~~~~~~

* **FXAA**: Fast Approximate Anti-Aliasing for edge smoothing
* **Temporal AA**: Temporal anti-aliasing using previous frame data
* **Blur**: Simple box blur effect
* **Sharpen**: Unsharp mask sharpening

Presets
-------

Pre-configured effect combinations for common use cases:

Cinematic Preset
~~~~~~~~~~~~~~~~

Optimized for cinematic rendering with warm bloom and balanced exposure::

    postfx.apply_preset("cinematic")
    # Equivalent to:
    # postfx.enable("bloom", threshold=1.2, strength=0.6)
    # postfx.enable("tonemap", exposure=1.1, gamma=2.2)

Quality Preset
~~~~~~~~~~~~~~

High-quality rendering with temporal anti-aliasing::

    postfx.apply_preset("quality")
    # Enables: temporal_aa, bloom, tonemap

Performance Preset
~~~~~~~~~~~~~~~~~~~

Optimized for performance with minimal effects::

    postfx.apply_preset("performance")
    # Enables: fxaa (low quality), tonemap

Sharp Preset
~~~~~~~~~~~~

Enhanced sharpness for technical visualization::

    postfx.apply_preset("sharp")
    # Enables: sharpen, fxaa, tonemap

Advanced Usage
--------------

Effect Chain Control
~~~~~~~~~~~~~~~~~~~~

The post-processing chain can be enabled/disabled globally::

    # Disable entire post-processing chain
    postfx.set_chain_enabled(False)
    
    # Re-enable the chain
    postfx.set_chain_enabled(True)
    
    # Check if chain is enabled
    if postfx.is_chain_enabled():
        print("Post-processing is active")

Parameter Introspection
~~~~~~~~~~~~~~~~~~~~~~~

Get information about available effects and their parameters::

    # List all available effects
    available = postfx.list_available()
    print(f"Available effects: {available}")
    
    # Get detailed information about an effect
    bloom_info = postfx.get_effect_info("bloom")
    print(f"Bloom description: {bloom_info['description']}")
    print(f"Bloom parameters: {bloom_info['parameters']}")
    
    # Check current parameter values
    threshold = postfx.get_parameter("bloom", "threshold")
    print(f"Current bloom threshold: {threshold}")

Performance Monitoring
~~~~~~~~~~~~~~~~~~~~~~

Monitor GPU timing for each effect::

    # Get timing statistics
    stats = postfx.get_timing_stats()
    for effect_name, time_ms in stats.items():
        print(f"{effect_name}: {time_ms:.2f} ms")

Custom Presets
~~~~~~~~~~~~~~

Create and apply custom preset configurations::

    # Create a custom preset
    custom_preset = postfx.create_preset("custom_hdr", [
        {'name': 'bloom', 'parameters': {'threshold': 2.0, 'strength': 0.4}},
        {'name': 'tonemap', 'parameters': {'exposure': 1.5, 'gamma': 2.4}}
    ])

Performance Considerations
--------------------------

The post-processing system is designed for high performance:

* **GPU Compute**: All effects run on GPU compute shaders
* **Resource Pooling**: Textures and buffers are reused across frames
* **Effect Ordering**: Effects are automatically ordered by priority for optimal GPU utilization
* **Timing Integration**: Built-in GPU timing helps identify performance bottlenecks

Performance targets:
* 60 FPS @ 1080p with three enabled effects in the chain
* <1% GPU time overhead for timing and resource management

Integration with Renderer
--------------------------

Post-processing integrates seamlessly with the forge3d rendering pipeline::

    import forge3d as f3d
    import forge3d.postfx as postfx
    
    # Set up renderer
    renderer = f3d.Renderer(1920, 1080)
    
    # Configure post-processing
    postfx.apply_preset("quality")
    postfx.set_parameter("bloom", "strength", 0.8)
    
    # Render with post-processing applied
    image = renderer.render_with_postfx()
    
    # Check performance
    stats = postfx.get_timing_stats()
    total_postfx_time = sum(stats.values())
    print(f"Post-processing took {total_postfx_time:.2f} ms")

Atmospherics, GI, and PostFX
----------------------------

The post-processing chain runs **after** the main renderer has composed:

* Analytic or HDRI sky selected via ``RendererConfig.atmosphere.sky`` or the
  terrain demo's ``--sky`` / ``--hdr`` flags.
* Volumetric fog configured via ``RendererConfig.atmosphere.volumetric`` or the
  terrain demo's ``--volumetric`` flag.
* Direct lighting and shadows.
* Global illumination modes from ``RendererConfig.gi.modes`` or the
  terrain demo's ``--gi`` flag.

In practice this means you should:

* First, shape the **HDR lighting and atmosphere** using sky, GI, and fog
  (via ``RendererConfig`` or CLI flags like ``--sky``, ``--gi``,
  ``--volumetric``).
* Then, use post-processing to handle **tone mapping, bloom, and AA** on the
  resulting HDR frame.

Example combining terrain CLI configuration with postfx usage::

    # Terrain demo (HDRI sky + IBL + volumetric fog)
    python examples/terrain_demo.py \
        --sky hdri \
        --hdr assets/snow_field_4k.hdr \
        --gi ibl,ssao \
        --volumetric "density=0.05,phase=hg,g=0.7"

    # Python: enable a postfx chain over the composed HDR image
    import forge3d as f3d
    import forge3d.postfx as postfx

    renderer = f3d.Renderer(1280, 720)
    postfx.apply_preset("quality")
    postfx.set_parameter("tonemap", "exposure", 1.1)

Notes:

* GI mode ``ibl`` still requires an HDR source, either from an environment
  light or from an HDRI sky (``sky=hdri`` + ``hdr_path`` / ``--hdr``).
* Enabling strong fog or bright skies pushes more energy into highlights;
  adjust tone-mapping exposure and bloom threshold to avoid clipping.

API Reference
-------------

.. automodule:: forge3d.postfx
   :members:
   :undoc-members:
   :show-inheritance: