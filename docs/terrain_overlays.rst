Terrain Overlays
================

This document describes the terrain overlay system (Option A - Draped Textures)
which allows raster images to be overlaid on terrain surfaces with full lighting
and shadow integration.

Overview
--------

The overlay system enables adding texture overlays (satellite imagery, heatmaps,
contour lines, etc.) to terrain surfaces. Unlike screen-space decals, these
overlays are **sampled in terrain UV space** and **blended into albedo before
lighting**, meaning they:

- ✅ Are lit by sun (diffuse term includes overlay color)
- ✅ Are shadowed (shadow_term multiplies diffuse result)
- ✅ Receive ambient occlusion (height_ao multiplies ambient term)
- ❌ Do NOT affect specular (specular depends on roughness, not albedo)

This approach produces the most natural integration with terrain rendering.

Quick Start
-----------

Python API
^^^^^^^^^^

.. code-block:: python

    from forge3d.terrain_params import (
        OverlaySettings,
        OverlayLayerConfig,
        OverlayBlendMode,
        make_terrain_params_config,
    )

    # Create overlay layer configuration
    satellite_layer = OverlayLayerConfig(
        name="satellite",
        source="satellite.png",
        extent=(0.0, 0.0, 1.0, 1.0),  # Full terrain coverage
        opacity=0.8,
        blend_mode=OverlayBlendMode.NORMAL,
        z_order=0,
    )

    # Create overlay settings
    overlay_settings = OverlaySettings(
        enabled=True,
        global_opacity=1.0,
        layers=[satellite_layer],
    )

    # Create terrain render parameters with overlay
    params = make_terrain_params_config(
        size_px=(1920, 1080),
        render_scale=1.0,
        terrain_span=1000.0,
        msaa_samples=4,
        z_scale=1.0,
        exposure=1.0,
        domain=(0.0, 1000.0),
        overlay=overlay_settings,
    )

IPC Commands (Interactive Viewer)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: json

    // Load overlay from file
    {"cmd": "load_overlay", "name": "satellite", "path": "overlay.png", 
     "extent": [0, 0, 1, 1], "opacity": 0.8, "z_order": 0}

    // Remove overlay by ID
    {"cmd": "remove_overlay", "id": 0}

    // Set overlay visibility
    {"cmd": "set_overlay_visible", "id": 0, "visible": false}

    // Set overlay opacity
    {"cmd": "set_overlay_opacity", "id": 0, "opacity": 0.5}

    // Set global overlay opacity multiplier
    {"cmd": "set_global_overlay_opacity", "opacity": 0.7}

    // Enable or disable overlay system
    {"cmd": "set_overlays_enabled", "enabled": true}

    // List all overlay IDs
    {"cmd": "list_overlays"}

Configuration Reference
-----------------------

OverlayLayerConfig
^^^^^^^^^^^^^^^^^^

Configuration for a single overlay layer.

.. list-table::
   :header-rows: 1

   * - Attribute
     - Type
     - Default
     - Description
   * - ``name``
     - str
     - (required)
     - Unique identifier for this layer
   * - ``source``
     - str
     - (required)
     - Path to image file (PNG, JPEG, etc.)
   * - ``extent``
     - tuple[float, float, float, float]
     - None (full coverage)
     - Extent in terrain UV space [u_min, v_min, u_max, v_max]
   * - ``opacity``
     - float
     - 1.0
     - Overlay opacity (0.0 = transparent, 1.0 = opaque)
   * - ``blend_mode``
     - str
     - "normal"
     - Blend mode: "normal", "multiply", or "overlay"
   * - ``visible``
     - bool
     - True
     - Whether this layer is rendered
   * - ``z_order``
     - int
     - 0
     - Stacking order (lower = behind)

OverlaySettings
^^^^^^^^^^^^^^^

Global overlay system configuration.

.. list-table::
   :header-rows: 1

   * - Attribute
     - Type
     - Default
     - Description
   * - ``enabled``
     - bool
     - False
     - Enable overlay system (default off for backward compatibility)
   * - ``global_opacity``
     - float
     - 1.0
     - Global opacity multiplier for all layers
   * - ``layers``
     - list[OverlayLayerConfig]
     - []
     - List of overlay layer configurations
   * - ``resolution_scale``
     - float
     - 1.0
     - Composite texture resolution relative to terrain

Blend Modes
^^^^^^^^^^^

Three blend modes are supported for compositing overlays with terrain albedo:

**Normal** (``OverlayBlendMode.NORMAL``)
    Standard alpha blend: ``mix(terrain_albedo, overlay_color, overlay_alpha)``

**Multiply** (``OverlayBlendMode.MULTIPLY``)
    Darkens terrain: ``terrain_albedo * overlay_color``
    Good for shadow/stain effects.

**Overlay** (``OverlayBlendMode.OVERLAY``)
    Photoshop-style overlay: increases contrast, darkens darks, brightens brights.
    Good for enhancing satellite imagery.

Technical Details
-----------------

Architecture
^^^^^^^^^^^^

The overlay system consists of these components:

1. **Overlay Stack (Rust)** - ``src/viewer/terrain/overlay.rs``
   
   - ``OverlayLayer``: Per-layer configuration and GPU texture
   - ``OverlayStack``: Manages layer collection, compositing, and GPU resources
   - CPU-based compositing flattens visible layers into a single RGBA texture

2. **Shader Integration** - ``src/viewer/terrain/shader_pbr.rs``
   
   - Overlay texture bound at ``@binding(5)``
   - Overlay sampler bound at ``@binding(6)``
   - Blending occurs before lighting in fragment shader

3. **Scene Integration** - ``src/viewer/terrain/scene.rs``
   
   - ``ViewerTerrainScene::overlay_stack``: Holds the overlay stack
   - API methods: ``add_overlay_raster``, ``add_overlay_image``, ``remove_overlay``, etc.

GPU Resources
^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Resource
     - Format
     - Usage
   * - Per-layer texture
     - Rgba8UnormSrgb
     - Individual layer storage
   * - Composite texture
     - Rgba8UnormSrgb
     - Flattened layer stack, sampled by shader
   * - Overlay sampler
     - Linear filtering, ClampToEdge
     - Texture sampling

Memory Budget
^^^^^^^^^^^^^

Overlay textures respect the 512 MiB host-visible heap constraint from AGENTS.md:

.. list-table::
   :header-rows: 1

   * - Resolution
     - Memory (RGBA8)
   * - 1k × 1k
     - 4 MB
   * - 2k × 2k
     - 16 MB
   * - 4k × 4k
     - 64 MB

Conservative estimate: 4 layers at 2k resolution = ~80 MB total.

UV Mapping
^^^^^^^^^^

Overlays use terrain UV space (0-1 range matching terrain grid):

.. code-block:: text

    Terrain coordinate space: [0, terrain_width] × [0, terrain_width]
    Terrain UV space: [0, 1] × [0, 1]

    For an overlay with extent [u_min, v_min, u_max, v_max]:
      - Pixels outside extent show terrain albedo unchanged
      - Pixels inside extent sample overlay and blend with terrain

Default Behavior
----------------

The overlay system is **disabled by default** to preserve backward compatibility.
Enabling overlays requires explicit configuration:

- Python: ``OverlaySettings(enabled=True, ...)``
- IPC: ``{"cmd": "set_overlays_enabled", "enabled": true}``

When disabled, the shader receives a 1×1 transparent fallback texture, ensuring
zero visual impact and identical rendering to pre-overlay builds.

Testing
-------

Run overlay-specific tests:

.. code-block:: bash

    python -m pytest tests/test_terrain_overlay_stack.py -v

Verify Rust compilation:

.. code-block:: bash

    cargo check --lib

See Also
--------

- ``docs/plan_overlays_option_a_draped_textures.md`` - Full implementation plan
- ``src/viewer/terrain/overlay.rs`` - Rust overlay stack implementation
- ``tests/test_terrain_overlay_stack.py`` - Python API unit tests
- ``examples/swiss_terrain_landcover_viewer.py`` - Real-world example with Switzerland DEM and land cover overlay
