Lighting System Overview
=========================

This document covers the **P1 Lighting System** infrastructure introduced in Milestone 1. The P1 system provides foundational GPU light buffers, triple-buffering, and TAA-friendly temporal seeding, with integration into shading pipelines deferred to future milestones.

.. contents:: Table of Contents
   :local:
   :depth: 2

Light Types
-----------

forge3d supports 7 light types, each with specific use cases:

Directional Lights
~~~~~~~~~~~~~~~~~~

Infinite distance light source (e.g., sun). Direction computed from azimuth and elevation angles.

**Use cases:** Outdoor scenes, sunlight, skylight

**Key parameters:**

- ``azimuth``: Horizontal angle in degrees (0-360)
- ``elevation``: Vertical angle in degrees (-90 to 90)
- ``intensity``: Light power multiplier
- ``color``: RGB color triplet [0-1]

Point Lights
~~~~~~~~~~~~

Omnidirectional light emanating from a point in space. Attenuates with distance.

**Use cases:** Lightbulbs, candles, explosions

**Key parameters:**

- ``position``: XYZ world-space coordinates
- ``range``: Maximum influence distance
- ``intensity``: Light power multiplier
- ``color``: RGB color triplet [0-1]

Spot Lights
~~~~~~~~~~~

Cone-shaped light with inner/outer falloff. Common for flashlights and stage lighting.

**Use cases:** Flashlights, spotlights, car headlights

**Key parameters:**

- ``position``: XYZ world-space coordinates
- ``direction``: Normalized direction vector
- ``inner_angle``: Inner cone angle in degrees (full intensity)
- ``outer_angle``: Outer cone angle in degrees (zero intensity)
- ``range``: Maximum influence distance
- ``intensity``: Light power multiplier
- ``color``: RGB color triplet [0-1]

Environment Lights
~~~~~~~~~~~~~~~~~~

Image-based lighting (IBL) using an HDR environment map. Provides ambient global illumination.

**Use cases:** Outdoor scenes, studio lighting, reflections

**Key parameters:**

- ``intensity``: IBL intensity multiplier
- ``env_texture_index``: Index into environment texture array
- ``hdr_path``: Path to HDR environment map (.hdr, .exr)

Area Rectangle Lights
~~~~~~~~~~~~~~~~~~~~~

Rectangular area light emitting from a planar surface. Produces soft shadows with realistic falloff.

**Use cases:** Windows, ceiling panels, light strips

**Key parameters:**

- ``position``: Center position XYZ
- ``normal``: Surface normal (direction light faces)
- ``half_width``: Half-extent in width dimension
- ``half_height``: Half-extent in height dimension
- ``intensity``: Light power multiplier
- ``color``: RGB color triplet [0-1]

Area Disk Lights
~~~~~~~~~~~~~~~~

Circular disk area light. Similar to rectangle but with radial symmetry.

**Use cases:** Round ceiling lights, portholes

**Key parameters:**

- ``position``: Center position XYZ
- ``normal``: Surface normal
- ``radius``: Disk radius
- ``intensity``: Light power multiplier
- ``color``: RGB color triplet [0-1]

Area Sphere Lights
~~~~~~~~~~~~~~~~~~

Spherical volumetric light. Omnidirectional emission from sphere surface.

**Use cases:** Light bulbs, glowing orbs, magical effects

**Key parameters:**

- ``position``: Center position XYZ
- ``radius``: Sphere radius
- ``intensity``: Light power multiplier
- ``color``: RGB color triplet [0-1]

GPU Layout (LightGPU)
---------------------

Each light is stored as an 80-byte structure (5 × vec4) in GPU memory using std430 layout:

.. code-block:: text

    LightGPU Layout (80 bytes, 16-byte aligned):
    
    vec4 #1 (bytes 0-15):
      kind:              u32    // Light type (0-6)
      intensity:         f32    // Power multiplier
      range:             f32    // Max distance (point/spot/area)
      env_texture_index: u32    // Environment map index
    
    vec4 #2 (bytes 16-31):
      color:             vec3   // RGB [0-1]
      _pad0:             f32    // Padding
    
    vec4 #3 (bytes 32-47):
      pos_ws:            vec3   // World-space position
      _pad1:             f32    // Padding
    
    vec4 #4 (bytes 48-63):
      dir_ws:            vec3   // Direction/normal
      _pad2:             f32    // Padding
    
    vec4 #5 (bytes 64-79):
      cone_cos:          vec2   // Spot: [cos(inner), cos(outer)]
      area_half:         vec2   // Area: [half_width/radius, half_height]

**Light Type Enum:**

- ``0``: Directional
- ``1``: Point
- ``2``: Spot
- ``3``: Environment
- ``4``: AreaRect
- ``5``: AreaDisk
- ``6``: AreaSphere

**Field Usage by Type:**

+-------------------+-------------+-------+------+-------------+----------+
| Field             | Directional | Point | Spot | Environment | Area     |
+===================+=============+=======+======+=============+==========+
| ``kind``          | ✓           | ✓     | ✓    | ✓           | ✓        |
+-------------------+-------------+-------+------+-------------+----------+
| ``intensity``     | ✓           | ✓     | ✓    | ✓           | ✓        |
+-------------------+-------------+-------+------+-------------+----------+
| ``range``         | ✗           | ✓     | ✓    | ✗           | ✓        |
+-------------------+-------------+-------+------+-------------+----------+
| ``env_texture_ix``| ✗           | ✗     | ✗    | ✓           | ✗        |
+-------------------+-------------+-------+------+-------------+----------+
| ``color``         | ✓           | ✓     | ✓    | ✓           | ✓        |
+-------------------+-------------+-------+------+-------------+----------+
| ``pos_ws``        | ✗           | ✓     | ✓    | ✗           | ✓        |
+-------------------+-------------+-------+------+-------------+----------+
| ``dir_ws``        | ✓           | ✗     | ✓    | ✗           | ✓        |
+-------------------+-------------+-------+------+-------------+----------+
| ``cone_cos``      | ✗           | ✗     | ✓    | ✗           | ✗        |
+-------------------+-------------+-------+------+-------------+----------+
| ``area_half``     | ✗           | ✗     | ✗    | ✗           | ✓        |
+-------------------+-------------+-------+------+-------------+----------+

Memory Budget
-------------

**MAX_LIGHTS Constant**

The P1 system supports up to **16 lights** simultaneously (``MAX_LIGHTS = 16``).

**GPU Memory Usage:**

- Light buffer: 16 × 80 bytes = **1,280 bytes** per frame
- Triple-buffered: 3 × 1,280 = **3,840 bytes** (~3.75 KB)
- Metadata buffer: 3 × 16 bytes = **48 bytes**
- Total: **~3.9 KB**

This is well within the 512 MiB host-visible budget and imposes negligible GPU overhead.

Triple-Buffering & TAA Seeds
-----------------------------

The light buffer uses **triple-buffering** to ensure TAA-friendly rendering:

Frame Management
~~~~~~~~~~~~~~~~

1. **Buffer Rotation**: Three GPU buffers rotate each frame (indices 0, 1, 2)
2. **Frame Counter**: Monotonic counter increments with each ``next_frame()`` call
3. **R2 Sequence Seeds**: Deterministic quasi-random seeds for temporal sampling

R2 Sequence (Additive Recurrence)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The system generates 2D R2 sequence samples for temporal anti-aliasing:

.. math::

   R_2(n) = \left( \text{frac}\left(0.5 + \frac{n}{\phi}\right), \text{frac}\left(0.5 + \frac{n}{\phi^2}\right) \right)

where :math:`\phi = 1.324717957...` (plastic constant).

**Properties:**

- Deterministic (same seed for frame N across runs)
- Low-discrepancy (good coverage of [0,1]² space)
- TAA-friendly (frame-to-frame variation minimizes aliasing)

**Usage in Shaders:**

Seeds are uploaded to GPU as ``vec2`` in the metadata uniform and can be used for:

- Temporal jitter offsets
- Stochastic light sampling
- Progressive refinement

Python API Usage
----------------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

    import forge3d as f3d
    
    # Create renderer
    renderer = f3d.Renderer(width=1920, height=1080)
    
    # Define lights
    lights = [
        {
            "type": "directional",
            "azimuth": 135.0,
            "elevation": 35.0,
            "intensity": 3.0,
            "color": [1.0, 0.9, 0.8],
        },
        {
            "type": "point",
            "position": [10.0, 5.0, 0.0],
            "range": 50.0,
            "intensity": 10.0,
            "color": [1.0, 1.0, 1.0],
        },
    ]
    
    # Upload to renderer
    renderer.set_lights(lights)

Advanced Example: Heterogeneous Scene
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    lights = [
        # Sun
        {
            "type": "directional",
            "azimuth": 225.0,
            "elevation": 45.0,
            "intensity": 5.0,
            "color": [1.0, 0.95, 0.9],
        },
        # Ceiling panel
        {
            "type": "area_rect",
            "position": [0.0, 10.0, 0.0],
            "direction": [0.0, -1.0, 0.0],
            "area_extent": [2.0, 1.5],  # Half-extents
            "intensity": 15.0,
            "color": [1.0, 1.0, 1.0],
        },
        # Spotlight
        {
            "type": "spot",
            "position": [5.0, 8.0, 5.0],
            "direction": [0.0, -1.0, 0.0],
            "inner_angle": 15.0,
            "outer_angle": 30.0,
            "range": 100.0,
            "intensity": 8.0,
            "color": [1.0, 0.9, 0.7],
        },
    ]
    
    renderer.set_lights(lights)

CLI Usage (terrain_demo.py)
----------------------------

The ``terrain_demo.py`` example supports ``--light`` flags for scene lighting:

Basic CLI Lights
~~~~~~~~~~~~~~~~

.. code-block:: bash

    python examples/terrain_demo.py assets/dem.tif \
        --light type=directional,intensity=3,color=1,0.9,0.8

Multiple Lights
~~~~~~~~~~~~~~~

.. code-block:: bash

    python examples/terrain_demo.py assets/Gore_Range_Albers_1m.tif \
        --light type=directional,dir=0.3,0.8,0.5,intensity=5 \
        --light type=point,pos=0,100,0,intensity=10,range=200 \
        --light type=spot,pos=50,80,50,dir=0,-1,0,cone=30,intensity=12

CLI Syntax
~~~~~~~~~~

Light specifications use ``key=value,key=value`` format:

**Common keys:**

- ``type``: Light type (directional, point, spot, area_rect, etc.)
- ``intensity`` or ``power``: Light intensity
- ``color`` or ``rgb``: RGB color (3 comma-separated floats)
- ``pos`` or ``position``: XYZ position (3 floats)
- ``dir`` or ``direction``: XYZ direction (3 floats)
- ``range``: Maximum distance
- ``cone`` or ``cone_angle``: Spot cone angle (degrees)
- ``area`` or ``area_extent``: Area dimensions (2 floats)

Debug Utility (lights_ssbo_debug.py)
-------------------------------------

The ``lights_ssbo_debug.py`` utility allows inspection of light buffer state without rendering.

Usage
~~~~~

.. code-block:: bash

    python examples/lights_ssbo_debug.py \
        --light type=directional,intensity=3 \
        --light type=point,pos=0,10,0,intensity=10

Output
~~~~~~

.. code-block:: text

    LightBuffer Debug Info:
      Count: 2 lights
      Frame: 0 (seed: [0.500, 0.755])
    
      Light 0: Directional
        Intensity: 3.00, Color: [1.00, 1.00, 1.00]
        Direction: [0.58, -0.57, 0.58]
    
      Light 1: Point
        Intensity: 10.00, Color: [1.00, 1.00, 1.00]
        Position: [0.00, 10.00, 0.00], Range: 100.00

This utility is useful for:

- Validating light configurations
- Debugging CLI parsing
- Acceptance testing without GPU
- Documenting expected behavior

Limitations and Future Work
----------------------------

P1 Scope
~~~~~~~~

The **P1 Lighting System** provides **infrastructure only**:

✅ **Included in P1:**

- GPU buffer management (LightBuffer)
- Triple-buffering with TAA seeds
- 7 light types with host/device parity
- Python and CLI parsing
- Debug inspection API
- Unit and GPU integration tests

❌ **NOT Included in P1:**

- BRDF shader integration
- Shadow mapping integration
- GI/IBL rendering integration
- Material-light interaction
- Physically-based attenuation

**Deferred to Future Milestones:**

- **P2**: Material shading integration (BRDF evaluation)
- **P3**: Shadow system integration (CSM, PCSS, VSM)
- **P4**: IBL integration (environment lighting)
- **P5**: Advanced features (volumetrics, caustics)

Shading Integration
~~~~~~~~~~~~~~~~~~~

P1 uploads lights to GPU but does **not** connect them to shading passes. To enable lighting in renders:

1. **Wait for P2+ integration**, or
2. **Manually bind light buffers** in custom shaders using:

   - Binding 3: Light array SSBO (read-only storage)
   - Binding 4: LightMetadata uniform (count, frame, seeds)
   - Binding 5: Environment params (reserved)

MAX_LIGHTS Scaling
~~~~~~~~~~~~~~~~~~

The current limit of 16 lights is suitable for most scenes. For scenes requiring more:

- **Clustering**: Partition scene into grid cells with local light lists (future)
- **Deferred culling**: GPU-side light culling per tile (future)
- **Increase MAX_LIGHTS**: Recompile with higher limit (simple, but impacts memory)

Best Practices
--------------

Light Count
~~~~~~~~~~~

- Use **1-4 lights** for simple scenes (sun + fill + accent)
- Use **4-8 lights** for indoor scenes (windows + fixtures)
- Avoid **>12 lights** unless clustering is implemented

Color Temperature
~~~~~~~~~~~~~~~~~

Recommended RGB values for common light sources:

- **Sunlight (5500K)**: ``[1.0, 0.95, 0.9]``
- **Tungsten (3200K)**: ``[1.0, 0.8, 0.6]``
- **Fluorescent (4500K)**: ``[0.95, 1.0, 1.0]``
- **Candlelight (1850K)**: ``[1.0, 0.6, 0.3]``

Intensity Scaling
~~~~~~~~~~~~~~~~~

Physically-based intensity ranges (when PBR integration is complete):

- **Sunlight**: 50,000 - 120,000 lux → intensity 5-12
- **Overcast sky**: 1,000 - 2,000 lux → intensity 1-2
- **Office lighting**: 300 - 500 lux → intensity 0.3-0.5
- **Moonlight**: 0.1 - 0.3 lux → intensity 0.0001-0.0003

Range Guidelines
~~~~~~~~~~~~~~~~

For point/spot lights, set range to avoid unnecessary shader work:

- **Small room**: range 10-20 units
- **Large hall**: range 50-100 units
- **Outdoor**: range 200-500 units

Performance Considerations
--------------------------

GPU Cost
~~~~~~~~

- Light buffer upload: **~10 μs** (3.9 KB transfer)
- Per-light shader cost: **~0.5-2 ms** depending on complexity
- Triple-buffering overhead: **Negligible** (~4 KB total)

CPU Cost
~~~~~~~~

- ``set_lights()`` parsing: **<1 ms** for typical scenes
- Frame advance (``next_frame()``): **<0.1 μs** (counter increment + R2 sample)

Recommendations
~~~~~~~~~~~~~~~

1. **Update lights only when changed** (not every frame)
2. **Prefer directional lights** (cheapest to evaluate)
3. **Use area lights sparingly** (most expensive, especially with shadows)
4. **Profile with GPU timestamps** when adding many lights

Related Documentation
---------------------

- :doc:`shadow_mapping` - Shadow mapping (P3 integration)
- :doc:`pbr_materials` - Material shading (P2 integration)
- :doc:`../api/path_tracing` - Path tracing (uses lights directly)
- :doc:`../memory/memory_budget` - Memory budget analysis

References
----------

**P1 Implementation Files:**

- ``src/lighting/types.rs`` - Light struct definitions
- ``src/lighting/light_buffer.rs`` - LightBuffer manager
- ``src/lighting/py_bindings.rs`` - Python bindings
- ``examples/lights_ssbo_debug.py`` - Debug utility
- ``tests/test_p1_light_buffer.rs`` - GPU integration tests

**Specifications:**

- ``p1.md`` - P1 milestone specification
- ``todo-1.md`` - P1 implementation checklist

Glossary
--------

.. glossary::

   BRDF
      Bidirectional Reflectance Distribution Function - describes how light reflects off surfaces
   
   CSM
      Cascaded Shadow Maps - shadow technique for large outdoor scenes
   
   HDR
      High Dynamic Range - images with >8-bit color depth for realistic lighting
   
   IBL
      Image-Based Lighting - using environment maps for ambient illumination
   
   SSBO
      Shader Storage Buffer Object - GPU buffer for arbitrary structured data
   
   TAA
      Temporal Anti-Aliasing - reduces aliasing by accumulating samples across frames
   
   R2 Sequence
      Low-discrepancy 2D sequence using the plastic constant for quasi-random sampling
