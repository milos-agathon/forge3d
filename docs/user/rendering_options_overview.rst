Rendering Options Overview
===========================

This page documents the comprehensive rendering configuration system in forge3d, including lighting, shading, shadows, global illumination, and atmospheric effects.

Configuration Structure
-----------------------

The renderer configuration is organized into five main sections:

1. **Lighting** - Light sources, exposure, and intensity
2. **Shading** - BRDF models and material properties
3. **Shadows** - Shadow mapping techniques and parameters
4. **GI (Global Illumination)** - Ambient occlusion and indirect lighting
5. **Atmosphere** - Sky models, HDR environments, and volumetric effects

Lighting Parameters
-------------------

Light Configuration
~~~~~~~~~~~~~~~~~~~

.. list-table:: Light Configuration Fields
   :header-rows: 1
   :widths: 20 15 15 50

   * - Field
     - Type
     - Default
     - Description
   * - ``type``
     - string
     - ``"directional"``
     - Light type: directional, point, spot, area-rect, area-disk, area-sphere, environment
   * - ``intensity``
     - float
     - ``5.0``
     - Light intensity multiplier
   * - ``color``
     - [float, float, float]
     - ``[1.0, 0.97, 0.94]``
     - RGB color (warm white default)
   * - ``direction``
     - [float, float, float]
     - ``[-0.35, -1.0, -0.25]``
     - Direction vector (required for directional lights)
   * - ``position``
     - [float, float, float]
     - ``None``
     - Position (required for point/spot/area lights)
   * - ``cone_angle``
     - float
     - ``None``
     - Cone angle in degrees [0, 180] for spot lights
   * - ``area_extent``
     - [float, float]
     - ``None``
     - Width and height for area lights (must be positive)
   * - ``hdr_path``
     - string
     - ``None``
     - Path to HDR environment map (required for environment lights)

Lighting Params
~~~~~~~~~~~~~~~

.. list-table:: Lighting Configuration
   :header-rows: 1
   :widths: 20 15 15 50

   * - Field
     - Type
     - Default
     - Description
   * - ``lights``
     - list
     - ``[LightConfig()]``
     - List of light sources (minimum one light required)
   * - ``exposure``
     - float
     - ``1.0``
     - Global exposure multiplier for tone mapping

**Valid Light Types:**

* ``directional`` (aliases: dir, sun)
* ``point`` (aliases: pointlight)
* ``spot`` (aliases: spotlight)
* ``area-rect`` (aliases: rect, rectlight, arearect)
* ``area-disk`` (aliases: disk, disklight, areadisk)
* ``area-sphere`` (aliases: sphere, spherelight, areasphere)
* ``environment`` (aliases: env, hdri)

Shading Parameters
------------------

.. list-table:: Shading Configuration
   :header-rows: 1
   :widths: 20 15 15 50

   * - Field
     - Type
     - Default
     - Description
   * - ``brdf``
     - string
     - ``"cooktorrance-ggx"``
     - BRDF model for material shading
   * - ``normal_maps``
     - bool
     - ``True``
     - Enable normal mapping
   * - ``metallic``
     - float
     - ``0.0``
     - Metallic factor [0.0, 1.0]
   * - ``roughness``
     - float
     - ``0.5``
     - Roughness factor [0.0, 1.0]
   * - ``sheen``
     - float
     - ``0.0``
     - Sheen factor for fabric-like materials
   * - ``clearcoat``
     - float
     - ``0.0``
     - Clearcoat layer intensity

**Valid BRDF Models:**

* ``lambert`` - Simple diffuse (fastest)
* ``phong`` - Classic specular
* ``blinn-phong`` (aliases: blinnphong)
* ``oren-nayar`` (aliases: orennayar) - Rough diffuse
* ``cooktorrance-ggx`` (aliases: ggx, cooktorranceggx) - Industry standard PBR
* ``cooktorrance-beckmann`` (aliases: beckmann, cooktorrancebeckmann)
* ``disney-principled`` (aliases: disney, disneyprincipled) - Disney BRDF
* ``ashikhmin-shirley`` (aliases: ashikhminshirley)
* ``ward`` - Anisotropic highlights
* ``toon`` - Stylized cel-shading
* ``minnaert`` - Lunar-like diffuse
* ``subsurface`` (aliases: sss) - Subsurface scattering approximation
* ``hair`` (aliases: kajiyakay, kajiya-kay) - Hair/fiber shading

Shadow Parameters
-----------------

.. list-table:: Shadow Configuration
   :header-rows: 1
   :widths: 20 15 15 50

   * - Field
     - Type
     - Default
     - Description
   * - ``enabled``
     - bool
     - ``True``
     - Enable shadow mapping
   * - ``technique``
     - string
     - ``"pcf"``
     - Shadow mapping technique
   * - ``map_size``
     - int
     - ``2048``
     - Shadow map resolution (must be power of two)
   * - ``cascades``
     - int
     - ``4``
     - Number of cascades [1, 4]
   * - ``contact_hardening``
     - bool
     - ``False``
     - Enable contact-hardening shadows
   * - ``pcss_blocker_radius``
     - float
     - ``6.0``
     - PCSS blocker search radius (must be non-negative)
   * - ``pcss_filter_radius``
     - float
     - ``4.0``
     - PCSS filter radius (must be non-negative)
   * - ``light_size``
     - float
     - ``1.0``
     - Light source size for PCSS (must be positive)
   * - ``moment_bias``
     - float
     - ``0.0005``
     - Bias for moment-based techniques (must be positive)

**Valid Shadow Techniques:**

* ``hard`` - Hard shadows (no filtering, fastest)
* ``pcf`` - Percentage-Closer Filtering (default, good quality)
* ``pcss`` - Percentage-Closer Soft Shadows (soft, contact-hardening)
* ``vsm`` - Variance Shadow Maps (moment-based)
* ``evsm`` - Exponential Variance Shadow Maps (improved VSM)
* ``msm`` - Moment Shadow Maps (highest quality moments)
* ``csm`` - Cascaded Shadow Maps (requires cascades >= 2)

**Validation Rules:**

* ``map_size`` must be power of two (256, 512, 1024, 2048, 4096, 8192)
* Filtered techniques (PCF, PCSS, VSM, EVSM, MSM, CSM) should use map_size >= 256
* ``cascades`` must be in range [1, 4]
* CSM technique requires ``cascades >= 2``
* PCSS requires positive ``light_size`` and non-negative radii
* Moment techniques (VSM, EVSM, MSM) require positive ``moment_bias``
* Total shadow atlas memory must not exceed 256 MiB

Global Illumination Parameters
-------------------------------

.. list-table:: GI Configuration
   :header-rows: 1
   :widths: 20 15 15 50

   * - Field
     - Type
     - Default
     - Description
   * - ``modes``
     - list[string]
     - ``[]``
     - List of GI techniques (can combine multiple)
   * - ``ambient_occlusion_strength``
     - float
     - ``0.0``
     - AO darkening strength [0.0, 1.0]

**Valid GI Modes:**

* ``none`` - No global illumination
* ``ibl`` - Image-Based Lighting (requires environment light or atmosphere HDR)
* ``irradiance-probes`` (aliases: probes, irradianceprobes) - Light probe grid
* ``ddgi`` - Dynamic Diffuse Global Illumination
* ``voxel-cone-tracing`` (aliases: vct, voxelconetracing) - Voxel-based GI
* ``ssao`` - Screen-Space Ambient Occlusion
* ``gtao`` - Ground-Truth Ambient Occlusion
* ``ssgi`` - Screen-Space Global Illumination
* ``ssr`` - Screen-Space Reflections

**Note:** Multiple GI modes can be combined, e.g., ``["ibl", "ssao", "ssr"]``

Atmosphere Parameters
---------------------

Sky Models
~~~~~~~~~~

.. list-table:: Atmosphere Configuration
   :header-rows: 1
   :widths: 20 15 15 50

   * - Field
     - Type
     - Default
     - Description
   * - ``enabled``
     - bool
     - ``True``
     - Enable atmospheric rendering
   * - ``sky``
     - string
     - ``"hosek-wilkie"``
     - Sky model for atmospheric scattering
   * - ``hdr_path``
     - string
     - ``None``
     - Path to HDR environment map
   * - ``volumetric``
     - VolumetricParams
     - ``None``
     - Volumetric fog/atmosphere settings

**Valid Sky Models:**

* ``hosek-wilkie`` (aliases: hosekwilkie) - Physically-based sky (default)
* ``preetham`` - Classic atmospheric model
* ``hdri`` (aliases: environment, envmap) - HDR environment map (requires ``hdr_path``)

Volumetric Parameters
~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Volumetric Fog Configuration
   :header-rows: 1
   :widths: 20 15 15 50

   * - Field
     - Type
     - Default
     - Description
   * - ``density``
     - float
     - ``0.02``
     - Fog density (must be non-negative)
   * - ``phase``
     - string
     - ``"isotropic"``
     - Phase function for light scattering
   * - ``anisotropy``
     - float
     - ``0.0``
     - HG anisotropy parameter [-0.999, 0.999]
   * - ``mode``
     - string
     - ``"raymarch"``
     - Volumetric rendering mode (raymarch or froxels)

**Valid Phase Functions:**

* ``isotropic`` - Uniform scattering in all directions
* ``henyey-greenstein`` (aliases: hg, henyeygreenstein) - Anisotropic scattering (requires ``anisotropy`` in [-0.999, 0.999])

CLI Examples
------------

Basic Configuration
~~~~~~~~~~~~~~~~~~~

Set BRDF model and shadow technique::

    python examples/terrain_demo.py --brdf toon --shadows pcf

Configure shadow map resolution and cascades::

    python examples/terrain_demo.py --shadow-map-res 4096 --cascades 2

Advanced Lighting
~~~~~~~~~~~~~~~~~

Custom directional light::

    python examples/terrain_demo.py \
        --light "type=directional,dir=0.2,0.8,-0.55,intensity=8,color=1,0.96,0.9"

Multiple lights (repeatable flag)::

    python examples/terrain_demo.py \
        --light "type=directional,dir=0,1,0,intensity=5" \
        --light "type=point,pos=100,50,0,intensity=10,color=1,0.5,0.2"

PCSS Soft Shadows
~~~~~~~~~~~~~~~~~

Enable PCSS with custom parameters::

    python examples/terrain_demo.py \
        --shadows pcss \
        --shadow-map-res 2048 \
        --pcss-blocker-radius 2.0 \
        --pcss-filter-radius 4.0 \
        --shadow-light-size 0.5

Global Illumination
~~~~~~~~~~~~~~~~~~~

Enable IBL with SSAO::

    python examples/terrain_demo.py \
        --gi ibl,ssao \
        --hdr assets/snow_field_4k.hdr

Atmospheric Effects
~~~~~~~~~~~~~~~~~~~

HDRI sky with volumetric fog::

    python examples/terrain_demo.py \
        --sky hdri \
        --hdr assets/sky.hdr \
        --volumetric "density=0.05,phase=hg,g=0.7"

Using Presets
~~~~~~~~~~~~~

Apply preset with overrides::

    python examples/terrain_demo.py \
        --preset outdoor_sun \
        --cascades 4 \
        --exposure 1.5 \
        --shadow-map-res 4096

Complete Example
~~~~~~~~~~~~~~~~

Full-featured render with all options::

    python examples/terrain_demo.py \
        --preset studio_pbr \
        --brdf cooktorrance-ggx \
        --shadows pcss \
        --shadow-map-res 4096 \
        --cascades 2 \
        --pcss-blocker-radius 1.5 \
        --gi ibl,ssao,ssr \
        --sky hdri \
        --hdr assets/snow_field_4k.hdr \
        --volumetric "density=0.02,phase=hg,g=0.6" \
        --exposure 1.2 \
        --size 2560 1440

Python API Examples
-------------------

Flat Keyword Arguments
~~~~~~~~~~~~~~~~~~~~~~

::

    import forge3d as f3d

    # Create renderer with flat overrides
    renderer = f3d.Renderer(
        1920, 1080,
        brdf="toon",
        shadows="pcf",
        cascades=2,
        gi=["ibl", "ssao"],
        hdr="assets/sky.hdr"
    )

    # Get normalized config
    config = renderer.get_config()
    print(f"BRDF: {config['shading']['brdf']}")
    print(f"Shadows: {config['shadows']['technique']}")
    print(f"GI modes: {config['gi']['modes']}")

Nested Configuration
~~~~~~~~~~~~~~~~~~~~

::

    import forge3d as f3d

    # Create renderer with nested config dict
    renderer = f3d.Renderer(
        1920, 1080,
        config={
            "lighting": {
                "exposure": 1.5,
                "lights": [
                    {
                        "type": "directional",
                        "direction": [0.2, 0.8, -0.55],
                        "intensity": 8.0,
                        "color": [1.0, 0.96, 0.9]
                    }
                ]
            },
            "shading": {
                "brdf": "cooktorrance-ggx",
                "metallic": 0.8,
                "roughness": 0.2
            },
            "shadows": {
                "technique": "pcss",
                "map_size": 4096,
                "cascades": 2,
                "light_size": 0.5
            },
            "gi": {
                "modes": ["ibl", "ssao"]
            },
            "atmosphere": {
                "sky": "hdri",
                "hdr_path": "assets/sky.hdr"
            }
        }
    )

Mixed Configuration
~~~~~~~~~~~~~~~~~~~

::

    import forge3d as f3d

    # Load base config from dict
    base_config = {
        "shading": {"brdf": "lambert"},
        "shadows": {"technique": "pcf"}
    }

    # Override specific parameters with kwargs
    renderer = f3d.Renderer(
        1920, 1080,
        config=base_config,
        cascades=4,           # Flat override
        gi=["ssao"],          # Flat override
        exposure=1.2          # Flat override
    )

    config = renderer.get_config()
    assert config["shading"]["brdf"] == "lambert"  # From base_config
    assert config["shadows"]["cascades"] == 4       # From override
    assert config["gi"]["modes"] == ["ssao"]        # From override

Loading from JSON
~~~~~~~~~~~~~~~~~

::

    import forge3d as f3d

    # Load config from JSON file
    renderer = f3d.Renderer(
        1920, 1080,
        config="renderer_config.json"
    )

    # Optionally override specific fields
    renderer = f3d.Renderer(
        1920, 1080,
        config="renderer_config.json",
        brdf="toon",          # Override from file
        cascades=2            # Override from file
    )

Using RendererConfig Directly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    import forge3d as f3d
    from forge3d.config import RendererConfig, load_renderer_config

    # Create config programmatically
    config = RendererConfig()
    config.shading.brdf = "toon"
    config.shadows.technique = "pcss"
    config.shadows.map_size = 4096
    config.gi.modes = ["ibl", "ssao"]

    # Validate and use
    config.validate()
    renderer = f3d.Renderer(1920, 1080, config=config)

    # Or use the loader with overrides
    config = load_renderer_config(
        None,  # Start with defaults
        {
            "brdf": "ggx",
            "shadows": "pcf",
            "cascades": 2,
            "gi": ["ibl", "ssao"]
        }
    )
    renderer = f3d.Renderer(1920, 1080, config=config)

Runtime Configuration Changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    import forge3d as f3d

    renderer = f3d.Renderer(1920, 1080)

    # Update lights at runtime
    renderer.set_lights([
        {
            "type": "directional",
            "direction": [0, -1, 0],
            "intensity": 5.0
        },
        {
            "type": "point",
            "position": [100, 50, 0],
            "intensity": 10.0
        }
    ])

    # Apply preset with overrides
    renderer.apply_preset("outdoor_sun", cascades=2)

    # Check current config
    config = renderer.get_config()
    print(f"Current config: {config}")

Common Patterns
---------------

Toon Shading Setup
~~~~~~~~~~~~~~~~~~

::

    renderer = f3d.Renderer(
        1920, 1080,
        brdf="toon",
        shadows="hard",
        gi=[]  # Disable GI for stylized look
    )

Studio Lighting
~~~~~~~~~~~~~~~

::

    renderer = f3d.Renderer(
        1920, 1080,
        config={
            "lighting": {
                "lights": [
                    {"type": "directional", "direction": [0.3, 0.8, -0.5], "intensity": 6.0},
                    {"type": "directional", "direction": [-0.5, 0.3, 0.8], "intensity": 3.0, "color": [0.5, 0.7, 1.0]},
                    {"type": "point", "position": [0, 100, 0], "intensity": 2.0}
                ]
            },
            "shadows": {"technique": "pcss", "map_size": 2048}
        }
    )

High-Quality Outdoor Scene
~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    renderer = f3d.Renderer(
        2560, 1440,
        brdf="cooktorrance-ggx",
        shadows="pcss",
        shadow_map_res=4096,
        cascades=4,
        gi=["ibl", "ssao", "ssr"],
        sky="hdri",
        hdr="assets/outdoor_4k.hdr",
        exposure=1.2
    )

Performance Mode
~~~~~~~~~~~~~~~~

::

    renderer = f3d.Renderer(
        1280, 720,
        brdf="lambert",
        shadows="pcf",
        shadow_map_res=1024,
        cascades=2,
        gi=[]  # Disable GI for performance
    )

Validation and Error Handling
------------------------------

Common Validation Errors
~~~~~~~~~~~~~~~~~~~~~~~~

**Power-of-two shadow map**::

    # ERROR: map_size must be power of two
    renderer = f3d.Renderer(1920, 1080, shadow_map_res=1000)
    # ValueError: shadows.map_size must be a power of two

**CSM requires multiple cascades**::

    # ERROR: CSM needs cascades >= 2
    renderer = f3d.Renderer(1920, 1080, shadows="csm", cascades=1)
    # ValueError: shadows.cascades must be >= 2 when using cascaded shadow maps

**Missing HDR for HDRI sky**::

    # ERROR: HDRI sky needs HDR path
    renderer = f3d.Renderer(1920, 1080, sky="hdri")
    # ValueError: atmosphere.sky=hdri requires atmosphere.hdr_path or environment light with hdr_path

**IBL without environment**::

    # ERROR: IBL needs HDR source
    renderer = f3d.Renderer(1920, 1080, gi=["ibl"])
    # ValueError: gi mode 'ibl' requires either an environment light or atmosphere.hdr_path

Catching Validation Errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    import forge3d as f3d
    from forge3d.config import load_renderer_config

    try:
        config = load_renderer_config(None, {
            "shadows": "csm",
            "cascades": 1  # Invalid for CSM
        })
    except ValueError as e:
        print(f"Configuration error: {e}")
        # Fix and retry
        config = load_renderer_config(None, {
            "shadows": "csm",
            "cascades": 2  # Valid
        })

    renderer = f3d.Renderer(1920, 1080, config=config)

Terrain & DEM Helpers
---------------------

The following helpers make it easier to go from a raster DEM to a terrain render
using the same building blocks as :mod:`examples/terrain_demo.py`:

* **Domain inference and robust ranges** (:mod:`forge3d.io`)::

      import forge3d as f3d
      from forge3d import io

      dem = io.load_dem("assets/Gore_Range_Albers_1m.tif")

      # Prefer metadata when available, otherwise fall back to a safe default
      domain_meta = io.infer_dem_domain(dem, fallback=(200.0, 2200.0))

      # Clamp to robust percentiles to ignore extreme outliers
      domain = io.robust_dem_domain(
          dem.data,
          q_lo=0.02,
          q_hi=0.98,
          fallback=domain_meta,
      )

* **DEM-aware colormap stops** (:mod:`forge3d.colormaps.core`)::

      from forge3d.colormaps.core import (
          interpolate_hex_colors,
          elevation_stops_from_hex_colors,
      )

      # Custom terrain palette in hex, optionally interpolated to many stops
      hex_colors = ["#e7d8a2", "#c5a06e", "#995f57", "#4a3c37"]
      hex_dense = interpolate_hex_colors(hex_colors, size=1024)

      # Place colors across the DEM elevation range (optionally quantile-based)
      stops = elevation_stops_from_hex_colors(
          domain,
          hex_dense,
          heightmap=dem.data,
          q_lo=0.02,
          q_hi=0.98,
      )

      # Convert to a Colormap1D used by the terrain renderer
      cm1d = f3d.Colormap1D.from_stops(stops=stops, domain=domain)

* **Terrain parameter config and height-curve LUT** (:mod:`forge3d.terrain_params`)::

      from forge3d.terrain_params import load_height_curve_lut, make_terrain_params_config

      lut = load_height_curve_lut("assets/height_curve.npy")  # 256 values in [0, 1]

      params_cfg = make_terrain_params_config(
          size_px=(1920, 1080),
          render_scale=1.0,
          msaa_samples=4,
          z_scale=2.0,
          exposure=1.0,
          domain=domain,
          albedo_mode="mix",
          colormap_strength=0.5,
          ibl_enabled=True,
          light_azimuth_deg=135.0,
          light_elevation_deg=35.0,
          sun_intensity=3.0,
          ibl_intensity=1.0,
          cam_radius=1000.0,
          cam_phi_deg=135.0,
          cam_theta_deg=45.0,
          height_curve_mode="smoothstep",
          height_curve_strength=0.6,
          height_curve_power=1.0,
          height_curve_lut=lut,
          overlays=[
              f3d.OverlayLayer.from_colormap1d(
                  cm1d,
                  strength=1.0,
                  offset=0.0,
                  blend_mode="Alpha",
                  domain=domain,
              )
          ],
      )

* **DEM water mask and sun direction** (:mod:`forge3d.render`, :mod:`forge3d.lighting`)::

      from forge3d.render import detect_dem_water_mask
      from forge3d.lighting import sun_direction_from_angles

      # Water detection in DEM space (normalized height + slope heuristic)
      water_mask = detect_dem_water_mask(
          dem.data,
          domain,
          level_normalized=0.35,
          slope_threshold=0.02,
          spacing=dem.resolution,
      )

      # Convert sun azimuth/elevation to a unit direction vector
      sun_dir = sun_direction_from_angles(azimuth_deg=135.0, elevation_deg=35.0)

See Also
--------

* :doc:`presets_overview` - Pre-configured rendering presets
* :doc:`../pbr_materials` - PBR material system documentation
* :doc:`../shadow_mapping` - Shadow mapping techniques
* :doc:`path_tracing` - Path tracing features
* :doc:`../examples/lighting_gallery` - Lighting examples
* :doc:`../examples/shadow_gallery` - Shadow technique comparisons
* :doc:`../examples/ibl_gallery` - IBL environment examples
