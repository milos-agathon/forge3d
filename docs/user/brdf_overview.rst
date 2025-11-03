BRDF Models Overview
=====================

This guide provides a comprehensive overview of the Bidirectional Reflectance Distribution Functions (BRDFs) available in forge3d. BRDFs determine how light reflects off surfaces and are fundamental to achieving realistic or stylized material appearance.

.. contents:: Table of Contents
   :local:
   :depth: 2

What is a BRDF?
---------------

A BRDF (Bidirectional Reflectance Distribution Function) describes how light reflects from a surface. It takes the incoming light direction and viewing direction as input and outputs the reflected light intensity. Different BRDF models approximate real-world material behavior with varying complexity and physical accuracy.

**Key Concepts:**

- **Diffuse reflection**: Light scatters equally in all directions (matte surfaces)
- **Specular reflection**: Light reflects in a mirror-like manner (shiny surfaces)
- **Microfacet theory**: Models surfaces as collections of microscopic mirrors
- **Energy conservation**: Realistic BRDFs ensure no more light is reflected than received

Choosing a BRDF Model
----------------------

forge3d supports 10 BRDF models, ranging from simple diffuse to advanced physically-based models:

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Model
     - Type
     - Performance
     - Best For
   * - Lambert
     - Diffuse
     - ★★★★★
     - Matte surfaces, fast rendering
   * - Phong
     - Empirical
     - ★★★★☆
     - Legacy content, simple specular
   * - Blinn-Phong
     - Empirical
     - ★★★★☆
     - Fast specular, real-time games
   * - Oren-Nayar
     - Diffuse
     - ★★★☆☆
     - Rough diffuse, clay, sand
   * - Cook-Torrance GGX
     - PBR
     - ★★★☆☆
     - Industry-standard PBR
   * - Disney Principled
     - PBR
     - ★★☆☆☆
     - Advanced PBR, film/production
   * - Toon
     - Stylized
     - ★★★★☆
     - Cel-shaded, cartoon rendering
   * - Minnaert
     - Diffuse
     - ★★★☆☆
     - Lunar surfaces, retroreflective
   * - Ward
     - Anisotropic
     - ★★☆☆☆
     - Brushed metal, hair, fabrics
   * - Ashikhmin-Shirley
     - Anisotropic
     - ★★☆☆☆
     - Advanced anisotropic materials

BRDF Model Details
------------------

Lambert
~~~~~~~

**Type**: Diffuse-only  
**Use cases**: Matte surfaces, chalk, unglazed clay, paper

Lambert is the simplest BRDF model, providing uniform diffuse reflection in all directions. It has no specular highlights and is the fastest to compute.

**Characteristics:**

- Perfect diffuse reflection (no specular)
- View-independent (looks the same from all angles)
- Energy conserving
- No parameters beyond base color

**When to use:**

- Fast preview rendering
- Matte materials without shine
- Stylized rendering without highlights
- Performance-critical applications

**Example:**

.. code-block:: python

   from forge3d.config import RendererConfig
   
   config = RendererConfig()
   config.shading.brdf = "lambert"
   # Alternatively, use global override
   config.brdf_override = "lambert"

Phong
~~~~~

**Type**: Empirical specular  
**Use cases**: Legacy content, simple shiny surfaces

Phong is a classic empirical BRDF with simple specular highlights. It's not physically accurate but computationally efficient.

**Characteristics:**

- Diffuse + specular components
- Specular lobe controlled by shininess exponent
- View-dependent highlights
- Not energy conserving

**Parameters:**

- **Roughness**: Controls specular sharpness (mapped from roughness to shininess)
- **Metallic**: Controls diffuse/specular balance

**When to use:**

- Quick visualization
- Legacy pipeline compatibility
- Simple shiny materials
- Artistic control over highlight size

**Limitations:**

- Not physically accurate
- No Fresnel effect
- Energy gain at grazing angles

Blinn-Phong
~~~~~~~~~~~

**Type**: Empirical specular  
**Use cases**: Real-time games, fast specular highlights

Blinn-Phong is a variation of Phong using the halfway vector, providing faster computation and more physically plausible highlights.

**Characteristics:**

- Similar to Phong but uses halfway vector
- Slightly more physically accurate
- Faster computation on some hardware
- Still not energy conserving

**When to use:**

- Real-time rendering
- Mobile platforms
- When Phong is too soft
- Fast preview rendering

Oren-Nayar
~~~~~~~~~~

**Type**: Rough diffuse  
**Use cases**: Clay, sand, rough concrete, moon surfaces

Oren-Nayar extends Lambert to model rough diffuse surfaces with view-dependent behavior. It accounts for surface roughness in diffuse reflection.

**Characteristics:**

- View-dependent diffuse reflection
- Brightens at grazing angles for rough surfaces
- More physically accurate than Lambert
- No specular component

**Parameters:**

- **Roughness**: Surface microsurface roughness (0 = Lambert, 1 = very rough)

**When to use:**

- Matte materials with visible texture
- Natural surfaces (sand, soil, clay)
- Fabric and cloth
- Retroreflective materials

Cook-Torrance GGX
~~~~~~~~~~~~~~~~~

**Type**: Physically-based (PBR)  
**Use cases**: Industry-standard PBR, realistic materials

Cook-Torrance with GGX distribution is the industry-standard PBR BRDF. It's physically accurate, energy conserving, and widely used in film and games.

**Characteristics:**

- Microfacet-based specular
- GGX (Trowbridge-Reitz) distribution
- Fresnel reflectance
- Energy conserving
- Physical accuracy

**Parameters:**

- **Base color**: Surface albedo
- **Metallic**: Metallic factor (0 = dielectric, 1 = metal)
- **Roughness**: Surface roughness (0 = mirror, 1 = rough)

**When to use:**

- Realistic rendering
- PBR workflow
- Production rendering
- Material authoring
- Most general-purpose rendering

**Material examples:**

- Polished metal (high metallic, low roughness)
- Plastic (low metallic, medium roughness)
- Worn metal (high metallic, high roughness)
- Glass (low metallic, very low roughness)

Disney Principled
~~~~~~~~~~~~~~~~~

**Type**: Physically-based (extended PBR)  
**Use cases**: Film production, advanced materials, complex surfaces

Disney Principled BRDF extends standard PBR with additional parameters for artistic control, including subsurface scattering, sheen, and clearcoat.

**Characteristics:**

- Extended PBR with artistic parameters
- Subsurface scattering approximation
- Sheen for fabric/velvet
- Clearcoat for layered materials
- Anisotropic reflections
- Physically plausible

**Additional parameters beyond GGX:**

- **Subsurface**: Subsurface scattering approximation
- **Specular**: Specular intensity control
- **Specular tint**: Tint specular with base color
- **Anisotropic**: Anisotropic reflection strength
- **Sheen**: Fabric/velvet sheen
- **Sheen tint**: Sheen color control
- **Clearcoat**: Additional clear layer
- **Clearcoat gloss**: Clearcoat roughness

**When to use:**

- Film and animation production
- Complex layered materials
- Fabric and cloth with sheen
- Car paint with clearcoat
- Artistic control over PBR

**Performance note:**

Disney Principled is more expensive than standard GGX. Use GGX if these advanced features aren't needed.

Toon
~~~~

**Type**: Stylized  
**Use cases**: Cartoon rendering, cel-shading, anime, NPR

Toon shading provides discrete lighting bands for a cel-shaded cartoon appearance.

**Characteristics:**

- Discrete lighting steps (bands)
- Hard transitions between light and dark
- Optional specular highlights
- Non-photorealistic

**Parameters:**

- **Roughness**: Controls specular band size

**When to use:**

- Cartoon/anime aesthetics
- Non-photorealistic rendering (NPR)
- Stylized games
- Illustrated look

**Artistic notes:**

Combine with rim lighting and outline rendering for classic cel-shaded appearance.

Minnaert
~~~~~~~~

**Type**: Limb-darkening diffuse  
**Use cases**: Lunar surfaces, retroreflective materials, velvet

Minnaert is a specialized diffuse model with limb darkening (or brightening), originally developed for astronomical rendering.

**Characteristics:**

- View-dependent diffuse
- Limb darkening effect
- Can be retroreflective
- No specular component

**Parameters:**

- **Roughness**: Limb darkening exponent (0.5 = Lambert, <0.5 = brightens at edges, >0.5 = darkens at edges)

**When to use:**

- Lunar and planetary surfaces
- Velvet and retroreflective materials
- Specialized astronomical rendering

Ward
~~~~

**Type**: Anisotropic specular  
**Use cases**: Brushed metal, hair, satin, grooved surfaces

Ward BRDF models anisotropic reflections where highlights stretch along surface grooves or fiber directions.

**Characteristics:**

- Anisotropic specular highlights
- Directional roughness
- Physically plausible
- Common in production rendering

**Parameters:**

- **Roughness**: Overall surface roughness
- **Anisotropy**: Anisotropic stretch (requires tangent vectors)

**When to use:**

- Brushed/polished metal
- Hair and fur
- Satin and brushed fabrics
- CDs and vinyl records
- Grooved or scratched surfaces

**Requirements:**

Requires tangent/bitangent vectors for anisotropy direction.

Ashikhmin-Shirley
~~~~~~~~~~~~~~~~~

**Type**: Anisotropic PBR  
**Use cases**: Advanced anisotropic materials, production rendering

Ashikhmin-Shirley is a more sophisticated anisotropic BRDF with improved energy conservation and physical accuracy.

**Characteristics:**

- Physically-based anisotropic specular
- Better energy conservation than Ward
- Separate diffuse and specular components
- Fresnel reflectance

**Parameters:**

- **Roughness**: Surface roughness
- **Anisotropy**: Anisotropic stretch factor

**When to use:**

- High-quality anisotropic materials
- Production rendering
- When Ward artifacts are visible
- Physically accurate anisotropic materials

Visual Comparison
-----------------

The following images demonstrate visual differences between key BRDF models:

.. figure:: ../../tests/golden/p2/lambert_sphere_256.png
   :width: 200px
   :align: left
   :alt: Lambert BRDF rendering
   
   Lambert: Flat diffuse with no specular highlights

.. figure:: ../../tests/golden/p2/ggx_sphere_256.png
   :width: 200px
   :align: left
   :alt: Cook-Torrance GGX BRDF rendering
   
   GGX: Physically-based specular highlights

.. figure:: ../../tests/golden/p2/disney_sphere_256.png
   :width: 200px
   :align: left
   :alt: Disney Principled BRDF rendering
   
   Disney: Extended PBR with advanced parameters

.. raw:: html

   <div style="clear: both;"></div>

Notice how:

- **Lambert** has uniform brightness with no highlights
- **GGX** shows realistic specular highlights with microfacet behavior
- **Disney** provides additional complexity with extended parameters

Configuration
-------------

Setting BRDF via Python API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Per-material BRDF:**

.. code-block:: python

   from forge3d.config import RendererConfig
   
   config = RendererConfig()
   config.shading.brdf = "cooktorrance-ggx"  # Material default

**Global BRDF override:**

.. code-block:: python

   config = RendererConfig()
   config.shading.brdf = "cooktorrance-ggx"  # Material setting
   config.brdf_override = "lambert"           # Override for ALL materials
   
   # Renderer will use Lambert regardless of material settings

**Available BRDF names:**

- ``"lambert"``
- ``"phong"``
- ``"blinn-phong"``
- ``"oren-nayar"``
- ``"cooktorrance-ggx"``
- ``"disney-principled"``
- ``"toon"``
- ``"minnaert"``
- ``"ward"``
- ``"ashikhmin-shirley"``

Override Precedence
~~~~~~~~~~~~~~~~~~~

The BRDF selection follows this precedence:

1. **Global override** (``config.brdf_override``) - Highest priority
2. **Material setting** (``config.shading.brdf``) - Fallback

This allows you to:

- Set per-material BRDFs for normal rendering
- Quickly test different BRDF models by overriding globally
- Compare BRDF models side-by-side

**Example workflow:**

.. code-block:: python

   # Base configuration with GGX
   config = RendererConfig()
   config.shading.brdf = "cooktorrance-ggx"
   
   # Compare different BRDFs
   for brdf in ["lambert", "cooktorrance-ggx", "disney-principled"]:
       test_config = config.copy()
       test_config.brdf_override = brdf
       # Render and compare...

Setting BRDF via Config File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**JSON configuration:**

.. code-block:: json

   {
     "shading": {
       "brdf": "cooktorrance-ggx"
     },
     "brdf_override": "lambert"
   }

Performance Considerations
--------------------------

BRDF Complexity
~~~~~~~~~~~~~~~

BRDFs have different computational costs:

**Fast (< 10 instructions):**

- Lambert
- Phong
- Blinn-Phong
- Toon

**Medium (10-30 instructions):**

- Oren-Nayar
- Cook-Torrance GGX
- Minnaert

**Complex (30+ instructions):**

- Disney Principled
- Ward
- Ashikhmin-Shirley

Optimization Tips
~~~~~~~~~~~~~~~~~

1. **Use Lambert for distant objects** - Diffuse-only for far LODs
2. **Prefer GGX over Disney** - Use Disney only when needed
3. **Batch by BRDF** - Sort draw calls by BRDF to minimize shader changes
4. **Profile your scene** - Measure actual GPU cost

GPU Shader Dispatch
~~~~~~~~~~~~~~~~~~~

forge3d uses dynamic BRDF dispatch, allowing runtime BRDF selection without shader recompilation. Each BRDF is a separate shader module loaded on demand.

**Benefits:**

- Switch BRDFs without recompilation
- Small per-frame overhead
- Flexible material system

Specialized Rendering
---------------------

Terrain Rendering
~~~~~~~~~~~~~~~~~

For terrain-specific shading with slope-based material blending and specialized BRDFs, see:

:doc:`../terrain_rendering`

The terrain renderer supports:

- Multi-layer material blending
- Slope-based texture selection
- Heightmap-driven shading
- Specialized terrain BRDFs

Path Tracing
~~~~~~~~~~~~

When using path tracing mode, BRDFs are evaluated using Monte Carlo integration for physically accurate global illumination. See:

:doc:`path_tracing`

Best Practices
--------------

Material Authoring
~~~~~~~~~~~~~~~~~~

1. **Start with GGX** - Industry standard for PBR
2. **Use metallic properly** - 0 or 1 for physical materials, not in-between
3. **Roughness range** - Most real materials are 0.2-0.8 roughness
4. **Energy conservation** - Ensure metallic + diffuse ≤ 1.0

Testing and Validation
~~~~~~~~~~~~~~~~~~~~~~

1. **Test under different lighting** - Verify materials under various light conditions
2. **Compare reference images** - Use golden images for regression testing
3. **Check edge cases** - Test at roughness = 0 and 1
4. **Validate energy** - No material should be brighter than incoming light

Debugging
~~~~~~~~~

If materials don't look right:

1. **Check BRDF selection** - Verify correct model is active
2. **Inspect parameters** - Ensure roughness/metallic in valid range
3. **Test with simple lighting** - Single directional light for debugging
4. **Compare with Lambert** - Does diffuse-only look correct?

See Also
--------

- :doc:`../pbr_materials` - PBR material system
- :doc:`lights_overview` - Lighting system
- :doc:`presets_overview` - Preset configurations
- :doc:`../terrain_rendering` - Terrain-specific shading
- :doc:`path_tracing` - Path tracing with BRDFs

References
----------

**Academic Papers:**

- Cook & Torrance (1982) - "A Reflectance Model for Computer Graphics"
- Oren & Nayar (1994) - "Generalization of Lambert's Reflectance Model"
- Walter et al. (2007) - "Microfacet Models for Refraction through Rough Surfaces" (GGX)
- Burley (2012) - "Physically-Based Shading at Disney"

**Further Reading:**

- `PBR Guide <https://learnopengl.com/PBR/Theory>`_ - OpenGL PBR tutorial
- `Disney BRDF Explorer <https://github.com/wdas/brdf>`_ - Interactive BRDF visualization
- `Filament Documentation <https://google.github.io/filament/Filament.html>`_ - Google's PBR renderer docs
