==================
BRDF Gallery Guide
==================

.. contents:: Table of Contents
   :depth: 3
   :local:

Overview
========

The BRDF (Bidirectional Reflectance Distribution Function) gallery generator creates visual comparisons of different material shading models across varying roughness values. This tool is essential for:

- Understanding material appearance in PBR workflows
- Validating BRDF implementations
- Creating reference images for documentation
- Visual debugging of shading parameters

The gallery uses offscreen GPU rendering to produce deterministic, reproducible images suitable for automated testing and CI pipelines.

Quick Start
===========

Basic Usage
-----------

Generate a default gallery with GGX and Disney models:

.. code-block:: bash

   python examples/brdf_gallery.py

This creates ``brdf_gallery.png`` with a 2×5 grid (2 models × 5 roughness values).

Custom Models
-------------

Specify which BRDF models to compare:

.. code-block:: bash

   # All four models
   python examples/brdf_gallery.py --models lambert,phong,ggx,disney
   
   # Single model
   python examples/brdf_gallery.py --models ggx

Custom Roughness Values
-----------------------

Control the roughness sweep:

.. code-block:: bash

   # More granular sweep
   python examples/brdf_gallery.py --roughness 0.0,0.2,0.4,0.6,0.8,1.0
   
   # Specific values
   python examples/brdf_gallery.py --roughness 0.1,0.5,0.9

BRDF Models
===========

Lambert
-------

**Type:** Diffuse-only model

**Description:**
The Lambert model represents ideal diffuse reflection where light scatters equally in all directions. This is the simplest shading model and serves as a baseline for comparison.

**Characteristics:**

- No specular highlights
- Uniform appearance from all viewing angles
- Roughness parameter has no effect
- Fast computation

**Use cases:**

- Matte materials (chalk, uncoated wood)
- Debugging diffuse-only workflows
- Performance-critical applications

**Model index:** 0

Phong
-----

**Type:** Empirical specular model

**Description:**
The Phong model is a classic computer graphics shading model that adds specular highlights to diffuse reflection. While not physically based, it's computationally efficient and intuitive.

**Characteristics:**

- Sharp, focused specular highlights
- Roughness controls specular power (shininess)
- Not energy-conserving
- Fast computation

**Roughness mapping:**
Specular power = ``(1 - roughness) × 128``

**Use cases:**

- Legacy content compatibility
- Fast preview rendering
- Stylized graphics

**Limitations:**

- Not physically accurate
- Highlights can appear too sharp
- Energy not conserved (breaks at grazing angles)

**Model index:** 1

Cook-Torrance GGX
-----------------

**Type:** Microfacet-based PBR model

**Description:**
GGX (also known as Trowbridge-Reitz) is the industry-standard physically-based shading model. It uses microfacet theory to model surface roughness accurately.

**Characteristics:**

- Physically accurate specular reflection
- Energy-conserving
- Natural-looking highlights with long tails
- Supports metallic workflow

**Roughness mapping:** α = roughness²

Where α is the microfacet distribution width parameter. This provides better perceptual linearity than using roughness directly.

**Normal Distribution Function:**

.. code-block:: text

   D_GGX(h, α) = α² / (π * ((n·h)² * (α² - 1) + 1)²)

**Use cases:**

- Standard PBR workflows
- Realistic material rendering
- Film and game production
- Product visualization

**Model index:** 4

Disney Principled
-----------------

**Type:** Extended PBR model

**Description:**
The Disney Principled BRDF is an artist-friendly extension of the microfacet model, designed by Walt Disney Animation Studios. It includes additional parameters for complex materials.

**Characteristics:**

- Based on GGX but with extensions
- Supports subsurface scattering
- Sheen parameter for fabrics
- Clearcoat for layered materials
- Artist-intuitive parameterization

**Roughness mapping:** α = roughness² (same as GGX)

**Use cases:**

- Character materials (skin, fabric)
- Complex layered surfaces
- Production rendering

**Note:** In the gallery, the basic implementation falls back to GGX for core specular. Full Disney BRDF includes additional parameters not shown in simple tile rendering.

**Model index:** 6

Roughness Parameter
===================

Understanding Roughness
-----------------------

Roughness controls the **microscopic surface irregularities** of a material:

- **Low roughness (0.0-0.3):** Smooth, mirror-like surfaces (polished metal, glass)
- **Medium roughness (0.4-0.6):** Semi-glossy surfaces (painted metal, plastic)
- **High roughness (0.7-1.0):** Matte surfaces (rough concrete, fabric)

Roughness Mapping: α = r²
--------------------------

For physically-based models (GGX, Disney), roughness is **squared** before use in the NDF:

.. code-block:: python

   alpha = roughness * roughness  # α = r²

**Why square it?**

1. **Perceptual linearity:** Changes in roughness slider produce visually uniform steps
2. **Artist-friendly:** Linear roughness maps to exponential specular spread
3. **Standard practice:** Matches Unreal Engine, Unity, and other PBR engines

**Example:**

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Roughness (r)
     - Alpha (α)
     - Visual appearance
   * - 0.1
     - 0.01
     - Very sharp specular, mirror-like
   * - 0.5
     - 0.25
     - Medium glossy, visible lobe
   * - 0.9
     - 0.81
     - Very diffuse, barely visible highlight

Clamping
--------

Roughness values outside [0, 1] are automatically clamped:

.. code-block:: python

   roughness = clamp(roughness, 0.0, 1.0)

- Values < 0 → 0.0 (perfect mirror)
- Values > 1 → 1.0 (fully matte)

NDF-Only Debug Mode
===================

Purpose
-------

NDF-only mode renders **only the Normal Distribution Function** term of the BRDF, useful for:

- Debugging roughness behavior
- Validating NDF implementation
- Understanding specular lobe shapes
- Educational visualization

Activation
----------

.. code-block:: bash

   python examples/brdf_gallery.py --ndf-only

Output
------

When enabled, the shader outputs grayscale values representing the NDF at each pixel.

**Characteristics:**

- Grayscale output (R = G = B)
- Shows only specular lobe shape
- No diffuse component
- No Fresnel term
- No geometric attenuation

Visual Interpretation
---------------------

In NDF-only mode, you can observe:

**Low roughness (0.1-0.3):**

- Sharp, bright center
- Rapid falloff
- Small highlight area

**Medium roughness (0.4-0.6):**

- Broader lobe
- Gentler falloff
- Visible specular shape

**High roughness (0.7-0.9):**

- Wide, diffuse lobe
- Low peak intensity
- Energy spread over large area

Monotonicity
------------

A correct NDF implementation should show **monotonic behavior**:

- **Lobe width increases** with roughness
- **Peak intensity decreases** with roughness

This is validated by the P7-08 unit tests.

Reading GGX/Phong Lobes
========================

**Milestone 6:** Understanding specular lobe characteristics for visual debugging.

Lobe Shape Comparison
---------------------

When comparing BRDF models, understanding their characteristic lobe shapes helps identify implementation issues:

**GGX (Trowbridge-Reitz):**

- **Shape:** Bell-shaped with long, heavy tails
- **Falloff:** Gradual, extends far from peak
- **Peak:** Lower intensity for high roughness
- **Tail behavior:** More energy in periphery than Phong
- **Physical accuracy:** Matches real-world measurements

**Phong:**

- **Shape:** Sharper, more focused lobe
- **Falloff:** Rapid, exponential dropoff
- **Peak:** Higher intensity, narrower width
- **Tail behavior:** Minimal energy beyond lobe
- **Characteristic:** Appears more "specular" even at same roughness

Visual Differences at Same Roughness
-------------------------------------

At **r = 0.5** (medium glossiness):

.. code-block:: bash

   python examples/brdf_gallery.py --models ggx,phong --roughness 0.5 --tile-size 256 256

**Expected observations:**

1. **GGX lobe:**
   
   - Broader, more spread out
   - Visible halo around bright center
   - Smoother falloff to background

2. **Phong lobe:**
   
   - Tighter, more concentrated
   - Sharper boundary between bright and dark
   - Faster transition to black

3. **Width comparison:**
   
   - GGX FWHM (Full Width Half Maximum): typically 30-40% wider
   - Phong falloff: ~2x faster beyond FWHM

Diagnosing Lobe Issues
----------------------

**Problem: Lobes look identical**

*Symptom:* GGX and Phong produce the same visual output

*Cause:* Models may be using the same NDF implementation

*Solution:*

.. code-block:: bash

   # Test with extreme roughness to amplify differences
   python examples/brdf_gallery.py --models ggx,phong --roughness 0.1,0.9 --ndf-only

*Expected:* Clear width difference, especially at r=0.1

**Problem: Lobe is too narrow**

*Symptom:* GGX appears sharper than expected

*Cause:* Alpha may not be squared (using r instead of r²)

*Solution:*

.. code-block:: bash

   # Check roughness visualization
   python examples/brdf_gallery.py --roughness-visualize --roughness 0.5
   # Should show medium gray (128/255), not darker

**Problem: Lobe is too wide**

*Symptom:* Even low roughness appears diffuse

*Cause:* May be double-squaring alpha (using r⁴ instead of r²)

*Solution:* Check shader code for alpha calculation

Lobe Width Analysis
-------------------

Quantitative lobe width analysis using FWHM (Full Width at Half Maximum):

.. list-table::
   :header-rows: 1
   :widths: 20 25 25 30

   * - Roughness
     - GGX Width
     - Phong Width
     - Ratio (GGX/Phong)
   * - 0.1
     - ~15-20 pixels
     - ~10-15 pixels
     - 1.3-1.5×
   * - 0.3
     - ~40-50 pixels
     - ~25-35 pixels
     - 1.4-1.6×
   * - 0.5
     - ~70-85 pixels
     - ~45-60 pixels
     - 1.4-1.5×
   * - 0.7
     - ~110-130 pixels
     - ~70-90 pixels
     - 1.4-1.6×
   * - 0.9
     - ~160-190 pixels
     - ~110-140 pixels
     - 1.4-1.5×

*Note:* Measurements assume 256×256 tile with standard camera/lighting setup.

Practical Lobe Reading Tips
----------------------------

1. **Use NDF-only mode** for clearest lobe visualization:

   .. code-block:: bash

      python examples/brdf_gallery.py --ndf-only --models ggx,phong --roughness 0.3,0.5,0.7

2. **Check monotonicity:** As roughness increases, lobes should:
   
   - Get wider (more pixels above threshold)
   - Get dimmer (lower peak intensity)
   - Maintain smooth falloff (no discontinuities)

3. **Compare horizontally:** Same roughness across models reveals implementation differences

4. **Compare vertically:** Same model across roughness values reveals parameter behavior

5. **Use full BRDF mode** to see how Fresnel and geometry terms modify the NDF

Gallery CLI Reference
=====================

Command Syntax
--------------

.. code-block:: bash

   python examples/brdf_gallery.py [OPTIONS]

Options
-------

``--models``
^^^^^^^^^^^^

Comma-separated list of BRDF models to include.

**Choices:** lambert, phong, ggx, disney

**Default:** ggx,disney

**Example:**

.. code-block:: bash

   python examples/brdf_gallery.py --models lambert,phong,ggx,disney

``--roughness``
^^^^^^^^^^^^^^^

Comma-separated roughness values in [0, 1].

**Default:** 0.1,0.3,0.5,0.7,0.9

**Example:**

.. code-block:: bash

   # Minimal sweep
   python examples/brdf_gallery.py --roughness 0.2,0.8
   
   # Fine-grained sweep
   python examples/brdf_gallery.py --roughness 0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0

``--tile-size``
^^^^^^^^^^^^^^^

Tile dimensions in pixels (width height).

**Default:** 256 256

**Example:**

.. code-block:: bash

   # High resolution
   python examples/brdf_gallery.py --tile-size 512 512
   
   # Quick preview
   python examples/brdf_gallery.py --tile-size 128 128

``--ndf-only``
^^^^^^^^^^^^^^

Enable NDF-only debug mode (grayscale output).

**Example:**

.. code-block:: bash

   python examples/brdf_gallery.py --ndf-only --models ggx --roughness 0.1,0.3,0.5,0.7,0.9

``--g-only``
^^^^^^^^^^^^

**Milestone 0:** Enable G-only debug mode (Smith geometry term as grayscale).

**Example:**

.. code-block:: bash

   python examples/brdf_gallery.py --g-only --models ggx --roughness 0.1,0.5,0.9

``--dfg-only``
^^^^^^^^^^^^^^

**Milestone 0:** Enable DFG-only debug mode (outputs D×F×G product before division).

**Example:**

.. code-block:: bash

   python examples/brdf_gallery.py --dfg-only --models ggx --roughness 0.3,0.7

``--roughness-visualize``
^^^^^^^^^^^^^^^^^^^^^^^^^

**Milestone 0:** Enable roughness visualization (outputs vec3(roughness) to validate parameter flow).

**Example:**

.. code-block:: bash

   python examples/brdf_gallery.py --roughness-visualize --roughness 0.2,0.5,0.8

``--exposure``
^^^^^^^^^^^^^^

**Milestone 4:** Exposure multiplier for final output (default: 1.0). Higher values brighten the image.

**Default:** 1.0

**Example:**

.. code-block:: bash

   # Brighten output
   python examples/brdf_gallery.py --exposure 1.5
   
   # Darken output
   python examples/brdf_gallery.py --exposure 0.7

``--light-intensity``
^^^^^^^^^^^^^^^^^^^^^

**Milestone 4:** Light source intensity (default: 0.8). Lower values prevent clipping at low roughness.

**Default:** 0.8 (tuned to keep peak < 0.95)

**Example:**

.. code-block:: bash

   # Reduce intensity to prevent clipping
   python examples/brdf_gallery.py --light-intensity 0.6 --roughness 0.1,0.2
   
   # Increase for darker materials
   python examples/brdf_gallery.py --light-intensity 1.0

``--out``
^^^^^^^^^

Output file path for the mosaic PNG.

**Default:** brdf_gallery.png

**Example:**

.. code-block:: bash

   python examples/brdf_gallery.py --out reports/brdf_comparison.png

Common Workflows
================

Material Comparison
-------------------

Compare all models at standard roughness values:

.. code-block:: bash

   python examples/brdf_gallery.py \
     --models lambert,phong,ggx,disney \
     --roughness 0.1,0.3,0.5,0.7,0.9 \
     --tile-size 256 256 \
     --out brdf_comparison.png

Result: 4×5 grid (20 tiles) comparing all models.

Roughness Study
---------------

Detailed roughness sweep for GGX:

.. code-block:: bash

   python examples/brdf_gallery.py \
     --models ggx \
     --roughness 0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \
     --tile-size 256 256 \
     --out ggx_roughness_sweep.png

Result: 1×11 grid showing smooth roughness progression.

NDF Validation
--------------

Generate NDF-only reference for testing:

.. code-block:: bash

   python examples/brdf_gallery.py \
     --models ggx,disney \
     --roughness 0.2,0.5,0.8 \
     --tile-size 128 128 \
     --ndf-only \
     --out ndf_validation.png

Result: 2×3 grid showing NDF lobe shapes.

Documentation Assets
--------------------

Generate reference images for documentation:

.. code-block:: bash

   # Full comparison
   python examples/brdf_gallery.py \
     --models lambert,phong,ggx,disney \
     --roughness 0.1,0.5,0.9 \
     --tile-size 256 256 \
     --out docs/images/brdf_full_comparison.png
   
   # GGX focus
   python examples/brdf_gallery.py \
     --models ggx \
     --roughness 0.1,0.3,0.5,0.7,0.9 \
     --tile-size 320 320 \
     --out docs/images/ggx_roughness.png

CI Golden Images
----------------

Generate small reference images for automated testing:

.. code-block:: bash

   python examples/brdf_gallery.py \
     --models ggx,disney,phong \
     --roughness 0.3,0.5,0.7 \
     --tile-size 128 128 \
     --out tests/golden/p7/mosaic_3x3_128.png

Result: Compact 3×3 grid (~400KB) suitable for version control.

Reproducibility
===============

Deterministic Rendering
-----------------------

The gallery generator uses fixed parameters to ensure reproducible output:

**Rendering:**

- Exposure: 1.0 (no adjustment)
- Tone mapping: Disabled
- Gamma: Linear (no correction applied)
- Mesh: UV sphere (64×32 subdivisions)

**Shader configuration:**

- Fixed camera position and lighting
- Deterministic random number generator (if applicable)
- Consistent texture filtering

Cross-GPU Consistency
---------------------

While the renderer is deterministic, minor pixel-level variations may occur across different GPU hardware due to:

- Floating-point precision differences
- Texture filtering implementations
- Driver optimizations

For golden image comparisons, use SSIM/PSNR thresholds to tolerate these minor variations (see P7-09 golden tests).

Reproducibility Tips
--------------------

1. **Use same GPU architecture** for critical comparisons
2. **Record GPU info** when creating reference images:

   .. code-block:: bash

      # Save GPU info with output
      nvidia-smi --query-gpu=name --format=csv,noheader > gpu_info.txt
      python examples/brdf_gallery.py --out gallery.png

3. **Pin driver versions** for long-term reproducibility
4. **Use golden tests** (P7-09) with tolerance thresholds
5. **Commit golden images** to version control for CI comparison

Troubleshooting
===============

Empty or Black Output
---------------------

**Symptoms:** All tiles are black or empty

**Causes:**

- Native module not built with GPU support
- GPU not available or drivers missing
- Mesh generation failure

**Solutions:**

.. code-block:: bash

   # Rebuild native module
   maturin develop --release
   
   # Check GPU availability
   python -c "import forge3d; print(forge3d.has_gpu())"
   
   # Check native function
   python -c "import forge3d._forge3d as f3d; print(hasattr(f3d, 'render_brdf_tile'))"

Incorrect Colors
----------------

**Symptoms:** Colors don't match expected BRDF behavior

**Causes:**

- Tone mapping enabled (should be disabled)
- Incorrect exposure value (should be 1.0)
- Shader implementation error

**Solutions:**

1. Verify exposure=1.0 and tone mapping disabled in shader
2. Compare against reference images
3. Run P7-08 unit tests to validate BRDF implementation

NDF Mode Not Grayscale
-----------------------

**Symptoms:** NDF-only mode shows colors instead of grayscale

**Causes:**

- ndf_only flag not passed correctly
- Shader not respecting ndf_only parameter

**Solutions:**

1. Verify ``--ndf-only`` flag is set
2. Check shader code respects ``ndf_only`` uniform
3. Run P7-08 NDF grayscale test

Performance Issues
------------------

**Symptoms:** Gallery generation is slow

**Solutions:**

1. **Reduce tile size:**

   .. code-block:: bash

      python examples/brdf_gallery.py --tile-size 128 128

2. **Reduce grid size:**

   .. code-block:: bash

      python examples/brdf_gallery.py --models ggx --roughness 0.3,0.7

3. **Check GPU utilization:**

   .. code-block:: bash

      nvidia-smi

Further Reading
===============

- **P7-08 Unit Tests:** Monotonicity validation and BRDF correctness
- **P7-09 Golden Tests:** Cross-GPU comparison with SSIM/PSNR
- **P7 Acceptance:** Manual verification procedures
- **examples/README_brdf_gallery.md:** Additional usage examples

Related APIs
============

Python API
----------

.. code-block:: python

   import forge3d as f3d
   
   # Render single BRDF tile
   tile = f3d.render_brdf_tile(
       model="ggx",
       roughness=0.5,
       width=256,
       height=256,
       ndf_only=False
   )
   
   # tile is numpy array (256, 256, 4) uint8
   
   # Save as PNG
   f3d.numpy_to_png("tile.png", tile)

Rust API
--------

.. code-block:: rust

   use forge3d::offscreen::brdf_tile::render_brdf_tile_offscreen;
   
   let buffer = render_brdf_tile_offscreen(
       device,
       queue,
       model_index,  // 4 for GGX
       roughness,    // 0.5
       width,        // 256
       height,       // 256
       ndf_only,     // false
       exposure,     // 1.0
   )?;
   
   // buffer is Vec<u8> of size height * width * 4

Appendix: BRDF Theory
=====================

Microfacet BRDF
---------------

The Cook-Torrance microfacet BRDF is:

.. code-block:: text

   f(l, v) = D(h) * F(v, h) * G(l, v, h) / (4 * (n·l) * (n·v))

Where:

- **D(h)** = Normal Distribution Function (controls lobe shape)
- **F(v,h)** = Fresnel term (reflection at grazing angles)
- **G(l,v,h)** = Geometric attenuation (self-shadowing)
- **h** = half vector between light and view
- **n** = surface normal

GGX/Trowbridge-Reitz NDF
-------------------------

.. code-block:: text

   D_GGX(h) = α² / (π * ((n·h)² * (α² - 1) + 1)²)

Properties:

- **α = 0:** Perfect mirror (Dirac delta)
- **α → 1:** Wide diffuse lobe
- **Long tail:** More realistic highlights than Blinn-Phong

Perceptual Roughness
--------------------

The mapping α = r² is chosen so that:

- **Linear slider** in r produces perceptually uniform changes
- **Matches** artist expectations from DCC tools
- **Standard** across PBR engines (Unreal, Unity, Substance)

References
----------

- Burley, "Physically Based Shading at Disney" (2012)
- Walter et al., "Microfacet Models for Refraction through Rough Surfaces" (2007)
- Karis, "Real Shading in Unreal Engine 4" (2013)
