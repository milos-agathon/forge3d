Troubleshooting Visual Issues
==============================

This guide provides systematic checklists for debugging visual artifacts,
incorrect rendering, and performance issues in forge3d. Organized by subsystem
for quick reference during QA and development.

.. contents:: Table of Contents
   :local:
   :depth: 2

General Debugging Workflow
---------------------------

1. **Isolate the subsystem**: Disable all features except the one exhibiting issues
2. **Check GPU budget**: Use ``scene.get_stats()`` to verify memory usage < 512 MiB
3. **Validate parameters**: Ensure all input values are within documented ranges
4. **Inspect shaders**: Review WGSL code for the affected pipeline
5. **Compare with golden images**: Run ``cargo test --test golden_images -- --ignored``
6. **Enable logging**: Set ``RUST_LOG=info`` or ``RUST_LOG=debug`` for detailed output

Lighting & Shadows
------------------

Black or Completely Dark Output
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms**: Rendered image is black or nearly black

**Checklist**:

- [ ] Verify light intensity is > 0
- [ ] Check light direction/position is valid (no NaN, no zero-length vectors)
- [ ] Ensure camera is not inside geometry
- [ ] Confirm exposure setting is reasonable (0.5-2.0 for typical scenes)
- [ ] Check that ``Scene`` has terrain or geometry added
- [ ] Verify colormap is loaded correctly (``cmap_name in SUPPORTED``)

**Fix**:

.. code-block:: python

   # Increase light intensity
   scene.set_light_intensity(5.0)

   # Adjust exposure
   scene.set_exposure(1.0)

Shadow Acne or Peter Panning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms**: Moire patterns or objects appearing to float above surfaces

**Checklist**:

- [ ] Increase shadow bias (0.001 - 0.01 range)
- [ ] Reduce slope-scale bias if using PCF/PCSS
- [ ] Check shadow map resolution (2048+ recommended for PCSS)
- [ ] Verify near/far plane distances for shadow frustum
- [ ] Ensure normal maps are applied correctly (if enabled)

**Fix**:

.. code-block:: python

   # Adjust shadow settings
   shadow_settings = forge3d.ShadowSettings(
       bias=0.005,
       normal_bias=0.01,
       slope_scale_bias=1.5
   )

Shadow Map Artifacts (Blocky Shadows)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms**: Shadows appear pixelated or blocky

**Checklist**:

- [ ] Increase shadow map resolution (512 → 1024 → 2048 → 4096)
- [ ] Switch from Hard to PCF or PCSS technique
- [ ] Enable contact hardening for PCSS
- [ ] Verify GPU memory budget allows higher resolution
- [ ] Check for auto-downscaling warnings in logs

**Fix**:

.. code-block:: python

   # Use higher resolution shadow map with soft shadows
   shadow_settings = forge3d.ShadowSettings(
       technique='pcss',
       map_size=2048,
        pcss_blocker_radius=6.0,
        pcss_filter_radius=12.0,
        light_size=0.35,
        moment_bias=0.0005
   )

BRDF & Materials
----------------

Materials Look Unrealistic
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms**: Materials appear too shiny, too dull, or incorrect color

**Checklist**:

- [ ] Verify roughness is in [0.04, 1.0] range (clamped for numerical stability)
- [ ] Check metallic is in [0.0, 1.0] range
- [ ] Ensure base color is in sRGB space (not linear)
- [ ] Confirm BRDF model matches material type:

  - Metals: ``cooktorrance-ggx`` or ``disney-principled``
  - Dielectrics: ``cooktorrance-ggx``, ``oren-nayar``, or ``lambert``
  - Cloth: ``ashikhmin-shirley`` or ``oren-nayar``
  - Stylized: ``toon``

- [ ] Verify normal maps are in tangent space (if using MikkTSpace)

**Fix**:

.. code-block:: python

   # Proper PBR material setup
   material = {
       'brdf': 'disney-principled',
       'base_color': [0.8, 0.8, 0.8],  # sRGB
       'roughness': 0.5,  # [0.04, 1.0]
       'metallic': 0.0,   # 0=dielectric, 1=metal
   }

Specular Highlights Too Bright/Dark
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms**: Incorrect specular reflection intensity

**Checklist**:

- [ ] Check Fresnel F0 value (metals: 0.5-1.0, dielectrics: 0.02-0.08)
- [ ] Verify light intensity is not excessive (< 10.0 typical)
- [ ] Ensure exposure is set correctly
- [ ] For ``cooktorrance-ggx``: verify GGX distribution is used
- [ ] Check that roughness is not too low (< 0.04 causes fireflies)

**Fix**:

.. code-block:: python

   # Clamp roughness to avoid fireflies
   roughness = max(0.04, user_roughness)

Anisotropic Rendering Issues
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms**: Anisotropic highlights appear circular instead of stretched

**Checklist**:

- [ ] Verify tangent vectors are provided (MikkTSpace recommended)
- [ ] Check BRDF model supports anisotropy:

  - ``ashikhmin-shirley``: Yes
  - ``ward``: Yes
  - ``cooktorrance-ggx``: Partial (isotropic only)

- [ ] Ensure anisotropy parameter is in [-1.0, 1.0] range
- [ ] Confirm tangent vectors are orthogonal to normals

Global Illumination (GI)
-------------------------

IBL Not Visible
~~~~~~~~~~~~~~~

**Symptoms**: Image-based lighting has no effect

**Checklist**:

- [ ] Verify HDR environment map is loaded (``hdr_path`` valid)
- [ ] Check IBL intensity > 0 (default: 1.0)
- [ ] Ensure ``gi_modes`` includes ``'ibl'``
- [ ] Confirm GPU memory budget allows IBL cubemap (check auto-downscaling)
- [ ] Verify BRDF LUT texture is generated correctly
- [ ] Check that exposure is not too low

**Fix**:

.. code-block:: python

   # Enable IBL with proper intensity
   ibl_settings = forge3d.IblSettings(
       hdr_path='assets/snow_field_4k.hdr',
       intensity=1.5,
       rotation=0.0
   )

SSAO Too Strong or Too Weak
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms**: Ambient occlusion is too dark, too subtle, or has haloing artifacts

**Checklist**:

- [ ] Adjust SSAO radius (0.5-5.0 range, default: 2.0)
- [ ] Modify intensity (0.0-2.0, default: 1.0)
- [ ] Check bias parameter (0.001-0.1, prevents self-occlusion)
- [ ] Increase sample count for higher quality (16-64)
- [ ] Switch to GTAO for more accurate results
- [ ] Enable bilateral blur to reduce noise
- [ ] Verify depth buffer is available

**Fix**:

.. code-block:: python

   # GTAO with moderate settings
   ssao = forge3d.SSAOSettings.gtao(
       radius=2.0,
       intensity=1.2,
       sample_count=32
   )

SSGI Noisy or Incorrect Indirect Lighting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms**: Screen-space GI shows speckles, flickering, or wrong colors

**Checklist**:

- [ ] Increase temporal accumulation alpha (0.8-0.95 for stability)
- [ ] Raise max ray marching steps (24-48 for quality)
- [ ] Enable IBL fallback for ray misses
- [ ] Use half-resolution mode for performance (with bilateral upsampling)
- [ ] Check that GBuffer pass is enabled
- [ ] Verify depth and normal textures are valid

**Fix**:

.. code-block:: python

   # Stable SSGI with temporal filtering
   ssgi = forge3d.SSGISettings(
       max_steps=32,
       intensity=1.0,
       ibl_fallback=0.3,
       temporal_alpha=0.9
   )

SSR Artifacts (Screen-Edge Cutoff, Stretching)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms**: Reflections disappear at screen edges or appear stretched

**Checklist**:

- [ ] Enable environment map fallback for ray misses
- [ ] Adjust thickness parameter (0.05-0.5 range)
- [ ] Increase hierarchical ray marching steps (32-64)
- [ ] Check Fresnel factor (0.0-1.0, default: 1.0)
- [ ] Verify depth hierarchy mipmap is built correctly
- [ ] Use temporal accumulation to reduce noise

**Fix**:

.. code-block:: python

   # SSR with environment fallback
   ssr = forge3d.SSRSettings(
       max_steps=48,
       thickness=0.1,
       fresnel_factor=1.0,
       env_fallback=0.5
   )

Atmospherics & Sky
------------------

Sky Looks Wrong (Too Bright, Wrong Color)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms**: Sky is unrealistic, wrong time of day, or oversaturated

**Checklist**:

- [ ] Verify sun direction is normalized
- [ ] Check turbidity parameter (2.0=clear, 6.0=hazy, 10.0=very hazy)
- [ ] Adjust ground albedo (0.0-1.0, typical: 0.2-0.3)
- [ ] Switch between Hosek-Wilkie (accurate) and Preetham (classic) models
- [ ] Modify exposure for HDR sky (0.5-2.0 range)
- [ ] Ensure sun intensity is reasonable (1.0-5.0)

**Fix**:

.. code-block:: python

   # Clear sky at noon
   sky = forge3d.SkySettings.hosek_wilkie(
       sun_direction=[0.3, 0.8, -0.5],  # normalized
       turbidity=2.0,
       ground_albedo=0.25
   )

Volumetric Fog Not Visible
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms**: Fog has no effect or is too subtle

**Checklist**:

- [ ] Increase density (0.01-0.1 range)
- [ ] Raise number of ray marching steps (32-64)
- [ ] Check fog color matches scene lighting
- [ ] Verify height falloff is appropriate
- [ ] Enable god-rays if sun is occluded by terrain
- [ ] Ensure temporal alpha is not too high (causes lag)

**Fix**:

.. code-block:: python

   # Visible uniform fog
   fog = forge3d.VolumetricSettings.uniform_fog(
       density=0.05,
       color=[0.8, 0.85, 0.9],
       num_steps=48
   )

God-Rays Not Rendering
~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms**: Volumetric shadows (light shafts) are missing

**Checklist**:

- [ ] Verify directional light tagged as ``'sun'``
- [ ] Ensure shadow maps are enabled
- [ ] Check that sun is partially occluded by geometry
- [ ] Increase volumetric step count (48-128)
- [ ] Adjust scattering phase function (Henyey-Greenstein, g=0.6-0.9)
- [ ] Confirm shadow map resolution is sufficient (1024+)

**Fix**:

.. code-block:: python

   # God-rays with forward scattering
   god_rays = forge3d.VolumetricSettings.with_god_rays(
       density=0.02,
       num_steps=64,
       phase='henyey-greenstein',
       anisotropy=0.7  # forward scattering
   )

Screen-Space Effects
--------------------

GBuffer Artifacts
~~~~~~~~~~~~~~~~~

**Symptoms**: Incorrect depth, normals, or material properties in GBuffer

**Checklist**:

- [ ] Verify depth buffer format is Depth32Float
- [ ] Check view-space normal calculation (normalize, flip if needed)
- [ ] Ensure material IDs are correctly assigned
- [ ] Confirm roughness/metallic values are in [0, 1]
- [ ] Verify depth linearization for orthographic/perspective cameras

Performance & Memory
--------------------

Out of Memory (OOM) Errors
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms**: GPU allocation fails, crashes, or auto-downscaling messages

**Checklist**:

- [ ] Check ``scene.get_stats()`` memory usage
- [ ] Reduce shadow map resolution (4096 → 2048 → 1024)
- [ ] Lower IBL cubemap resolution (512 → 256 → 128)
- [ ] Decrease froxel grid dimensions (if using volumetrics)
- [ ] Disable screen-space effects (SSAO, SSGI, SSR) if not needed
- [ ] Verify budget is set correctly (``--gpu-budget-mib 512``)

**Fix**:

.. code-block:: python

   # Check memory before rendering
   stats = scene.get_stats()
   print(f"GPU memory: {stats['gpu_memory_mb']:.1f} / {stats['gpu_memory_budget_mb']:.0f} MiB")

   if stats['gpu_utilization'] > 0.9:
       print("Warning: Approaching memory budget limit")

Slow Rendering (>90s for 640×360)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms**: Rendering takes too long, CI times out

**Checklist**:

- [ ] Reduce shadow map resolution
- [ ] Disable expensive GI modes (SSGI, SSR)
- [ ] Lower ray marching step counts (SSAO, SSGI, volumetrics)
- [ ] Use half-resolution screen-space effects
- [ ] Disable volumetric fog if not needed
- [ ] Switch from PCSS to PCF shadows
- [ ] Reduce MSAA sample count (4x → 2x → 1x)

**Fix**:

.. code-block:: python

   # Fast preview settings
   preset = forge3d.presets.minimal()  # Lambert + single light + no shadows

Platform-Specific Issues
-------------------------

Windows (MSVC)
~~~~~~~~~~~~~~

- [ ] Ensure Visual Studio Build Tools are installed
- [ ] Check DX12/Vulkan backend availability (``WGPU_BACKENDS=DX12``)
- [ ] Verify GPU drivers are up to date
- [ ] Test with ``--release`` build (debug can be very slow)

Linux (GNU/GCC)
~~~~~~~~~~~~~~~

- [ ] Install Vulkan development packages (``libvulkan-dev``)
- [ ] Check ``VK_ICD_FILENAMES`` environment variable
- [ ] Verify Mesa/proprietary drivers are compatible
- [ ] Test with ``WGPU_BACKENDS=VULKAN``

macOS (Clang)
~~~~~~~~~~~~~

- [ ] Ensure Metal backend is used (``WGPU_BACKENDS=METAL``)
- [ ] Check macOS version ≥ 10.15 (Catalina)
- [ ] Verify universal2 wheel is used for Apple Silicon
- [ ] Test on both Intel and ARM64 architectures

Shader Compilation Errors
--------------------------

WGSL Validation Failures
~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms**: Shader fails to compile or validate

**Checklist**:

- [ ] Check WGSL syntax (naga validator is strict)
- [ ] Verify uniform buffer alignment (16-byte boundaries)
- [ ] Ensure struct sizes are multiples of 16 bytes
- [ ] Confirm bind group layouts match shader bindings
- [ ] Run ``cargo test --test test_shader_params_p5_p8`` for packing tests

Spirv-Cross Errors
~~~~~~~~~~~~~~~~~~

**Symptoms**: SPIR-V translation fails on some backends

**Checklist**:

- [ ] Avoid non-portable WGSL features
- [ ] Test on multiple backends (Vulkan, DX12, Metal)
- [ ] Check wgpu version compatibility
- [ ] Verify target Vulkan version is 1.2+

Validation & Testing
--------------------

Golden Image Test Failures
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms**: SSIM < 0.98 compared to reference images

**Checklist**:

- [ ] Regenerate golden images if shaders changed intentionally
- [ ] Check for platform-specific rendering differences
- [ ] Inspect diff images in ``tests/golden/diff/``
- [ ] Verify test resolution is 1280×920
- [ ] Ensure deterministic random number generation (fixed seed)

CI Failures
~~~~~~~~~~~

**Symptoms**: Tests pass locally but fail in CI

**Checklist**:

- [ ] Check CI runner has GPU access (Windows/Linux only)
- [ ] Verify all system dependencies are installed
- [ ] Ensure tests run serially (``--test-threads=1``)
- [ ] Check for race conditions in GPU resource allocation
- [ ] Review CI logs for specific error messages
- [ ] Test with same OS/Python/Rust versions as CI

Getting Help
------------

If you've exhausted all troubleshooting steps:

1. **File a GitHub Issue**: Include:
   - Platform (OS, GPU, driver version)
   - Minimal reproduction code
   - Output of ``scene.get_stats()``
   - Rendered image (if visual artifact)
   - ``RUST_LOG=debug`` output

2. **Check Documentation**:
   - ``docs/index.rst`` - Main documentation
   - ``CHANGELOG.md`` - Recent changes
   - ``CLAUDE.md`` - Repository overview

3. **Run Diagnostics**:

   .. code-block:: bash

      python examples/diagnostics.py
      cargo test --workspace --all-features -- --nocapture

References
----------

- `WGSL Specification <https://www.w3.org/TR/WGSL/>`_
- `wgpu Documentation <https://wgpu.rs/>`_
- `PBR Guide <https://learnopengl.com/PBR/Theory>`_
- `Real-Time Rendering, 4th ed. <http://www.realtimerendering.com/>`_
