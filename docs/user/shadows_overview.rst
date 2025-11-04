Shadow System Overview
======================

The shadow system in forge3d provides high-quality real-time shadows using cascaded shadow maps (CSM) with multiple filtering techniques. This document covers shadow techniques, configuration, troubleshooting, and performance optimization.

.. contents:: Table of Contents
   :depth: 3
   :local:

Shadow Techniques
-----------------

forge3d supports six shadow techniques with varying quality and performance characteristics:

Hard Shadows
~~~~~~~~~~~~

**Quality**: ★☆☆ | **Performance**: ★★★

Hard shadows use a single depth comparison per pixel, producing sharp shadow boundaries with no softening.

**Visual Characteristics**:
- Sharp, aliased edges
- No penumbra (transition region)
- Binary visibility (fully lit or fully shadowed)
- Suitable for stylized or retro aesthetics

**Use Cases**:
- Performance-critical applications
- Stylized rendering (toon shading, pixel art)
- Distant shadows where softness isn't visible
- Debugging shadow coverage

**Example**::

    python examples/terrain_demo.py \
        --dem assets/terrain.tif \
        --shadows hard

PCF (Percentage Closer Filtering)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Quality**: ★★☆ | **Performance**: ★★☆

PCF applies a fixed-size filter kernel to the shadow map, averaging multiple depth comparisons to produce soft edges.

**Visual Characteristics**:
- Softened shadow boundaries
- Fixed penumbra width (independent of distance)
- Reduced aliasing compared to Hard
- Industry-standard filtering quality

**Technical Details**:
- Filter kernel: 3×3 or 5×5 Poisson disk samples
- Comparison samples: 9-25 per pixel
- Constant softness regardless of occluder distance

**Use Cases**:
- General-purpose shadows
- Good quality/performance balance
- Consistent shadow appearance
- Real-time applications with moderate GPU budget

**Example**::

    python examples/terrain_demo.py \
        --dem assets/terrain.tif \
        --shadows pcf

PCSS (Percentage Closer Soft Shadows)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Quality**: ★★★ | **Performance**: ★☆☆

PCSS estimates penumbra size based on occluder distance, producing contact-hardening shadows with variable softness.

**Visual Characteristics**:
- Variable penumbra width (distance-dependent)
- Contact-hardening: hard near surfaces, soft farther away
- Realistic soft shadows
- Highest visual quality

**Technical Details**:
- **Blocker search**: Find average occluder depth in search region
- **Penumbra estimation**: Calculate filter size based on light size and distance
- **PCF filtering**: Apply variable-width filter
- **Parameters**:
  - ``pcss_blocker_radius``: Search radius for occluders (default: 0.05)
  - ``pcss_filter_radius``: Maximum filter radius (default: 0.08)
  - ``light_size``: Effective light source size (default: 0.5)

**Use Cases**:
- Cinematic rendering
- Architectural visualization
- Close-up scenes where shadow quality is critical
- Applications with GPU budget for high-quality shadows

**Example**::

    python examples/terrain_demo.py \
        --dem assets/terrain.tif \
        --shadows pcss \
        --pcss-blocker-radius 0.05 \
        --pcss-filter-radius 0.08 \
        --light-size 0.5

VSM (Variance Shadow Maps)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Quality**: ★★☆ | **Performance**: ★★☆

VSM stores depth mean and variance in moment maps, enabling pre-filtered shadows with Chebyshev's inequality for visibility estimation.

**Visual Characteristics**:
- Soft shadows with pre-filtering
- Reduced aliasing
- Potential light leaking in high-contrast scenes
- Suitable for scenes with moderate depth complexity

**Technical Details**:
- Stores two moments: E[x], E[x²]
- Uses Chebyshev's inequality: P(x ≥ t) ≤ σ²/(σ² + (t - μ)²)
- Requires moment atlas (RGBA32Float)
- Memory: Depth (4 bytes) + Moments (16 bytes per cascade)

**Use Cases**:
- Scenes with low depth complexity
- Applications requiring pre-filtered shadows
- When light leaking is acceptable

**Example**::

    python examples/terrain_demo.py \
        --dem assets/terrain.tif \
        --shadows vsm \
        --moment-bias 0.0005

EVSM (Exponential Variance Shadow Maps)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Quality**: ★★☆ | **Performance**: ★☆☆

EVSM applies exponential warping to depth values, reducing light leaking compared to VSM at the cost of precision.

**Visual Characteristics**:
- Softer shadows than VSM
- Reduced light leaking
- Requires careful bias tuning
- Better for complex scenes than VSM

**Technical Details**:
- Stores four moments: positive and negative exponents
- Warp exponents: typically c₁ = 40, c₂ = 5
- Requires RGBA32Float moment atlas
- Higher memory usage than VSM

**Use Cases**:
- Complex scenes where VSM leaks too much
- Applications requiring pre-filtered soft shadows
- When PCSS is too expensive

**Example**::

    python examples/terrain_demo.py \
        --dem assets/terrain.tif \
        --shadows evsm \
        --moment-bias 0.0005

MSM (Moment Shadow Maps)
~~~~~~~~~~~~~~~~~~~~~~~~~

**Quality**: ★★★ | **Performance**: ★☆☆

MSM stores higher-order moments (up to 4th) for more accurate visibility estimation with minimal light leaking.

**Visual Characteristics**:
- Highest quality among moment-based techniques
- Minimal light leaking
- Soft, artifact-free shadows
- Most expensive moment technique

**Technical Details**:
- Stores four moments: E[x], E[x²], E[x³], E[x⁴]
- Uses moment reconstruction for visibility
- Requires RGBA32Float moment atlas
- Highest computational cost

**Use Cases**:
- High-end cinematic rendering
- Scenes where light leaking is unacceptable
- Applications with sufficient GPU budget

**Example**::

    python examples/terrain_demo.py \
        --dem assets/terrain.tif \
        --shadows msm \
        --moment-bias 0.0005

Technique Comparison
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 15 10 10 15 20 30

   * - Technique
     - Quality
     - Perf
     - Memory
     - Penumbra
     - Best For
   * - **Hard**
     - ★☆☆
     - ★★★
     - Depth only
     - None (sharp)
     - Performance, stylized
   * - **PCF**
     - ★★☆
     - ★★☆
     - Depth only
     - Fixed width
     - General purpose
   * - **PCSS**
     - ★★★
     - ★☆☆
     - Depth only
     - Variable
     - High quality, realism
   * - **VSM**
     - ★★☆
     - ★★☆
     - Depth + moments
     - Pre-filtered
     - Simple scenes
   * - **EVSM**
     - ★★☆
     - ★☆☆
     - Depth + moments
     - Pre-filtered
     - Complex scenes
   * - **MSM**
     - ★★★
     - ★☆☆
     - Depth + moments
     - Pre-filtered
     - Cinematic quality

Cascaded Shadow Maps (CSM)
---------------------------

forge3d uses cascaded shadow maps to provide high shadow resolution near the camera while covering large distances.

How CSM Works
~~~~~~~~~~~~~

The camera frustum is split into multiple cascades, each with its own shadow map:

1. **Split Generation**: Frustum divided into 2-4 cascades
2. **Independent Rendering**: Each cascade rendered separately
3. **Runtime Selection**: Pixel selects appropriate cascade based on view depth
4. **Blending**: Optional cross-fade at cascade boundaries

**Benefits**:
- High resolution near camera (close shadows crisp)
- Low resolution far from camera (distant shadows still present)
- Efficient use of shadow map resolution
- Scalable to large view distances

Cascade Configuration
~~~~~~~~~~~~~~~~~~~~~

**Cascade Count**:
- Default: 3 cascades
- Range: 1-4 cascades
- More cascades = better resolution distribution
- More cascades = higher overhead

**Split Scheme**:
- Uses practical split with λ = 0.75
- Blends logarithmic (detail near) and uniform (coverage far)
- Formula: ``split[i] = λ × log_split + (1 - λ) × uniform_split``

**Example**: 3 cascades, near=0.1, far=200.0

.. code-block:: text

   Cascade 0: [0.10, 10.50] - High detail, 10.4 units
   Cascade 1: [10.50, 52.75] - Medium detail, 42.25 units
   Cascade 2: [52.75, 200.00] - Low detail, 147.25 units

Cascade Stabilization
~~~~~~~~~~~~~~~~~~~~~~

Cascade stabilization prevents shadow shimmering when the camera moves:

**Without Stabilization**:
- Shadow map texels move with camera
- Creates flickering/crawling artifacts
- Noticeable in PCF/PCSS where edges are softened

**With Stabilization**:
- Shadow map snapped to world texel grid
- Texels remain stable during camera movement
- Slight resolution reduction for stability

**Implementation**:
- Calculate world-space texel size per cascade
- Snap view-projection matrix to texel-aligned grid
- Enables smooth camera motion without shimmer

**Configuration**::

    # Stabilization enabled by default in CsmConfig
    stabilize_cascades: true

Atlas Budget and Memory
-----------------------

Shadow maps consume GPU memory that must be managed within a budget.

Memory Budget
~~~~~~~~~~~~~

**Default Budget**: 256 MiB

**Memory Components**:

1. **Depth Atlas**:
   - Format: Depth32Float (4 bytes per texel)
   - Memory: ``cascade_count × resolution² × 4 bytes``

2. **Moment Atlas** (VSM/EVSM/MSM only):
   - Format: RGBA32Float (16 bytes per texel)
   - Memory: ``cascade_count × resolution² × 16 bytes``

**Total Memory**::

    memory = depth_atlas + moment_atlas (if applicable)
    
    # Hard/PCF/PCSS (depth only)
    memory = cascade_count × resolution² × 4
    
    # VSM/EVSM/MSM (depth + moments)
    memory = cascade_count × resolution² × (4 + 16)

Memory Calculations
~~~~~~~~~~~~~~~~~~~

**Example 1**: PCF, 3 cascades, 2048×2048

.. code-block:: text

   Depth = 3 × 2048² × 4 = 48 MiB
   Moment = 0 (PCF doesn't use moments)
   Total = 48 MiB ✓ (within 256 MiB budget)

**Example 2**: EVSM, 4 cascades, 2048×2048

.. code-block:: text

   Depth = 4 × 2048² × 4 = 64 MiB
   Moment = 4 × 2048² × 16 = 256 MiB
   Total = 320 MiB ✗ (exceeds budget)
   → System downscales to 1536×1536 automatically

**Example 3**: PCSS, 2 cascades, 4096×4096

.. code-block:: text

   Depth = 2 × 4096² × 4 = 128 MiB
   Moment = 0
   Total = 128 MiB ✓

Budget Enforcement
~~~~~~~~~~~~~~~~~~

When configuration exceeds budget:

1. **Downscaling**: Resolution reduced automatically
2. **Warning Logged**: System logs new resolution
3. **Stable**: Single-step downscale (no thrashing)

**Example Log**::

    [WARN] Shadow map memory (320 MiB) exceeds budget (256 MiB)
    [INFO] Downscaling shadow map: 2048 → 1536

Resolution Guidelines
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 40

   * - Use Case
     - Cascades
     - Resolution
     - Memory
   * - Performance
     - 2
     - 1024
     - 8 MiB (depth only)
   * - Balanced
     - 3
     - 2048
     - 48 MiB (depth only)
   * - Quality
     - 4
     - 2048
     - 64 MiB (depth only)
   * - Cinematic
     - 4
     - 4096
     - 256 MiB (depth only)

CLI Flags
---------

Shadow Configuration
~~~~~~~~~~~~~~~~~~~~

Basic Options
^^^^^^^^^^^^^

``--shadows <technique>``
   Shadow filtering technique.
   
   **Choices**: ``none``, ``hard``, ``pcf``, ``pcss``, ``vsm``, ``evsm``, ``msm``
   
   **Default**: ``none``
   
   **Example**::
   
       --shadows pcss

``--shadow-map-res <size>``
   Shadow map resolution per cascade.
   
   **Range**: 256-8192 (power of 2 recommended)
   
   **Default**: 2048
   
   **Example**::
   
       --shadow-map-res 4096

``--cascades <count>``
   Number of CSM cascades.
   
   **Range**: 1-4
   
   **Default**: 3
   
   **Example**::
   
       --cascades 4

PCSS Parameters
^^^^^^^^^^^^^^^

``--pcss-blocker-radius <radius>``
   Search radius for blocker detection (world units).
   
   **Range**: 0.01-1.0
   
   **Default**: 0.05
   
   **Effect**: Larger = more aggressive penumbra estimation
   
   **Example**::
   
       --pcss-blocker-radius 0.08

``--pcss-filter-radius <radius>``
   Maximum PCF filter radius (world units).
   
   **Range**: 0.01-2.0
   
   **Default**: 0.08
   
   **Effect**: Larger = softer shadows (up to clamp limit)
   
   **Example**::
   
       --pcss-filter-radius 0.1

``--light-size <size>``
   Effective light source size (affects penumbra width).
   
   **Range**: 0.1-10.0
   
   **Default**: 0.5
   
   **Effect**: Larger = wider penumbrae
   
   **Example**::
   
       --light-size 1.0

Moment Technique Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``--moment-bias <bias>``
   Bias for moment-based techniques (VSM/EVSM/MSM).
   
   **Range**: 0.00001-0.01
   
   **Default**: 0.0005
   
   **Effect**: Reduces light leaking, increases peter-panning
   
   **Example**::
   
       --moment-bias 0.001

Complete Example
~~~~~~~~~~~~~~~~

High-quality PCSS shadows with 4 cascades::

    python examples/terrain_demo.py \
        --dem assets/Gore_Range_Albers_1m.tif \
        --shadows pcss \
        --shadow-map-res 2048 \
        --cascades 4 \
        --pcss-blocker-radius 0.05 \
        --pcss-filter-radius 0.08 \
        --light-size 0.5

Balanced PCF shadows::

    python examples/terrain_demo.py \
        --dem assets/terrain.tif \
        --shadows pcf \
        --shadow-map-res 2048 \
        --cascades 3

Performance-optimized hard shadows::

    python examples/terrain_demo.py \
        --dem assets/terrain.tif \
        --shadows hard \
        --shadow-map-res 1024 \
        --cascades 2

Troubleshooting
---------------

Shadow Acne
~~~~~~~~~~~

**Symptom**: Moiré patterns, sparkles, or self-shadowing artifacts on surfaces.

**Cause**: Shadow map depth precision insufficient for surface depth test.

**Solutions**:

1. **Increase Depth Bias** (in code, not CLI):
   
   .. code-block:: rust
   
      config.csm.depth_bias = 0.005; // Increase from 0.001

2. **Increase Slope Bias**:
   
   .. code-block:: rust
   
      config.csm.slope_bias = 0.01; // Increase from 0.005

3. **Higher Resolution**: More precision per texel
   
   ::
   
       --shadow-map-res 4096

4. **More Cascades**: Better resolution distribution
   
   ::
   
       --cascades 4

**Visual Example**:

.. code-block:: text

   Before (acne):        After (fixed):
   ▓░▓░▓░▓░▓            ▓▓▓▓▓▓▓▓▓
   ░▓░▓░▓░▓░            ▓▓▓▓▓▓▓▓▓
   (sparkly)            (smooth)

Peter-Panning
~~~~~~~~~~~~~

**Symptom**: Shadows detached from objects (floating shadows).

**Cause**: Excessive depth bias pushes shadow casters away from surfaces.

**Solutions**:

1. **Decrease Depth Bias**:
   
   .. code-block:: rust
   
      config.csm.depth_bias = 0.001; // Decrease from 0.005

2. **Decrease Peter-Panning Offset**:
   
   .. code-block:: rust
   
      config.csm.peter_panning_offset = 0.001; // Decrease

3. **Balance Trade-off**: Find minimum bias that avoids acne

**Visual Example**:

.. code-block:: text

   Peter-panning:        Correct:
   
   ████                  ████
     ▓▓▓▓               ▓▓▓▓▓▓
   (gap)                (contact)

Light Leaking (VSM/EVSM/MSM)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom**: Light bleeding through occluders in moment-based techniques.

**Cause**: Variance underestimation or high depth complexity.

**Solutions**:

1. **Increase Moment Bias**:
   
   ::
   
       --moment-bias 0.001

2. **Switch to EVSM/MSM**: Better than VSM for complex scenes

3. **Use PCSS Instead**: No light leaking with depth-based techniques

Cascade Transitions
~~~~~~~~~~~~~~~~~~~

**Symptom**: Visible seams or popping at cascade boundaries.

**Cause**: Resolution mismatch or abrupt cascade selection.

**Solutions**:

1. **Enable Cascade Blending** (in code):
   
   .. code-block:: rust
   
      config.csm.cascade_blend_range = 0.1; // Blend 10% of range

2. **More Cascades**: Smoother transitions
   
   ::
   
       --cascades 4

3. **Higher Resolution**: Less visible difference between cascades

Shadow Shimmering
~~~~~~~~~~~~~~~~~

**Symptom**: Shadows flicker or crawl when camera moves.

**Cause**: Shadow map texels moving with camera.

**Solution**: Cascade stabilization (enabled by default)

.. code-block:: rust

   config.csm.stabilize_cascades = true;

Performance Tips
----------------

Optimization Strategies
~~~~~~~~~~~~~~~~~~~~~~~

1. **Choose Appropriate Technique**
   
   - **Performance-critical**: Hard (fastest)
   - **Balanced**: PCF (good quality/perf)
   - **Quality-critical**: PCSS (best looking)

2. **Resolution Tuning**
   
   - Start with 2048, adjust based on profiling
   - Lower resolution for distant/unimportant shadows
   - Higher resolution for close-up/hero objects

3. **Cascade Count**
   
   - Use 2 cascades for close-range scenes
   - Use 3 cascades for general outdoor scenes
   - Use 4 cascades only for vast distances

4. **Reduce Shadow Distance**
   
   Shorter shadow distance = less area to cover = better resolution
   
   .. code-block:: rust
   
      config.csm.max_distance = 100.0; // Instead of 1000.0

Performance Targets
~~~~~~~~~~~~~~~~~~~

Based on P3-14 benchmarks (Apple M3 Max):

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 40

   * - Configuration
     - Update Time
     - Target Use
     - Notes
   * - Hard, 2048, 2 cascades
     - <0.01 ms
     - 60+ FPS
     - Negligible overhead
   * - PCF, 2048, 3 cascades
     - <0.01 ms
     - 60+ FPS
     - Negligible overhead
   * - PCSS, 2048, 3 cascades
     - <0.01 ms
     - 60+ FPS
     - Negligible overhead

**Note**: Update time only (CPU-side). Full shadow rendering (GPU) adds 2-4 ms.

**Frame Budget**: Shadows should use <4 ms of 16.6 ms frame budget (24%).

Memory Optimization
~~~~~~~~~~~~~~~~~~~

1. **Use Depth-Only Techniques**
   
   Hard/PCF/PCSS use 4× less memory than moment techniques::
   
       PCF:  48 MiB (3×2048², depth only)
       EVSM: 240 MiB (3×2048², depth+moments)

2. **Reduce Cascade Count**
   
   Halving cascades halves memory::
   
       4 cascades: 64 MiB
       2 cascades: 32 MiB

3. **Lower Resolution**
   
   Halving resolution quartersMemory::
   
       2048: 48 MiB
       1024: 12 MiB

Quality/Performance Trade-offs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Scenario
     - Fast (>120 FPS)
     - Quality (30-60 FPS)
   * - **Technique**
     - Hard or PCF
     - PCSS or MSM
   * - **Resolution**
     - 1024-1536
     - 2048-4096
   * - **Cascades**
     - 2
     - 3-4
   * - **Memory**
     - 8-24 MiB
     - 48-256 MiB

Best Practices
~~~~~~~~~~~~~~

1. **Profile First**: Measure before optimizing
2. **Start Simple**: Begin with PCF, 2048, 3 cascades
3. **Iterate**: Adjust based on visual quality needs
4. **Test on Target Hardware**: Performance varies by GPU
5. **Monitor Memory**: Stay within 256 MiB budget

Advanced Topics
---------------

Shadow Map Filtering
~~~~~~~~~~~~~~~~~~~~

**Depth Sampling**: Shadow maps use ``sampler_comparison`` for hardware PCF.

**Moment Filtering**: VSM/EVSM/MSM use standard ``sampler`` with linear filtering.

**Poisson Disk Sampling**: PCF/PCSS use Poisson-distributed samples for natural softness.

Bias Configuration
~~~~~~~~~~~~~~~~~~

**Depth Bias**: Constant offset along view direction

.. code-block:: rust

   config.csm.depth_bias = 0.0005; // World units

**Slope Bias**: Angle-dependent offset (steeper = more bias)

.. code-block:: rust

   config.csm.slope_bias = 0.001; // Scale factor

**Peter-Panning Offset**: Additional offset to improve contact

.. code-block:: rust

   config.csm.peter_panning_offset = 0.002; // World units

PCSS Radius Clamping
~~~~~~~~~~~~~~~~~~~~

PCSS radii are automatically clamped based on cascade texel size:

.. code-block:: text

   max_blocker_radius = min_texel_size × 6.0
   max_filter_radius = min_texel_size × 12.0

This prevents searching beyond reasonable texel neighborhoods.

Further Reading
---------------

- :doc:`brdf_overview` - Material and lighting models that interact with shadows
- :doc:`lights_overview` - Light source configuration affecting shadow casting
- :doc:`rendering_options_overview` - General rendering configuration

**API Documentation**:

- ``forge3d.shadows.ShadowManager`` - Shadow system control
- ``forge3d.lighting.types.ShadowTechnique`` - Technique enumeration
- ``forge3d.shadows.ShadowManagerConfig`` - Configuration options

**Implementation Details**:

- ``src/shadows/manager.rs`` - Shadow manager implementation
- ``src/shadows/csm.rs`` - Cascaded shadow map logic
- ``src/shaders/shadows.wgsl`` - Shadow sampling shaders

**Testing and Validation**:

- ``tests/test_p3_13_shadow_unit_tests.rs`` - Unit tests
- ``tests/test_p3_14_shadow_perf.rs`` - Performance validation
- ``tests/test_p3_15_shadow_goldens.rs`` - Visual quality tests

Acknowledgments
---------------

Shadow system implementation includes:

- Cascaded shadow map stabilization (Michal Valient, GPU Gems)
- PCSS algorithm (NVIDIA, GPU Gems 2)
- Variance shadow maps (Donnelly & Lauritzen, GDC 2006)
- Exponential variance shadow maps (Lauritzen & McCool, I3D 2008)
- Moment shadow maps (Christoph Peters, EGSR 2015)
