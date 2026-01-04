<!-- docs/api/atmospherics_p6.md -->
<!-- Design skeleton for P6 Atmospherics & Sky (Hosek–Wilkie/Preetham sky, volumetric fog, god-rays). -->
<!-- RELEVANT FILES:
  - Specs: p6.md, todo-6.md, AGENTS.md
  - WGSL: src/shaders/sky.wgsl, src/shaders/volumetric.wgsl, src/shaders/fog_upsample.wgsl
  - Rust core: src/lighting/mod.rs, src/lighting/types.rs, src/lighting/shadow_map.rs,
               src/render/params.rs, src/core/mod.rs, src/viewer/mod.rs
  - Python: python/forge3d/config.py, python/forge3d/lighting.py
  - Tests (initial): tests/test_media_fog.py, tests/test_media_hg.py,
                     tests/test_p6_sky.rs, tests/test_p6_fog.rs
-->

# P6 – Atmospherics & Sky (Design Skeleton)

This note captures the **intended architecture and data flow** for P6 sky and volumetric media in `forge3d`.
It is the reference for subsequent milestones; behavior is ultimately defined by tests and `p6.md`.

Goals from `p6.md` / `todo-6.md`:

- Physical sky (`Hosek–Wilkie`, `Preetham`) driven by the **sun directional light**.
- Volumetric single-scattering fog with **Henyey–Greenstein** phase.
- God-rays (volumetric shadows) with jitter + temporal reprojection.
- Configurable via renderer config / Python / CLI, without breaking existing paths.

No new behavior is introduced by this document; it describes **current wiring** and **planned tests**.

---

## 1. High-Level Architecture

### 1.1 Components

- **Sky compute pass**
  - WGSL: `src/shaders/sky.wgsl` (`cs_render_sky`).
  - Host: `src/viewer/mod.rs` sky resources (`SkyUniforms`, `sky_params`, `sky_camera`, `sky_output`).
  - Output: full-resolution `sky_output` texture (`Rgba8Unorm`).

- **Volumetric fog & god-rays**
  - WGSL: `src/shaders/volumetric.wgsl`:
    - `cs_volumetric` – view-ray marching path.
    - `cs_build_froxels` / `cs_apply_froxels` – froxelized alternative.
  - Host: `src/viewer/mod.rs` fog resources (`VolumetricUniformsStd140`, `FogCameraUniforms`, fog output/history, froxel 3D texture, samplers).
  - Output: `fog_output` (`Rgba16Float`, full-res) and optional froxel volume.

- **Fog upsample/composite helper**
  - WGSL: `src/shaders/fog_upsample.wgsl` (`cs_main`).
  - Used when fog is computed at half resolution and upsampled with optional bilateral depth-aware filter.

- **Lighting & atmospherics types** (GPU-aligned)
  - `src/lighting/types.rs`:
    - `SkyModel` (Off / Preetham / Hosek–Wilkie).
    - `SkySettings` (sun direction, turbidity, ground albedo, model, sun intensity, exposure).
    - `VolumetricPhase` (Isotropic / Henyey–Greenstein).
    - `VolumetricSettings` (density, height falloff, phase g, max steps, near/far, absorption,
      scattering and ambient colors, sun intensity, temporal alpha, use_shadows, jitter, phase function).
    - `AtmosphericsSettings { sky: Option<SkySettings>, volumetric: Option<VolumetricSettings> }`.

- **Renderer config & Python facade**
  - Rust: `src/render/params.rs`:
    - Enums: `SkyModel`, `VolumetricPhase`, `VolumetricMode`.
    - Structs: `VolumetricParams`, `AtmosphereParams`, `RendererConfig`.
  - Python: `python/forge3d/config.py`:
    - Dataclasses: `VolumetricParams`, `AtmosphereParams`, `RendererConfig`.
    - `load_renderer_config` / `split_renderer_overrides` map user / CLI kwargs into nested config.

- **Shadow maps for god-rays**
  - `src/lighting/shadow_map.rs`:
    - `ShadowMap` (depth texture + sampler, memory budgeting).
    - `ShadowMatrixCalculator` (directional / spot light matrices).
  - Volume shader sees:
    - `@group(1)` `shadow_map`, `shadow_sampler`, `shadow_matrix`.

---

## 2. Render Pipeline Integration

### 2.1 Viewer render order (interactive path)

`src/viewer/mod.rs::Viewer::render` integrates P6 as follows:

1. **Sky compute (optional)**
   - If `self.sky_enabled`:
     - Build camera matrices (view, proj, inv_view, inv_proj) from `CameraController` and `view_config`.
     - Upload to `sky_camera` uniform buffer.
     - Compute sun direction in **world space** by transforming a view-space sun vector through `inv_view`.
     - Pack `SkyUniforms` (sun direction, `sky_model_id`, `sky_turbidity`, `sky_ground_albedo`,
       `sky_sun_intensity`, `sky_exposure`) into `sky_params`.
     - Dispatch `cs_render_sky` over full framebuffer to write `sky_output`.

2. **Geometry + GBuffer**
   - Build GBuffer using existing GI pipeline (`ScreenSpaceEffectsManager`).
   - Depth for GI (`gi.gbuffer().depth_view`) and main depth (`self.z_view`) are populated.

3. **Screen-space GI / SSR / compositing**
   - Existing GI passes (AO/SSGI/SSR) operate on the GBuffer.
   - Composition compute pass receives:
     - Sky color from `sky_output_view`.
     - Lit color / GI buffers.
     - Optional fog buffer (`fog_view`, see below).
     - Depth.
   - Final result is written to `lit_output` and then to the swapchain.

4. **Volumetric fog compute (optional)**
   - If `self.fog_enabled` after geometry and depth:
     - Build `FogCameraUniforms` with view/proj, inverses, view-proj, eye position, near/far.
     - Pack `VolumetricUniformsStd140` from viewer controls:
       - `fog_density`, `fog_g`, `fog_steps`, `fog_temporal_alpha`, `fog_use_shadows`,
         `fog_half_res_enabled`, fixed defaults for height falloff, absorption, colors.
       - Sun direction from camera-space template transformed into world space.
     - Bind depth, shadow, and history textures via `fog_bgl0`, `fog_bgl1`, `fog_bgl2`.
     - Dispatch either:
       - `cs_volumetric` (full-res ray-march), or
       - Froxel path (`cs_build_froxels` then `cs_apply_froxels`) depending on chosen mode.
     - Result stored in `fog_output` (full or half res) and history updated.

5. **Fog upsample & final composite**
   - If half-res fog is enabled, `cs_main` in `fog_upsample.wgsl` upsamples to full-res fog using depth-aware weights.
   - Composite shader mixes sky, lit color, and fog into the final HDR buffer.

This design keeps **sky/fog as separate compute passes** feeding into the existing compositing path,
without changing terrain, PBR, or GI semantics.

---

## 3. Configuration Types and Mapping

### 3.1 Rust `RendererConfig` (`src/render/params.rs`)

Relevant enums and structs:

- `SkyModel` (config-serializable):
  - `HosekWilkie`, `Preetham`, `Hdri`.

- `VolumetricPhase`:
  - `Isotropic`, `HenyeyGreenstein`.

- `VolumetricMode`:
  - `Raymarch`, `Froxels`.

- `VolumetricParams`:
  - `density: f32` – base fog density.
  - `phase: VolumetricPhase` – scattering phase function.
  - `anisotropy: f32` – HG `g` parameter.
  - `mode: VolumetricMode` – ray-march vs froxels.
  - `height_falloff: f32` – exponential falloff with height.
  - `max_steps: u32` – integration steps.
  - `start_distance / max_distance: f32` – integration range.
  - `absorption: f32` – extinction.
  - `scattering_color: [f32; 3]` – fog tint.
  - `ambient_color: [f32; 3]` – ambient sky.
  - `temporal_alpha: f32` – reprojection weight.
  - `use_shadows: bool` – enable god-rays.
  - `jitter_strength: f32` – per-frame jitter.

- `AtmosphereParams`:
  - `enabled: bool` – global toggle.
  - `sky: SkyModel` – analytic or HDRI.
  - `hdr_path: Option<String>` – environment map for HDRI sky.
  - `volumetric: Option<VolumetricParams>` – enable volumetric fog when present.

- `RendererConfig` includes `atmosphere: AtmosphereParams` alongside lighting/shadows/GI.
  - `RendererConfig::validate` enforces consistency:
    - `sky=hdri` requires either `atmosphere.hdr_path` or an environment light.
    - Volumetric HG phase requires `anisotropy` in [-0.999, 0.999].

### 3.2 Python `RendererConfig` (`python/forge3d/config.py`)

- `_SKY_MODELS` maps strings to `{"hosek-wilkie","preetham","hdri"}`.
- `_PHASE_FUNCTIONS` maps strings to `{"isotropic","henyey-greenstein"}`.

Dataclasses:

- `VolumetricParams`:
  - `density: float = 0.02`.
  - `phase: str = "isotropic"`.
  - `anisotropy: float = 0.0` (also settable via `g` key).
  - `mode: str = "raymarch"`.
  - `preset: Optional[str] = None` (future convenience presets).

- `AtmosphereParams`:
  - `enabled: bool = True`.
  - `sky: str = "hosek-wilkie"`.
  - `hdr_path: Optional[str]`.
  - `volumetric: Optional[VolumetricParams]`.

- `RendererConfig` dataclass mirrors Rust `RendererConfig` fields, and `validate()`
  enforces the same constraints (shadow atlas budget, IBL/HDRI requirements, HG anisotropy range).

### 3.3 CLI-style overrides (intended mapping)

Using `load_renderer_config(config=None, overrides={...})` in Python:

- Sky selection and parameters:
  - `{"sky": "hosek-wilkie"}` → `atmosphere.sky = "hosek-wilkie"`.
  - `{"sky": "preetham"}` → `atmosphere.sky = "preetham"`.

- Volumetric fog example corresponding to
  `--volumetric 'density=0.015,phase=hg,g=0.7,max_steps=48'`:

  ```python
  overrides = {
      "atmosphere": {
          "volumetric": {
              "density": 0.015,
              "phase": "hg",   # mapped to "henyey-greenstein"
              "g": 0.7,         # anisotropy
              "mode": "raymarch",
              "max_steps": 48,
          },
      },
  }
  cfg = load_renderer_config(overrides=overrides)
  ```

- Viewer environment overrides (already wired in `src/viewer/mod.rs`):
  - `FORGE3D_SKY_MODEL` → initial `SkyUniforms.model` (Preetham/Hosek–Wilkie).
  - `FORGE3D_SKY_TURBIDITY`, `FORGE3D_SKY_GROUND`, `FORGE3D_SKY_EXPOSURE`, `FORGE3D_SKY_INTENSITY`
    → initial sky uniform values before viewer UI adjustments.

Later milestones will align viewer controls with the Rust/Python `RendererConfig`
so that P6 behavior is driven consistently from config, not only environment.

---

## 4. Tests and Acceptance Criteria

### 4.1 Existing tests reused for P6

- `tests/test_media_fog.py`
  - `height_fog_factor(d, density)` is monotonic, in [0, 1].
  - `single_scatter_estimate` is non-negative, increasing, and asymptotic.

- `tests/test_media_hg.py`
  - `hg_phase` is normalized over the sphere (Monte Carlo check).
  - `sample_hg` produces directions whose PDF matches `hg_phase`.

- `tests/test_p6_sky.rs`
  - GPU compute test of `cs_render_sky`.
  - Renders a small sky texture at midday vs low sun elevation, compares horizon-band
    `(R - B)` averages; asserts **warmer horizon** at low sun (sunset) than at midday.

- `tests/test_p6_fog.rs`
  - GPU compute test of `cs_volumetric`.
  - Validates:
    - Average fog alpha increases when `density` increases from 0.0 to >0.
    - Temporal smoothing (`temporal_alpha > 0`) reduces inter-frame L1 difference
      relative to a purely stochastic jittered fog.

These tests already connect WGSL media behavior to the analytic helpers in
`python/forge3d/lighting.py` and set the baseline for P6 sky/fog quality.

### 4.2 Planned tests by milestone (design intent)

The following tests are **planned** and referenced here so later milestones
can implement them without re-deriving the plan:

- **Milestone 1 – Sky integration**
  - Extend `tests/test_p6_sky.rs`:
    - Parameter sweep over sun elevation to assert monotonic shift in chromaticity
      (e.g., red/blue ratio trend) from noon to sunset.
  - New Python test `tests/test_p6_sky_config.py`:
    - Round-trip `RendererConfig.atmosphere.sky` ↔ Python `RendererConfig` for
      Hosek–Wilkie vs Preetham vs HDRI.

- **Milestone 2 – Height fog integration**
  - New test `tests/test_p6_fog_config.py`:
    - Validate mapping of `RendererConfig.atmosphere.volumetric` → `VolumetricSettings`
      and then into WGSL `VolumetricParams` (bounds, clamping, defaults).
  - Scene-based Python test (small synthetic depth field):
    - Render with fog off/on; assert depth-dependent attenuation correlation
      (e.g., far pixels have higher fog alpha than near pixels).

- **Milestone 3 – God-rays & temporal reprojection**
  - New Rust test `tests/test_p6_godrays.rs`:
    - Drive a synthetic depth & shadow map where the sun is occluded by a
      simple “ridge”; check that fog scattering is higher in the occluded
      region (visible beams) than in an unshadowed control.
  - Multi-frame Rust test `tests/test_p6_temporal.rs`:
    - Render a short sequence with fixed camera and jittered fog;
      assert temporal reprojection reduces per-pixel temporal variance,
      while keeping mean alpha stable.

- **Milestone 4 – End-to-end viewer acceptance**
  - Python/CLI integration test `tests/test_p6_viewer_sky_fog.py`:
    - Launch headless viewer scene with controlled `RendererConfig`:
      - Sweep sun elevation from sunrise → noon.
      - Enable volumetric fog/god-rays.
      - Capture either AOVs or final frames into golden images.
    - Assert:
      - **Sunrise/sunset color shift** by comparing sky pixel statistics.
      - **Beams visible when sun occluded by terrain** by comparing beam band intensity
        across a depth-masked region.
      - **Temporal stability** via frame-to-frame delta thresholds.

These tests are not yet implemented; they define the **target test surface**
so later P6 milestones can be judged precisely.

---

## 5. Memory & Performance Considerations

- **Sky**:
  - `sky_output` is a single `Rgba8Unorm` 2D texture at viewport resolution.
  - Counts toward host-visible budget only if created in a host-visible heap
    (usual case is GPU-only; memory tracker need not be updated).

- **Volumetric fog**:
  - `VolumetricSettings::froxel_memory_budget()` estimates froxel storage cost
    (typical 16×8×64 grid ≈ 64 KiB at 8 bytes/froxel).
  - Full-res `fog_output` and history are `Rgba16Float` 2D textures; these
    must collectively respect the 512 MiB host-visible budget documented in
    `docs/memory_budget.rst`.

- **Design rule**:
  - Prefer GPU-only allocations for sky/fog textures.
  - If any P6 resources become host-visible, update the memory tracker
    in `src/core/memory_tracker.rs` and extend `tests/test_memory_budget.py`.

---

## 6. Summary

- Sky and volumetric fog are implemented as **separate compute passes** (`sky.wgsl`,
  `volumetric.wgsl`, `fog_upsample.wgsl`) plugged into the viewer’s render loop
  before and after geometry respectively.
- Configuration flows from Python `RendererConfig` → Rust `RendererConfig` →
  lighting/atmospherics GPU structs → WGSL uniforms.
- Existing tests already cover analytic fog/phase behavior and basic GPU sky/fog
  correctness; this document defines the **remaining test plan** for P6 acceptance.
- No behavior is changed by this file; it serves as the design skeleton required
  by **Milestone 0** in `todo-6.md`.
