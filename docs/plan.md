## Strict plan with deliverables & milestones

### 1. Design decisions

#### 1.1 Global rules

* Global: keep defaults identical (all new features opt-in); budget all new textures against ≤512 MiB host-visible; avoid cross-backend feature requirements; do not add new tools/scripts unless unavoidable; read AGENTS.md before you start working to get familiar with codebase rules.

#### 1.2 Preset schema discipline

* Preset schema discipline: before adding any new CLI/config keys, enumerate preset keys in examples/terrain_demo.py (_apply_json_preset param_map) and config parsing in python/forge3d/config.py::load_renderer_config and python/forge3d/terrain_demo.py overrides; update argparse + preset param_map + CLI override precedence map together; define boolean precedence (--flag, --no-flag, default, preset).

#### 1.3 Feature-specific design notes

* F1 AOVs + OpenEXR: implement in shader pass (MRT or storage writes) plus output layer (PNG/EXR), only when aovs enabled; AOV set must include albedo/normal/depth/roughness/metallic/AO/sun-vis/mask/ID.
* F2 DoF: reuse core/dof compute on HDR color + depth; add optional focus-plane tilt for tilt-shift; keep determinism via analytic CoC and fixed kernel.
* F3 Motion blur: reuse terrain/accumulation.rs for temporal accumulation with shutter interval; deterministic sample sequence tied to seed and sample index.
* F4 Lens effects: add post-process pass after tonemap (LDR) for distortion/CA/vignette; deterministic math only.
* F5 Denoising: keep CPU A-trous (no external services) and add optional GPU denoise for AO/sun-vis/volumetrics; deterministic kernels and guidance AOVs.
* F6 Volumetrics: reuse volumetric.wgsl compute with depth + sun + shadows; allow raymarch/froxels; deterministic jitter sequence.
* F7 Sky + aerial perspective: reuse sky.wgsl compute; integrate with sun/camera and optional IBL generation; tie fog aerial perspective to sky/inscatter colors.

---

### 2. Per-feature implementation spec

#### F1 AOVs + OpenEXR

* F1 AOVs + OpenEXR Integration point(s): src/terrain/renderer.rs (add TerrainAovTargets, new render path or MRT variant), src/terrain/render_params.rs (add AovSettingsNative), src/shaders/terrain_pbr_pom.wgsl (AOV writes), src/renderer/readback.rs (typed readback), src/util/image_write.rs or new src/util/exr_write.rs, src/path_tracing/io.rs::AovWriter::write_exr, python/forge3d/terrain_params.py (AOV dataclass), python/forge3d/terrain_demo.py and examples/terrain_demo.py (CLI/preset keys), python/forge3d/render.py (wire save_aovs);
* GPU resources: per-AOV textures (albedo/normal/roughness+metallic in Rgba16Float, depth in R32Float, AO/sun-vis in R32Float, mask/ID in Rgba8Unorm or R32Uint if supported), optional staging buffers;
* Pass scheduling: write AOVs in main PBR pass (MRT or storage), readback after render/tonemap, no impact when aovs empty;
* Config surface: TerrainRenderParams.aovs, aov_format (png|exr|raw), aov_dir, aov_channels, CLI --aovs, --aov-dir, --aov-format;
* Failure modes: EXR writer unavailable (images feature), memory budget exceeded, unsupported format on backend, missing material IDs—error early with clear message.

#### F2 Depth of Field

* F2 Depth of Field Integration point(s): src/terrain/renderer.rs (instantiate core::dof::DofRenderer), src/terrain/render_params.rs (add DofSettingsNative), src/shaders/dof.wgsl (tilt-shift plane params), python/forge3d/terrain_params.py (DofSettings dataclass), python/forge3d/terrain_demo.py + examples/terrain_demo.py (CLI flags/preset keys);
* GPU resources: HDR color Rgba16Float, depth view, DOF output Rgba16Float, uniforms buffer;
* Pass scheduling: after PBR + volumetric, before tonemap/bloom (or explicitly chosen order, documented);
* Config surface: dof.enabled, dof.f_stop, dof.focus_distance, dof.focal_length, dof.method, dof.quality, dof.tilt_pitch/yaw;
* Failure modes: DOF enabled without depth, invalid focus distance, unsupported sample count.

#### F3 Motion blur

* F3 Motion blur Integration point(s): src/terrain/renderer.rs (temporal loop), src/terrain/accumulation.rs (reuse accumulation + jitter), python/forge3d/terrain_params.py (MotionBlurSettings), src/terrain/render_params.rs (MotionBlurSettingsNative), CLI in python/forge3d/terrain_demo.py/examples/terrain_demo.py;
* GPU resources: existing accumulation texture (Rgba32Float), per-sample color target;
* Pass scheduling: N sub-frames across shutter interval, accumulate, resolve at end;
* Config surface: motion_blur.enabled, samples, shutter_open, shutter_close, camera_path (optional), seed;
* Failure modes: high sample count exceeds time budget; if object motion blur requested without animated transforms, error (no silent fallback).

#### F4 Lens/sensor effects

* F4 Lens/sensor effects Integration point(s): new src/terrain/lens_effects.rs + src/shaders/lens_effects.wgsl, hook in src/terrain/renderer.rs post-tonemap, parse in src/terrain/render_params.rs, add LensEffectsSettings in python/forge3d/terrain_params.py, CLI + preset keys in python/forge3d/terrain_demo.py/examples/terrain_demo.py;
* GPU resources: LDR color input, one LDR output, uniforms;
* Pass scheduling: after tonemap and before final PNG readback;
* Config surface: lens.distortion, lens.chromatic_aberration, lens.vignette_strength, lens.vignette_radius/softness;
* Failure modes: sampling outside bounds, non-finite params, intermediate texture not allocated.

#### F5 Optional denoising

* F5 Optional denoising Integration point(s): python/forge3d/denoise.py (CPU A-trous), python/forge3d/path_tracing.py (wire GPU AOV guidance when available), add GPU denoise pass in src/terrain/renderer.rs (AO/sun-vis/volumetrics), optional helper in src/core/screen_space_effects.rs or new src/terrain/denoise.rs, config in python/forge3d/terrain_params.py + src/terrain/render_params.rs;
* GPU resources: ping-pong textures per buffer, guidance textures (normal/depth/albedo);
* Pass scheduling: denoise AO/sun-vis before shading; denoise volumetric before composite;
* Config surface: denoise.enabled, denoise.method, denoise.iterations, denoise.sigma_*;
* Failure modes: denoise enabled without guidance AOVs, CPU/GPU mismatch, performance regressions.

#### F6 Volumetrics + light shafts

* F6 Volumetrics + light shafts Integration point(s): share viewer fog creation (src/viewer/init/fog_init.rs) by extracting a core::volumetric module, wire into src/terrain/renderer.rs with new pass using src/shaders/volumetric.wgsl; parse AtmosphereParams.volumetric from python/forge3d/config.py into terrain params;
* GPU resources: fog_output, fog_history, optional half-res/froxel textures, shadow map + uniforms;
* Pass scheduling: after depth, before tonemap; composite using existing HDR path;
* Config surface: existing --volumetric string + structured overrides (mode, density, steps, phase, anisotropy, use_shadows);
* Failure modes: use_shadows without CSM, memory budget exceeded, unsupported mode -> hard error.

#### F7 Physically based sky + aerial perspective

* F7 Physically based sky + aerial perspective Integration point(s): reuse src/shaders/sky.wgsl via new core::sky module (mirroring src/viewer/init/sky_init.rs), integrate sky compute into src/terrain/renderer.rs (background replace, optional IBL source), add sky parameters to terrain params;
* GPU resources: sky output texture (Rgba16Float), sky uniform buffers, optional low-res cubemap for IBL;
* Pass scheduling: generate sky before main render, composite where depth is far;
* Config surface: atmosphere.sky, sky_turbidity, sky_ground_albedo, sky_exposure, sky_sun_intensity;
* Failure modes: sky enabled with missing sun direction or invalid params; IBL generation unsupported for some backends -> error.

---

### 3. Definition of Done

* Default images unchanged with all new features off; baseline PNG hashes/metrics identical using pinned presets.
* Terrain AOV export supports albedo/normal/depth/roughness/metallic/AO/sun-vis/mask-ID, with correct sizes and formats.
* EXR output implemented for HDR AOVs and HDR frames; PNG pipeline unchanged; Frame.save rejects unsupported formats explicitly.
* DOF enabled on terrain changes output with measurable blur (edge-width or PSNR/SSIM delta); tilt-shift produces directional blur with only tilt changed.
* Motion blur shows measurable temporal blur with shutter interval >0; A/B test uses identical scene/camera/preset, only shutter changes.
* Lens effects produce measurable distortion/CA/vignette deltas (edge displacement, channel shift, luminance falloff stats).
* Denoiser reduces variance for AO/sun-vis/volumetric buffers with numeric metrics (variance/ROI stats); CPU/GPU options documented.
* Volumetrics show forced-impact delta (sun-only or no-sun scene), density=0 matches baseline within numeric tolerance.
* Sky model renders deterministically with sun/camera integration; A/B diff vs sky off with numeric stats.
* Tests added under tests/ for AOV export, EXR writer, DOF, motion blur, lens effects, volumetrics, sky; no tolerance relaxation.
* Example scripts updated to exercise each feature; outputs include A/B metrics or pixel diffs.
* Performance measurement recorded using existing runners (e.g., examples/terrain_demo.py timed runs, python/tools/perf_sanity.py for GPU baseline) with target ranges documented.
* Shader validation clean (wgsl-analyzer/naga), and build logs captured for shader provenance.
* Canonical verification suite defined and run before concluding: python -m pytest, cargo test, cargo check (or documented exceptions).

---

### 4. Milestones & deliverables (strict)

#### M1: Terrain AOV plumbing

* Scope: F1 core AOV textures for albedo/normal/depth
* Files: src/terrain/renderer.rs, src/terrain/render_params.rs, src/shaders/terrain_pbr_pom.wgsl, python/forge3d/terrain_params.py, python/forge3d/terrain_demo.py, examples/terrain_demo.py
* Deliverables: AOV outputs (PNG/raw), tests in tests/, sample renders
* Acceptance: AOV outputs exist + correct shapes
* Risks: MRT limits—mitigate by storage textures.

#### M2: EXR output + HDR save

* Scope: F1 EXR writer and HDR frame export
* Files: src/path_tracing/io.rs, src/util/image_write.rs or new src/util/exr_write.rs, src/lib.rs, python/forge3d/path_tracing.py
* Deliverables: EXR writer, docs update, tests
* Acceptance: EXR loads with expected named channels (beauty, depth, normal, albedo, roughness, metallic, ao, sun_vis, id/mask) and correct dimensions.
* Risks: platform packaging—mitigate with feature-gated exr crate.

#### M3: DoF + tilt-shift

* Scope: F2
* Files: src/core/dof/mod.rs, src/shaders/dof.wgsl, src/terrain/renderer.rs, src/terrain/render_params.rs, python/forge3d/terrain_params.py, CLI files
* Deliverables: DoF controls, tests, example
* Acceptance: numeric blur delta on forced-impact scene
* Risks: depth mismatch—mitigate with depth visualization test.

#### M4: Motion blur

* Scope: F3 object motion blur is not included; camera shutter accumulation only.
* Files: src/terrain/renderer.rs, src/terrain/accumulation.rs, src/shaders/accumulation_blend.wgsl, python/forge3d/terrain_params.py, CLI files
* Deliverables: shutter accumulation, tests, sample renders
* Acceptance: measurable blur on moving camera
* Risks: perf—mitigate with sample cap + half-res option.

#### M5: Lens effects + denoise

* Scope: F4 + F5
* Files: new src/terrain/lens_effects.rs, src/shaders/lens_effects.wgsl, src/terrain/renderer.rs, python/forge3d/terrain_params.py, python/forge3d/denoise.py
* Deliverables: lens post pass + denoise integration
* Acceptance: numeric CA/vignette/diff metrics and variance reduction
* Risks: halo/artifacts—mitigate with clamp + edge-aware filtering.

#### M6: Volumetrics + sky

* Scope: F6 + F7
* Files: src/shaders/volumetric.wgsl, src/shaders/sky.wgsl, src/terrain/renderer.rs, new shared src/core/volumetric.rs/src/core/sky.rs (or adapted from viewer), python/forge3d/config.py, CLI files
* Deliverables: offline volumetrics + analytic sky integration
* Acceptance: forced-impact delta + baseline preservation at density=0
* Risks: memory budget—mitigate with half-res and opt-in defaults.

---

## 5. Interactive Viewer Post-Processing Implementation Plan

This section defines the surgical implementation plan for post-processing effects in the interactive terrain viewer (`examples/terrain_viewer_interactive.py`). These effects extend M3-M6 with GPU shader implementations.

### Current State

| Effect | Config Plumbing | IPC Protocol | Shader/Pass | Status |
|--------|-----------------|--------------|-------------|--------|
| Vignette | ✅ | ✅ | ✅ | **Working** |
| Barrel Distortion | ✅ | ✅ | ❌ | Config only |
| Chromatic Aberration | ✅ | ✅ | ❌ | Config only |
| DoF | ✅ | ✅ | ❌ | Config only |
| Motion Blur | ✅ | ✅ | ❌ | Config only |
| Volumetrics | ✅ | ✅ | ❌ | Config only |

---

### P1: Full-Screen Post-Process Pass Infrastructure

**Objective**: Create reusable full-screen post-process pass infrastructure for lens effects requiring UV remapping.

**Scope**: Separate post-process pass with ping-pong textures for effects that need to sample the rendered image at offset coordinates.

**Files to modify**:
- `src/viewer/terrain/mod.rs` – add post_process module
- `src/viewer/terrain/post_process.rs` – new file: post-process pass manager
- `src/shaders/viewer_post_process.wgsl` – new file: full-screen quad + lens effects shader

**Deliverables**:
1. `PostProcessPass` struct with ping-pong textures
2. Full-screen triangle/quad vertex shader
3. Uniforms for screen dimensions, lens parameters
4. Integration point in `ViewerTerrainScene::render()` after main pass

**Acceptance Criteria**:
- [ ] Post-process pass renders to intermediate texture
- [ ] Final output identical to input when all effects disabled (bit-exact)
- [ ] Memory budget: ≤1 additional Rgba8Unorm texture at render resolution

**Risks**: Texture format mismatch, alpha handling. Mitigate with explicit format matching.

---

### P2: Barrel Distortion + Chromatic Aberration

**Objective**: Implement lens distortion and chromatic aberration in the post-process shader.

**Scope**: UV-remapping distortion (barrel/pincushion) and per-channel UV offset for CA.

**Files to modify**:
- `src/shaders/viewer_post_process.wgsl` – add distortion + CA functions
- `src/viewer/terrain/post_process.rs` – pass distortion/CA uniforms
- `src/viewer/terrain/render.rs` – wire lens_effects.distortion and .chromatic_aberration

**Algorithm**:
```wgsl
// Barrel distortion (Brown-Conrady model, simplified)
fn apply_distortion(uv: vec2<f32>, k: f32) -> vec2<f32> {
    let centered = uv - 0.5;
    let r2 = dot(centered, centered);
    let factor = 1.0 + k * r2;
    return centered * factor + 0.5;
}

// Chromatic aberration (radial RGB split)
fn apply_ca(uv: vec2<f32>, ca_strength: f32) -> vec3<f32> {
    let centered = uv - 0.5;
    let r_uv = centered * (1.0 + ca_strength) + 0.5;
    let g_uv = uv;
    let b_uv = centered * (1.0 - ca_strength) + 0.5;
    return vec3<f32>(
        textureSample(input_tex, samp, r_uv).r,
        textureSample(input_tex, samp, g_uv).g,
        textureSample(input_tex, samp, b_uv).b
    );
}
```

**Deliverables**:
1. Barrel distortion with `--lens-distortion` parameter (-0.5 to 0.5)
2. Chromatic aberration with `--lens-ca` parameter (0.0 to 0.1)
3. Edge clamping to prevent sampling outside texture bounds

**Acceptance Criteria**:
- [ ] `--lens-distortion 0.2` produces visible barrel effect (center expands, edges compress)
- [ ] `--lens-ca 0.05` produces visible RGB fringing at edges
- [ ] Distortion=0 + CA=0 produces bit-exact output vs post-process disabled
- [ ] Screenshot A/B diff shows measurable pixel displacement

**Test Command**:
```bash
python examples/terrain_viewer_interactive.py --dem assets/dem_rainier.tif --pbr --lens-effects --lens-distortion 0.15 --lens-ca 0.03 --lens-vignette 0.3
```

---

### P3: Depth of Field (Separable Blur)

**Objective**: Implement depth-based blur using Circle of Confusion (CoC) and separable Gaussian blur.

**Scope**: Two-pass separable blur weighted by CoC; requires depth buffer access.

**Files to modify**:
- `src/viewer/terrain/dof.rs` – new file: DoF pass manager
- `src/shaders/viewer_dof.wgsl` – new file: CoC calculation + separable blur
- `src/viewer/terrain/render.rs` – add depth texture to post-process bind group
- `src/viewer/terrain/mod.rs` – add dof module

**Algorithm**:
```wgsl
// Circle of Confusion calculation
fn calculate_coc(depth: f32, focus_dist: f32, focal_length: f32, f_stop: f32) -> f32 {
    let aperture = focal_length / f_stop;
    let coc = aperture * abs(depth - focus_dist) / depth * (focal_length / (focus_dist - focal_length));
    return clamp(coc, 0.0, max_blur_radius);
}

// Separable blur (horizontal then vertical)
// Weight samples by CoC of center pixel
```

**Uniforms**:
- `focus_distance: f32` – distance to focal plane (world units)
- `f_stop: f32` – aperture (2.8 = wide/shallow, 16 = narrow/deep)
- `focal_length: f32` – lens focal length (mm, for CoC scaling)
- `tilt_pitch: f32` – tilt-shift plane tilt (degrees)
- `tilt_yaw: f32` – tilt-shift plane yaw (degrees)
- `quality: u32` – blur kernel radius (4/8/16 samples)

**Deliverables**:
1. Two-pass separable blur pipeline
2. CoC texture (R16Float) storing per-pixel blur radius
3. Tilt-shift support via tilted focal plane
4. Quality presets (low=4, medium=8, high=16 samples)

**Acceptance Criteria**:
- [ ] `--dof --dof-f-stop 2.8 --dof-focus-distance 500` blurs foreground/background
- [ ] Focus plane visible as sharp band in image
- [ ] Tilt-shift (`--dof-tilt-pitch 30`) produces diagonal focus plane
- [ ] A/B edge-width histogram shows blur delta vs DoF disabled

**Test Command**:
```bash
python examples/terrain_viewer_interactive.py --dem assets/dem_rainier.tif --pbr --dof --dof-f-stop 2.8 --dof-focus-distance 300 --dof-quality high
```

---

### P4: Motion Blur (Temporal Accumulation)

**Objective**: Implement camera motion blur via temporal accumulation across shutter interval.

**Scope**: Render N sub-frames with interpolated camera, blend with equal weights.

**Files to modify**:
- `src/viewer/terrain/motion_blur.rs` – new file: motion blur accumulator
- `src/viewer/terrain/render.rs` – add multi-frame accumulation loop
- `src/viewer/cmd/handler.rs` – store motion blur config, trigger multi-frame render

**Algorithm**:
1. Parse shutter interval: `shutter_open` (0.0) to `shutter_close` (0.5 = 180° shutter)
2. For `i` in `0..samples`:
   - `t = shutter_open + (shutter_close - shutter_open) * (i + 0.5) / samples`
   - Interpolate camera: `phi += cam_phi_delta * t`, etc.
   - Render frame to accumulation buffer
3. Resolve: divide accumulated color by sample count

**Uniforms**:
- `samples: u32` – number of sub-frames (8-32)
- `shutter_open: f32` – shutter open time (0.0 = frame start)
- `shutter_close: f32` – shutter close time (0.5 = 180° shutter angle)
- `cam_phi_delta: f32` – camera phi change during shutter
- `cam_theta_delta: f32` – camera theta change during shutter
- `cam_radius_delta: f32` – camera radius change during shutter

**Deliverables**:
1. Accumulation buffer (Rgba32Float)
2. Multi-frame render loop with camera interpolation
3. Final resolve pass (divide by sample count)

**Acceptance Criteria**:
- [ ] `--motion-blur --mb-samples 16 --mb-cam-phi-delta 5` produces directional blur
- [ ] Blur direction matches camera motion direction
- [ ] samples=1 produces identical output to motion blur disabled
- [ ] Performance: 16 samples ≈ 16x single-frame time (acceptable for snapshots)

**Test Command**:
```bash
python examples/terrain_viewer_interactive.py --dem assets/dem_rainier.tif --pbr --motion-blur --mb-samples 16 --mb-shutter-angle 180 --mb-cam-phi-delta 5
```

---

### P5: Volumetric Fog + Light Shafts

**Objective**: Implement height-based fog and light shaft (god ray) rendering.

**Scope**: Single-pass ray march from camera through fog volume; sun occlusion for light shafts.

**Files to modify**:
- `src/viewer/terrain/volumetrics.rs` – new file: volumetric fog pass
- `src/shaders/viewer_volumetrics.wgsl` – new file: fog ray march shader
- `src/viewer/terrain/render.rs` – composite fog after main pass

**Algorithm**:
```wgsl
// Height-based fog density
fn fog_density(world_pos: vec3<f32>, base_density: f32, height_falloff: f32) -> f32 {
    return base_density * exp(-world_pos.y * height_falloff);
}

// Ray march through fog
fn raymarch_fog(ray_origin: vec3<f32>, ray_dir: vec3<f32>, max_dist: f32, steps: u32) -> vec4<f32> {
    var inscatter = vec3<f32>(0.0);
    var transmittance = 1.0;
    let step_size = max_dist / f32(steps);
    
    for (var i = 0u; i < steps; i++) {
        let t = (f32(i) + 0.5) * step_size;
        let pos = ray_origin + ray_dir * t;
        let density = fog_density(pos, u.fog_density, u.height_falloff);
        
        // Light shaft: check sun visibility at this position
        let sun_vis = sample_shadow_map(pos);  // or heightfield trace
        let light = sun_vis * u.sun_color * phase_function(dot(ray_dir, u.sun_dir));
        
        inscatter += transmittance * light * density * step_size;
        transmittance *= exp(-density * u.absorption * step_size);
    }
    return vec4<f32>(inscatter, transmittance);
}
```

**Uniforms**:
- `enabled: bool` – volumetrics on/off
- `mode: string` – "uniform" | "height" | "froxels"
- `density: f32` – base fog density (0.001 - 0.1)
- `scattering: f32` – scattering coefficient (0.0 - 1.0)
- `absorption: f32` – absorption coefficient (0.0 - 0.5)
- `height_falloff: f32` – exponential height falloff (0.001 - 0.1)
- `light_shafts: bool` – enable god rays
- `shaft_intensity: f32` – light shaft brightness multiplier
- `steps: u32` – ray march steps (16-64)
- `half_res: bool` – render at half resolution

**Deliverables**:
1. Volumetric fog compute/fragment shader
2. Height-based density falloff
3. Light shaft integration with sun direction
4. Half-resolution option for performance
5. Fog composite blend over scene

**Acceptance Criteria**:
- [ ] `--volumetrics --vol-density 0.02` produces visible atmospheric haze
- [ ] `--vol-light-shafts` produces visible god rays toward sun
- [ ] density=0 produces identical output to volumetrics disabled
- [ ] A/B luminance histogram shows fog contribution

**Test Command**:
```bash
python examples/terrain_viewer_interactive.py --dem assets/dem_rainier.tif --pbr --volumetrics --vol-mode height --vol-density 0.015 --vol-light-shafts --vol-shaft-intensity 2.0
```

---

### Implementation Order & Dependencies

```
P1 (Post-Process Infra) ──┬──> P2 (Distortion + CA)
                          │
                          └──> P3 (DoF) [needs depth buffer]
                          
P4 (Motion Blur) ─────────────> [independent, multi-frame]

P5 (Volumetrics) ─────────────> [independent, needs sun/depth]
```

**Recommended order**: P1 → P2 → P3 → P5 → P4

P1 is prerequisite for P2/P3. P4 and P5 are independent.

---

### Verification Suite

After each milestone, run:

```bash
# Unit tests
python -m pytest tests/test_volumetrics_sky.py -v

# Cargo check
cargo check --features extension-module

# Visual verification (capture screenshots)
python examples/terrain_viewer_interactive.py --dem assets/dem_rainier.tif --pbr [EFFECT_FLAGS] --snapshot test_output.png
```

Compare screenshots with baseline using:
```bash
python scripts/compare_images.py baseline.png test_output.png --metric ssim
```

---

## Assumptions

* Offline terrain renders go through TerrainRenderer.render_terrain_pbr_pom and Frame.save rather than the interactive viewer; verify by inspecting python/forge3d/terrain_demo.py and src/terrain/renderer.rs.
* EXR is currently unsupported in Rust output paths and must be implemented for HDR export; verify src/lib.rs::Frame.save and src/path_tracing/io.rs::AovWriter::write_exr.
* AOV/EXR is implemented for the terrain pipeline first; path tracer parity is explicitly out of scope for this plan.
---

## What remains / risks

* No tests or renders executed in this pass; integration risk for GPU memory and multi-target outputs remains.
* Volumetrics/sky reuse from viewer may require shared modules; risk of duplication vs refactor scope.
* EXR dependency and cross-platform packaging need validation in CI.

---

## Remaining checklist (Next actions)

- [x] Confirm whether to prioritize terrain pipeline only or also path tracer for AOV/EXR parity.  
  Decision: Terrain pipeline first. Keep naming/channel conventions compatible with future path tracer parity, but do not expand scope now.

- [x] Decide EXR writer approach and expected channel layout.  
  Decision: Rust-native EXR writer (feature-gated). Single EXR with named channels:
  beauty (RGB[A]), depth (Z), normal (Nx,Ny,Nz), albedo (RGB), roughness, metallic, ao, sun_vis, id/mask.

- [x] Approve milestone ordering and scope exclusions.  
  Decision: Keep milestone order. Explicitly exclude object motion blur in this plan (camera shutter accumulation only). Track object blur as future work.

- [x] After implementation, run verification suite and record raw outputs.  
  Decision: Required gate for each milestone: capture raw logs for `python -m pytest`, `cargo test`, `cargo check` (and `cargo clippy` if in AGENTS.md), plus a deterministic sample render set and metrics.

