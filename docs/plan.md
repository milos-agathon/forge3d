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

