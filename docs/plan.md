• Repo Integration Investigation
                                                                                                                                                                              
  - Terrain pipeline is driven from Python helpers python/forge3d/terrain_demo.py / python/forge3d/terrain_pbr_pom.py (run/render_terrain_pbr_pom_sequence), which build      
    configs then call forge3d.TerrainRenderer.render_terrain_pbr_pom.                                                                                                         
  - Render parameters and validation live in python/forge3d/terrain_params.py (dataclasses for light/IBL/shadows/triplanar/POM/fog/reflection/detail,                         
    ShadowSettings.validate_for_terrain, AO weight fields).                                                                                                                   
  - Core renderer is src/terrain/renderer.rs (TerrainScene, TerrainRenderer::render_terrain_pbr_pom, render_internal, compute_coarse_ao_from_heightmap, shadow pass setup).   
    Internal render target is a single color texture (Rgba8Unorm, optional MSAA, optional blit for render_scale). No depth buffer/G-buffer; world position is reconstructed   
    from UV/height inside the shader. Shadow maps are separate depth attachments via CsmRenderer.                                                                             
  - Shader src/shaders/terrain_pbr_pom.wgsl defines TerrainUniforms, TerrainShadingUniforms, OverlayUniforms, bind group 0 entries (0–15), shadow sampling, IBL               
    (lighting_ibl.wgsl), POM, and AO hook via ao_debug_tex/u_overlay.params3.x multiplying diffuse+IBL. No outputs beyond the final color.                                    
  - Shadows: src/shadows + src/core/shadow_mapping/types.rs feed CsmRenderer used by terrain; shadow bind group is @group(3) in the terrain pipeline.                         
  - IBL: src/lighting/ibl_wrapper.rs builds prefiltered environment maps and uniforms; WGSL side uses group(2) eval_ibl/eval_ibl_split.                                       
  - Screen-space effects infra exists in src/core/screen_space_effects.rs (compute SSAO/GTAO/SSGI/SSR) with raw/blurred AO textures and getters;                              
    TerrainScene::set_ao_debug_view can bind an external AO view into the terrain shader.                                                                                     
  - Readback/output: src/lib.rs Frame -> to_numpy/save uses src/renderer/readback.rs (read_texture_tight) and src/util/image_write.rs. Color-space conversion is shader-side  
    (tonemap + optional sRGB EOTF flag). Tests tests/test_terrain_render_color_space.py and tests/verify_terrain_pbr_pom_shader.py guard current behavior.                    
                                                                                                                                                                              
  Subsystem map:                                                                                                                                                              
                                                                                                                                                                              
  Subsystem | File path | Key symbols | What it does | Why it matters for AO/sun_vis                                                                                          
                                                                                                                                                                              
  - Python entry | python/forge3d/terrain_demo.py, python/forge3d/terrain_pbr_pom.py | run, _build_renderer_config, render_terrain_pbr_pom | Loads DEM/HDR, builds params,    
    calls native renderer | New flags/presets must thread through here without breaking aliases                                                                               
  - Params/validation | python/forge3d/terrain_params.py | TerrainRenderParamsConfig, ShadowSettings.validate_for_terrain, AO weight fields | Validates all terrain knobs,    
    clamps memory budgets | New AO/sun visibility settings and CLI wiring land here                                                                                           
  - Renderer core | src/terrain/renderer.rs | TerrainScene, render_terrain_pbr_pom, render_internal, compute_coarse_ao_from_heightmap, create_shadow_bind_group | Allocates   
    textures, encodes shadow pass, main render, blit; owns bind group layouts (0–5) | Compute passes and new textures must be created/bound here; current AO hook uses        
    binding(12)                                                                                                                                                               
  - Terrain shader | src/shaders/terrain_pbr_pom.wgsl | TerrainUniforms, OverlayUniforms, calculate_shadow_terrain, eval_ibl usage, DBG_RAW_SSAO | Performs triplanar/POM,    
    direct sun, IBL; samples ao_debug_tex; modulates direct light via shadows | AO/sun_vis compositing and new bindings must be added here; no depth/G-buffer exists          
  - Shadow pipeline | src/shadows/*, src/core/shadow_mapping/types.rs, TerrainScene.shadow_depth_pipeline | CsmRenderer, CsmUniforms, depth maps | Builds cascaded shadow maps    consumed by terrain shader | Sun visibility must coexist with or gate this; bind group(3) already occupied                                                                
  - IBL resources | src/lighting/ibl_wrapper.rs, src/shaders/lighting_ibl.wgsl | IBL, IblGpuResources, WGSL eval_ibl/eval_ibl_split | Prefilters env maps and exposes bind    
    group(2) | AO must modulate IBL diffuse; rotation/intensity already wired                                                                                                 
  - Screen-space effects | src/core/screen_space_effects.rs | ScreenSpaceEffectsManager, SSAO pipelines, get_ssao_view | Compute SSAO/GTAO buffers with storage textures |    
    Potential reuse for AO texture or validation; existing binding(12) can point at these outputs                                                                             
  - Readback/output | src/lib.rs, src/renderer/readback.rs | Frame::to_numpy/save, read_texture_tight | Copies single-sample color to CPU/PNG | Validation renders will use   
    this path; color-space handled in shader                                                                                                                                  
                                                                                                                                                                              
  Missing supporting buffers: no depth buffer, no world-position/normal render targets; only heightmap (R32F), optional water mask, and shadow maps exist. Any AO/sun         
  visibility must derive geometry from the heightfield + matrices or add new intermediate textures.                                                                           
                                                                                                                                                                              
  Feasibility & Preconditions                                                                                                                                                 
                                                                                                                                                                              
  - Heightfield Ray AO:                                                                                                                                                       
      - Inputs needed: heightmap texture (R32F, non-filtered), spacing/height scale (TerrainUniforms.spacing_h_exag), height min/max and POM height curve                     
        (TerrainShadingUniforms), camera view/proj, render resolution, optional water mask for excluding water.                                                               
      - Currently available: all above via existing uniform buffers and textures; no world-position buffer, so AO must raymarch the heightfield.                              
      - New GPU resources: AO output texture (R8Unorm or R16Float) at internal resolution (or downscaled), storage binding + sampled view/sampler; optional temporal/blur     
        buffers if denoising is added.                                                                                                                                        
      - Coordinate space: terrain XY spans spacing_h_exag.xy around (0,0); height uses clamp0 + height_curve; UV is [0,1] with Python-side flipud already applied.            
      - Memory guard: 4K R8 AO ≈ 8 MB; add blur targets doubles that; stays within 512 MiB budget if reused/re-sized.                                                         
      - Choice: use heightfield ray-march in DEM texture space (B) because no G-buffer/depth exists and fullscreen triangle makes interpolated world_pos unreliable;          
        heightfield path aligns with shadow-normalization already in shader.                                                                                                  
  - Heightfield Ray Sun Visibility / Soft Shadows:                                                                                                                            
      - Inputs needed: heightmap, spacing/height scale, sun direction/intensity (TerrainUniforms.sun_exposure), camera matrices for view-dependent marching, cascade distances        for consistency with CSM (optional), resolution and step budget, optional jitter seeds for softness.                                                                  
      - Currently available: sun direction/intensity via uniforms, heightmap and spacing via group(0); no per-pixel depth, so must reconstruct from UV/heightfield.           
      - New GPU resources: sun visibility texture (R8Unorm/R16Float) at internal resolution (optional lower scale) + sampler; optional multi-sample accumulation buffer for   
        soft shadows.                                                                                                                                                         
      - Coordinate space: same DEM UV/world mapping as AO; must match shadow map conventions (normalize_for_shadow uses tex_coord-derived world xy/z).                        
      - Memory guard: 4K R8 sun_vis ≈ 8 MB; one extra buffer for soft accumulation doubles cost; within budget.                                                               
      - Choice: heightfield ray-march in DEM space (B) to stay portable and consistent with existing CSM normalization; G-buffer path is unavailable.                         
                                                                                                                                                                              
  Plan A — Heightfield Ray AO                                                                                                                                                 
                                                                                                                                                                              
  ### A.1 Goal                                                                                                                                                                
                                                                                                                                                                              
  - Compute heightfield ray-traced ambient occlusion per pixel to darken ambient/IBL diffuse; opt-in; defaults preserve current output.                                       
                                                                                                                                                                              
  ### A.2 Definition of Done (measurable)                                                                                                                                     
                                                                                                                                                                              
  - New config fields (Python + Rust) to enable AO, set samples/step length/max distance/resolution scale; default disabled.                                                  
  - New compute pipeline created once and reused per resolution change; validated by naga/wgsl-analyzer clean compile.                                                        
  - AO texture generated and bound into render pass; shader applies it only when enabled.                                                                                     
  - Automated test: render forced-impact scene (steep ridge) with AO on/off; assert measurable delta (e.g., SSIM drop > 0.05 or mean luminance difference in occluded ROI >   
    5%).                                                                                                                                                                      
  - Perf measurement: record compute pass GPU time at 1080p (target ≤ 2 ms with default settings) via wgpu timestamps or profiler hook.                                       
  - Memory check: log AO texture size; assert within budget in test (<=32 MB total for AO resources).                                                                         
  - Image artifact check: AO debug output path wired to DBG_RAW_SSAO using new texture.                                                                                       
  - Existing test suite passes unchanged.                                                                                                                                     
  - Render artifact saved (PNG) for AO-on vs off plus numeric metric stored/logged.                                                                                           
                                                                                                                                                                              
  ### A.3 Data flow & resources                                                                                                                                               
                                                                                                                                                                              
  - Inputs: heightmap R32F (binding 1), height sampler (2), terrain/shading uniforms (0,5), overlay uniforms (8) for AO weight, optional water mask (11) to zero AO over      
    water, camera matrices for ray direction, AO params uniform (new).                                                                                                        
  - Outputs: AO texture (R8Unorm or R16Float) at internal resolution (or scale factor); sampled with linear sampler.                                                          
  - Bind groups: reuse group(0) binding(12/13) for AO texture/sampler (currently debug); add small AO params uniform (e.g., group(0) binding(16) if needed for params) without    colliding existing indices.                                                                                                                                               
  - Pass order: compute AO before shadow + main render; AO texture stays bound for render pass.                                                                               
  - Composite: shader replaces ao_debug_tex sample with computed AO; multiplies IBL diffuse (and optional diffuse edge term) by ao_weight-scaled factor.                      
                                                                                                                                                                              
  ### A.4 Algorithm choice (portable)                                                                                                                                         
                                                                                                                                                                              
  - Heightfield ray-march: for each pixel UV, march along a small set of horizon directions (e.g., 6–8 directions) with fixed texel step up to max_distance in world units    
    (converted to UV via spacing). Accumulate max elevation angle to derive occlusion.                                                                                        
  - Complexity: O(pixels * directions * steps); default directions=6, steps=16 → ~96 samples/pixel.                                                                           
  - Softening: optional one-pass bilateral blur using normals from height derivatives; default off.                                                                           
  - Quality knobs: resolution_scale (default 0.5 for perf), directions (6), steps (16), max_distance (e.g., 200 m), blur_enable (false), ao_strength mapped to ao_weight.     
                                                                                                                                                                              
  ### A.5 Rust/wgpu integration points                                                                                                                                        
                                                                                                                                                                              
  - Add compute pipeline creation and cache in src/terrain/renderer.rs (pipeline cache struct).                                                                               
  - Add AO params uniform struct and buffer in render_internal.                                                                                                               
  - Encode compute pass in render_internal before shadow rendering: bind heightmap, uniforms, AO output storage texture.                                                      
  - Manage AO textures in TerrainScene (reuse, resize on resolution change).                                                                                                  
  - Bind AO texture into existing bind group (0) via ao_debug_view/coarse_ao_view plumbing; ensure layout matches shader changes.                                             
                                                                                                                                                                              
  ### A.6 Python API & presets surface                                                                                                                                        
                                                                                                                                                                              
  - Introduce HeightAoSettings dataclass (enabled: bool, resolution_scale: float, directions: int, steps: int, max_distance: float, strength: float, blur: bool) in python/   
    forge3d/terrain_params.py, default enabled=False.                                                                                                                         
  - Thread through TerrainRenderParamsConfig and TerrainRenderParams.                                                                                                         
  - CLI: optional flags --height-ao (bool), --height-ao-strength, --height-ao-resolution-scale, --height-ao-distance, --height-ao-steps, --height-ao-directions, --height-ao- 
    blur with defaults matching disabled behavior; update preset param_map/override precedence if present.                                                                    
  - Update stubs (python/forge3d/__init__.pyi) and docs/presets without breaking existing presets.                                                                            
                                                                                                                                                                              
  ### A.7 Validation & benchmarking                                                                                                                                           
                                                                                                                                                                              
  - Scene: existing assets assets/Gore_Range_Albers_1m.tif at 1080p; camera angled to cast self-occlusion; render AO off vs on.                                               
  - Outputs: ao_off.png, ao_on.png, ao_heat.png (DBG_RAW_SSAO).                                                                                                               
  - Metrics: SSIM/PSNR between off/on; ROI mean darkening near concave regions (>5%); AO histogram sanity (values in [0,1]).                                                  
  - Perf: measure compute dispatch duration (wgpu timestamps or frame timing log) and memory usage log.                                                                       
  - Failure checks: banding, light leaks (bright AO), temporal noise if blur off, mismatch with water mask.                                                                   
                                                                                                                                                                              
  ### A.8 Milestones & deliverables                                                                                                                                           
                                                                                                                                                                              
  - M1: Shader/API plumbing — Files: terrain_pbr_pom.wgsl, terrain_params.py, __init__.pyi. Deliverables: new bindings/params default-off, docs draft. Acceptance: naga       
    compile passes; defaults keep tests green. Risks: bind group layout mismatch; mitigate with feature toggle + pipeline layout update.                                      
  - M2: Compute pipeline + resource mgmt — Files: terrain/renderer.rs. Deliverables: AO pipeline creation, texture reuse/resizing, params uniform. Acceptance: AO pass runs   
    without errors at 1080p, memory log <32 MB. Risks: lifetime hazards; mitigate with reuse + resize guards.                                                                 
  - M3: Shader composite + debug — Files: terrain_pbr_pom.wgsl, renderer.rs. Deliverables: AO sampling replaces debug texture path, debug mode outputs AO. Acceptance: AO     
    visible in DBG_RAW_SSAO; base render unchanged when disabled. Risks: unintended darkening; mitigate with default weight 0 and clamps.                                     
  - M4: Tests & metrics — Files: tests/test_terrain_renderer.py (new case) or new test file, scripts/terrain_validation.py update. Deliverables: AO on/off integration test,  
    metric computation script. Acceptance: test asserts >5% ROI delta and SSIM drop; existing tests pass. Risks: nondeterminism; mitigate by fixed seeds/directions.          
  - M5: Docs/presets — Files: docs/terrain_pbr_pom_shader_reference.md, docs/examples/terrain_demo_quickstart.rst, presets if any. Deliverables: user-facing description, CLI 
    help updates. Acceptance: docs build passes, defaults unchanged. Risks: doc drift; mitigate with exact flag names.                                                        
                                                                                                                                                                              
  Plan B — Heightfield Ray Sun Visibility / Soft Shadows                                                                                                                      
                                                                                                                                                                              
  ### B.1 Goal                                                                                                                                                                
                                                                                                                                                                              
  - Compute heightfield ray-traced sun visibility factor to modulate direct-sun lighting (hard or soft); opt-in; defaults preserve current output and existing CSM.           
                                                                                                                                                                              
  ### B.2 Definition of Done (measurable)                                                                                                                                     
                                                                                                                                                                              
  - New config fields to enable heightfield sun visibility, choose hard/soft mode, sample count, max distance, resolution scale; default disabled.                            
  - Compute pipeline built and reused; naga/wgsl-analyzer clean.                                                                                                              
  - Sun visibility texture generated and sampled in shader to scale direct sun term (and optional IBL diffuse) only when enabled.                                             
  - Automated test: forced-impact scene with sun low angle; compare CSM-only vs CSM+sun_vis (or sun_vis-only if CSM disabled); assert luminance drop in occluded ROI >10% and 
    SSIM drop >0.08.                                                                                                                                                          
  - Perf measurement: compute pass timing at 1080p (target ≤ 3 ms default soft settings).                                                                                     
  - Memory log for sun_vis resources (target <32 MB).                                                                                                                         
  - Debug visualization path for visibility (grayscale output or debug mode) proves branch executed.                                                                          
  - Existing tests unaffected when feature off.                                                                                                                               
                                                                                                                                                                              
  ### B.3 Data flow & resources                                                                                                                                               
                                                                                                                                                                              
  - Inputs: heightmap, height sampler, terrain/shading uniforms (spacing, height exag, clamp), sun direction/intensity (u_terrain.sun_exposure), camera matrices, optional    
    cascade split info for aligning distance falloff, optional jitter seed.                                                                                                   
  - Outputs: sun visibility texture (R8Unorm/R16Float) at internal resolution (or scaled); linear sampler.                                                                    
  - Bind groups: add group(0) binding for sun visibility texture + sampler (e.g., binding(16)/(17)) and small params uniform; ensure pipeline layout updated in renderer.rs.  
  - Pass order: compute sun_vis after AO (reuse same internal resolution) but before shadow sampling/main render; texture stays bound through render pass.                    
  - Composite: shader multiplies direct sun lighting by sun_vis (optionally blends with existing shadow_factor); by default, if enabled alongside CSM, combine                
    multiplicatively to deepen terrain self-shadowing; if CSM disabled, sun_vis replaces shadow_factor.                                                                       
                                                                                                                                                                              
  ### B.4 Algorithm choice (portable)                                                                                                                                         
                                                                                                                                                                              
  - Heightfield ray-march along sun direction projected into UV plane: step in UV proportional to spacing and sun direction XY; track max horizon elevation to compute        
    visibility = smoothstep(horizon - sun_elev).                                                                                                                              
  - Hard shadows: single ray per pixel, visibility 0/1 (with epsilon bias).                                                                                                   
  - Soft shadows: N jittered rays or PCF-like multi-offset sampling (e.g., 4–8 offsets) averaged; optional penumbra estimation based on height delta.                         
  - Complexity: O(pixels * samples * steps); default samples=4, steps=24.                                                                                                     
  - Quality knobs: resolution_scale (default 0.5), samples (4), steps (24), max_distance (e.g., 400 m), bias/offset to reduce self-shadowing, softness toggle.                
                                                                                                                                                                              
  ### B.5 Rust/wgpu integration points                                                                                                                                        
                                                                                                                                                                              
  - Add sun visibility compute pipeline and resources to TerrainScene cache in src/terrain/renderer.rs.                                                                       
  - Manage visibility textures with resize on resolution change; optional combined resource manager with AO.                                                                  
  - Encode compute dispatch before main render; share command encoder; bind outputs to new bind group entries.                                                                
  - Update pipeline layout creation (create_render_pipeline) to include new bindings in group(0) and set bind group entries when building the main bind group.                
                                                                                                                                                                              
  ### B.6 Python API & presets surface                                                                                                                                        
                                                                                                                                                                              
  - Add SunVisibilitySettings dataclass (enabled: bool, mode: str {“hard”, “soft”}, resolution_scale: float, samples: int, steps: int, max_distance: float, softness: float,  
    bias: float) default disabled.                                                                                                                                            
  - Wire through TerrainRenderParamsConfig/TerrainRenderParams, expose in __init__.pyi.                                                                                       
  - CLI flags (defaults off): --sun-vis, --sun-vis-mode, --sun-vis-resolution-scale, --sun-vis-samples, --sun-vis-steps, --sun-vis-distance, --sun-vis-softness, --sun-vis-   
    bias.                                                                                                                                                                     
  - Presets: none enable by default; add optional preset demonstrating feature without touching existing ones.                                                                
                                                                                                                                                                              
  ### B.7 Validation & benchmarking                                                                                                                                           
                                                                                                                                                                              
  - Scene: low sun angle over mountainous DEM; render cases: baseline (CSM only), sun_vis on (with/without CSM).                                                              
  - Outputs: sunvis_off.png, sunvis_on.png, sunvis_debug.png (visibility map).                                                                                                
  - Metrics: ROI mean luminance drop in self-occluded slopes (>10%), SSIM/PSNR comparisons, histogram of visibility values.                                                   
  - Perf: log compute pass timing and total frame time delta; report samples/steps used.                                                                                      
  - Failure modes: light leaks at grazing angles, acne from bias, mismatch with CSM cascade edges, shimmer from jitter; check via debug map and stability across frames (if   
    temporal enabled later).                                                                                                                                                  
                                                                                                                                                                              
  ### B.8 Milestones & deliverables                                                                                                                                           
                                                                                                                                                                              
  - M1: API + shader hooks — Files: terrain_params.py, terrain_pbr_pom.wgsl, __init__.pyi. Deliverables: config structs/flags, shader bindings placeholders gated by enabled. 
    Acceptance: naga compile clean; feature off preserves output. Risks: layout churn; mitigate with default-off + guarded sampling.                                          
  - M2: Compute pipeline + resources — Files: terrain/renderer.rs. Deliverables: sun_vis textures, pipeline cache, dispatch encoding. Acceptance: visibility map produced     
    without errors at 1080p; memory log <32 MB. Risks: race with main pass; mitigate via single encoder ordering.                                                             
  - M3: Shader compositing — Files: terrain_pbr_pom.wgsl, renderer.rs. Deliverables: direct lighting multiplied by sun_vis, debug output for visibility. Acceptance: debug    
    mode shows expected occlusion; baseline unchanged when disabled. Risks: double-darkening with CSM; mitigate via blend policy and clamps.                                  
  - M4: Tests/metrics — Files: new integration test + validation script updates. Deliverables: sun_vis on/off render diff with numeric thresholds. Acceptance: test passes    
    with measurable delta; existing suite green. Risks: nondeterminism from jitter; mitigate with deterministic jitter seeds in tests.                                        
  - M5: Docs/help — Files: docs and CLI help. Deliverables: user guide for sun_vis flags, interaction with shadows. Acceptance: docs build succeeds; defaults unchanged.      
    Risks: user confusion vs CSM; mitigate with explicit precedence doc.

  What remains / risks

  - Need design decisions on blend policy with existing CSM (multiply vs replace) and default resolution scales to hit perf targets.   
  - Shader bind-group expansion risks touching multiple tests; careful gating and default-off needed.
  - Temporal stability/denoise strategies are deferred; might be required if soft shadows shimmer.
  - No automated perf harness yet; will need timing hooks or timestamp queries to meet perf DoD.

  Remaining checklist

  - [ ] Implement heightfield ray AO compute path and bind into terrain pipeline
  - [ ] Implement heightfield sun visibility compute path and shader compositing
  - [ ] Add Python config/CLI/docs/tests for both features with defaults off

---

## Agent Observations (prompt_raytrace.md compliance review)

### Verification of plan accuracy against codebase

**Bind group layout (group(0)):** Verified bindings 0–15 are occupied in `terrain_pbr_pom.wgsl`:
- 0: TerrainUniforms, 1–2: height tex/sampler, 3–4: material tex/sampler
- 5: TerrainShadingUniforms, 6–7: colormap, 8: OverlayUniforms, 9–10: height curve LUT
- 11: water mask, 12–13: ao_debug tex/sampler, 14–15: detail normal tex/sampler
- **Bindings 16+ are free** for new AO/sun_vis resources—plan's proposal to use 16/17 is valid.

**Existing AO infrastructure:** `TerrainScene` already has `coarse_ao_texture`/`coarse_ao_view` computed via `compute_coarse_ao_from_heightmap()` (CPU-side horizon sampling). The new GPU compute path should **replace** this with a more sophisticated ray-march, reusing the same binding slot (12) and fallback chain.

**Screen-space effects:** `ScreenSpaceEffectsManager` in `src/core/screen_space_effects.rs` provides SSAO/GTAO but requires a G-buffer (depth + normals). Terrain pipeline has **no G-buffer**, so this infrastructure cannot be reused directly. The heightfield ray-march approach in Plan A/B is correct.

**IBL modulation:** `eval_ibl` / `eval_ibl_split` in `lighting_ibl.wgsl` are called from terrain shader. AO currently modulates via `ao_weight` in `OverlayUniforms.params3.x`—plan correctly identifies this hook.

### Completeness against prompt_raytrace.md requirements

| Requirement | Status | Notes |
|-------------|--------|-------|
| Step 1: Repo reconnaissance table | ✓ Complete | Subsystem map with file paths, symbols, and integration relevance |
| Step 2: Feasibility & preconditions | ✓ Complete | Both features analyzed; heightfield ray-march (option B) justified |
| Plan X.1 Goal | ✓ Complete | Both plans state opt-in with defaults preserving output |
| Plan X.2 Definition of Done | ✓ Complete | 9 items each; includes tests, metrics, perf, memory checks |
| Plan X.3 Data flow & resources | ✓ Complete | Inputs/outputs/bindings specified |
| Plan X.4 Algorithm choice | ✓ Complete | Complexity estimates, quality knobs with defaults |
| Plan X.5 Rust/wgpu integration | ✓ Complete | Files/functions listed |
| Plan X.6 Python API | ✓ Complete | Dataclass fields, CLI flags, backward compat strategy |
| Plan X.7 Validation | ✓ Complete | Scene, outputs, metrics, failure modes |
| Plan X.8 Milestones | ✓ Complete | 5 milestones each with files/deliverables/acceptance/risks |

### Refinements recommended

1. **Binding strategy clarification:** Plan A says reuse binding(12/13) for AO; Plan B proposes new binding(16/17) for sun_vis. This is correct—AO replaces existing debug texture, sun_vis needs new slots.

2. **Compute shader file location:** Plans should specify new WGSL files will be created:
   - `src/shaders/heightfield_ao.wgsl` (compute)
   - `src/shaders/heightfield_sun_vis.wgsl` (compute)

3. **Resolution scale interaction:** Both plans default to `resolution_scale=0.5`. Clarify whether AO and sun_vis share the same scaled resolution or can differ independently.

4. **CSM interaction policy:** Plan B mentions "multiply vs replace" as open question. Recommend: when `sun_vis` enabled alongside CSM, use `final_shadow = min(csm_shadow, sun_vis)` to let heightfield self-shadow override CSM in occluded areas, avoiding double-darkening.

5. **Texture format choice:** Plan suggests R8Unorm or R16Float. Recommend **R8Unorm** for both (8 MB at 4K) unless soft shadows need sub-1% precision—saves memory and bandwidth.

6. **Blur strategy:** Plan A mentions optional bilateral blur; recommend deferring blur to a later milestone (M6) to keep initial implementation minimal per guardrails.

### Implementation order recommendation

Execute **Plan A (AO) first**—it establishes the compute pipeline infrastructure and texture management patterns that Plan B will reuse. Milestones can interleave:
1. A.M1 + B.M1 (API/shader plumbing for both)
2. A.M2 (AO compute pipeline)
3. A.M3 + A.M4 (AO composite + tests)
4. B.M2 (sun_vis compute pipeline, reusing patterns)
5. B.M3 + B.M4 (sun_vis composite + tests)
6. A.M5 + B.M5 (docs)