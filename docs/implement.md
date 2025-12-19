# O1: Subsystem Map

| Subsystem | File Path(s) | Key Symbols | Notes |
|-----------|--------------|-------------|-------|
| **Terrain Offline Entry** | `python/forge3d/terrain_demo.py`, `python/forge3d/terrain_pbr_pom.py` | [run()](cci:1:python/forge3d/terrain_demo.py:739:0-1109:12), [render_terrain_pbr_pom()](cci:1:src/terrain/renderer.rs:747:4-787:5) | Python entry point, wraps native TerrainRenderer |
| **TerrainRenderer (Rust)** | `src/terrain/renderer.rs` | [TerrainRenderer](cci:2:src/terrain/renderer.rs:148:0-150:1), [TerrainScene](cci:2:src/terrain/renderer.rs:36:0-117:1), [render_terrain_pbr_pom()](cci:1:src/terrain/renderer.rs:747:4-787:5) | GPU PBR+POM pipeline, ~5300 lines |
| **Terrain Shader** | `src/shaders/terrain_pbr_pom.wgsl` | `vs_main`, `fs_main`, `calculate_pbr_brdf` | Main terrain PBR+POM shader (~3400 lines) |
| **Tonemapping** | `src/core/tonemap.rs`, `src/shaders/postprocess_tonemap.wgsl` | [TonemapProcessor](cci:2:src/core/tonemap.rs:10:0-21:1), `ToneMappingOperator`, ACES/Reinhard/Uncharted2 | HDR→LDR conversion with exposure control |
| **HDR Pipeline** | `src/core/hdr.rs`, `src/core/hdr_types.rs` | [HdrRenderTarget](cci:2:src/core/hdr.rs:20:0-30:1), [HdrConfig](cci:2:src/core/hdr_types.rs:42:0-50:1), [ToneMappingUniforms](cci:2:src/core/hdr_types.rs:69:0-74:1) | Rgba16Float HDR targets, CPU/GPU readback |
| **Bloom Effect** | `src/core/bloom.rs`, `src/shaders/bloom_*.wgsl` | [BloomEffect](cci:2:src/core/bloom.rs:50:0-71:1), [BloomConfig](cci:2:src/core/bloom.rs:13:0-18:1) | Brightpass + H/V blur compute pipelines |
| **Volumetric Fog** | `src/viewer/init/fog_init.rs`, `src/shaders/volumetric.wgsl` | [FogResources](cci:2:src/viewer/init/fog_init.rs:8:0-40:1), `VolumetricParams`, `fog_density_at_height()` | Height fog, Henyey-Greenstein scattering, froxel grid |
| **IBL System** | `src/core/ibl.rs` | `IBLQuality`, prefilter/irradiance compute | Equirect→cubemap, specular prefilter, BRDF LUT |
| **Water Surface** | `src/core/water_surface.rs`, `src/shaders/water_surface.wgsl` | [WaterSurfaceParams](cci:2:src/core/water_surface.rs:30:0-64:1), `WaterSurfaceMode`, Fresnel/waves | P4 water with reflections, foam overlay |
| **Reflections** | `src/core/reflections.rs`, `src/terrain/renderer.rs:100-108` | `water_reflection_*` resources, planar reflections | P4 planar reflections for water |
| **Terrain Analysis** | `src/terrain/analysis.rs` | [slope_aspect_compute()](cci:1:src/terrain/analysis.rs:37:0-85:1), [contour_extract()](cci:1:src/terrain/analysis.rs:145:0-202:1) | CPU slope/aspect/contour extraction |
| **Vector Overlays** | `src/core/overlays.rs`, `src/vector/polygon.rs`, `src/vector/line.rs` | [OverlayRenderer](cci:2:src/core/overlays.rs:38:0-53:1), [PolygonRenderer](cci:2:src/vector/polygon.rs:10:0-18:1), `LineRenderer` | Screen-space overlay compositing, vector fill/stroke |
| **Heightfield AO** | `src/terrain/renderer.rs:67-74` | `height_ao_compute_pipeline`, compute ray-AO | Compute-based heightfield ambient occlusion |
| **Sun Visibility** | `src/terrain/renderer.rs:76-82`, `src/shaders/heightfield_sun_vis.wgsl` | `sun_vis_compute_pipeline` | Compute-based terrain self-shadowing |
| **CSM Shadows** | `src/shadows/` | `CsmRenderer`, PCF/PCSS filtering | Cascaded shadow maps for terrain |
| **PNG Readback** | `src/core/async_readback.rs`, `src/util/image_write.rs` | `read_hdr_texture()`, `read_ldr_texture()` | Async GPU→CPU readback, image encoding |
| **Specular AA** | `src/shaders/terrain_pbr_pom.wgsl:2380-2414` | Toksvig variance-based roughness boost | Screen-derivative normal variance → roughness |

---

# O2: Feature Presence Audit (F1–F8)

| Feature ID | Status | Evidence (paths + symbols) | User-facing knobs | Gaps / Risks |
|------------|--------|---------------------------|-------------------|--------------|
| **F1: Color Mgmt / Tonemap** | **PRESENT** | `src/core/tonemap.rs` [TonemapProcessor](cci:2:src/core/tonemap.rs:10:0-21:1); `src/shaders/postprocess_tonemap.wgsl` ACES/Reinhard/Uncharted2; `src/core/hdr_types.rs` `ToneMappingOperator` | [exposure](cci:1:src/core/tonemap.rs:157:4-160:5), [gamma](cci:1:src/terrain/render_params.rs:1117:4-1120:5), `tone_mapping` operator via [HdrConfig](cci:2:src/core/hdr_types.rs:42:0-50:1) | **No white balance**; **no LUT support** in shader; tonemap operator not exposed to Python terrain API—only hardcoded Reinhard in terrain path |
| **F2: Accumulation AA** | **ABSENT** | No accumulation buffer found; no subpixel jitter; no sample averaging | None | **Complete implementation needed**: camera jitter, HDR accumulation buffer, configurable sample count |
| **F3: Atmosphere/Fog** | **PARTIAL** | `src/viewer/init/fog_init.rs` [FogResources](cci:2:src/viewer/init/fog_init.rs:8:0-40:1); `src/shaders/volumetric.wgsl` height fog + HG phase; `src/terrain/render_params.rs:202-227` [FogSettingsNative](cci:2:src/terrain/render_params.rs:205:0-214:1) | `fog.density`, `fog.height_falloff`, `fog.inscatter` via Python | **Interactive viewer only**—not wired to offline [TerrainRenderer](cci:2:src/terrain/renderer.rs:148:0-150:1); no aerial perspective (distance-based desaturation); god-rays shadow sampling TODO in shader |
| **F4: Bloom/Glare** | **PARTIAL** | `src/core/bloom.rs` [BloomEffect](cci:2:src/core/bloom.rs:50:0-71:1); shaders exist at [bloom_brightpass.wgsl](cci:7:src/shaders/bloom_brightpass.wgsl:0:0-0:0), [bloom_blur_h.wgsl](cci:7:src/shaders/bloom_blur_h.wgsl:0:0-0:0), [bloom_blur_v.wgsl](cci:7:src/shaders/bloom_blur_v.wgsl:0:0-0:0) | `threshold`, `softness`, `strength`, `radius` in [BloomConfig](cci:2:src/core/bloom.rs:13:0-18:1) | **execute() is stub** (line 327-332: "placeholder implementation"); not integrated into terrain offline path; no glare streaks |
| **F5: Specular AA** | **PRESENT** | `src/shaders/terrain_pbr_pom.wgsl:2380-2414` Toksvig-like variance→roughness; `src/terrain/renderer.rs:4604-4654` `spec_aa_enabled`, `specaa_sigma_scale` | `VF_SPEC_AA_ENABLED` env var, `VF_SPECAA_SIGMA_SCALE` env var | Effectively disabled (threshold=1.0 in beauty mode); roughness-to-mip mapping present in IBL prefilter; **no LEAN/aniso variants** |
| **F6: Water Material** | **PRESENT** | `src/core/water_surface.rs` [WaterSurfaceParams](cci:2:src/core/water_surface.rs:30:0-64:1); `src/shaders/water_surface.wgsl`; `src/terrain/renderer.rs:100-110` planar reflections | `reflection.enabled`, `fresnel_power`, `wave_strength`, `shore_atten_width` in Python | P4 water with env + planar reflections; **SSR not integrated for water**; foam partial |
| **F7: Terrain Layering** | **PARTIAL** | `src/terrain/analysis.rs` [slope_aspect_compute()](cci:1:src/terrain/analysis.rs:37:0-85:1); `src/shaders/terrain_pbr_pom.wgsl:150-162` `layer_heights`, `layer_roughness`, `layer_metallic` | Material layers via `TerrainShadingUniforms` | **Snow/wetness/rock driven by attributes NOT implemented**—layers exist but are height-based only; no slope/aspect/concavity-driven material blending in shader |
| **F8: Vector Overlays** | **PARTIAL** | `src/core/overlays.rs` [OverlayRenderer](cci:2:src/core/overlays.rs:38:0-53:1); `src/vector/polygon.rs` [PolygonRenderer](cci:2:src/vector/polygon.rs:10:0-18:1); `src/vector/line.rs` `LineRenderer`; `src/shaders/overlays.wgsl` | [overlays](cci:1:src/terrain/render_params.rs:1190:4-1193:5) list in Python params; contour params in [OverlayUniforms](cci:2:src/terrain/renderer.rs:183:0-191:1) | Vectors render in 2D screen-space—**no depth-correct draping onto terrain**; no halo/shadow for readability; contour extraction CPU-only, not styled |

---

## Key Findings: Top 5 Biggest Gaps for Blender-like Offline Quality

1. **F2 Accumulation AA (ABSENT)**: No stochastic jitter + accumulation pipeline. This is critical for offline quality—Blender Cycles accumulates many samples. High-impact, moderate effort.

2. **F4 Bloom (PARTIAL/stub)**: BloomEffect.execute() is a placeholder. Multi-scale HDR bloom is essential for filmic highlights. Medium effort to complete.

3. **F7 Terrain Layering (PARTIAL)**: Only height-based layers. Missing slope/aspect/curvature-driven snow, rock, wetness. High visual impact for terrain realism.

4. **F3 Atmosphere (PARTIAL)**: Fog exists in interactive viewer but not wired to offline path. Missing aerial perspective (distance-based desaturation/blue shift). Medium effort.

5. **F8 Vector Overlays (PARTIAL)**: No depth-correct draping—vectors render screen-space only. For cartographic quality, vectors must respect terrain occlusion with halos.

---

# O3: Implementation Plan

## Design Decisions

1. **Preserve Defaults**: All new features gated behind opt-in flags (default=off) to protect existing output.
2. **Reuse Pipeline Concepts**: Accumulation reuses existing HDR target; bloom integrates into post chain; fog reuses volumetric infrastructure.
3. **Offline-First**: Optimize for quality/determinism over FPS; sample counts and pass counts can be high.
4. **Python Config Surface**: Every new knob exposed via existing param dataclasses in [python/forge3d/terrain_params.py](cci:7:python/forge3d/terrain_params.py:0:0-0:0).

## Data Flows

```
┌─────────────────────────────────────────────────────────────────────┐
│  Offline Terrain Render Pipeline (proposed)                         │
├─────────────────────────────────────────────────────────────────────┤
│ For sample_idx in 1..N (accumulation):                              │
│   1. Apply camera jitter (subpixel offset)                          │
│   2. Render terrain PBR+POM → HDR target (Rgba16Float)              │
│   3. Apply fog/atmosphere pass (if enabled)                         │
│   4. Composite vector overlays with depth test (if any)             │
│   5. Accumulate into HDR accumulation buffer                        │
│ End loop                                                            │
│ 6. Divide accumulation by N                                         │
│ 7. Apply bloom (brightpass → blur → composite)                      │
│ 8. Tonemap (ACES/Filmic + exposure + optional LUT)                  │
│ 9. Readback → PNG                                                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

# O4: Milestones & Deliverables

## Milestone 1: Accumulation AA Infrastructure
**Scope**: F2 (complete)  
**Files touched/added**:
- `src/terrain/renderer.rs` — add `AccumulationBuffer`, jitter sequence
- `src/terrain/render_params.rs` — add `aa_samples: u32`, `aa_seed: Option<u64>`
- `python/forge3d/terrain_params.py` — expose `aa_samples`, `aa_seed`
- New file: `src/terrain/accumulation.rs` (<300 lines)

**Deliverables**:
- Rust accumulation buffer with HDR format
- Halton/R2 jitter sequence (deterministic with seed)
- Python API: `aa_samples=1` (default, no AA), `aa_samples=16/64/256`
- Test: `tests/test_accumulation_aa.py` with SSIM comparison (aa=1 vs aa=64)

**Acceptance Criteria**:
- `aa_samples=1` produces identical output to current baseline (hash match)
- `aa_samples=64` on high-frequency terrain shows measurable noise reduction: edge SSIM > 0.95 vs aa=1 on synthetic checker
- No memory leak: peak VRAM delta < 2× single frame

**Risks & Mitigations**:
- **Risk**: Large sample counts slow offline. **Mitigation**: Document expected time, allow early termination.

---

## Milestone 2: Complete Bloom Pipeline
**Scope**: F4 (complete implementation)  
**Files touched/added**:
- `src/core/bloom.rs` — implement [execute()](cci:1:src/core/bloom.rs:279:4-337:5) with resource pool integration
- `src/terrain/renderer.rs` — wire bloom into terrain offline path
- `src/terrain/render_params.rs` — add `BloomSettingsNative`
- `python/forge3d/terrain_params.py` — expose `bloom.enabled`, `bloom.threshold`, `bloom.strength`

**Deliverables**:
- Working 3-pass bloom: brightpass → H blur → V blur → composite
- Multi-scale option (2-3 mip levels) for filmic quality
- Test: `tests/test_bloom_effect.py` with HDR highlight scene

**Acceptance Criteria**:
- `bloom.enabled=False` produces identical output (hash match baseline)
- `bloom.enabled=True` with bright sun produces visible glow: ROI luminance in bloom region > 1.2× non-bloom
- PSNR of bloom vs no-bloom ≥ 25 dB (not too aggressive)

**Risks & Mitigations**:
- **Risk**: Bloom too aggressive. **Mitigation**: Conservative defaults (threshold=1.5, strength=0.3).

---

## Milestone 3: Atmosphere/Fog for Offline Path
**Scope**: F3 (wire existing fog + add aerial perspective)  
**Files touched/added**:
- `src/terrain/renderer.rs` — wire [FogResources](cci:2:src/viewer/init/fog_init.rs:8:0-40:1) into offline render
- `src/shaders/terrain_pbr_pom.wgsl` — add aerial perspective distance-based desaturation
- `python/forge3d/terrain_params.py` — expose `fog.aerial_perspective_strength`

**Deliverables**:
- Height fog working in offline terrain renders
- Aerial perspective: distance-based desaturation + blue shift
- Test: `tests/test_fog_offline.py` with depth-varying scene

**Acceptance Criteria**:
- `fog.density=0` identical to baseline
- `fog.density=0.01` produces visible haze: distant pixels desaturated (saturation < 0.7× foreground)
- Fog integrates with shadows (god-rays optional, deferred if perf issue)

**Risks & Mitigations**:
- **Risk**: Fog compute pass too slow. **Mitigation**: Half-res fog with upscale (already in code).

---

## Milestone 4: Terrain Material Layering
**Scope**: F7 (slope/aspect/curvature-driven materials)  
**Files touched/added**:
- `src/shaders/terrain_pbr_pom.wgsl` — add `compute_terrain_attributes()` for slope/aspect/curvature; add material blending logic
- `src/terrain/render_params.rs` — add `MaterialLayerSettingsNative` (snow threshold, wetness, rock exposure)
- `python/forge3d/terrain_params.py` — expose `materials.snow_*`, `materials.wetness_*`, `materials.rock_*`

**Deliverables**:
- Snow deposition: altitude + slope + aspect-driven (south-facing less snow)
- Rock/talus exposure: steep slopes show rock
- Wetness/darkening: concavity proxy (placeholder: slope curvature)
- Test: `tests/test_terrain_materials.py` with known DEM

**Acceptance Criteria**:
- Default params produce identical output (all material layers disabled)
- `snow.enabled=True` shows white on high-altitude, low-slope areas: ROI albedo > 0.8
- Rock exposure visible on slopes > 45°

**Risks & Mitigations**:
- **Risk**: Curvature computation expensive. **Mitigation**: Precompute CPU-side or use simple screen-space proxy.

---

## Milestone 5: Depth-Correct Vector Overlays
**Scope**: F8 (depth draping + halos)  
**Files touched/added**:
- `src/core/overlays.rs` — add depth-tested overlay pass
- `src/shaders/overlays.wgsl` — sample terrain depth, apply offset bias
- `src/vector/line.rs` — add halo/outline rendering mode
- `python/forge3d/terrain_params.py` — expose `overlay.depth_test`, `overlay.halo_*`

**Deliverables**:
- Vectors respect terrain occlusion (hidden behind ridges)
- Halo/shadow outline for readability
- Optional ink-like contour rendering from curvature
- Test: `tests/test_vector_drape.py` with road crossing ridge

**Acceptance Criteria**:
- `overlay.depth_test=False` identical to baseline
- `overlay.depth_test=True` occludes vectors behind terrain: occluded pixels < 5% vs non-occluded baseline
- Halos visible: line edge contrast > 2× vs terrain

**Risks & Mitigations**:
- **Risk**: Depth bias tuning tricky. **Mitigation**: Expose bias as knob, document good defaults.

---

## Milestone 6: Tonemap Enhancements (LUT + White Balance)
**Scope**: F1 gaps  
**Files touched/added**:
- `src/shaders/postprocess_tonemap.wgsl` — add LUT sampling, white balance uniform
- `src/core/tonemap.rs` — add LUT texture loading, white balance params
- `src/terrain/render_params.rs` — expose tonemap operator to terrain path
- `python/forge3d/terrain_params.py` — expose `tonemap.operator`, `tonemap.lut_path`, `tonemap.white_balance`

**Deliverables**:
- 3D LUT support (cube format)
- White balance (temperature/tint)
- Tonemap operator selection from Python (ACES/Reinhard/Uncharted2)
- Test: `tests/test_tonemap_lut.py`

**Acceptance Criteria**:
- No LUT produces identical output
- LUT applied: visual difference measurable (PSNR ≠ ∞)
- ACES vs Reinhard produces different output on HDR scene

**Risks & Mitigations**:
- **Risk**: LUT format compatibility. **Mitigation**: Support .cube format (standard).

---

## Milestone 7: Integration & Validation
**Scope**: End-to-end testing, documentation, example renders  
**Files touched/added**:
- `examples/blender_quality_demo.py` — showcase all features
- `docs/api/blender_features.md` — documentation
- `tests/test_blender_quality_integration.py` — integration test

**Deliverables**:
- Example render demonstrating all features enabled
- Documentation with parameter reference
- Regression test suite passing
- Before/after comparison renders

**Acceptance Criteria**:
- All existing tests pass (no regressions)
- Integration test: render with all features enabled completes without error
- SSIM of integration render vs baseline ≥ 0.85 (features change output but don't break)
- Memory usage < 512 MiB host-visible (per AGENTS.md)

**Risks & Mitigations**:
- **Risk**: Feature interactions cause issues. **Mitigation**: Test combinations, prioritize single-feature tests.

---

# O5: Non-Goals and Guardrails

## Non-Goals
- **Full multi-bounce path tracing**: Not needed for terrain; compute AO + sun-vis already exist
- **Real-time FPS optimization**: Offline quality prioritized over frame rate
- **SSR for water**: Planar reflections sufficient for terrain water; SSR is complex and marginal benefit
- **Procedural clouds**: Out of scope; static HDRI environment sufficient
- **Full GI solution**: IBL + AO provides adequate ambient; no radiosity/SSGI for terrain

## Guardrails (Invariants)
1. **No default output changes**: Every new feature defaults to OFF; `aa_samples=1`, `bloom.enabled=False`, `fog.density=0`, etc.
2. **No test regressions**: All existing tests in `tests/` must pass before and after each milestone
3. **Portable-first**: All shaders must compile on Vulkan 1.2 / WebGPU; no platform-specific extensions
4. **Memory budget**: ≤512 MiB host-visible heap (per AGENTS.md); monitor with [memory_tracker](cci:7:src/core/memory_tracker:0:0-0:0)
5. **Files <300 lines**: New files must stay under 300 lines per AGENTS.md; split as needed
6. **Deterministic seeding**: Accumulation AA and any stochastic features must support deterministic seed for reproducibility

---

## Summary

I've completed the comprehensive audit of forge3d for Blender-like offline rendering features per your request.

### Key Results

**Feature Status Overview**:
| Feature | Status |
|---------|--------|
| F1: Color Management/Tonemap | **PRESENT** (missing LUT, white balance) |
| F2: Accumulation AA | **ABSENT** (complete implementation needed) |
| F3: Atmosphere/Fog | **PARTIAL** (viewer-only, needs offline wiring) |
| F4: Bloom/Glare | **PARTIAL** (stub implementation) |
| F5: Specular AA | **PRESENT** (Toksvig in shader) |
| F6: Water Material | **PRESENT** (P4 planar reflections working) |
| F7: Terrain Layering | **PARTIAL** (height-only, no slope/aspect) |
| F8: Vector Overlays | **PARTIAL** (no depth-correct draping) |

**Biggest Gaps for Blender-like Quality**:
1. **Accumulation AA** — completely missing, highest impact
2. **Bloom** — stub implementation, needs completion
3. **Terrain Material Layering** — no slope/aspect-driven materials
4. **Atmosphere** — not wired to offline path
5. **Vector Draping** — no depth-correct terrain occlusion

**Plan**: 7 milestones with explicit files, deliverables, acceptance criteria (SSIM/PSNR metrics), and risks. All features opt-in to preserve existing baselines per AGENTS.md rules.