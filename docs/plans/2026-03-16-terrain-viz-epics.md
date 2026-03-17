# Forge3D Terrain Visualization: Corrected Epic Backlog

**Date:** 2026-03-16  
**Sources:** `forge3d_vs_blender_gap_analysis.md`, `docs/plans/forge3d_vs_unreal_gap_analysis.md`, direct source audit  
**Scope:** Terrain-first 3D visualization only. This document excludes editor-authoring workflows, character systems, gameplay systems, and generic DCC parity work unless they directly improve terrain rendering, terrain review, or terrain export.

---

## 1. Purpose of This Revision

The previous terrain-epics draft was directionally useful but not precise enough. It overstated several gaps that are only **partial** rather than **absent**.

The source audit changes the assessment in four important ways:

1. **Procedural terrain noise is already present, but ad hoc.** `src/shaders/terrain_pbr_pom.wgsl` already contains `hash`, `value_noise`, procedural detail normals, and albedo-noise wiring. The real gap is a reusable, higher-quality terrain noise module, not "add noise from scratch."
2. **A Hosek-Wilkie sky path already exists, but mostly in the viewer path.** `src/shaders/sky.wgsl` and `src/viewer/state/sky_state.rs` implement a configurable sky model. By contrast, the terrain renderer parses `SkySettings` but does not appear to drive a full terrain-sky render path from those settings. The real gap is path parity, not sky invention.
3. **Instanced rendering exists, but not as terrain-native scattering.** `src/scene/py_api/instanced_mesh.rs` and `src/render/instancing.rs` provide instanced mesh paths, but there is no terrain-aware placement, mask-driven scattering, terrain-specific culling, or terrain-viewer integration.
4. **EXR channel infrastructure exists, but terrain compositing is incomplete.** `src/util/exr_write.rs` can write arbitrary named EXR channels. However, the public save path still emits one file per AOV, and terrain AOV rendering in `src/terrain/renderer/aov.rs` is still plumbing-heavy rather than a verified populated render path.

This revised plan therefore distinguishes among:

- **Present:** already implemented and materially usable
- **Partial:** implemented in one path, lightly implemented, or wired only as plumbing
- **Missing:** not implemented in a usable form

---

## 2. Selection Rules

A feature is included in the build backlog only if it meets all of the following:

1. It materially improves terrain scenes, terrain atmosphere, terrain population, terrain review, or terrain export.
2. It fits Forge3D's architecture as an additive subsystem or a terrain-path integration task.
3. It does not require turning Forge3D into a Blender-class authoring tool or an Unreal-class game engine.
4. It can be implemented without degrading clipmap terrain streaming, COG/COPC/3D Tiles throughput, or IPC responsiveness.

---

## 3. Feasibility Scale

| Rating | Meaning | Effort | Risk |
|---|---|---:|---|
| **F1** | Natural fit; mostly wiring or contained extension | 5-15 pd | Negligible |
| **F2** | New terrain subsystem or medium extension | 15-45 pd | Low |
| **F3** | Cross-cutting subsystem touching several render paths | 45-90 pd | Medium |

Only F1-F3 work is considered here. F4-F5 items from the Blender and Unreal reports are handled explicitly in the defer/reject section.

---

## 4. Verified Terrain-Relevant Gap Matrix

| Feature | Blender / Unreal reference | Verified repo status | Feasibility | Desirability | Decision |
|---|---|---|---|---|---|
| **Terrain sky rendering and aerial perspective** | Blender: Hosek-Wilkie / Nishita; Unreal: atmosphere quality | **Partial.** Viewer sky is implemented in `src/shaders/sky.wgsl` and `src/viewer/state/sky_state.rs`. Terrain params parse sky fields in `src/terrain/render_params/decode_atmosphere.rs`, but the terrain path appears to use fog + aerial perspective approximations rather than a full terrain-sky path. | F1-F2 | High | **Build** |
| **Terrain AOVs and compositing export** | Blender: AOVs, multi-layer EXR, Cryptomatte-adjacent workflows | **Partial.** `render_with_aov()` exists and `src/terrain/renderer/aov.rs` allocates textures, but this is not yet a verified populated terrain-AOV path. EXR multi-channel writing exists in `src/util/exr_write.rs`, but public save paths still write per-AOV files. | F1-F2 | High | **Build** |
| **Terrain-native scattering / foliage-style placement** | Blender: Instance on Points; Unreal: foliage instancing | **Partial.** Instanced mesh APIs exist in `src/scene/py_api/instanced_mesh.rs` and `src/render/instancing.rs`, but there is no terrain-aware placement, terrain masks, slope filters, or terrain-viewer draw path. | F2-F3 | High | **Build** |
| **Procedural terrain material variation** | Blender: procedural texture nodes; Unreal: richer terrain material systems | **Partial.** `src/shaders/terrain_pbr_pom.wgsl` already includes `value_noise`, procedural detail normals, and albedo variation, but not a reusable or higher-fidelity terrain noise system. | F1-F2 | High | **Build** |
| **Local light probes** | Blender: irradiance volumes + reflection cubemaps; Unreal: probe-style GI fallback | **Missing, with enum stub only.** `GiMode::IrradianceProbes` exists in `src/render/params/gi.rs`, but no corresponding terrain implementation was found. | F2-F3 | High | **Build** |
| **Heterogeneous terrain volumetrics** | Blender: heterogeneous volume shading; Unreal: localized atmospheric FX | **Partial.** Terrain volumetrics exist in `src/viewer/terrain/volumetrics.rs` and `src/shaders/viewer_volumetrics.wgsl`, but they are height/uniform based with pseudo-noise, not 3D density volumes. | F2 | Medium-High | **Build later** |
| **GPU weather particles** | Blender: particles; Unreal: GPU particles / weather FX | **Missing.** No terrain-oriented emitter/render/update path was found. | F3 | Medium | **Build later** |
| **FFT ocean / spectrum water** | Blender: ocean modifier; Unreal: higher-end water simulation expectations | **Missing.** `src/shaders/water_surface.wgsl` uses analytic wave composition and foam breakup, not FFT/Tessendorf/JONSWAP. | F2 | Medium, but domain-specific | **Conditional** |
| **OCIO color management** | Blender: OCIO pipeline | **Missing.** No active OpenColorIO integration was found. | F3 | Medium | **Conditional** |
| **Projected decals** | Unreal: deferred decals | **Missing.** No terrain decal path was found. Existing raster/vector overlays already cover many terrain annotation needs. | F2 | Medium-Low | **Defer** |
| **Compute tessellation for terrain** | Unreal: tessellation/displacement | **Missing.** No terrain compute-subdivision path was found. Current clipmaps + POM + detail normals already cover many practical cases. | F2 | Medium-Low | **Defer** |
| **Adaptive sampling + OIDN** | Blender: adaptive sampling + OIDN | **Missing.** No path-tracer adaptive scheduler or OIDN integration was found. Existing denoise paths are A-trous in Rust and NumPy. | F2 | Medium, but not terrain-path critical today | **Defer until terrain export foundation lands** |

---

## 5. What Is Already Strong and Must Not Regress

The following are differentiators. No terrain epic should regress them:

- Clipmap terrain streaming and quadtree/page-table behavior
- COG, COPC/EPT, and 3D Tiles ingestion throughput
- Memory-budget tracking via `memory_tracker`
- Terrain overlays and vector draping
- Terrain PBR + POM path
- Terrain water reflection path already present in `terrain_pbr_pom.wgsl`
- Python and IPC control surfaces

Any terrain roadmap item that threatens those systems is incorrectly scoped.

---

## 6. Build Backlog

### Epic TV1 - Terrain Atmosphere Path Parity

**Why this is in scope:** The viewer already has a configurable Hosek-Wilkie / Preetham sky implementation, but the terrain renderer currently exposes sky settings without proving that the terrain path actually honors them. This is a high-value, low-risk parity fix.

**Feasibility:** F1-F2  
**Estimate:** 10-18 pd  
**Priority:** P0

| Task | Scope | Definition of done |
|---|---|---|
| **TV1.1 Wire `SkySettings` into the terrain render path** | Reuse the existing viewer-sky implementation, or port the minimum required logic, so `TerrainRenderParams.sky` changes terrain output instead of being dead plumbing. | Enabling `params.sky.enabled` changes output in the terrain renderer; `turbidity`, `ground_albedo`, `sun_intensity`, `sun_size`, and `sky_exposure` each produce observable, test-covered output changes; disabling sky is pixel-identical to the current baseline. |
| **TV1.2 Remove duplicated atmosphere semantics** | Make terrain aerial perspective derive from the same sky/atmosphere configuration rather than a parallel fog-only approximation. | Terrain fog/aerial perspective parameter semantics are documented once; there is no separate terrain-only interpretation of the same sky inputs; at least one terrain scene verifies low-haze vs high-haze output differences. |
| **TV1.3 Add terrain-sky regression tests** | Add config-level and image-level tests for the terrain path. | At least 3 terrain golden scenes cover clear sky, hazy sky, and low-sun conditions; one test fails if sky settings are parsed but ignored. |

### Epic TV2 - Terrain Output and Compositing Foundation

**Why this is in scope:** Terrain rendering currently has the surface API for AOVs and the low-level EXR utility pieces, but not a trustworthy terrain-output pipeline for compositing or QA.

**Feasibility:** F1-F2  
**Estimate:** 18-28 pd  
**Priority:** P0

| Task | Scope | Definition of done |
|---|---|---|
| **TV2.1 Make terrain AOV targets real** | Turn `render_with_aov()` into a populated terrain render path for beauty, albedo, normal, and depth. | `TerrainRenderer.render_with_aov()` returns populated buffers, not placeholder textures; normal output is world-space and normalized; depth is documented and stable; sizes exactly match the beauty frame. |
| **TV2.2 Add single-file terrain EXR export** | Use `src/util/exr_write.rs` to export terrain beauty + AOVs into one multi-channel EXR file. | A single EXR contains at least `beauty`, `albedo`, `normal`, and `depth`; the file round-trips with channel names intact; the terrain API exposes an explicit save path instead of forcing one file per pass. |
| **TV2.3 Harden terrain regression coverage** | Promote terrain visual tests from optional plumbing to a required quality gate. | A GPU-capable CI lane runs terrain golden comparisons for at least PBR terrain, terrain + water, and terrain + atmosphere; shader changes in terrain paths trigger those checks. |

### Epic TV3 - Terrain Scatter and Population

**Why this is in scope:** Blender's Instance on Points and Unreal's foliage tooling matter in terrain visualization because terrain scenes look unfinished without repeatable population systems. Forge3D already has instancing primitives; the missing work is terrain-native placement and render integration.

**Feasibility:** F2-F3  
**Estimate:** 28-45 pd  
**Priority:** P1

| Task | Scope | Definition of done |
|---|---|---|
| **TV3.1 Add terrain placement generators** | Implement deterministic placement strategies for terrain scenes: seeded random, grid+jitter, and mask-driven density. | Public API accepts a terrain source plus placement parameters and returns reproducible instance transforms; slope and elevation filters are supported; the same seed always generates the same transform set. |
| **TV3.2 Reuse existing instanced mesh infrastructure in terrain scenes** | Connect terrain-generated transforms to the existing instanced mesh renderer instead of creating a second instancing system. | Terrain scenes can render instanced assets through one shared instancing path; at least one terrain-viewer example and one offscreen example use the same asset + transform contract. |
| **TV3.3 Add terrain-specific culling and LOD policy** | Add per-batch distance culling and simple LOD switching suitable for terrain populations. | 50k placed instances render without breaking the frame budget on a representative mid-range GPU target; culling and LOD state are measurable in stats; instance memory is tracked by `memory_tracker`. |

### Epic TV4 - Terrain Material Variation Upgrade

**Why this is in scope:** Terrain already has procedural detail hooks, but they are local, ad hoc, and terrain-shader-specific. The gap versus Blender is not "zero procedural texturing"; it is "insufficiently reusable and too low fidelity."

**Feasibility:** F1-F2  
**Estimate:** 14-24 pd  
**Priority:** P1

| Task | Scope | Definition of done |
|---|---|---|
| **TV4.1 Extract the current terrain noise helpers into a shared WGSL unit** | Move the existing hash/value-noise logic out of terrain-only inline code into a shared shader module with stable naming and call sites. | Terrain visuals are pixel-identical at default settings after the refactor; duplicate terrain-only noise logic is removed from consumer shaders. |
| **TV4.2 Add a bounded higher-quality noise set** | Add only the noises that materially improve terrain: FBM, ridged FBM, and one cellular-distance variant. Do not build a general-purpose shader-graph substitute. | Terrain layers can independently use low-frequency macro variation and high-frequency detail variation; at least snow/wetness/rock blending can consume the new functions; the old detail path remains available as the zero-regression default. |
| **TV4.3 Expose terrain noise controls cleanly** | Add explicit terrain parameters for amplitude, scale, and octave count where justified. | Python terrain config exposes only the parameters that affect rendered output; setting amplitude to zero preserves the baseline image; a perf test demonstrates the new noise path stays within a documented budget. |

### Epic TV5 - Local Probe Lighting for Terrain Scenes

**Why this is in scope:** The Blender report correctly identifies light probes as one of the highest-value quality upgrades. Terrain scenes with buildings, underpasses, or local water reflections need more than one global IBL map.

**Feasibility:** F2-F3  
**Estimate:** 30-55 pd  
**Priority:** P2

| Task | Scope | Definition of done |
|---|---|---|
| **TV5.1 Turn `GiMode::IrradianceProbes` into a real terrain feature** | Implement baked diffuse irradiance probes for terrain scenes instead of keeping the enum as a configuration stub. | Selecting `irradiance-probes` changes indirect diffuse lighting in a terrain scene; probe data can be baked, stored, and sampled; when probes are absent, the system falls back to current IBL behavior. |
| **TV5.2 Add probe placement, invalidation, and memory accounting** | Define how probes are positioned, rebaked, and budgeted. | Probe bounds, resolution, and max count are explicit API inputs; rebake/invalidation behavior is documented; GPU memory for probes is tracked. |
| **TV5.3 Add local reflection probes only after diffuse probes are stable** | Add a bounded specular-probe path for terrain scenes with water or glossy built structures. | Reflection probes are optional and runtime-selectable; no reflection-probe path ships without a verified diffuse-probe baseline; at least one terrain + structure scene demonstrates better local specular than global IBL alone. |

### Epic TV6 - Heterogeneous Terrain Volumetrics

**Why this is in scope:** Current terrain volumetrics are useful but still fundamentally height/uniform fog with pseudo-noise. Localized fog banks, smoke plumes, and volcanic or industrial emissions are legitimate terrain-visualization use cases.

**Feasibility:** F2  
**Estimate:** 18-30 pd  
**Priority:** P2

| Task | Scope | Definition of done |
|---|---|---|
| **TV6.1 Add 3D density-volume support** | Extend the existing terrain volumetric pass to consume a bounded 3D density texture. | The terrain viewer can render a localized density field with documented world-space placement; the legacy height/uniform fog modes still work unchanged. |
| **TV6.2 Add procedural terrain-volume generators** | Add a small preset layer for valley fog, plume, and localized haze volumes. | At least 3 named density presets can be instantiated from Python; each preset maps onto the same underlying 3D density path rather than bespoke shader forks. |
| **TV6.3 Add quality and budget tests** | Prevent the new volume path from becoming an uncontrolled memory/perf sink. | Density textures participate in memory tracking; at least one test records expected perf and memory bounds for a representative volume size. |

### Epic TV7 - Weather Particle Foundation

**Why this is in scope:** Weather and airborne particulates are terrain-adjacent, not game-engine fluff. Rain, snow, dust, and ash directly improve terrain storytelling and simulation visualization.

**Feasibility:** F3  
**Estimate:** 25-40 pd  
**Priority:** P3

| Task | Scope | Definition of done |
|---|---|---|
| **TV7.1 Add a minimal GPU particle core** | Implement spawn, update, and billboard render stages for terrain scenes. | A single emitter can spawn, update, and render particles entirely on GPU; particle buffers are tracked; the system can be disabled with zero terrain-path impact when unused. |
| **TV7.2 Add terrain-aware collision / kill logic** | Make particles aware of terrain height rather than simulating in free space only. | Rain/snow/dust particles can collide with or die on the heightfield without CPU readback; at least one weather preset uses this behavior. |
| **TV7.3 Ship only terrain-relevant presets** | Limit the initial preset set to rain, snow, dust, and ash. | The v1 API does not include generic game VFX concepts; presets are parameterized and documented around terrain/weather use cases only. |

---

## 7. Conditional Epics

These are legitimate features, but they should not enter the core terrain backlog unless the product direction explicitly demands them.

### Epic TV8 - Coastal / Hydrology Water Upgrade

**Why conditional:** FFT/Tessendorf-class water is desirable for coastal, estuary, and hydrology-heavy scenes, but it is not universally required across terrain visualization. The current water path is already usable for many inland and cartographic scenarios.

**Feasibility:** F2  
**Estimate:** 20-35 pd  
**Enter backlog only if:** coastal visualization, open-water cinematics, or wave-spectrum fidelity become a named product requirement.

### Epic TV9 - OCIO Color-Managed Terrain Output

**Why conditional:** OCIO matters for VFX and print pipelines, but it carries the highest regression risk because it touches final pixel output across all terrain renders.

**Feasibility:** F3  
**Estimate:** 25-45 pd  
**Enter backlog only if:** Forge3D needs a formal color-managed interchange path with Blender/Nuke/ACES-based pipelines.

---

## 8. Deferred But Reasonable

These are not rejected forever. They are simply not the right near-term terrain backlog.

| Feature | Why deferred |
|---|---|
| **Projected decals** | Existing raster and vector overlays already cover most terrain annotation workflows. Revisit only if terrain needs projected textured detail on arbitrary mesh surfaces. |
| **Compute tessellation for terrain** | Current clipmaps, POM, and detail normals already cover many practical close-range needs. The risk/reward ratio is weaker than probes, scattering, or output hardening. |
| **Adaptive sampling + OIDN** | Valuable, but currently attached to the broader offline/path-tracing story rather than the primary terrain render path. Revisit after terrain AOV/EXR output is trustworthy and if film-quality offline terrain rendering becomes a core product goal. |
| **Virtual Shadow Maps** | Desirable for shadow stability, but the terrain path already supports multiple shadow techniques. It is not ahead of probes or terrain population in terrain-specific value. |

---

## 9. Explicitly Out of Scope for Terrain

The following appear in the Blender or Unreal comparison documents but should not be disguised as terrain-roadmap work:

| Feature family | Why it stays out |
|---|---|
| **Nanite-class virtualized geometry** | Architecturally mismatched with Forge3D's terrain-first rendering model and unnecessary for the terrain problem it already solves with clipmaps and tiles. |
| **Lumen-class fully dynamic GI** | Major renderer-architecture project with poor terrain-first ROI compared with probe lighting. |
| **Skeletal animation, morph targets, advanced character shading** | Not terrain visualization work. |
| **Material graph editor, Geometry Nodes, in-editor sculpting, full modifier stack** | These are editor-product commitments, not terrain-rendering epics. |
| **Gameplay, AI, multiplayer replication, audio systems** | Unreal-class engine concerns, not terrain visualization roadmap items. |
| **Grease Pencil, video editing, Blender-style authoring UI** | Blender-class DCC concerns, not Forge3D terrain priorities. |

---

## 10. Recommended Execution Order

| Phase | Epics | Reason |
|---|---|---|
| **Phase 1** | TV1, TV2 | Fix the terrain-path correctness issues first: atmosphere parity, real AOVs, trustworthy export, and regression coverage. |
| **Phase 2** | TV3, TV4 | Improve what users see immediately in terrain scenes: population and material variation. |
| **Phase 3** | TV5 | Add the first real local-lighting upgrade after terrain output and scene population are solid. |
| **Phase 4** | TV6, TV7 | Add localized atmosphere and weather once the core terrain image is correct and compositable. |
| **Conditional branch** | TV8, TV9 | Only if coastal/hydrology or color-managed pipeline requirements become explicit roadmap items. |

---

## 11. Effort Summary

### Core terrain backlog

| Epic | Low | High |
|---|---:|---:|
| TV1 - Terrain Atmosphere Path Parity | 10 pd | 18 pd |
| TV2 - Terrain Output and Compositing Foundation | 18 pd | 28 pd |
| TV3 - Terrain Scatter and Population | 28 pd | 45 pd |
| TV4 - Terrain Material Variation Upgrade | 14 pd | 24 pd |
| TV5 - Local Probe Lighting for Terrain Scenes | 30 pd | 55 pd |
| TV6 - Heterogeneous Terrain Volumetrics | 18 pd | 30 pd |
| TV7 - Weather Particle Foundation | 25 pd | 40 pd |
| **Total core backlog** | **143 pd** | **240 pd** |

### Conditional terrain backlog

| Epic | Low | High |
|---|---:|---:|
| TV8 - Coastal / Hydrology Water Upgrade | 20 pd | 35 pd |
| TV9 - OCIO Color-Managed Terrain Output | 25 pd | 45 pd |
| **Total conditional** | **45 pd** | **80 pd** |

---

## 12. Bottom-Line Assessment

Against Blender and Unreal, the **meaningful** terrain-visualization gaps are not "become a full DCC" and not "become a full game engine." They are:

1. Finish the terrain paths that are currently only half-wired.
2. Add terrain-native scene population and local-lighting tools.
3. Improve atmosphere, export, and compositing in ways that preserve Forge3D's terrain strengths.

That is the correct terrain backlog. Everything else from the Blender and Unreal reports should either remain deferred or be handled in a separate non-terrain roadmap.
