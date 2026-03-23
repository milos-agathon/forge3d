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

---

## 13. Implementation Status Snapshot (2026-03-22)

Source audit of `main` branch as of commit `8653a6f`.

| Epic | Status | Evidence |
|---|---|---|
| **TV1 - Terrain Atmosphere Path Parity** | **Shipped** | `src/terrain/renderer/atmosphere.rs` fully integrates `SkySettings` into terrain compute pipeline via `sky.wgsl`. Commit `e735359`. |
| **TV2 - Terrain Output and Compositing Foundation** | **Shipped** | `src/terrain/renderer/aov.rs` (702 lines) populates albedo/normal/depth MRTs. Multi-channel EXR export via `build_terrain_exr_channels()`. Commit `d833137`. |
| **TV3 - Terrain Scatter and Population** | **Shipped** | `src/terrain/scatter.rs` implements seeded random, grid+jitter, and mask-density placement with slope filters, distance culling, and LOD selection. Memory tracked via `TerrainScatterMemoryReport`. Commit `d341097`. |
| **TV4 - Terrain Material Variation Upgrade** | **Shipped** | `src/shaders/terrain_noise.wgsl` (108 lines) extracts shared noise module with FBM, ridged FBM, and cellular distance. Python controls exposed. Commit `e580d8a`. |
| **TV5 - Local Probe Lighting for Terrain Scenes** | **Shipped** | SH L2 irradiance probes with probe baker, GPU types, `SHL2` in `src/terrain/probes/`, and `terrain_probes.wgsl` shader integration (fs_main blending, debug modes). Commit `3c8ac2e`. |
| **TV13 - Terrain Population LOD Pipeline** | **Shipped** | QEM mesh simplification in `src/geometry/simplify.rs`, auto-LOD chain generation via `generate_lod_chain()` and `auto_lod_levels()`, HLOD spatial clustering with merged proxy meshes in `src/terrain/scatter.rs`. Stats (`hlod_cluster_draws`, `hlod_covered_instances`, `effective_draws`) and memory tracking (`hlod_buffer_bytes`) plumbed through renderer, viewer, and IPC paths. Branch `epic-13`. |
| **TV6 - Heterogeneous Terrain Volumetrics** | **Partial** | Viewer volumetrics (`viewer_volumetrics.wgsl`) implement height-based exponential fog with Henyey-Greenstein phase and god-rays, but density is height-derived only — no 3D density texture or spatial variation. Terrain renderer defers to sky pass. |
| **TV7 - Weather Particle Foundation** | **Not started** | No GPU particle emitter, update, or render code in any terrain path. |
| **TV8 - Coastal / Hydrology Water Upgrade** | **Not started** | `water_surface.wgsl` uses analytic wave composition. No FFT/Tessendorf path. |
| **TV9 - OCIO Color-Managed Terrain Output** | **Not started** | No OCIO integration found. |

---

## 14. Cross-Report Addendum: Additional Terrain-Relevant Gaps

The original TV1-TV9 backlog was derived from a first-pass Blender comparison and the Unreal report's terrain-adjacent sections. Now that TV1-TV4 and TV10 are shipped and TV5 is in progress, a second pass against `docs/plans/forge3d_vs_unreal_gap_analysis.md` reveals the remaining terrain-visualization gaps that still meet the selection rules in §2 and are not covered by already-shipped work.

### 14.1 Selection Methodology

Each candidate was evaluated against:

1. **Terrain specificity** — Does this gap directly affect terrain rendering, terrain review, or terrain export? General rendering improvements that happen to touch terrain are excluded.
2. **Current code state** — What plumbing, stubs, or partial implementations already exist? A gap with existing infrastructure is cheaper and less risky than one requiring greenfield work.
3. **Unblocked by shipped work** — Several items were previously deferred pending TV1-TV4. Those deferral conditions have now been met.
4. **Blender/Unreal parity value** — Is the gap visible in a direct terrain render comparison, or is it an internal architecture concern?

### 14.2 Additional Terrain Gap Matrix

| Feature | Blender / Unreal reference | Verified repo status | Feasibility | Desirability | Decision |
|---|---|---|---|---|---|
| **Terrain subsurface materials (snow/ice/earth SSS)** | Blender: Cycles SSS for natural materials; Unreal: Subsurface Profile shading model (§1.2) | **Shipped.** TV10 adds per-layer terrain subsurface controls in `python/forge3d/terrain_params.py`, native/uniform plumbing in `src/terrain/render_params/` and `src/terrain/renderer/`, bounded shader support in `src/shaders/terrain_pbr_pom.wgsl`, and docs/example/golden coverage. | F2 | High | **Shipped** |
| **Terrain shadow quality (VSM integration)** | Unreal: Virtual Shadow Maps (§2.2); Blender: CSM cascade quality is a known terrain limitation | **Partial.** `ShadowTechnique::VSM` and `EVSM` variants exist in `src/lighting/shadow.rs`. Terrain renderer uses CSM only. `page_table.rs` provides the page-table pattern VSM would reuse. | F2-F3 | Medium-High | **Build** |
| **Terrain offline render quality (adaptive sampling + denoiser)** | Blender: adaptive sampling + OIDN; Unreal: movie render queue quality passes (§12.2) | **Partial.** Path tracer with ReSTIR, importance sampling, and A-trous denoiser exists. No adaptive sampling scheduler. No OIDN or production-grade learned denoiser. Existing A-trous is edge-aware but not terrain-optimized. Deferral condition ("after terrain export foundation lands") is now satisfied — TV2 shipped. | F2 | Medium-High | **Build** |
| **Terrain population LOD pipeline** | Blender: modifier-stack mesh simplification; Unreal: auto LOD generation + HLOD (§3.2, §11.2) | **Missing for scatter.** TV3's scatter system is shipped but requires users to provide LOD levels manually. No automatic mesh simplification (quadric error metrics). No HLOD merging for distant terrain objects. `src/geometry/` has subdivision but not simplification. | F2 | Medium-High | **Build** |
| **Terrain flow and trajectory visualization** | Blender: particle paths, curve rendering; Unreal: ribbon/trail effects (§13.2, rated F1) | **Partial.** `src/geometry/` has ribbon generation, tube generation, and curve handling. No animated terrain-aware flow rendering. No integration with terrain height sampling for draped flow paths. | F1-F2 | Medium | **Conditional** |

---

## 15. Additional Build Backlog

### Epic TV10 - Terrain Subsurface Materials

**Status:** Shipped in `1.17.0`.

**Why this is in scope:** Snow, glacial ice, wet earth, and dense vegetation all exhibit subsurface light transport. Blender's Cycles renders these materials with dedicated SSS; Unreal's Subsurface Profile shading model handles the same cases. Forge3D's `subsurface` material parameter already flows through `MaterialShading` but is not consumed by the terrain PBR shader. With TV4 (material variation) shipped, the material layer system now distinguishes snow, rock, and wetness layers — each of which would directly benefit from terrain-specific SSS.

**Feasibility:** F2
**Estimate:** 12-20 pd
**Priority:** P1 (completed)

| Deliverable | Scope | Shipped outcome |
|---|---|---|
| **TV10.1 Wire subsurface parameters into terrain PBR shader** | Extend `terrain_pbr_pom.wgsl` to read terrain-layer subsurface state inside the terrain lighting accumulation. The shipped path stays terrain-first: a bounded per-pixel approximation built from wrap lighting, backscatter, and curvature-weighted diffusion. | Snow and wet terrain layers now show visibly different light transport from rock under the same lighting conditions, while explicit zero-strength subsurface remains pixel-stable against the pre-TV10 baseline. No full-screen post-pass was added. |
| **TV10.2 Add per-layer subsurface controls to terrain material config** | Expose `subsurface_strength` and `subsurface_color` per material layer (snow, rock, wetness) in the Python terrain params config, with defaults that improve alpine scenes out of the box without forcing the feature on when layers are disabled. | `MaterialLayerSettings` now preserves independent per-layer subsurface values; snow defaults to a non-zero terrain subsurface response, rock stays neutral by default, and the public docs page records the parameter semantics and valid ranges. |
| **TV10.3 Add terrain SSS regression tests** | Add config-level, runtime render, and golden-image coverage that verifies terrain SSS output changes when expected and does not regress the zero-strength baseline. | TV10 ships with dedicated config/API tests, runtime render tests across two lighting setups, a real-DEM demo for Mount Rainier and Gore Range, and three dedicated goldens that lock both SSS-on scenes and the zero-strength baseline. |

### Epic TV11 - Terrain Shadow Quality (VSM Integration)

**Why this is in scope:** Cascaded shadow map transitions create visible banding artifacts on large terrain surfaces, particularly at medium viewing distances where cascade boundaries cross the terrain plane. This is a known terrain rendering quality gap versus both Blender (which composites shadow in a single offline pass) and Unreal (which uses page-based virtual shadow maps for per-pixel resolution). Forge3D already has `ShadowTechnique::VSM` and `EVSM` in `src/lighting/shadow.rs` and a page-table pattern in `src/terrain/clipmap/page_table.rs` — the architectural primitives exist but are not connected to the terrain shadow path.

**Feasibility:** F2-F3
**Estimate:** 22-38 pd
**Priority:** P2

| Task | Scope | Definition of done |
|---|---|---|
| **TV11.1 Add VSM shadow path to terrain renderer** | Wire `ShadowTechnique::VSM` into the terrain shadow pipeline as a runtime-selectable alternative to the current CSM path. Reuse the existing VSM filtering code in `shadow.rs`. | Terrain scenes can render with VSM enabled via config; cascade banding artifacts are measurably reduced or eliminated at medium viewing distances; CSM remains the default and is unaffected when VSM is not selected. |
| **TV11.2 Add page-based shadow allocation for terrain** | Adapt the terrain page-table pattern to drive shadow-page allocation, so shadow resolution tracks the visible terrain surface rather than fixed cascades. | Shadow resolution is proportional to screen-space pixel density on the terrain surface; shadow memory is tracked by `memory_tracker`; the page allocation path does not interfere with the existing terrain clipmap page table. |
| **TV11.3 Add shadow quality A/B comparison tests** | Add terrain-specific shadow comparison tests that quantify the CSM-to-VSM quality difference and prevent shadow regressions. | At least 2 terrain scenes compare CSM vs VSM output for shadow boundary quality; a perf test records frame-time impact of VSM on a representative terrain scene; CSM baseline golden images are not broken by the addition. |

### Epic TV12 - Terrain Offline Render Quality

**Why this is in scope:** TV2 (terrain AOV export) is shipped. The deferral condition in §8 ("Defer until terrain export foundation lands") is satisfied. Blender's offline rendering pipeline delivers adaptive sampling (concentrating samples on noisy pixels) and OIDN denoising as standard terrain-output quality tools. Forge3D's path tracer has ReSTIR, importance sampling, and an A-trous denoiser, but lacks adaptive sample scheduling and a production-grade learned denoiser. For terrain offline renders — the primary output path for print, publication, and compositing workflows — these are the remaining quality gaps between Forge3D terrain output and Blender Cycles terrain output.

**Feasibility:** F2
**Estimate:** 18-30 pd
**Priority:** P2

| Task | Scope | Definition of done |
|---|---|---|
| **TV12.1 Add adaptive sampling scheduler for terrain path tracing** | Implement per-tile or per-pixel variance estimation that concentrates additional samples on high-variance regions of the terrain render. | Adaptive mode converges to a target noise threshold with fewer total samples than uniform sampling on a representative terrain scene; the scheduler is optional and defaults to off for backward compatibility; sample counts are queryable from Python. |
| **TV12.2 Integrate a learned denoiser for terrain offline output** | Add OIDN (Intel Open Image Denoise) or equivalent as a feature-gated post-pass for terrain path-traced output, guided by the AOV channels (albedo, normal, depth) that TV2 now provides. | The denoiser produces visibly cleaner terrain output than the existing A-trous pass alone at equivalent sample counts; the denoiser is optional and feature-gated; it consumes the same AOV channels that TV2 exports; the existing A-trous path remains available as the zero-dependency fallback. |
| **TV12.3 Add terrain offline quality regression tests** | Add tests that verify adaptive sampling convergence and denoiser output quality on terrain scenes. | At least 1 terrain scene verifies that adaptive sampling reaches target variance in fewer samples than uniform; at least 1 scene compares denoised vs raw output PSNR; the existing path-tracing golden tests are not regressed. |

### Epic TV13 - Terrain Population LOD Pipeline

**Why this is in scope:** TV3 (terrain scatter) is shipped and supports multi-level LOD selection, but users must provide pre-authored LOD meshes manually. Both Blender (modifier-stack mesh simplification) and Unreal (automatic LOD generation + HLOD merging) provide automatic mesh simplification. For terrain population at scale — thousands of placed trees, rocks, or structures — manual LOD authoring is impractical. Automatic simplification and HLOD generation would make TV3's scatter system viable for dense terrain scenes without requiring users to pre-process every asset.

**Feasibility:** F2
**Estimate:** 16-28 pd
**Priority:** P2

| Task | Scope | Definition of done |
|---|---|---|
| **TV13.1 Add automatic mesh simplification** | Implement quadric error metric (QEM) mesh simplification as a preprocessing utility in `src/geometry/`. Input: a `MeshBuffers` at full resolution. Output: a simplified `MeshBuffers` at a target face ratio. | A representative mesh can be simplified to 50%, 25%, and 10% face counts with documented quality/fidelity trade-offs; the simplifier preserves UV seams and hard edges where possible; the output feeds directly into the existing instanced mesh path used by TV3. |
| **TV13.2 Add auto-LOD chain generation for scatter assets** | Given a scatter asset, automatically generate a configurable number of LOD levels using TV13.1, and register them with the TV3 scatter system's existing `select_level_index` path. | Scatter batches can be configured with `auto_lod: true` and a target LOD count; generated LOD levels are consumed by the existing TV3 distance-based LOD selection; users can still provide manual LOD meshes to override auto-generation. |
| **TV13.3 Add HLOD merging for distant terrain populations** | For scatter objects beyond a configurable distance threshold, merge individual instances into a single simplified imposter mesh rather than drawing each instance. | A dense scatter batch (10k+ instances) shows measurably reduced draw call count beyond the HLOD distance; the merged mesh is regenerated when scatter parameters change; HLOD memory is tracked by `memory_tracker`. |

---

## 16. Additional Conditional Epics

### Epic TV14 - Terrain Flow and Trajectory Visualization

**Why conditional:** Flow paths, wind vectors, erosion patterns, and trajectory rendering are legitimate terrain-visualization use cases (hydrology, meteorology, geomorphology). The geometry module already has ribbon generation, tube generation, and curve handling — the infrastructure gap is small. However, this feature is domain-specific to scientific terrain workflows and may not justify the terrain pipeline integration cost for general terrain visualization.

**Feasibility:** F1-F2
**Estimate:** 10-18 pd
**Enter backlog only if:** hydrology, meteorology, or trajectory visualization becomes a named product requirement for terrain scenes.

### Epic TV15 - Compute Tessellation for Terrain

**Why conditional:** Close-range terrain detail is limited by fixed mesh topology. Compute-driven adaptive tessellation would provide real geometric displacement for terrain close-ups, surpassing the current POM approximation. However, the existing combination of clipmaps + POM + detail normals (enhanced by TV4's noise system) already covers most practical terrain scenarios. The ROI is lower than other terrain quality upgrades.

**Feasibility:** F2
**Estimate:** 20-35 pd
**Enter backlog only if:** close-range terrain detail fidelity becomes a named product requirement that POM cannot satisfy, or if WebGPU gains hardware tessellation stages that reduce implementation cost.

---

## 17. Updated Effort Summary

### Shipped terrain backlog

| Epic | Status |
|---|---|
| TV1 - Terrain Atmosphere Path Parity | **Shipped** |
| TV2 - Terrain Output and Compositing Foundation | **Shipped** |
| TV3 - Terrain Scatter and Population | **Shipped** |
| TV4 - Terrain Material Variation Upgrade | **Shipped** |
| TV10 - Terrain Subsurface Materials | **Shipped** |

### Active terrain backlog

| Epic | Status | Low | High |
|---|---|---:|---:|
| TV5 - Local Probe Lighting for Terrain Scenes | In progress (worktree) | 30 pd | 55 pd |
| TV6 - Heterogeneous Terrain Volumetrics | Partial (height fog only) | 18 pd | 30 pd |
| TV7 - Weather Particle Foundation | Not started | 25 pd | 40 pd |
| **Subtotal active (original)** | | **73 pd** | **125 pd** |

### New terrain backlog (this addendum)

| Epic | Low | High |
|---|---:|---:|
| TV11 - Terrain Shadow Quality (VSM Integration) | 22 pd | 38 pd |
| TV12 - Terrain Offline Render Quality | 18 pd | 30 pd |
| TV13 - Terrain Population LOD Pipeline | 16 pd | 28 pd |
| **Total new backlog** | **56 pd** | **96 pd** |

### Conditional terrain backlog (updated)

| Epic | Low | High |
|---|---:|---:|
| TV8 - Coastal / Hydrology Water Upgrade | 20 pd | 35 pd |
| TV9 - OCIO Color-Managed Terrain Output | 25 pd | 45 pd |
| TV14 - Terrain Flow and Trajectory Visualization | 10 pd | 18 pd |
| TV15 - Compute Tessellation for Terrain | 20 pd | 35 pd |
| **Total conditional** | **75 pd** | **133 pd** |

### Grand total remaining terrain work

| Category | Low | High |
|---|---:|---:|
| Active original (TV5-TV7) | 73 pd | 125 pd |
| New build (TV11-TV13) | 56 pd | 96 pd |
| **Total build backlog** | **129 pd** | **221 pd** |
| Conditional (TV8-TV9, TV14-TV15) | 75 pd | 133 pd |
| **Total including conditional** | **204 pd** | **354 pd** |

---

## 18. Updated Execution Order

| Phase | Epics | Reason |
|---|---|---|
| **Phase 1** *(shipped)* | TV1, TV2, TV3, TV4 | Terrain-path correctness, output, population, and material foundations. **Complete.** |
| **Phase 2** *(active)* | TV5 | Local probe lighting remains the only open item in this phase now that TV10 is shipped. |
| **Phase 3** | TV6, TV11, TV13 | Heterogeneous volumetrics, shadow quality, and population LOD. These improve the visible quality of terrain scenes that already have atmosphere (TV1), population (TV3), and probes (TV5). |
| **Phase 4** | TV7, TV12 | Weather particles and offline render quality. These are the final terrain-image quality features before the core terrain rendering story is complete. |
| **Conditional branch** | TV8, TV9, TV14, TV15 | Only if coastal/hydrology, color-managed pipelines, flow visualization, or close-range displacement become explicit product requirements. |

---

## 19. Addendum Bottom-Line Assessment

With TV1-TV4 and TV10 shipped, and TV5 still in progress, the remaining terrain-visualization gap against Blender and Unreal has narrowed to four categories:

1. **Terrain material realism** — probe-quality local lighting (TV5, in progress).
2. **Terrain shadow and atmosphere quality** — VSM shadow integration (TV11), heterogeneous volumetrics (TV6), and weather particles (TV7).
3. **Terrain output quality** — adaptive sampling and production denoiser for offline renders (TV12), now unblocked by TV2.
4. **Terrain population at scale** — automatic LOD generation and HLOD merging for scatter objects (TV13), extending TV3.

These are the gaps that remain visible in a direct terrain-render comparison. Everything else from the Blender and Unreal reports is either already shipped, in progress, correctly deferred, or not terrain-visualization work.

---

## 20. Third-Pass Addendum: Terrain Review and Delivery Gaps

The prior addenda correctly focused on image-quality gaps. A third pass across the same terrain-first scope surfaces a different class of missing capability: features that make terrain scenes easier to compare, review, animate, and deliver once the renderer is already credible.

### 20.1 Method Note

The user-requested root file `forge3d_vs_blender_gap_analysis.md` is not present in this worktree. This pass therefore uses:

- the Blender-derived findings already embedded in sections 4-19 of this document,
- `docs/api/blender_features.md`,
- `docs/notes/implement.md`, and
- `docs/plans/forge3d_vs_unreal_gap_analysis.md`.

This pass also treats the active TV5 probe files now present in the workspace (`src/terrain/renderer/probes.rs`, `src/shaders/terrain_probes.wgsl`, `tests/test_terrain_probes.py`) as **in-progress implementation evidence**, not as shipped backlog closure.

### 20.2 Additional Terrain Gap Matrix

| Feature | Blender / Unreal reference | Verified repo status | Feasibility | Desirability | Decision |
|---|---|---|---|---|---|
| **Terrain scene variants and review layers** | Unreal: Data Layers / scene variants; Blender-adjacent: saved scene/view-layer state for alternate outputs | **Partial.** `python/forge3d/bundle.py` and `src/bundle/manifest.rs` persist terrain metadata, presets, and camera bookmarks, but not named runtime variants or grouped layer states. `src/viewer/ipc/protocol/request.rs` exposes per-overlay toggles, not named variant activation. | F1-F2 | High | **Build** |
| **Terrain camera rig toolkit** | Unreal: camera rig system; Blender: path/constrained camera workflows | **Partial.** `src/animation/mod.rs` supports cubic-Hermite camera keyframes and `src/animation/render_queue.rs` enumerates frame export, but there are no orbit/rail/target-follow rig primitives, terrain-clearance constraints, or rig baking helpers. | F1-F2 | High | **Build** |
| **Terrain delivery queue and bounded timeline** | Unreal: Movie Render Queue + Sequencer; Blender: repeatable batch render workflow | **Partial.** Terrain AOV export exists after TV2 and camera keyframe export exists, but there is no shot manifest, no multi-shot batch queue, and no bounded track system for terrain-specific property changes such as active variant, sun, sky, or overlay visibility. | F2-F3 | High | **Build** |
| **Collaborative terrain review** | Unreal: collaborative viewing / shared review; Blender equivalent is weaker, but review sessions are a real product gap | **Partial.** The existing viewer IPC surface is already TCP + NDJSON and includes terrain, overlay, label, and camera commands, but there is no multi-client session model, no broadcast/state replay, and no shared annotation ownership semantics. | F2 | Medium-High | **Build later** |
| **Terrain night-light fidelity** | Unreal: IES profiles; Blender: measured-light workflows | **Missing.** The lighting stack has the right shader/light infrastructure for additive profile sampling, but there is no IES asset ingestion or terrain-scene hookup. This is valuable only for night scenes with built infrastructure. | F1 | Medium-Low | **Conditional** |

### 20.3 Selection Rationale

These items meet the section 2 rules because they materially improve terrain review or terrain export without requiring Forge3D to become an editor product. They also sit above, rather than instead of, the TV5-TV15 rendering backlog:

1. **TV16 and TV17** improve how terrain scenes are compared and authored for flyovers.
2. **TV18** turns the existing renderer into a repeatable delivery pipeline instead of a single-shot API.
3. **TV19** is review-specific and should stay explicitly below the rendering and delivery fundamentals.
4. **IES lighting** remains conditional because it helps only a narrower subset of terrain scenes than variants, rigs, or delivery tooling.

---

## 21. Additional Build Backlog

### Epic TV16 - Terrain Scene Variants and Review Layers

**Why this is in scope:** The current bundle/runtime model is good at storing one terrain state plus bookmarks, but weak at comparing alternate states of the same scene. Unreal's Data Layers are relevant here because terrain review often requires switching among alternate overlays, scatter states, lighting presets, and annotation sets without duplicating the entire scene.

**Feasibility:** F1-F2  
**Estimate:** 10-16 pd  
**Priority:** P1

| Task | Scope | Definition of done |
|---|---|---|
| **TV16.1 Add named layer and variant schema** | Extend bundle/runtime state with named review layers and named variants that reference existing terrain assets, overlays, scatter batches, labels, and presets without duplicating payload files. | A bundle can persist at least 3 named variants that share one DEM and one asset set; variants round-trip through save/load without checksum ambiguity; the schema distinguishes between per-layer visibility and whole-variant activation. |
| **TV16.2 Add atomic apply/list APIs** | Add Python and IPC surfaces to list variants, query the active variant, toggle individual layers, and apply a variant in one state transition. | One API call can switch the active variant; invalid layer or variant IDs fail explicitly; applying a variant that reuses the same terrain asset does not force a terrain reload. |
| **TV16.3 Add variant regression tests** | Verify persistence, deterministic switching, and blast radius. | Tests prove that switching variants changes only the declared overlays/scatter/labels/preset fields, preserves camera state unless explicitly variant-owned, and survives bundle round-trip without silent state loss. |

### Epic TV17 - Terrain Camera Rig Toolkit

**Why this is in scope:** Forge3D already has camera keyframes, but terrain flyovers are still authored one keyframe at a time. Blender and Unreal both reduce this friction with path- and rig-driven cameras. For terrain visualization, the missing value is not character animation; it is reusable flyover authoring with terrain-aware clearance.

**Feasibility:** F1-F2  
**Estimate:** 12-20 pd  
**Priority:** P1

| Task | Scope | Definition of done |
|---|---|---|
| **TV17.1 Add bounded rig primitives** | Implement only terrain-relevant rig types: orbit, rail/path, and target-follow with minimum altitude-above-terrain clearance. | Each rig type can produce a deterministic camera path from a small parameter set; the target-follow rig never penetrates below a configured terrain clearance threshold on a representative DEM. |
| **TV17.2 Bake rigs into existing `CameraAnimation`** | Keep the current keyframe/interpolation system as the execution layer by compiling rig specs into editable camera keyframes rather than inventing a parallel playback runtime. | A rig can be baked into `CameraAnimation`, inspected, edited, and rendered through the existing render queue; manual keyframe overrides survive re-render without custom rig-only code paths. |
| **TV17.3 Add rig examples and tests** | Cover both ergonomics and safety. | At least two example flyovers exist (`orbit` and `rail`); tests verify deterministic frame count, stable timing, and no below-terrain camera samples for the clearance-constrained rigs. |

### Epic TV18 - Terrain Delivery Queue and Bounded Timeline

**Why this is in scope:** TV2 solved terrain AOV export and the animation module can already enumerate camera frames, but there is still no coherent terrain delivery surface comparable to Unreal's Movie Render Queue plus a constrained subset of Sequencer. The missing feature is not a full DCC timeline; it is a repeatable shot queue for terrain outputs.

**Feasibility:** F2-F3  
**Estimate:** 18-28 pd  
**Priority:** P2

| Task | Scope | Definition of done |
|---|---|---|
| **TV18.1 Define a terrain shot manifest** | Introduce a serializable shot/job spec containing camera animation or rig, active variant, render size, frame range, output formats, and AOV selection. | One manifest can describe multiple shots with explicit output directories and pass selections; invalid shot specs fail before rendering begins; per-shot metadata records the exact terrain preset and variant used. |
| **TV18.2 Add bounded timeline tracks** | Support only the property tracks that materially affect terrain delivery: camera, sun, sky, exposure/tonemap, active variant, and overlay visibility. Do not add generic arbitrary-property tracks, events, or audio. | A shot can animate at least camera, sun, sky, and active variant over time; evaluation is deterministic frame-to-frame; v1 explicitly excludes audio, gameplay events, and arbitrary object tracks. |
| **TV18.3 Build pass-aware batch rendering** | Run multi-shot renders with progress, resume, and structured outputs using the TV2 terrain beauty/AOV pipeline. | A queue can render multiple shots to a stable directory layout with beauty PNG plus optional multi-channel EXR/AOV outputs; interrupted runs can resume from the current shot/frame without re-rendering completed outputs; progress is queryable from Python. |

### Epic TV19 - Collaborative Terrain Review

**Why this is in scope:** Terrain visualization is often a review workflow, not just a render. Unreal's collaborative-viewing direction is relevant here, but Forge3D should stop far short of multiplayer simulation. The practical goal is shared camera, shared annotations, and shared variant state over the existing NDJSON control surface.

**Feasibility:** F2  
**Estimate:** 16-26 pd  
**Priority:** P3

| Task | Scope | Definition of done |
|---|---|---|
| **TV19.1 Add a session coordinator** | Add a minimal multi-client session layer above the existing viewer IPC protocol, with one authoritative scene state and client join/leave semantics. | At least two clients can connect to the same terrain review session and observe synchronized camera, active variant, and overlay/label state without manual polling glue in user code. |
| **TV19.2 Add shared bookmark and annotation propagation** | Make camera bookmarks, labels, callouts, and review annotations session-visible and persistable. | An annotation or bookmark created by one client appears on all connected clients, can be saved into a bundle/session artifact, and includes minimal author/timestamp metadata. |
| **TV19.3 Define recovery and conflict rules** | Keep the semantics explicit and testable. | Reconnecting clients receive the current session snapshot; last-writer-wins or host-authoritative conflict behavior is documented and test-covered; there is no hidden slide into full replicated simulation. |

---

## 22. Revised Prioritization for Terrain Visualization

These epics do not replace TV5-TV15. They are the next terrain-first layer once render quality is credible enough to support review and delivery workflows.

### Additional effort summary

| Epic | Low | High |
|---|---:|---:|
| TV16 - Terrain Scene Variants and Review Layers | 10 pd | 16 pd |
| TV17 - Terrain Camera Rig Toolkit | 12 pd | 20 pd |
| TV18 - Terrain Delivery Queue and Bounded Timeline | 18 pd | 28 pd |
| TV19 - Collaborative Terrain Review | 16 pd | 26 pd |
| **Total new workflow backlog** | **56 pd** | **90 pd** |

### Updated remaining terrain work

| Category | Low | High |
|---|---:|---:|
| Existing build backlog before this addendum | 141 pd | 241 pd |
| New workflow backlog (TV16-TV19) | 56 pd | 90 pd |
| **Revised total build backlog** | **197 pd** | **331 pd** |
| Existing conditional backlog (TV8-TV9, TV14-TV15) | 75 pd | 133 pd |
| Terrain night-light fidelity (conditional) | 5 pd | 10 pd |
| **Revised total including conditional** | **277 pd** | **474 pd** |

### Revised execution order

| Phase | Epics | Reason |
|---|---|---|
| **Phase 5** | TV16, TV17 | Scene variants and rig authoring unlock the most immediately useful terrain review/flyover workflows with low architectural risk. |
| **Phase 6** | TV18 | A delivery queue only becomes coherent after there is a stable variant model and a reusable camera-authoring layer. |
| **Phase 7** | TV19 | Collaborative review should land only after single-user review and delivery workflows are explicit and persistence semantics are stable. |
| **Conditional branch** | Terrain night-light fidelity | Only if night-time built-environment terrain scenes become a named product requirement. |

---

## 23. Final Assessment After the Third Pass

After TV5-TV15, the highest-value remaining terrain gaps are no longer only about shading. The package is still missing the workflow layer that makes terrain scenes easy to compare, animate, batch-render, and review with other people.

The honest priority order is:

1. **Finish image-quality fundamentals already on the board**: TV5, TV11, TV12, TV13.
2. **Add scene variants and camera rigs next**: TV16 and TV17 are the most leverage-heavy workflow additions and have the lowest scope risk.
3. **Add a bounded delivery queue before any broad editor ambitions**: TV18 closes a real Blender/Unreal workflow gap without dragging Forge3D into full editor scope.
4. **Keep collaboration review-only**: TV19 is worthwhile, but only as synchronized review state on top of the existing viewer protocol, not as replication or gameplay infrastructure.

That is the surgically precise terrain backlog after a third systematic pass. The remaining out-of-scope items from Blender and Unreal are still out of scope for the same reasons recorded earlier: they either require an editor product, a game-engine runtime, or a renderer-architecture rewrite.
