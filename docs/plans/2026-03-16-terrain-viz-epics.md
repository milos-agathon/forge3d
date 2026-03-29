# Forge3D Terrain Visualization Epic Backlog

**Date:** 2026-03-29
**Sources:** `docs/plans/forge3d_vs_unreal_gap_analysis.md`, direct audit of `src/terrain`, `python/forge3d`, `src/viewer`, `tests/`, and `examples/`, plus a 2026-03-29 re-audit of the current runtime/API surfaces. `docs/superpowers/specs/*` and `docs/superpowers/plans/*` were treated as design and implementation-plan documents, not proof of shipped runtime support.
**Scope:** Terrain-first 3D visualization only. This roadmap covers terrain rendering, terrain population, terrain review, and terrain delivery. It excludes editor-product work, gameplay systems, character systems, and generic DCC parity unless they directly improve terrain scenes.

---

## 1. Baseline

Forge3D does not have a terrain-rendering foundation problem. It already has the core pieces of a credible terrain visualization stack. This document tracks the current implementation status of the terrain epics and keeps the remaining backlog limited to the terrain-specific gaps that still materially affect image quality, scene scale, review workflow, or delivery workflow.

### Implemented foundations

- **TV1:** terrain atmosphere path parity is shipped.
- **TV2:** terrain AOV capture and multi-channel EXR export are shipped.
- **TV3:** terrain scatter is shipped, including deterministic placement, culling, and manual LOD selection.
- **TV4:** terrain material variation is shipped, including shared terrain noise functions and bounded procedural controls.
- **TV5:** terrain local probe lighting is shipped, including diffuse irradiance probes, local reflection probes, placement/rebake invalidation, debug views, and probe memory reporting.
- **TV6:** terrain volumetrics are shipped. The codebase has a public volumetrics settings surface, native decode, dedicated terrain-viewer volumetrics pass/shader, and both screen and offscreen application paths with examples and tests.
- **TV10:** terrain subsurface materials are shipped. The public Python material-layer surface, native decode path, terrain shader integration, and regression coverage are all present in this worktree.
- **TV12:** offline terrain render quality is shipped. Deterministic accumulation, adaptive sampling, and optional OIDN-based denoising are wired through the public Python API.
- **TV13:** terrain population LOD pipeline is shipped, including QEM mesh simplification, auto-generated scatter LOD chains, and HLOD clustering/runtime integration.
- **TV21:** terrain-mesh blending and contact integration are shipped, including per-batch blend/contact controls across the native renderer and viewer paths.
- **TV22:** scatter wind animation is shipped, including GPU deformation, viewer/offscreen wiring, validation, and regression coverage.
- Terrain material virtual texturing is shipped, including paged albedo-family streaming, queryable residency stats, an end-to-end demo, and regression coverage. The public contract already reserves normal and mask families for a later native extension.
- Terrain rendering already includes clipmap terrain, COG/COPC/3D Tiles ingestion, PBR + POM, planar water reflections, vector overlays, cloud shadows, bloom, tone mapping, and height-aware lighting controls.
- Terrain viewer post-processing already includes TAA, depth of field, and camera motion blur.
- Generic virtual-texture and page-table foundations are now exercised by the shipped terrain-material VT path rather than existing only as renderer-side scaffolding.

### Repository Hygiene Note

As of 2026-03-25, the clean local worktrees for `epic-12-clean`, `epic-21`, `epic-22`, and `tv12-ship-release` were removed after confirming their commits already exist on `origin`. The local `epic-12` worktree and branch were then removed after audit: its branch tip `98fb618` was already contained in `origin/main` via the `tv12-ship-release` merge, and the leftover worktree-only edits were stale drift rather than unshipped TV12 work.

The sections below separate implemented epics from work that is still open.

As of the 2026-03-29 re-audit, none of the epics listed below as still open have moved to shipped runtime coverage in this worktree. The main change from the earlier draft is tighter wording around the existing foundations: bundles persist terrain metadata, presets, and camera bookmarks; camera animation supports one keyframed path plus frame export; water, curve/ribbon generation, and terrain post effects provide adjacent building blocks but do not by themselves close the remaining terrain-review, delivery, weather, or renderer-gap epics.

---

## 2. Selection Rules

A feature stays in this roadmap only if it satisfies all of the following:

1. It materially improves terrain image quality, terrain scale, terrain population, terrain review, or terrain delivery.
2. It fits Forge3D as an additive subsystem or a terrain-path extension.
3. It does not require turning Forge3D into a full editor product or a general-purpose game engine.
4. It can be implemented without regressing clipmap streaming, COG/COPC/3D Tiles throughput, memory accounting, or the Python and IPC control surfaces.

---

## 3. Feasibility Scale

| Rating | Meaning | Effort | Risk |
|---|---|---:|---|
| **F1** | Natural fit; mostly wiring or a contained extension | 5-15 pd | Negligible |
| **F2** | New terrain subsystem or medium extension | 15-45 pd | Low |
| **F3** | Cross-cutting subsystem touching several render paths | 45-90 pd | Medium |

Only F1-F3 work is considered here. Higher-cost engine-scale work stays out of the terrain roadmap.

---

## 4. Verified Epic Status Matrix

| Feature | Blender / Unreal / top-engine reference | Verified repo status | Feasibility | Desirability | Decision |
|---|---|---|---|---|---|
| **Local reflection probes** | Blender: reflection cubemaps; Unreal: local reflection capture | **Implemented.** Terrain local reflection probes are exposed through `ReflectionProbeSettings`, reuse the diffuse-probe placement discipline, feed the terrain and water specular fallback path, expose memory/debug reporting, and are covered by regression tests plus a real-DEM demo. | F2 | Medium-High | **Done** |
| **Terrain subsurface materials** | Blender: Cycles SSS for snow/ice/earth; Unreal: Subsurface Profile shading | **Implemented.** Terrain-layer subsurface controls exist in `terrain_params.py`, are decoded in the native terrain path, and are consumed by the terrain shader with dedicated regression tests. | F2 | High | **Done** |
| **Offline terrain quality pipeline** | Blender: accumulation AA, adaptive sampling, OIDN; Unreal: movie render quality passes | **Implemented.** `OfflineQualitySettings`, `render_offline`, adaptive scheduling, and optional OIDN denoise support are present and covered by dedicated TV12 tests. | F2 | High | **Done** |
| **Heterogeneous terrain volumetrics** | Blender: heterogeneous volume shading; Unreal: localized atmospheric FX | **Implemented in shipped scope.** Terrain volumetric fog and light shafts are exposed through `VolumetricsSettings`, decoded natively, applied in dedicated terrain-viewer screen/offscreen passes, and covered by examples/tests. The old backlog wording around bounded 3D density volumes was stale relative to the shipped feature surface. | F2 | Medium-High | **Done** |
| **Weather particles** | Blender: particles; Unreal: GPU weather FX | **Missing.** TV6 volumetric density volumes can suggest plume/ash-like atmospherics, but there is still no terrain-oriented GPU weather emitter/update/render path for rain, snow, dust, or ash particles. | F3 | Medium | **Build later** |
| **Terrain population LOD pipeline** | Blender: simplification workflows; Unreal: auto LOD + HLOD | **Implemented.** `simplify_mesh`, `generate_lod_chain`, `auto_lod_levels`, and HLOD clustering/runtime integration are present with documentation, a real-DEM demo, and regression coverage. | F2 | High | **Done** |
| **Terrain material virtual texturing** | Unreal: virtual textures / RVT-style terrain material streaming; top engines: large-scene material paging | **Implemented in shipped v1 scope.** Terrain VT settings, source registration, feedback-driven residency, queryable stats, a real-DEM demo, and regression coverage are present in the terrain path. The shipped runtime currently pages the albedo family; normal and mask families remain forward-compatible in the Python contract but are not yet decoded natively. | F2-F3 | High | **Done** |
| **Terrain-mesh blending and contact integration** | Unreal: terrain/mesh blending, contact integration, terrain-aware surface transitions | **Implemented.** Terrain/mesh seam softening, terrain-aware contact darkening, per-batch controls, viewer IPC wiring, a demo, and regression coverage are present in the shipped codebase. | F2 | Medium-High | **Done** |
| **Scatter wind animation** | Unreal: foliage wind; Blender: animated vegetation workflows | **Implemented.** Scatter wind settings, GPU deformation, viewer/offscreen wiring, time-driven animation plumbing, validation, examples, and regression coverage are present. | F1-F2 | Medium-High | **Done** |
| **Terrain scene variants and review layers** | Unreal: Data Layers; Blender-adjacent: alternate scene states | **Partial.** Bundles persist terrain metadata, camera bookmarks, and presets, but there is still no named terrain-variant schema, grouped review-layer visibility model, or atomic apply/list API. | F1-F2 | High | **Build** |
| **Terrain camera rig toolkit** | Unreal: camera rigs; Blender: path- and constraint-driven flyovers | **Partial.** `CameraAnimation`, example flyovers, and offline frame enumeration exist, but reusable orbit/rail/target-follow terrain rig primitives and bake-to-animation APIs do not. | F1-F2 | High | **Build** |
| **Terrain shot queue and bounded timeline** | Unreal: Movie Render Queue + Sequencer; Blender: repeatable batch render workflows | **Partial.** Terrain AOV output and single-animation frame export exist, but there is no terrain shot manifest, bounded track model, pass-aware multi-shot queue, or resume semantics. | F2-F3 | High | **Build** |
| **Page-based terrain shadowing** | Unreal: Virtual Shadow Maps | **Missing.** Design and implementation-plan docs exist for TV11, but the runtime terrain shadow path audited here still does not expose `shadow_backend`, paged-shadow settings, a shadow page cache, or paged-shadow stats. | F3 | Medium | **Defer** |
| **Coastal / hydrology water upgrade** | Blender: Ocean modifier; Unreal: higher-end water simulation | **Missing.** Current water remains an analytic wave/foam path with `ocean`/`lake`/`river` presets rather than an FFT/spectrum-driven coastal or hydrology workflow. | F2 | Medium | **Conditional** |
| **OCIO color-managed terrain output** | Blender: OCIO pipeline | **Missing.** No active OpenColorIO terrain output path or public OCIO surface is present. | F3 | Medium | **Conditional** |
| **Terrain flow and trajectory visualization** | Blender: curve/path rendering; Unreal: ribbon/trail effects | **Partial.** Curve, ribbon, and tube geometry foundations exist, but there is still no terrain-aware flow or trajectory visualization workflow on top of them. | F1-F2 | Medium | **Conditional** |
| **Compute tessellation for terrain** | Unreal / top engines: tessellation and displacement for close terrain detail | **Missing.** Generic mesh subdivision exists, but there is no terrain-specific compute tessellation pipeline or public terrain API for adaptive tessellation. | F2 | Medium | **Conditional** |
| **Collaborative terrain review** | Unreal: collaborative viewing | **Partial.** The viewer IPC surface remains a good substrate, but there is still no multi-client terrain review session model, shared-state broadcast path, or shared annotation/camera workflow. | F2 | Medium | **Conditional** |
| **Terrain temporal upscaling** | Unreal: TSR / temporal upscalers; top engines: resolution scaling for heavy terrain scenes | **Missing.** TAA exists, and some screen-space subsystems upscale their own buffers, but there is still no lower-resolution terrain render plus temporal upscale path for the terrain renderer or terrain viewer. | F3 | Medium | **Conditional** |

---

## 5. Core Build Backlog

The epics below are the remaining terrain-first work that is both feasible and worth building now. Local reflection probes, TV13, TV21, and TV22 are no longer part of this section because they now have shipped runtime coverage.

### Review and Delivery

### Epic TV16 - Terrain Scene Variants and Review Layers

**Why this is in scope:** Terrain review usually involves comparing alternate states of one scene, not reloading separate projects for every overlay, scatter, or lighting choice.

**Feasibility:** F1-F2  
**Estimate:** 10-16 pd
**Priority:** P2

| Task | Scope | Definition of done |
|---|---|---|
| **TV16.1 Add named terrain variants and review layers** | Extend bundles and runtime state with named variants and grouped layer visibility. | A bundle can persist named variants that share one terrain asset set, and the schema distinguishes between per-layer visibility and whole-variant activation. |
| **TV16.2 Add atomic apply/list APIs** | Make variants usable from Python and IPC without ad hoc state juggling. | Clients can list variants, query the active variant, toggle layers, and apply a variant in one state transition; invalid IDs fail explicitly. |
| **TV16.3 Add persistence and state-isolation tests** | Ensure variant switching is deterministic and bounded. | Tests prove that switching variants changes only declared overlay/scatter/label/preset state and survives bundle round-trip without silent state loss. |

### Epic TV17 - Terrain Camera Rig Toolkit

**Why this is in scope:** Keyframes exist, but terrain flyovers are still too manual. The missing layer is reusable rig authoring for orbit, rail, and constrained follow cameras.

**Feasibility:** F1-F2  
**Estimate:** 12-20 pd
**Priority:** P2

| Task | Scope | Definition of done |
|---|---|---|
| **TV17.1 Add terrain-relevant rig primitives** | Implement orbit, rail/path, and target-follow rigs with terrain-clearance control. | Each rig produces a deterministic camera path from a small parameter set, and clearance-constrained rigs do not intersect the terrain on representative DEMs. |
| **TV17.2 Bake rigs into `CameraAnimation`** | Keep one playback/runtime model by compiling rigs into editable camera keyframes. | Rigs bake into `CameraAnimation`, can be inspected and edited, and render through the existing frame queue without a second animation runtime. |
| **TV17.3 Add rig examples and tests** | Cover usability and safety. | At least two example flyovers and regression tests verify deterministic timing, frame counts, and clearance behavior. |

### Epic TV18 - Terrain Shot Queue and Bounded Timeline

**Why this is in scope:** Forge3D can render one terrain sequence, but it still lacks a coherent terrain delivery workflow for multi-shot output.

**Feasibility:** F2-F3
**Estimate:** 18-28 pd
**Priority:** P3

| Task | Scope | Definition of done |
|---|---|---|
| **TV18.1 Define a terrain shot manifest** | Introduce a serializable shot/job format for terrain outputs. | One manifest can describe multiple shots with render size, frame range, active variant, and pass selection; invalid manifests fail before rendering starts. |
| **TV18.2 Add bounded timeline tracks** | Support only terrain-relevant animated properties. | A shot can animate camera, sun, sky, exposure/tonemap, active variant, and overlay visibility; v1 explicitly excludes audio, gameplay events, and arbitrary object tracks. |
| **TV18.3 Build pass-aware multi-shot rendering** | Run queued terrain renders with structured outputs and progress reporting. | A queue can render multiple shots to a stable directory layout with beauty PNG plus optional EXR/AOV outputs and can resume interrupted runs without re-rendering completed frames. |

### Epic TV7 - Weather Particle Foundation

**Why this is in scope:** Rain, snow, dust, ash, and airborne particulate events are terrain-visualization features, not game-engine fluff.

**Feasibility:** F3  
**Estimate:** 25-40 pd  
**Priority:** P4

| Task | Scope | Definition of done |
|---|---|---|
| **TV7.1 Add a minimal GPU particle core** | Implement GPU spawn, update, and billboard render for terrain scenes. | A terrain scene can spawn, update, and render at least one weather emitter entirely on GPU, with tracked memory usage and a clean disable path. |
| **TV7.2 Add terrain-aware collision / kill behavior** | Make particles interact with the terrain heightfield without CPU readback. | Rain, snow, dust, or ash particles can collide with or die on terrain surfaces based on heightfield sampling. |
| **TV7.3 Ship only terrain-relevant presets** | Keep the v1 scope narrow. | The initial preset set is limited to rain, snow, dust, and ash and does not expand into generic game-VFX concepts. |

---

## 6. Conditional and Deferred Epics

The items below are valid, but they are not core near-term terrain backlog. They are intentionally not decomposed into task tables yet. Doing that now would add speculative work to the plan before the entry conditions are met.

| Epic | Feasibility | Estimate | Status | Why it is not core right now |
|---|---|---:|---|---|
| **TV8 - Coastal / Hydrology Water Upgrade** | F2 | 20-35 pd | Conditional | Worth doing only if coastal, estuary, open-water, or hydrology visualization becomes a named product requirement. |
| **TV9 - OCIO Color-Managed Terrain Output** | F3 | 25-45 pd | Conditional | Valuable for VFX interchange, but high regression risk because it changes final pixel semantics across terrain outputs. |
| **TV11 - Page-Based Terrain Shadowing** | F3 | 22-40 pd | Deferred | Real work with real value, but lower ROI than terrain VT, offline quality, population scale, and terrain/mesh integration. |
| **TV14 - Terrain Flow and Trajectory Visualization** | F1-F2 | 10-18 pd | Conditional | Useful for hydrology, meteorology, or geomorphology workflows, but not core to general terrain visualization. |
| **TV15 - Compute Tessellation for Terrain** | F2 | 20-35 pd | Conditional | Current clipmaps + POM + detail normals cover most practical close-up terrain needs. |
| **TV19 - Collaborative Terrain Review** | F2 | 16-26 pd | Conditional | Useful, but it should follow scene variants, camera rigs, and the single-user shot/delivery workflow rather than precede them. |
| **TV23 - Terrain Temporal Upscaling and Upscaled Viewer Path** | F3 | 18-32 pd | Conditional | TAA already exists. Add a true upscaled viewer path only if interactive high-resolution terrain performance becomes a stated product goal. |

---

## 7. Explicitly Out of Scope

The following should not be smuggled back into the terrain roadmap:

| Feature family | Why it stays out |
|---|---|
| **Nanite-class virtualized geometry** | Architecturally mismatched with Forge3D's terrain-first rendering model and unnecessary for the terrain problem clipmaps and tiles already solve well. |
| **Lumen-class fully dynamic GI** | Much larger renderer work with worse terrain-first ROI than local probe lighting. |
| **Material graph editors, Geometry Nodes equivalents, in-editor sculpting, full DCC authoring** | These are editor-product commitments, not terrain-rendering epics. |
| **Skeletal animation, hair/fur pipelines, advanced character systems** | Not terrain visualization work. |
| **Gameplay frameworks, AI, audio systems, multiplayer replication** | Engine-product concerns, not terrain roadmap items. |

---

## 8. Effort Summary

### Implemented foundations

| Epic | Status |
|---|---|
| TV1 - Terrain Atmosphere Path Parity | Implemented |
| TV2 - Terrain Output and Compositing Foundation | Implemented |
| TV3 - Terrain Scatter and Population | Implemented |
| TV4 - Terrain Material Variation Upgrade | Implemented |
| TV5 - Terrain Local Probe Lighting | Implemented |
| TV6 - Heterogeneous Terrain Volumetrics | Implemented |
| TV10 - Terrain Subsurface Materials | Implemented |
| TV12 - Terrain Offline Render Quality | Implemented |
| TV13 - Terrain Population LOD Pipeline | Implemented |
| Terrain Material Virtual Texturing | Implemented |
| TV21 - Terrain-Mesh Blending and Contact Integration | Implemented |
| TV22 - Scatter Wind Animation | Implemented |

### Core build backlog

| Epic | Low | High |
|---|---:|---:|
| TV16 - Terrain Scene Variants and Review Layers | 10 pd | 16 pd |
| TV17 - Terrain Camera Rig Toolkit | 12 pd | 20 pd |
| TV18 - Terrain Shot Queue and Bounded Timeline | 18 pd | 28 pd |
| TV7 - Weather Particle Foundation | 25 pd | 40 pd |
| **Total core backlog** | **65 pd** | **104 pd** |

### Conditional and deferred backlog

| Epic | Low | High |
|---|---:|---:|
| TV8 - Coastal / Hydrology Water Upgrade | 20 pd | 35 pd |
| TV9 - OCIO Color-Managed Terrain Output | 25 pd | 45 pd |
| TV11 - Page-Based Terrain Shadowing | 22 pd | 40 pd |
| TV14 - Terrain Flow and Trajectory Visualization | 10 pd | 18 pd |
| TV15 - Compute Tessellation for Terrain | 20 pd | 35 pd |
| TV19 - Collaborative Terrain Review | 16 pd | 26 pd |
| TV23 - Terrain Temporal Upscaling and Upscaled Viewer Path | 18 pd | 32 pd |
| **Total conditional / deferred backlog** | **131 pd** | **231 pd** |

---

## 9. Execution Order

| Phase | Epics | Reason |
|---|---|---|
| **Phase 1** | TV16, TV17 | Add review-state management and reusable flyover authoring now that the remaining renderer-only backlog is narrower. |
| **Phase 2** | TV18 | Build the shot queue and bounded timeline after variants and rig authoring exist. |
| **Phase 3** | TV7 | Weather particles matter, but they are less foundational than the review and delivery tooling backlog. |
| **Later / optional** | TV8, TV9, TV11, TV14, TV15, TV19, TV23 | Only pursue when product direction or scene requirements clearly justify them. |

---

## 10. Bottom Line

Forge3D no longer has a terrain-roadmap problem of "missing everything." Terrain local probe lighting, terrain material virtual texturing, TV6, TV10, TV12, TV13, TV21, and TV22 are now in the shipped column, so the real remaining gaps are narrower and more concrete. The 2026-03-29 repo re-audit did not change that status split:

1. **Review and delivery workflow:** TV16, TV17, and TV18.
2. **Terrain atmosphere and weather richness:** later TV7.
3. **Conditional renderer upgrades that still are not in runtime code despite adjacent design work:** especially TV11.

Terrain-local specular probes, terrain material paging, terrain population scale, terrain/mesh integration, and scatter wind now have shipped first-pass runtime coverage rather than sitting in the remaining core backlog.

That is the coherent terrain backlog. Everything else from Blender, Unreal, or other top-tier engines should remain conditional, deferred, or out of scope unless the product direction changes.
