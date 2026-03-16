# Forge3D vs Unreal Engine 5: Feature Gap Analysis & Feasibility Assessment

**Date:** 2026-03-14
**Scope:** Rigorous feature-by-feature comparison against UE5.4/5.5, with feasibility ratings for each gap within Forge3D's WebGPU/Rust/Python architecture.
**Methodology:** Each missing feature is rated on a 5-point feasibility scale considering architectural compatibility, effort, and risk of regression.

---

## Feasibility Rating Scale

| Rating | Meaning | Effort | Risk to Existing Functions |
|--------|---------|--------|---------------------------|
| **F1 — Natural Fit** | Aligns with existing architecture; incremental extension | 1–4 weeks | Negligible |
| **F2 — Moderate Extension** | Requires new modules but no architectural rework | 1–3 months | Low |
| **F3 — Significant Investment** | New subsystem; touches multiple existing modules | 3–6 months | Medium — requires integration testing |
| **F4 — Major Undertaking** | Fundamental new capability; may need architectural scaffolding | 6–12 months | Medium-High — careful API design needed |
| **F5 — Architectural Mismatch** | Conflicts with core design assumptions or would require near-rewrite | 12+ months | High — may compromise existing strengths |

---

## 1. RENDERING — Advanced Material Systems

### 1.1 What Forge3D Has
- PBR materials (albedo, roughness, metallic, IOR, emissive)
- 13-model BRDF dispatch: Lambert, Phong, Blinn-Phong, Oren-Nayar, Cook-Torrance (GGX + Beckmann), Disney Principled, Ashikhmin-Shirley, Ward, Toon, Minnaert, plus Subsurface/Hair enums that currently dispatch to fallback shaders (Disney Principled and Ashikhmin-Shirley respectively) rather than dedicated implementations
- MaterialShading struct exposes clearcoat, subsurface, and anisotropy parameters (consumed by Disney Principled path)
- Material presets for buildings (18 types)
- Environment mapping / IBL
- Normal mapping (feature-gated)
- Terrain PBR + parallax occlusion mapping (POM) via dedicated `terrain_pbr_pom.wgsl` shader
- Colormap-driven terrain shading (100+ colormaps)
- Dual-source OIT (Order Independent Transparency) with WBOIT fallback, multiple quality levels (Low/Medium/High/Ultra)
- Planar reflections with clip plane, Fresnel blending, and quality presets (suitable for water surfaces)

### 1.2 What Unreal Has That Forge3D Lacks

| Feature | UE5 System | Feasibility | Notes |
|---------|-----------|-------------|-------|
| **Layered/substrate materials** | Substrate (formerly Shading Models) — modular material layers with independent BSDF stacks | **F3** | Forge3D's raster material system is parameter-driven rather than truly layered. Adding a compositor is feasible, but it requires a new material-composition abstraction rather than a few extra BRDF enums. |
| **Full subsurface profile / screen-space diffusion** | Subsurface Profile shading model for skin, wax, foliage | **F2** | Forge3D's `BrdfModel::Subsurface` enum exists but dispatches to Disney Principled rather than a dedicated SSS shader. No separable screen-space diffusion pass or profile-based scattering exists. The parameter plumbing is in place; the gap is the post-pass. |
| **Hair/fur rendering** | Strand-based and card-based hair systems | **F4** | `BrdfModel::Hair` exists but dispatches to Ashikhmin-Shirley fallback. No strand/card geometry, groom asset pipeline, Marschner model, or dedicated visibility pipeline. |
| **Dedicated cloth shading model** | Ashikhmin/Charlie BRDF for fabric | **F1** | Ashikhmin-Shirley BRDF is fully implemented in `shaders/brdf/ashikhmin_shirley.wgsl` and partially serves this role. A dedicated Charlie cloth BRDF with sheen would be a small shader addition. |
| **Material node graph / visual editor** | Material Editor with hundreds of nodes | **F5** | Forge3D is a library, not an IDE. A full material graph editor would be a separate application. The shader pipeline could support compiled material graphs, but building the editor is out of scope. |
| **Decals** | Deferred decals projected onto surfaces | **F2** | Requires a deferred or pre-composite pass reading G-buffer/depth data. The current renderers are orchestrated with explicit passes, so this is still feasible but not a simple drop-in framegraph pass. |

### 1.3 Risk Assessment
The BRDF dispatch infrastructure is solid (13 enum variants, with explicit shader dispatch paths for 11 of them), but Subsurface and Hair currently fall back to proxy shaders. The main remaining material gaps are: layering (F3), dedicated SSS diffusion pass (F2), and standalone cloth/hair shader implementations rather than fallbacks. OIT and planar reflections are already handled. These are shader/data-model extensions with low regression risk. A material graph editor (F5) should still be avoided — it doesn't align with Forge3D's library-first architecture.

---

## 2. RENDERING — Global Illumination & Lighting

### 2.1 What Forge3D Has
- Cascaded shadow maps (CSM)
- SSAO, SSGI, SSR (screen-space)
- IBL with HDR environment maps
- Rectangular area lights via LTC (simplified LUT approximation, not fitted polynomial)
- Shared `shaders/lights.wgsl` sampling helpers for rect, disk, and sphere area-light shapes
- Separate CPU-side `lighting/area_lights.rs` enum/manager that also includes a Cylinder variant, but no dedicated cylinder sampling path is present in `shaders/lights.wgsl`
- Soft light radius calculations
- Bloom, tone mapping (5 operators: Reinhard, Reinhard Extended, ACES, Uncharted2, Exposure + LUT + white balance), TAA with neighborhood clamping and motion vectors
- Sun ephemeris (NOAA Solar Calculator) for accurate directional lighting from geographic coordinates + UTC
- Atmospheric scattering
- Volumetric fog: full ray-marched system with Henyey-Greenstein phase function, light shafts/god-rays, temporal jitter, cascaded shadow integration — present in both offline terrain rendering and interactive viewer
- Volumetric clouds (procedural billboards + IBL integration) and cloud shadows (quality tiers from 256² to 2048², animation presets for calm/stormy)
- Path tracing (megakernel + wavefront) for offline, with ReSTIR (spatial + temporal reuse), importance sampling/guiding, alias table sampling, A-trous edge-aware denoiser, firefly clamping, and multiple AOV channels

### 2.2 What Unreal Has That Forge3D Lacks

| Feature | UE5 System | Feasibility | Notes |
|---------|-----------|-------------|-------|
| **Fully dynamic GI (Lumen-class)** | Software/hardware ray-traced GI with infinite bounces, screen and world probes | **F4** | Forge3D has SSGI (screen-space) and offline path tracing. A real-time probe-based GI system (irradiance probes + screen traces) is a major but architecturally sound addition. The path tracing infrastructure provides the ray-marching foundation. |
| **Virtual Shadow Maps** | Clipmap-based shadow pages, per-pixel shadow resolution | **F3** | Current CSM system works but has cascade transitions. VSM would replace it with a page-table approach. Forge3D already has `page_table.rs` for terrain tiles — the pattern transfers. |
| **Hardware ray tracing (real-time)** | DXR/Vulkan RT for reflections, shadows, AO in real-time | **F4** | WebGPU does not yet expose ray tracing extensions (though `wgpu` has experimental support). Blocked by the WebGPU spec timeline. Forge3D's compute-based path tracer is the interim solution. |
| **IES light profiles** | Measured light distribution for architectural lighting | **F1** | Load IES files as 1D/2D textures, sample in light shader. Trivial extension to the existing area light sampling infrastructure in `shaders/lights.wgsl`. |
| **Fitted polynomial LTC LUT** | Production-quality LTC approximation | **F1** | Current LTC LUT uses simplified approximation (noted in `ltc_lut.rs`). Replacing with Heitz fitted polynomial tables is a data swap, not a code rewrite. |
| **Light propagation volumes / voxel GI** | Alternative GI techniques | **F3** | Would require a voxelization pass and light injection. Significant but self-contained, but it would have to be wired into today's explicit render loops rather than a central live framegraph. |

### 2.3 Risk Assessment
The lighting system is substantially more complete than a first pass suggests — explicit LTC rectangular area lights, shared WGSL sampling helpers for rect/disk/sphere area-light shapes, ray-marched volumetrics with phase functions, and a path tracer with ReSTIR/guiding. IES profiles and LTC LUT upgrade are clean additive features. Lumen-class GI and VSM are larger projects with usable foundations in the compute stack, current shadow system, and terrain page-table code, but they would require explicit integration into the existing render loops. Hardware RT is externally blocked by the WebGPU spec.

---

## 3. RENDERING — Geometry & LOD

### 3.1 What Forge3D Has
- Clipmap-based terrain LOD with quad-tree
- Tile-based streaming with GPU page table
- 3D Tiles SSE-based LOD
- Point cloud octree LOD
- Mesh generation primitives (sphere, box, cylinder, torus, cone, etc.)
- Subdivision and adaptive refinement
- Displacement mapping, mesh welding, validation, thick polyline generation, ribbon/tube generation, curve handling
- UV unwrapping (planar X/Y/Z projection, spherical projection)
- GPU instancing (feature-gated)
- Hierarchical Z-buffer (HZB) for efficient occlusion queries and screen-space effect optimization
- GPU BVH (LBVH with bitonic sort, GPU refit, SAH CPU fallback) and TLAS for path tracing

### 3.2 What Unreal Has That Forge3D Lacks

| Feature | UE5 System | Feasibility | Notes |
|---------|-----------|-------------|-------|
| **Virtualized micropolygon geometry (Nanite-class)** | Arbitrary polygon counts with automatic streaming LOD, no manual LOD chains | **F5** | Nanite is a cluster-based, GPU-driven, visibility-buffer renderer. It would require replacing the forward/deferred pipeline with a visibility buffer, implementing cluster hierarchy generation, and a software rasterizer fallback. This fundamentally conflicts with Forge3D's terrain-first rendering model. The ROI is low — Forge3D's terrain clipmap and 3D Tiles LOD already solve the domain-specific LOD problem. |
| **Automatic mesh LOD generation** | Simplification with preservation of silhouette | **F2** | Mesh decimation (e.g., quadric error metrics) as a preprocessing step. Forge3D's `geometry` module has subdivision; adding simplification is the inverse. Output feeds into existing rendering. |
| **Tessellation / displacement** | Hardware tessellation for terrain detail | **F2** | WebGPU does not have hardware tessellation stages, but compute-shader tessellation is possible. Forge3D already has displacement mapping in the geometry module. A GPU-side adaptive tessellation compute pass is feasible. |
| **Foliage instancing system** | HISM-based foliage painting with wind, LOD, collision | **F3** | Forge3D has GPU instancing (feature-gated). Extending this with hierarchical culling, wind vertex animation, and a foliage placement API is a moderate system. No existing pipeline conflicts. |
| **Procedural mesh generation at runtime** | Dynamic mesh creation from code | **F1** | Already has extensive procedural geometry (`geometry/`). Exposing runtime mesh creation to the viewer pipeline is incremental. |

### 3.3 Risk Assessment
Nanite-class virtualized geometry (F5) should **not be pursued** — it's architecturally misaligned and unnecessary for Forge3D's GIS/terrain domain where clipmap LOD and 3D Tiles already excel. Automatic LOD generation and foliage instancing are safe, orthogonal additions.

---

## 4. PHYSICS

### 4.1 What Forge3D Has
- **None.** Forge3D has no physics simulation system. It has terrain ray casting for picking and height queries, but no rigid body, collision, or dynamics simulation.

### 4.2 What Unreal Has

Chaos Physics: rigid body dynamics, soft body, cloth simulation, destruction, vehicle physics, ragdoll, fluid simulation, physics LOD, parallel computation, substepping.

### 4.3 Gap Assessment

| Feature | Feasibility | Notes |
|---------|-------------|-------|
| **Basic rigid body physics** | **F3** | Integrate an existing Rust physics library (Rapier3D) rather than building from scratch. Rapier provides rigid body, collision detection, joints, and CCD. The scene graph would need a physics component. No existing system conflicts — physics is purely additive. |
| **Cloth simulation** | **F3** | Position-based dynamics (PBD) or mass-spring on GPU compute. Would need a new vertex skinning pass. |
| **Destruction** | **F4** | Voronoi fracture + physics. Requires mesh splitting at runtime, which the geometry module partially supports. |
| **Fluid simulation** | **F4** | SPH or PBF on GPU compute. Self-contained but complex. |
| **Vehicle physics** | **F4** | Requires rigid body foundation first. Layered on top. |

### 4.4 Recommendation
Physics is the **largest categorical gap** but also the least relevant to Forge3D's primary use case (GIS visualization, terrain rendering, cartographic output). If physics is needed for specific use cases (e.g., geological simulation, debris flow modeling), integrating **Rapier3D** via a feature-gated module is the most pragmatic path. It would be a self-contained `src/physics/` module with no impact on the rendering pipeline.

**Risk to existing functions: Negligible** — physics is a simulation layer that feeds positions to the renderer but doesn't modify the rendering pipeline.

---

## 5. AUDIO

### 5.1 What Forge3D Has
- **None.** No audio system of any kind.

### 5.2 What Unreal Has
MetaSounds (node-based DSP), spatial 3D audio, HRTF, convolution reverb, Quartz metronome, sound cues, audio buses, occlusion/attenuation, procedural music generation.

### 5.3 Gap Assessment

| Feature | Feasibility | Notes |
|---------|-------------|-------|
| **Basic spatial audio** | **F2** | Integrate `kira` or `rodio` (Rust audio libraries). Spatial positioning from camera + source coordinates. Self-contained `src/audio/` module. |
| **Full audio DSP graph** | **F4** | Building a MetaSounds equivalent is massive. Using `kira`'s audio graph or `fundsp` provides a reasonable subset. |
| **Audio occlusion** | **F3** | Requires ray casting from listener to source through scene geometry. Forge3D's BVH/picking system could feed occlusion queries. |

### 5.4 Recommendation
Audio is **domain-irrelevant** for Forge3D's core use cases (terrain visualization, cartographic output, scientific rendering). If ambient soundscapes are desired (e.g., for interactive 3D flyovers), a lightweight `rodio` integration as an optional feature-gated module would suffice. This is F2 effort with zero risk to existing systems.

---

## 6. ANIMATION

### 6.1 What Forge3D Has
- Camera keyframe animation with cubic Hermite interpolation
- Frame-by-frame render queue for offline export
- Static glTF mesh import (positions, normals, UVs, indices only)
- No skeletal animation, no IK, no blend spaces, no morph targets

### 6.2 What Unreal Has That Forge3D Lacks

| Feature | Feasibility | Notes |
|---------|-------------|-------|
| **Skeletal animation** | **F3** | Requires: skeleton hierarchy, bone matrices, GPU skinning shader, animation clip playback, and importer support for skins/joints/weights. The current glTF importer is static-mesh-only, so the data-ingest layer must expand before playback exists. |
| **Animation blending / state machines** | **F3** | Layered on skeletal animation. Blend trees and state machines are CPU-side logic feeding bone transforms. |
| **Morph targets / blend shapes** | **F3** | Vertex delta buffers interpolated on GPU are straightforward once the asset pipeline carries morph-target data, but the current glTF importer does not parse that data yet. |
| **Inverse kinematics** | **F3** | FABRIK or CCD solvers are CPU algorithms that output bone transforms. Self-contained, feeds into the skeletal system. |
| **Motion matching** | **F4** | Data-driven animation selection. Requires a large animation database and nearest-neighbor search. |
| **Procedural animation (Control Rig)** | **F4** | Node-based procedural rigging. Would need a runtime graph evaluator. |
| **Root motion** | **F2** | Extract translation from animation clip and apply to actor position. Straightforward once skeletal playback exists. |

### 6.3 Recommendation
If Forge3D expands into architectural visualization or urban simulation (CityJSON buildings with animated elements), skeletal animation becomes relevant. The main gap is not just GPU skinning; it starts at asset ingest, because current glTF support only imports static mesh primitives. This is still tractable, but it is a broader F3 project than the previous draft implied.

**Risk to existing functions: Negligible** — animation is a new pipeline branch, not a modification of existing paths.

---

## 7. ARTIFICIAL INTELLIGENCE

### 7.1 What Forge3D Has
- **None.** No AI, pathfinding, behavior trees, or navigation mesh.

### 7.2 What Unreal Has
Behavior trees, State Trees, Smart Objects, NavMesh, EQS, AI Perception, Mass Entity (ECS), crowd simulation.

### 7.3 Gap Assessment
| Feature | Feasibility | Notes |
|---------|-------------|-------|
| **Navigation mesh generation** | **F3** | Could integrate `recast-rs` (Rust bindings for Recast/Detour). Terrain heightfield → navmesh is a natural fit. Self-contained module. |
| **Behavior trees** | **F2** | Pure logic layer. Rust crates exist (`bonsai-bt`). No rendering involvement. |
| **Crowd simulation** | **F3** | Agent-based simulation with instanced rendering. Forge3D's GPU instancing provides the rendering side. |
| **AI perception** | **F3** | Spatial queries against scene data. Forge3D's BVH/R-tree structures could support this. |

### 7.4 Recommendation
AI is **entirely out of scope** for a terrain visualization engine. If Forge3D were to evolve toward simulation (e.g., urban pedestrian flow, evacuation modeling), these would be domain-specific additions. The rendering architecture imposes no constraints — AI is a CPU-side simulation feeding positions to instanced rendering.

**Risk to existing functions: None.**

---

## 8. NETWORKING & MULTIPLAYER

### 8.1 What Forge3D Has
- IPC-based viewer control (TCP + NDJSON) for Python↔Viewer communication
- No multiplayer, replication, or server architecture

### 8.2 What Unreal Has
Actor replication, dedicated servers, client prediction, lag compensation, session management, RPCs, networked physics.

### 8.3 Gap Assessment

| Feature | Feasibility | Notes |
|---------|-------------|-------|
| **Collaborative viewing (shared camera/annotations)** | **F2** | Extend existing IPC protocol to support multiple clients. The NDJSON protocol is already network-ready. Add session management and state broadcast. |
| **Full state replication** | **F4** | Would require a replication framework for scene state. Significant but Forge3D's scene is relatively simple (terrain + overlays) compared to a game world. |
| **Real-time multiplayer simulation** | **F5** | Fundamentally not what Forge3D is. Would require a game server architecture. |

### 8.4 Recommendation
The pragmatic path is **collaborative viewing** (F2): multiple users viewing the same terrain with shared annotations, camera bookmarks, and vector overlays. This extends the existing IPC system naturally. Full multiplayer replication is unnecessary.

**Risk to existing functions: Low** — the IPC module is already isolated. Extension is additive.

---

## 9. USER INTERFACE

### 9.1 What Forge3D Has
- HUD with debug info and statistics
- Python-side Jupyter widget integration
- No in-engine GUI framework

### 9.2 What Unreal Has
UMG (Unreal Motion Graphics), Slate, Common UI, rich text blocks, localization, data binding.

### 9.3 Gap Assessment

| Feature | Feasibility | Notes |
|---------|-------------|-------|
| **Immediate-mode GUI overlay** | **F2** | Integrate `egui` (Rust immediate-mode GUI) for in-viewer panels, sliders, and controls. `egui` has a `wgpu` backend. This is the standard approach for Rust+WebGPU applications. |
| **Retained-mode GUI framework** | **F4** | Building a UMG equivalent is massive. Not recommended. |
| **Localization** | **F1** | String table loading is a library concern, not a rendering concern. Trivial addition. |

### 9.4 Recommendation
**`egui` integration** is the clear path. It's well-proven with wgpu, provides panels, sliders, color pickers, and text input, and renders as an overlay pass that doesn't interfere with the 3D pipeline. Several Forge3D systems (terrain params, lighting controls, camera settings) would immediately benefit from interactive sliders.

**Risk to existing functions: Negligible** — egui renders as a final overlay pass after the 3D scene.

---

## 10. SCRIPTING & EXTENSIBILITY

### 10.1 What Forge3D Has
- Full Python API via PyO3
- IPC command protocol for viewer control
- Configuration objects (TerrainRenderParams, etc.)
- Colormap extensibility hooks via manual registry/provider loading; packaging reserves an entry-point group, but runtime entry-point discovery is not wired yet
- No visual scripting

### 10.2 What Unreal Has That Forge3D Lacks

| Feature | Feasibility | Notes |
|---------|-------------|-------|
| **Visual scripting (Blueprints)** | **F5** | Not aligned with library architecture. Forge3D's Python API *is* the scripting layer. |
| **Hot reload / live coding** | **F2** | Python scripts can already be modified and re-sent via IPC. Adding file-watching and auto-reload is incremental. |
| **Plugin system** | **F2** | A `forge3d.colormaps` entry-point group is declared in packaging, but runtime auto-discovery is not implemented. Turning that into a real plugin API for overlays/shaders/importers is still future scaffolding, not a shipped system. |
| **Native plugin ABI / C++ extension development** | **F3** | Forge3D can be extended natively today from Rust crates. A stable C/C++ plugin ABI does not exist and would require an explicit FFI surface plus compatibility guarantees. |

### 10.3 Recommendation
Forge3D's Python API is already a more accessible scripting layer than Blueprints for its target audience (scientists, cartographers, GIS professionals). The investment should go toward **formalizing the current extension hooks into a real plugin architecture** (F2) and **hot reload** (F2), not visual scripting.

**Risk to existing functions: None.**

---

## 11. WORLD BUILDING & STREAMING

### 11.1 What Forge3D Has
- Clipmap-based infinite terrain
- Tile-based quad-tree streaming
- GPU page table for tile→slot mapping
- Cloud Optimized GeoTIFF (COG) streaming
- 3D Tiles hierarchical streaming
- Point cloud octree streaming (COPC, EPT)
- Terrain-specific streaming: page-table infrastructure (`page_table.rs`) and clipmap streaming (`clipmap/streaming.rs`) provide world-partition–like behaviour **for terrain data only** — not a generalized runtime cell system for arbitrary scene entities
- Scene bundles (.forge3d) for portable scenes

### 11.2 What Unreal Has That Forge3D Lacks

| Feature | Feasibility | Notes |
|---------|-----------|-------|
| **Data Layers (scene variants)** | **F1** | Add named layer sets to scene bundles. Terrain + overlay combinations can be toggled. Minimal change to bundle manifest. |
| **Procedural Content Generation (PCG)** | **F3** | A graph-based procedural system for placing features (buildings, vegetation, infrastructure) on terrain. Forge3D's geometry module provides primitives; the gap is the placement/distribution logic and a graph evaluator. |
| **In-editor terrain sculpting** | **F4** | Forge3D loads pre-existing DEMs. Interactive sculpting would need a brush system writing back to the heightfield GPU texture. The viewer infrastructure supports mouse input; the gap is the brush/sculpt pipeline. |
| **Foliage painting** | **F3** | Instanced mesh placement with density brushes. Requires the foliage instancing system from §3. |
| **HLOD (Hierarchical LOD)** | **F2** | Merge distant geometry into simplified imposters. The 3D Tiles traversal already does something similar. Generalizing to arbitrary meshes is moderate. |

### 11.3 Risk Assessment
Forge3D's streaming architecture is actually **more advanced than Unreal's** for its domain: COG streaming, COPC octrees, and 3D Tiles are GIS-native streaming that Unreal lacks entirely. The gaps are in interactive content authoring (sculpting, painting) rather than runtime streaming.

---

## 12. CINEMATICS & VIRTUAL PRODUCTION

### 12.1 What Forge3D Has
- Camera keyframe animation with cubic Hermite interpolation
- Frame-by-frame render queue for offline export
- High-res snapshots up to 8K
- Path tracing for photorealistic offline rendering
- Path-tracing AOV render/save utilities (beauty, normals, depth, albedo + custom channels)
- A-trous edge-aware denoiser guided by albedo/normal/depth (GPU compute + NumPy fallback)
- Map plate compositor for publication layouts
- SVG/PDF vector export

### 12.2 What Unreal Has That Forge3D Lacks

| Feature | UE5 System | Feasibility | Notes |
|---------|-----------|-------------|-------|
| **Timeline sequencer** | Multi-track timeline for cameras, actors, audio, events | **F3** | Expand the animation module from camera-only to multi-track. Add event tracks, property animation tracks. The IPC protocol could carry timeline commands. |
| **Movie render queue with passes** | Configurable render passes (beauty, depth, normals, motion vectors) | **F2** | Path tracing already has AOV plumbing and file output. The remaining gap is unified offline UX and parity for terrain/viewer renders; terrain-side `render_with_aov` currently exposes the API surface but still defers full MRT capture. |
| **Virtual production / LED volume** | nDisplay, Live Link, In-Camera VFX | **F5** | Entirely different domain. Requires multi-display synchronization, camera tracking integration, real-time compositing. Not feasible or relevant. |
| **Camera rig system** | Dolly, crane, rail cameras | **F2** | Parameterized camera paths beyond keyframes. Spline-based camera rigs with constraints. Extends the existing camera module. |
| **Motion capture integration** | Live Link data streams | **F3** | Would need a protocol adapter (e.g., OSC, VRPN) feeding into the animation system. Self-contained. |

### 12.3 Risk Assessment
Camera rigs (F2) and unified offline pass export (F2) are immediately valuable for terrain flyovers and cartographic animation workflows. A multi-track sequencer (F3) would significantly enhance creator workflow, but it would sit on top of today's camera-only animation foundation rather than a broader animation runtime.

---

## 13. VISUAL EFFECTS (VFX)

### 13.1 What Forge3D Has
- Bloom post-processing
- Depth of field (physically-based with f-stop, hyperfocal distance, circle of confusion)
- Motion blur (temporal accumulation with configurable shutter interval)
- Volumetric clouds and cloud shadows
- Atmospheric scattering
- Volumetric fog with ray-marched god-rays
- Water surface rendering (`water_surface.rs`, 756 lines): animated waves, foam, reflections, refractions, shoreline foam with procedural noise, water masks for terrain-aware rendering, multiple modes (Transparent, Reflective, Animated)
- Order-independent transparency (dual-source OIT + WBOIT fallback)
- No particle system

### 13.2 What Unreal Has That Forge3D Lacks

| Feature | Feasibility | Notes |
|---------|-------------|-------|
| **GPU particle system** | **F3** | Compute-shader particle simulation with spawn, update, and render stages. Forge3D's compute pipeline and OIT system support this. Useful for: rain, snow, dust, volcanic ash, wildfire smoke on terrain. |
| **Fluid simulation (Niagara Fluids)** | **F4** | Grid-based fluid simulation on GPU compute. Significant but self-contained. Useful for: water flow visualization, atmospheric phenomena. Note: Forge3D's water surface system handles surface-level water rendering; this gap is about volumetric fluid dynamics. |
| **Particle collision** | **F3** | Requires depth buffer readback or SDF-based collision. Forge3D's terrain heightfield and HZB provide natural collision surfaces. |
| **Ribbon/trail effects** | **F1** | The geometry module already has ribbon/tube generation with curve handling. Animating these along paths is incremental — the infrastructure is already there. |

### 13.3 Recommendation
A **GPU particle system** is the highest-value VFX addition for terrain visualization — weather effects, volcanic simulations, dust storms, and smoke plumes are directly relevant to geoscience use cases. This is a self-contained compute module with no impact on existing rendering.

**Risk to existing functions: Negligible.**

---

## 14. ASSET PIPELINE

### 14.1 What Forge3D Has
- OBJ, glTF, STL import
- CityJSON, OSM buildings import
- KTX2 texture loading
- HDR/EXR image formats
- Scene bundles (.forge3d)
- Mapbox Style Spec import
- COG, COPC, EPT, 3D Tiles streaming

### 14.2 What Unreal Has That Forge3D Lacks

| Feature | Feasibility | Notes |
|---------|-------------|-------|
| **FBX import** | **F2** | Add `fbx` crate or use Assimp bindings. FBX carries skeletal animation, materials, and LOD chains. |
| **USD (Universal Scene Description)** | **F3** | Increasingly important for interop with DCC tools. Rust bindings exist but are immature. |
| **Datasmith-class CAD import** | **F4** | Direct CAD (STEP, IGES) import requires geometry kernel integration. Heavy. |
| **Automatic LOD chain generation** | **F2** | Mesh simplification as preprocessing. See §3. |
| **Asset hot-reload** | **F2** | File watching + GPU resource invalidation. The tile cache and resource tracker already handle resource lifecycle. |
| **Texture compression (BC/ASTC at import)** | **F2** | Forge3D has `compressed_textures.rs`. Adding offline compression (basis_universal via `basis-universal` crate) for imported textures is moderate. |

### 14.3 Risk Assessment
Asset pipeline additions are inherently **low-risk** — they're input processing that produces data for existing renderers. FBX and USD import are the highest-value additions for interoperability.

---

## 15. INPUT SYSTEM

### 15.1 What Forge3D Has
- Keyboard and mouse input via winit
- Orbit and FPS camera controllers
- Picking (click, lasso, hover)
- IPC commands from Python

### 15.2 What Unreal Has That Forge3D Lacks

| Feature | Feasibility | Notes |
|---------|-------------|-------|
| **Gamepad support** | **F1** | The current input stack (`viewer_input.rs`) handles keyboard, mouse, cursor, and wheel events only — no gamepad plumbing or dependency exists in `Cargo.toml`. This is still a small additive feature, but it requires introducing dedicated gamepad input handling rather than just mapping an already-present event path. |
| **Touch input (mobile)** | **F2** | winit supports touch events. Pinch-to-zoom, pan gestures. Requires touch gesture recognition layer. |
| **Input remapping** | **F1** | Configuration-driven key bindings. Extend `viewer_config.rs`. |
| **VR/XR input** | **F4** | Requires OpenXR integration. WebGPU + OpenXR is still emerging. |
| **Enhanced input abstraction** | **F2** | Action-based input system (press, hold, tap, combo). Replaces raw event handling with semantic actions. |

### 15.3 Risk Assessment
Input extensions are **zero-risk** to existing rendering. Gamepad and remapping are quick wins. VR/XR is externally constrained by the WebGPU+OpenXR ecosystem.

---

## 16. PLATFORM & DEPLOYMENT

### 16.1 What Forge3D Has
- Windows (primary, pre-built wheels)
- Linux (x86_64 + aarch64 wheels)
- macOS (via wgpu Metal backend)
- Python package on PyPI
- No mobile, no console, no web

### 16.2 What Unreal Has That Forge3D Lacks

| Feature | Feasibility | Notes |
|---------|-------------|-------|
| **WebAssembly/WebGPU browser target** | **F3** | wgpu compiles to WASM+WebGPU. The Rust core could target wasm32. Python bindings wouldn't work — would need a JS/TS API layer. Significant but architecturally viable for the rendering core. |
| **iOS / Android** | **F3** | wgpu supports Metal (iOS) and Vulkan (Android). The challenge is the Python dependency — would need a native API or embedded Python. |
| **Console (PS5, Xbox, Switch)** | **F5** | Requires proprietary SDKs and NDA toolchains. Entirely out of scope. |
| **VR/AR headset rendering** | **F4** | Stereo rendering + head tracking. wgpu can render to two viewports. The viewer would need VR camera handling. |

### 16.3 Recommendation
**WebAssembly** is the highest-impact platform expansion — it would enable browser-based terrain visualization with the existing WGSL shaders. The blockers are: (1) Python API wouldn't transfer, needing a JS wrapper, and (2) wgpu WASM support is still maturing. This is a medium-term strategic initiative, not a quick addition.

**Risk to existing functions: None** — cross-compilation doesn't modify the desktop codebase.

---

## 17. SCENE MANAGEMENT & ECS

### 17.1 What Forge3D Has
- `Scene` as the central monolithic container (**77 declared fields** starting at `scene/mod.rs:50`, across ~6 438 lines) holding terrain pipeline, MSAA textures, height samplers, all screen-space effect state, reflection/DOF/cloud/water/ground-plane renderers, light arrays, OIT state, overlays, text overlays, 3D text, and GPU instancing — each subsystem stored as `Option<RendererType>` or direct state
- Standalone hierarchical scene graph (`scene_graph.rs`, 522 lines): parent-child nodes, local/world transforms, dirty-flag propagation, cycle detection, visitor pattern
- Matrix stack for hierarchy rendering
- Tile cache with resource tracking (resource_tracker: buffers + textures with auto-cleanup on drop)
- GPU memory budget management (per-resource and per-category stats)

### 17.2 What Unreal Has That Forge3D Lacks

| Feature | Feasibility | Notes |
|---------|-------------|-------|
| **Full ECS (Entity Component System)** | **F4** | Forge3D has a standalone scene graph and subsystem separation, but the live runtime centers on a 77-field monolithic `Scene` struct (~6 438 lines) with tight coupling. Moving to archetypes/queries/processors would require decomposing this struct — a substantial refactor with high regression risk to every renderer that currently reaches into `Scene` fields directly. |
| **Garbage collection / reference counting** | **F1** | Rust's ownership model already handles this. GPU resources are tracked by `resource_tracker.rs`. No gap. |
| **Actor spawning/destruction** | **F2** | Dynamic entity lifecycle. Extend the scene to support add/remove at runtime beyond the current overlay API. |
| **Level hierarchy / sub-scenes** | **F2** | Scene composition (main terrain + sub-scenes for buildings, infrastructure). Extend the bundle system. |

### 17.3 Risk Assessment
A formal ECS could improve scalability for high-entity-count scenarios, but it would be a meaningful architectural refactor, not a non-invasive add-on. The current design is better described as subsystem-oriented than ECS-inspired.

---

## 18. GAMEPLAY SYSTEMS

### 18.1 What Unreal Has
Gameplay Ability System (GAS), gameplay tags, attribute system, data tables, cooldowns, effects.

### 18.2 Assessment
These are **entirely irrelevant** to Forge3D. GAS exists for action games with abilities, health, mana, buffs/debuffs. Forge3D is a visualization engine. Including any of these systems would bloat the codebase with zero benefit.

**Recommendation: Do not implement. F5 — architectural mismatch with purpose.**

---

## 19. TOOLS & EDITOR

### 19.1 What Forge3D Has
- Python scripting as the primary authoring tool
- IPC-based viewer with debug HUD
- Benchmark utilities
- Memory metrics (`memory_tracker/` with per-resource and per-category stats, auto-cleanup on drop)
- GPU timing infrastructure (`gpu_timing.rs`, 399 lines): timestamp queries, debug markers for RenderDoc/Nsight/RGP, `gpu_time!()` convenience macro, async result readback. Pipeline statistics query scaffolding exists but collection is **commented out** (`get_results()` returns `pipeline_stats: None`). Timing is **partially integrated**: wired into GI re-execution/capture helpers in `viewer_helpers.rs`, but not into the main per-frame viewer render loop. **Not exported via Python bindings.**
- Multi-threaded command recording (`multi_thread/`) and async compute prepasses (`async_compute/`) for GPU pipeline parallelization
- Staging rings, double buffering, big buffer management for efficient GPU uploads
- No standalone editor application

### 19.2 What Unreal Has That Forge3D Lacks

| Feature | Feasibility | Notes |
|---------|-------------|-------|
| **Standalone editor application** | **F5** | Building an Unreal Editor equivalent is a multi-year, multi-team project. Not aligned with library-first architecture. |
| **GPU profiler with polished timeline/HUD surfacing** | **F2** | `GpuTimingManager` (399 lines) with timestamp queries, debug markers, and `gpu_time!()` macro exists. Pipeline statistics query scaffolding is present but **collection is commented out**. Timing is partially integrated (GI re-execution helpers in `viewer_helpers.rs`) but **not in the main per-frame render loop** and **not exposed via Python**. The gap is completing pipeline-stat collection, broadening integration across render paths, and surfacing via HUD/Python. |
| **Memory profiler visualization** | **F1** | `memory_tracker.rs` already tracks allocations. Add a visualization overlay (egui panel) or Python export. |
| **Live property editing** | **F2** | IPC already supports parameter updates. An egui panel reflecting `TerrainRenderParams` with real-time sliders would close this gap. |
| **Asset browser** | **F4** | A visual asset browser is an editor feature. Not aligned with library architecture. |

### 19.3 Recommendation
**Profiling surfacing** (F2) and **live property editing** via egui (F2) are the highest-value tool additions. They enhance the developer experience without requiring a standalone editor. A standalone editor is explicitly out of scope.

---

## CONSOLIDATED PRIORITY MATRIX

### Tier 1: High Value, Low Risk, Feasible (F1–F2)

These are comparatively low-risk extensions. `Epic Coverage` names the delivery epic that owns each feature.

| Feature | Feasibility | Domain Relevance | Epic Coverage |
|---------|-------------|-----------------|---------------|
| Terrain-path clear coat / cloth support | F1 | Architectural viz | `P8` |
| IES light profiles | F1 | Architectural viz | `P3` |
| Fitted polynomial LTC LUT upgrade | F1 | Lighting quality | `P3` |
| Data Layers (scene variants) | F1 | Core GIS | `P9` |
| Gamepad support | F1 | Interactive exploration | `P10` |
| Input remapping | F1 | Usability | `P10` |
| Memory profiler visualization | F1 | Developer tooling | `P11` |
| Terrain/viewer AOV export parity | F2 | Cartographic output | `P2` |
| Terrain-path anisotropic materials | F2 | Material quality | `P8` |
| Full subsurface profile / screen-space SSS | F2 | Material quality | `P8` |
| Decals | F2 | Annotation | `P12` |
| Automatic mesh LOD | F2 | Performance | `P14` |
| Ribbon/trail effects (animated) | F1 | Visualization | `P13` |
| Camera rig system | F2 | Cinematic | `P13` |
| egui integration (UI overlay) | F2 | Usability | `P5` |
| Hot reload / live coding | F2 | Developer experience | `P15` |
| Plugin architecture | F2 | Extensibility | `P16` |
| FBX import | F2 | Interoperability | `P7` |
| GPU profiling HUD / timeline surfacing | F2 | Developer tooling | `P1` |
| Collaborative viewing | F2 | Multi-user | `P6` |
| HLOD | F2 | Performance | `P14` |
| Texture compression pipeline | F2 | Performance | `P14` |

### Tier 2: High Value, Medium Risk, Significant Effort (F3)

These require new modules but are **architecturally compatible**. `Epic Coverage` names the build epic that owns each feature.

| Feature | Feasibility | Domain Relevance | Epic Coverage |
|---------|-------------|-----------------|---------------|
| GPU particle system | F3 | Weather, geoscience VFX | `P17` |
| Skeletal animation + GPU skinning | F3 | Architectural viz | `P18` |
| Morph targets / blend shapes | F3 | Animation | `P18` |
| Virtual Shadow Maps | F3 | Shadow quality | `P19` |
| Physics (Rapier3D integration) | F3 | Simulation | `P20` |
| PCG (Procedural Content Generation) | F3 | Terrain population | `P21` |
| Foliage instancing | F3 | Vegetation | `P17` |
| Multi-track timeline sequencer | F3 | Cinematic | `P22` |
| WebAssembly target | F3 | Browser deployment | `P23` |
| USD import | F3 | Interoperability | `P24` |
| Layered/substrate materials | F3 | Material quality | `P25` |
| Navigation mesh | F3 | Simulation | `P20` |

### Tier 3: Low ROI or Architectural Mismatch (F4–F5)

These should be **explicitly deferred or rejected**. `Epic Coverage` names the boundary/decision epic that owns the deferral and revisit criteria.

| Feature | Feasibility | Reason | Epic Coverage |
|---------|-------------|--------|---------------|
| Nanite-class virtualized geometry | F5 | Requires a renderer architecture shift toward clustered/visibility-buffer virtualized geometry | `P26` |
| Material node graph editor | F5 | Requires a full material-authoring application and graph compiler surface | `P27` |
| Visual scripting (Blueprints) | F5 | Python API is the scripting layer | `P28` |
| Full multiplayer replication | F5 | Requires a dedicated replication, prediction, and authoritative session model | `P29` |
| Gameplay Ability System | F5 | Requires a gameplay-specific ability, attribute, and effect framework | `P28` |
| Console platform support | F5 | Proprietary SDKs | `P30` |
| Standalone editor application | F5 | Library-first architecture | `P27` |
| Virtual production / LED volume | F5 | Requires multi-display synchronization, camera tracking, and real-time compositing infrastructure | `P30` |
| Hair/fur rendering | F4 | Niche for terrain engine | `P31` |
| Hardware ray tracing | F4 | Requires runtime access to real-time ray-tracing features not present in the current target stack | `P26` |
| Fluid simulation | F4 | High effort, niche use | `P32` |
| Motion matching | F4 | Requires skeletal foundation first | `P31` |
| VR/XR | F4 | Ecosystem immaturity | `P30` |
| In-editor terrain sculpting | F4 | Library, not editor | `P27` |
| CAD import (Datasmith-class) | F4 | Heavy geometry kernel | `P33` |

---

## EXECUTION EPICS

The execution program below gives **every Tier 1, Tier 2, and Tier 3 feature an owning epic**. `P1`-`P16` are delivery epics for Tier 1, `P17`-`P25` are build epics for Tier 2, and `P26`-`P33` are explicit boundary epics that own Tier 3 deferrals as named decisions with revisit criteria.

### Planning Assumptions

- **Impact**
  - **High**: directly improves a core output path, developer loop, or a capability already exposed at the API boundary
  - **Medium**: meaningful improvement for a specific workflow, but not foundational
  - **Low**: niche or mostly preparatory
- **Effort**
  - **S**: ~1-2 engineering weeks
  - **M**: ~2-5 engineering weeks
  - **L**: ~1-3 engineering months

### Epic Overview

| Priority | Epic | Impact | Effort | Covers |
|---------|------|--------|--------|--------|
| **P1** | GPU Profiling Surfacing | **High** | **M** | GPU profiling HUD / timeline surfacing |
| **P2** | Terrain AOV Completion + Viewer AOV Baseline | **High** | **M** | Terrain/viewer AOV export parity |
| **P3** | Lighting Fidelity Polish (LTC + IES) | **Medium** | **M** | IES light profiles; Fitted polynomial LTC LUT upgrade |
| **P4** | Plugin Discovery v1 (Colormaps Only) | **Medium** | **S** | Colormap-plugin discovery slice that seeds `P16` |
| **P5** | Viewer Overlay Platform + Live Controls | **Medium** | **M** | egui integration (UI overlay) |
| **P6** | Collaborative Viewing v1 | **Medium** | **M** | Collaborative viewing |
| **P7** | FBX Import Phase 1 (Static Mesh Only) | **Medium** | **M** | FBX import |
| **P8** | Material Fidelity Pass | **Medium** | **M** | Terrain-path clear coat / cloth support; Terrain-path anisotropic materials; Full subsurface profile / screen-space SSS |
| **P9** | Data Layers & Scene Variants | **Medium** | **S** | Data Layers (scene variants) |
| **P10** | Input Modernization | **Medium** | **S** | Gamepad support; Input remapping |
| **P11** | Memory Profiler Surfacing | **Medium** | **S** | Memory profiler visualization |
| **P12** | Decals & Projected Markup | **Medium** | **M** | Decals |
| **P13** | Motion FX & Camera Toolkit | **Medium** | **M** | Ribbon/trail effects; Camera rig system |
| **P14** | Asset Optimization Pipeline | **Medium** | **M** | Automatic mesh LOD; HLOD; Texture compression pipeline |
| **P15** | Developer Loop Modernization | **Medium** | **M** | Hot reload / live coding |
| **P16** | Plugin Architecture v1 | **Medium** | **M** | General plugin architecture beyond colormaps |
| **P17** | Real-Time VFX & Foliage Systems | **Medium** | **L** | GPU particle system; Foliage instancing |
| **P18** | Character Animation Foundation | **Medium** | **L** | Skeletal animation + GPU skinning; Morph targets / blend shapes |
| **P19** | Virtual Shadow Maps | **Medium** | **L** | Virtual Shadow Maps |
| **P20** | Simulation Foundations | **Medium** | **L** | Physics (Rapier3D integration); Navigation mesh |
| **P21** | Procedural World Generation | **Medium** | **L** | PCG (Procedural Content Generation) |
| **P22** | Sequencer & Cinematic Timeline | **Medium** | **L** | Multi-track timeline sequencer |
| **P23** | WebAssembly Target | **Medium** | **L** | WebAssembly target |
| **P24** | USD Interoperability | **Medium** | **L** | USD import |
| **P25** | Layered Material System | **Medium** | **L** | Layered/substrate materials |
| **P26** | Renderer Architecture Frontier | **Low** | **M** | Nanite-class virtualized geometry; Hardware ray tracing |
| **P27** | Editor Product Boundary | **Low** | **M** | Material node graph editor; Standalone editor application; In-editor terrain sculpting |
| **P28** | Scripting & Gameplay Framework Boundary | **Low** | **S** | Visual scripting (Blueprints); Gameplay Ability System |
| **P29** | Networked Simulation Boundary | **Low** | **M** | Full multiplayer replication |
| **P30** | Platform & Immersive Expansion Boundary | **Low** | **M** | Console platform support; VR/XR; Virtual production / LED volume |
| **P31** | Advanced Character Systems Boundary | **Low** | **M** | Hair/fur rendering; Motion matching |
| **P32** | Advanced Simulation Boundary | **Low** | **M** | Fluid simulation |
| **P33** | CAD/Datasmith Interoperability Boundary | **Low** | **M** | CAD import (Datasmith-class) |

### Epic P1 — GPU Profiling Surfacing

**Impact:** High  
**Effort:** M

**Goal:** Turn the existing timestamp infrastructure into a real developer feature for the interactive viewer and Python callers.

**Specific tasks**

1. Instrument the **main per-frame viewer render loop**, not just GI helper paths, so the viewer can report timings for the passes that dominate real interactive cost.
2. Expose the latest timing snapshot through a supported API/IPC surface for Python and tooling consumers.
3. Make pipeline-statistics support explicit: either ship it as capability-gated functionality or keep it out of the surfaced feature set.
4. Add a lightweight in-viewer timing readout for interactive profiling.

**Definition of done**

- The interactive viewer can report per-pass GPU timings for the **main per-frame render loop**, not just GI re-execution helpers.
- Python and IPC clients can fetch timing data through a stable API surface.
- Unsupported devices return explicit capability metadata rather than silent omission.

### Epic P2 — Terrain AOV Completion + Viewer AOV Baseline

**Impact:** High  
**Effort:** M

**Goal:** Make terrain AOV export real rather than placeholder, then give the viewer a minimal but useful pass-export baseline.

**Specific tasks**

1. Replace the current terrain placeholder path with real albedo, normal, and depth capture.
2. Bring the viewer export path to a minimum parity baseline with terrain for `beauty`, `normal`, and `depth`.
3. Keep the terrain and viewer outputs consistent enough to support offline workflows without separate feature-specific export APIs.

**Definition of done**

- `TerrainRenderer.render_with_aov()` returns populated albedo, normal, and depth data.
- Viewer snapshot/export can emit at least beauty, normal, and depth through public API/IPC.
- The exported buffers have defined semantics rather than placeholder/empty content.

### Epic P3 — Lighting Fidelity Polish (LTC + IES)

**Impact:** Medium  
**Effort:** M

**Goal:** Upgrade the lighting stack where the code already exposes clear placeholder seams.

**Specific tasks**

1. Replace the current placeholder-grade LTC lookup path with production-grade fitted data for the existing rectangular LTC path.
2. Add IES profile support for the raster light types that already fit the current lighting model.
3. Keep the scope narrow: improve measured/fitted lighting fidelity without expanding into a generalized new lighting architecture.

**Definition of done**

- LTC textures are populated with non-placeholder data at runtime.
- At least one supported raster light type can use an IES profile end-to-end.
- The improvement is user-visible, not just internally plumbed.

### Epic P4 — Plugin Discovery v1 (Colormaps Only)

**Impact:** Medium  
**Effort:** S

**Goal:** Turn the existing packaging declaration into a real runtime extension mechanism for colormaps.

**Specific tasks**

1. Add runtime discovery for the existing `forge3d.colormaps` entry-point group.
2. Merge discovered colormaps with the current built-in and provider-based loading path.
3. Ensure third-party plugin failures do not break built-in colormap availability.

**Definition of done**

- An installed third-party colormap plugin is discoverable without manual import.
- Built-in colormaps still load when plugin discovery fails.
- The scope remains explicitly limited to **colormaps**, not a generalized plugin ABI.

### Epic P5 — Viewer Overlay Platform + Live Controls

**Impact:** Medium  
**Effort:** M

**Goal:** Establish an egui-backed viewer overlay platform, then ship a lightweight live-control surface without turning Forge3D into an editor product.

**Specific tasks**

1. Integrate an egui-backed final-pass viewer overlay so panels can be added without changing the runtime ownership model.
2. Keep the first release tightly scoped to high-value controls: exposure, sun/light intensity, fog, TAA, OIT, and snapshot actions.
3. Ensure the overlay edits the same underlying runtime state used by commands and Python control paths.

**Definition of done**

- The viewer binary can toggle an egui panel at runtime.
- The overlay host can support future diagnostics and tooling panels without a second UI stack.
- Panel changes immediately affect the live render.
- The control surface does not introduce a second configuration model.

### Epic P6 — Collaborative Viewing v1

**Impact:** Medium  
**Effort:** M

**Goal:** Add shared viewing semantics without jumping straight to full multiplayer replication.

**Specific tasks**

1. Define a minimal shared session model for camera state, overlays, annotations, and point-cloud/view parameters.
2. Allow multiple clients to observe and synchronize that shared state without breaking existing single-client control flows.
3. Keep v1 at the collaborative-viewing layer; do not expand into full gameplay-style replication.

**Definition of done**

- Two clients can attach to one viewer session and stay synchronized by revision-based polling.
- Existing single-client scripts continue to work unchanged.
- v1 does not require a WebSocket/server-push rewrite.

### Epic P7 — FBX Import Phase 1 (Static Mesh Only)

**Impact:** Medium  
**Effort:** M

**Goal:** Improve DCC interoperability without prematurely opening the skeletal-animation scope.

**Specific tasks**

1. Add FBX import for the same static-mesh boundary Forge3D already consumes today: positions, normals, UVs, and indices.
2. Expose that import path consistently to Python and the viewer.
3. Keep phase 1 explicitly out of skeletal animation, morph targets, and material-graph translation.

**Definition of done**

- A representative static FBX mesh can be imported into `MeshBuffers` from Python.
- Unsupported FBX constructs fail with explicit error messages.
- Phase 1 does not expand the asset boundary beyond what the current renderer already consumes.

**Additional Tier 1 Delivery Epics**

### Epic P8 — Material Fidelity Pass

**Impact:** Medium  
**Effort:** M  
**Covers:** Terrain-path clear coat / cloth support; Terrain-path anisotropic materials; Full subsurface profile / screen-space SSS

**Goal:** Close the highest-value raster material gaps on the existing terrain/viewer path without expanding into layered materials.

**Definition of done**

- Clear-coat / cloth and anisotropy parameters are consumed by supported raster material shaders instead of remaining mostly inert data.
- A dedicated screen-space subsurface pass exists for supported materials with documented inputs, limits, and defaults.

### Epic P9 — Data Layers & Scene Variants

**Impact:** Medium  
**Effort:** S  
**Covers:** Data Layers (scene variants)

**Goal:** Add named scene variants and layer-based visibility so GIS and review workflows can switch among alternate states without duplicating scenes.

**Definition of done**

- Scene bundles and runtime state can define named layers or variants with stable IDs.
- Python and IPC can toggle layer visibility / active variants without rebuilding the scene.

### Epic P10 — Input Modernization

**Impact:** Medium  
**Effort:** S  
**Covers:** Gamepad support; Input remapping

**Goal:** Modernize the interaction layer so navigation works across keyboard, mouse, and gamepad with configurable bindings.

**Definition of done**

- The viewer supports gamepad-driven camera/navigation with documented default mappings.
- Input bindings can be remapped through config or API rather than being hard-coded.

### Epic P11 — Memory Profiler Surfacing

**Impact:** Medium  
**Effort:** S  
**Covers:** Memory profiler visualization

**Goal:** Turn the existing memory tracking data into a visible developer tool for the overlay and Python-facing diagnostics.

**Definition of done**

- Live memory budget, utilization, and major allocation categories are visible in a supported UI/API surface.
- Memory snapshots can be exported or queried without attaching an external debugger.

### Epic P12 — Decals & Projected Markup

**Impact:** Medium  
**Effort:** M  
**Covers:** Decals

**Goal:** Add projected decals as a first-class annotation and material-detail mechanism on the current raster path.

**Definition of done**

- At least one decal path works on supported terrain or mesh surfaces with explicit blending rules.
- The API surface documents how decals interact with depth, normals, and existing annotation systems.

### Epic P13 — Motion FX & Camera Toolkit

**Impact:** Medium  
**Effort:** M  
**Covers:** Ribbon/trail effects (animated); Camera rig system

**Goal:** Package lightweight cinematic tools that fit Forge3D's visualization workflows without importing a full cinematic editor.

**Definition of done**

- Animated ribbon or trail primitives can be generated from runtime paths or sampled trajectories.
- Reusable camera rigs expose orbit, rail, or crane-style behaviors to Python and the viewer.

### Epic P14 — Asset Optimization Pipeline

**Impact:** Medium  
**Effort:** M  
**Covers:** Automatic mesh LOD; HLOD; Texture compression pipeline

**Goal:** Reduce asset cost through a preprocessing pipeline that emits optimized assets for the existing renderer instead of changing the renderer architecture.

**Definition of done**

- Mesh simplification and HLOD generation produce assets the current runtime can stream or render without a parallel geometry stack.
- Texture compression outputs supported compressed formats with a documented fallback path for unsupported devices.

### Epic P15 — Developer Loop Modernization

**Impact:** Medium  
**Effort:** M  
**Covers:** Hot reload / live coding

**Goal:** Shorten the development loop for shaders, selected assets, and configuration without destabilizing the viewer.

**Definition of done**

- At least shaders and selected runtime assets/configuration can reload during development without process restart.
- Reload failures surface explicit diagnostics and preserve a usable viewer state.

### Epic P16 — Plugin Architecture v1

**Impact:** Medium  
**Effort:** M  
**Covers:** Plugin architecture

**Goal:** Generalize runtime extensibility beyond the colormap-only discovery slice introduced in `P4`.

**Definition of done**

- At least one non-colormap extension point is discoverable through a documented registration contract.
- Plugin loading, versioning, and failure-isolation rules are explicit enough for third-party extensions to ship safely.

**Tier 2 Build Epics**

### Epic P17 — Real-Time VFX & Foliage Systems

**Impact:** Medium  
**Effort:** L  
**Covers:** GPU particle system; Foliage instancing

**Goal:** Add GPU-driven VFX and vegetation systems that reuse the existing instancing and terrain-placement foundations.

**Definition of done**

- A representative GPU particle effect path exists for weather or geoscience-style visualization.
- Foliage placement, culling, and LOD operate through the instancing path rather than a bespoke renderer.

### Epic P18 — Character Animation Foundation

**Impact:** Medium  
**Effort:** L  
**Covers:** Skeletal animation + GPU skinning; Morph targets / blend shapes

**Goal:** Establish the ingest, playback, and shader foundations for animated character or articulated asset support.

**Definition of done**

- Importers can carry skeleton, skin, clip, and morph-target data into runtime structures.
- GPU skinning and morph-target playback work on representative assets through supported public APIs.

### Epic P19 — Virtual Shadow Maps

**Impact:** Medium  
**Effort:** L  
**Covers:** Virtual Shadow Maps

**Goal:** Introduce a page-based shadowing path that reduces cascade artifacts without immediately removing the current CSM implementation.

**Definition of done**

- A representative scene renders with VSM enabled and shows materially reduced cascade-transition artifacts.
- VSM rollout is feature-gated or runtime-selectable alongside the current shadow path.

### Epic P20 — Simulation Foundations

**Impact:** Medium  
**Effort:** L  
**Covers:** Physics (Rapier3D integration); Navigation mesh

**Goal:** Add the baseline simulation substrates needed for collision-aware motion and path planning without entangling the renderer.

**Definition of done**

- Rigid-body physics and collision queries work through a clear feature-gated integration boundary.
- Navigation-mesh generation and query APIs work on representative terrain or scene geometry.

### Epic P21 — Procedural World Generation

**Impact:** Medium  
**Effort:** L  
**Covers:** PCG (Procedural Content Generation)

**Goal:** Introduce deterministic procedural generation that feeds existing placement, scene, and instancing systems.

**Definition of done**

- At least one rule-based or graph-based generator can populate terrain content reproducibly from inputs and seeds.
- Generated outputs flow into current scene/instancing paths rather than a parallel runtime model.

### Epic P22 — Sequencer & Cinematic Timeline

**Impact:** Medium  
**Effort:** L  
**Covers:** Multi-track timeline sequencer

**Goal:** Extend the existing camera animation path into a coordinated multi-track timeline for cinematic and review workflows.

**Definition of done**

- Multiple animated channels can be authored, scrubbed, and exported through one timeline abstraction.
- Existing camera keyframe animation interoperates with, or is subsumed by, the sequencer model.

### Epic P23 — WebAssembly Target

**Impact:** Medium  
**Effort:** L  
**Covers:** WebAssembly target

**Goal:** Produce a supported browser-deployable subset of Forge3D with clear capability boundaries.

**Definition of done**

- A documented core viewer/runtime subset compiles to WebAssembly.
- Unsupported subsystems and asset-loading constraints are explicit in build and runtime documentation.

### Epic P24 — USD Interoperability

**Impact:** Medium  
**Effort:** L  
**Covers:** USD import

**Goal:** Add USD ingestion at the same runtime asset boundary Forge3D already consumes for static scenes and geometry.

**Definition of done**

- Representative USD content imports into current scene or mesh structures through supported APIs.
- Unsupported schema, material, or animation cases fail explicitly instead of silently degrading.

### Epic P25 — Layered Material System

**Impact:** Medium  
**Effort:** L  
**Covers:** Layered/substrate materials

**Goal:** Add a compositional material abstraction that can stack a small set of surface layers without requiring a full editor product.

**Definition of done**

- The material system can compose at least a minimal set of layers with stable runtime semantics.
- Compiled layered materials map onto the current raster pipeline without introducing an editor-only dependency.

**Tier 3 Boundary Epics**

### Epic P26 — Renderer Architecture Frontier

**Impact:** Low  
**Effort:** M  
**Covers:** Nanite-class virtualized geometry; Hardware ray tracing

**Goal:** Keep high-cost renderer-architecture ideas as explicit watchlist items with written reasons for deferral.

**Definition of done**

- An ADR or equivalent records why Nanite-class geometry and real-time hardware RT are not active roadmap commitments today.
- Revisit triggers are tied to concrete architecture or WebGPU capability changes rather than vague future intent.

### Epic P27 — Editor Product Boundary

**Impact:** Low  
**Effort:** M  
**Covers:** Material node graph editor; Standalone editor application; In-editor terrain sculpting

**Goal:** Define the boundary between runtime capabilities Forge3D should grow and editor-product commitments it should continue to reject.

**Definition of done**

- Scope guidance explicitly states why editor-heavy features stay out of the near-term execution backlog.
- Any enabling runtime work is broken into smaller implementation epics instead of hiding editor scope inside engine tasks.

### Epic P28 — Scripting & Gameplay Framework Boundary

**Impact:** Low  
**Effort:** S  
**Covers:** Visual scripting (Blueprints); Gameplay Ability System

**Goal:** Make the scripting and gameplay-framework boundary explicit so library users do not infer a hidden game-engine roadmap.

**Definition of done**

- The roadmap names Python/API surfaces as the primary scripting model for current product goals.
- Gameplay-framework requests require a separate product-direction decision before entering the build backlog.

### Epic P29 — Networked Simulation Boundary

**Impact:** Low  
**Effort:** M  
**Covers:** Full multiplayer replication

**Goal:** Distinguish collaborative viewing from full replicated simulation and keep the latter deferred unless the product direction changes.

**Definition of done**

- Replication, prediction, authority, and session-hosting requirements are documented as distinct from `P6`.
- Revisit criteria explicitly require a move toward networked simulation use cases rather than shared review.

### Epic P30 — Platform & Immersive Expansion Boundary

**Impact:** Low  
**Effort:** M  
**Covers:** Console platform support; VR/XR; Virtual production / LED volume

**Goal:** Record why platform and immersive-expansion features remain out of scope given ecosystem maturity, staffing, and product fit.

**Definition of done**

- The roadmap documents the external dependencies and ecosystem constraints behind these deferrals.
- Revisit triggers identify what strategic change would justify promoting one of these into a build epic.

### Epic P31 — Advanced Character Systems Boundary

**Impact:** Low  
**Effort:** M  
**Covers:** Hair/fur rendering; Motion matching

**Goal:** Keep advanced character features sequenced behind the animation foundation instead of allowing them to appear as near-term commitments.

**Definition of done**

- The roadmap makes these explicit post-foundation follow-ons rather than active implementation promises.
- Prerequisites from `P18` are documented so the dependency chain is concrete.

### Epic P32 — Advanced Simulation Boundary

**Impact:** Low  
**Effort:** M  
**Covers:** Fluid simulation

**Goal:** Document why high-cost fluid simulation remains deferred and what evidence would be needed to justify it.

**Definition of done**

- A narrow research note or ADR captures likely implementation approaches and the current ROI argument against them.
- The main terrain/rendering roadmap carries no hidden dependency on fluid simulation landing.

### Epic P33 — CAD/Datasmith Interoperability Boundary

**Impact:** Low  
**Effort:** M  
**Covers:** CAD import (Datasmith-class)

**Goal:** Keep heavy CAD/Datasmith ingestion out of the near-term roadmap while leaving a clear path for future reassessment.

**Definition of done**

- The interoperability section documents the geometry-kernel and translation burden that keeps CAD import deferred.
- Any future revisit begins with narrower interchange targets instead of committing immediately to a full Datasmith-class pipeline.

---

## ARCHITECTURAL INTEGRITY ASSESSMENT

### What Would NOT Be Compromised

The following core systems are likely to remain largely insulated from most Tier 1 and Tier 2 additions:

1. **Terrain rendering pipeline** — Clipmap, tiling, LOD, streaming, COG
2. **Point cloud system** — COPC, EPT, octree traversal
3. **3D Tiles system** — Tileset streaming, B3DM/PNTS
4. **Vector overlay stack** — Polygons, lines, points, extrusion
5. **Label system** — MSDF, collision detection, decluttering
6. **Path tracing** — Megakernel, wavefront, AOV
7. **Python API** — PyO3 bindings, IPC protocol
8. **Scene bundles** — Save/load, manifest, checksums
9. **Style system** — Mapbox GL parsing, data-driven expressions
10. **Export pipeline** — SVG, PDF, PNG, EXR

### Why the Architecture Is Resilient

Forge3D's architecture is modular, but not in exactly the way the previous draft claimed:

- **Feature flags** gate optional systems at compile time, so new modules don't increase binary size for users who don't need them
- **Framegraph utilities exist** (`framegraph_impl/` with topological sort, barrier planning, resource aliasing), but the main viewer loop (`render/main_loop.rs`) directly calls `encoder.begin_render_pass()` — the framegraph is infrastructure, not the orchestrator
- **IPC protocol** decouples the viewer from Python with 66 `IpcRequest` variants across terrain, mesh, labels, vectors, point clouds, picking, rendering, camera, and lighting — new features can be exposed via new IPC commands without changing existing ones
- **Scene organization is monolithic** — a 77-field `Scene` struct (~6 438 lines) where each subsystem is an `Option<RendererType>` field. A standalone `SceneGraph` (522 lines) with full hierarchy, transforms, and cycle detection exists alongside but is architecturally separate from the main `Scene`
- **Multi-threaded command recording and async compute** provide GPU parallelization infrastructure, though integration breadth varies across renderer paths
- **Resource tracker** manages GPU memory centrally, so new systems participate in the same budget without ad-hoc allocation

### The Key Architectural Constraint

Forge3D runs on **WebGPU via wgpu**, not Vulkan/DX12 directly. This means:

- No hardware ray tracing (until WebGPU RT extension lands)
- No hardware tessellation stages (compute-shader tessellation is the workaround)
- No mesh shaders / task shaders (until WebGPU supports them)
- No bindless resources (WebGPU binding model is more constrained)

These are **external constraints**, not design flaws. As the WebGPU spec evolves, these gaps close automatically.

---

## SUMMARY

**Audit methodology:** Every "What Forge3D Has" claim in this document was verified against actual source code (shader files, Rust modules, Python bindings). Feature counts, struct sizes, and implementation states are derived from reading the files, not from README descriptions or comments.

Forge3D is missing large categories of features that exist in Unreal Engine — physics, audio, AI, gameplay systems, skeletal animation, full GUI framework, visual scripting, and multi-platform deployment. However, the codebase is substantially more capable than a surface read suggests. The source audit revealed:

- **Materials**: 13-model BRDF dispatch with explicit shader dispatch paths for 11 variants, though Subsurface and Hair dispatch to proxy shaders
- **Lighting**: LTC rectangular area lights, shared WGSL sampling helpers for rect/disk/sphere area-light shapes, a separate CPU-side area-light enum that also includes Cylinder, soft light radius, NOAA ephemeris
- **Post-processing**: Complete pipeline — bloom, TAA with neighborhood clamping, 5 tone map operators + LUT + white balance, physically-based DOF, motion blur
- **Volumetrics**: Full ray-marched volumetric fog with Henyey-Greenstein phase function and god-rays, not just exponential distance fog
- **Water**: Full water surface rendering (756-line module) with waves, foam, reflections, refractions, shoreline effects — closes a major UE5 gap
- **Path tracing**: ReSTIR with spatial/temporal reuse, importance sampling/guiding, A-trous denoiser, firefly clamping, multiple AOVs — approaching production-grade offline rendering
- **GPU infrastructure**: Multi-threaded command recording, async compute, HZB, GPU BVH (LBVH + SAH), staging rings, double buffering, virtual texture foundation, 399-line GPU timing system with timestamp queries and RenderDoc/Nsight markers (pipeline-stat collection scaffolded but commented out; partially integrated into GI helpers, not into the main per-frame loop)
- **Scene**: 77-field monolithic Scene struct (~6 438 lines) with a separate 522-line hierarchical scene graph — functional but tightly coupled

The real question is less "does the repo have anything here?" and more "how complete and consistently integrated is it across render paths?" The GPU timing system exists and is partially wired into GI helpers, but not into the main per-frame viewer loop; pipeline-stat collection is scaffolded but commented out. The framegraph has topological sort and barrier planning but the main loop bypasses it. Entry-point extensibility is declared but not loaded at runtime. These are integration gaps, not capability gaps.

The features that remain most relevant and feasible are: dedicated SSS/hair shaders (replacing current proxy fallbacks), GPU particles for weather, egui for interactive controls, collaborative viewing, profiling surfacing (completing pipeline-stat collection, broadening timestamp integration beyond GI helpers, exposing via Python), FBX/USD import, and a multi-track sequencer. Most can be added with limited blast radius, but the 77-field Scene struct means any feature that needs to interact with multiple subsystems will touch a tightly coupled core.

Forge3D can grow toward substantially higher rendering quality in its domain without becoming a general-purpose game engine. The practical enablers are feature flags, subsystem separation, IPC (66 request variants), resource tracking, and targeted render-path refactors. The recommendation is to pursue corrected Tier 1 items first (several are now F1 data-swap tasks rather than new code), plan Tier 2 items deliberately, and explicitly reject Tier 3 items as out of scope.
