# Rendering beyond conventional path tracing for forge3d

Research date: 2026-07-12. Sources are primary papers, standards, official project documentation, and the forge3d tree.

## Bottom line

There is no universally "more powerful" successor to path tracing. Conventional unidirectional path tracing is a general light-transport estimator; methods that beat it either sample difficult paths better (BDPT, photon mapping/VCM, MLT, ReSTIR), reuse prior work (temporal reservoirs or caches), learn an approximation, or solve a different problem. For 3D maps, the largest gains do **not** come from a more elaborate integrator. They come from rendering only visible, screen-relevant geodata and amortizing expensive shading across pixels and frames.

For forge3d, the best order is:

1. **Finish the map-native raster stack:** streamed geometry clipmaps + virtual textures + 3D Tiles screen-space-error traversal + GPU culling/indirect draws.
2. **Add temporal reconstruction/upscaling to the existing raster hybrid:** lower-resolution shading, motion vectors, disocclusion handling, then TAAU-style reconstruction.
3. **Use selective software-ray effects, not full-frame PT:** terrain heightfield shadows/AO, building/landmark BVH reflections or GI, denoised and temporally accumulated.
4. **Keep ReSTIR as an opt-in hero/offline path:** useful for many lights or low-sample indirect illumination, but not the default outdoor-map renderer.
5. **Add Gaussian splats only as a captured-scene layer type:** valuable for photogrammetry/digital twins, not a replacement for semantic terrain, vectors, labels, or editable buildings.
6. **Defer neural radiance caching, BDPT/VCM/MLT, and differentiable rendering** until a measured workload demands them.

## Why map-native rasterization ranks first

Geometry clipmaps keep nested regular grids around the viewer and refill them incrementally. The original method demonstrated a 20-billion-sample US height field with sub-pixel screen error and interactive flight; its key properties are bounded working set, continuity, and stable cost ([Losasso and Hoppe 2004](https://hhoppe.com/proj/geomclipmap/)). OGC 3D Tiles applies the same screen-relevance principle to heterogeneous geospatial content: hierarchical LOD, bounding volumes, and geometric error drive refinement in screen pixels ([OGC 3D Tiles 1.1](https://docs.ogc.org/cs/22-025r4/22-025r4.html); [Cesium specification source](https://github.com/CesiumGS/3d-tiles/blob/main/specification/README.adoc)).

This is more useful to a map than tracing more paths through geometry that should not be resident or drawn. It is also the most WebGPU-native option: vertex/fragment rendering, compute preprocessing, storage buffers/textures, indirect drawing, and texture compression are in the portable API. The WebGPU feature index contains no standardized acceleration structures or ray-query feature ([WebGPU specification](https://gpuweb.github.io/gpuweb/#feature-index)).

forge3d is already close: `src/terrain/clipmap/`, `src/terrain/renderer/streaming.rs`, `src/terrain/renderer/virtual_texture.rs`, and `src/terrain/page_table/` implement the major pieces. The lazy next step is integration and profiling, not another terrain representation.

## Ranked method assessment

| Rank | Method | What it adds over plain PT | 3D-map payoff | WebGPU/wgpu feasibility | forge3d decision |
|---:|---|---|---|---|---|
| 1 | Clipmaps + virtual texturing + HLOD/3D Tiles + GPU culling | Avoids processing non-resident/non-visible detail; stable screen-space cost | Transformative for continental terrain, imagery, buildings, points | **High**: ordinary raster/compute/storage/indirect primitives | Consolidate the existing stack and make it the default large-scene path |
| 2 | Temporal raster hybrid / temporal upscaling | Reuses prior frames so expensive shading runs below display resolution | Large frame-time win while preserving labels, terrain detail, shadows, atmosphere | **High–medium**: compute implementation is portable; quality depends on correct motion/depth/reactive masks | Extend existing TAA and previous-frame matrices into a minimal TAAU path |
| 3 | Specialized software-ray hybrid | Spends rays only where raster approximations fail | Better contact shadows, reflections, AO/GI around terrain/buildings and hero landmarks | **Medium**: WGSL compute BVH or heightfield traversal works; no portable hardware RT | Reuse existing heightfield/BVH compute paths; keep raster primary visibility |
| 4 | ReSTIR DI/GI/PT | Reuses important light/path samples across pixels and frames at very low spp | Strong for night cities/many emissive lights and hero GI; modest for one sun + sky | **Medium algorithmically**, **low for hardware-RT assumptions** | Finish only behind a measured many-light or hero/offline use case |
| 5 | 3D Gaussian splatting | Real-time novel-view rendering of captured radiance fields | Excellent photogrammetric sites/digital twins; weak semantics, relighting, editing, and cartography | **Medium–high**: compute sorting/culling + instanced alpha splats are demonstrated in WebGPU | Add as a separate streamed layer when users have splat datasets |
| 6 | Neural/learned reconstruction or radiance cache | Approximates expensive shading or reconstructs sparse/low-res output | Upscaling can help; online radiance learning has weak map-specific ROI | **Upscaling: medium**; **online cache: low–medium** without tuned matrix/cooperative ops | Prefer non-neural temporal reconstruction first; revisit after profiling |
| 7 | BDPT, photon mapping/VCM, MLT | Samples caustics, narrow portals, and difficult specular paths better than camera PT | Mostly irrelevant to sunlit terrain; useful only for architectural interiors/water caustics | **Possible but expensive** in portable compute; large implementation/validation cost | Do not implement for the current map renderer |
| 8 | Differentiable rendering | Computes gradients of images with respect to geometry/material/light parameters | Useful for calibration, reconstruction, or inverse problems—not display rendering | **Low** as an in-engine WebGPU feature; mature systems use LLVM/CUDA/OptiX | Keep external to forge3d unless inverse rendering becomes a product goal |

## ReSTIR: useful, but narrower than the name suggests

ReSTIR DI resamples candidate lights spatially and temporally; the original paper reports equal-error speedups of 6–60× for scenes with up to 3.4 million dynamic emissive triangles ([Bitterli et al. 2020](https://research.nvidia.com/labs/rtr/publication/bitterli2020spatiotemporal/)). ReSTIR GI applies reuse to multi-bounce indirect paths and reports 9.3–166× MSE improvement over one-sample-per-pixel PT in its test scenes ([Ouyang et al. 2021](https://research.nvidia.com/publication/2021-06_restir-gi-path-resampling-real-time-path-tracing)). GRIS/ReSTIR PT extends the theory to correlated samples and varied path domains ([Lin et al. 2022](https://research.nvidia.com/labs/rtr/publication/lin2022generalized/)). Newer work improves difficult cases—caustics through bidirectional paths ([ReSTIR BDPT 2025](https://research.nvidia.com/labs/rtr/publication/hedstrom2025restir/)) and reuse across changing LOD topology ([LOD-aware ReSTIR 2026](https://research.nvidia.com/labs/rtr/publication/wang2026levelofdetail/)).

The catch is visibility. ReSTIR chooses samples; it does not make ray intersections free. Portable WebGPU has no acceleration-structure/ray-query feature. forge3d's pinned `wgpu = "0.19"` does define native-only, Vulkan-only `RAY_QUERY` and `RAY_TRACING_ACCELERATION_STRUCTURE` flags, but forge3d does not request them; current wgpu still labels ray queries experimental and native-only. Making them a required path would violate the package's DX12/Metal/WebGPU portability target ([current wgpu feature documentation](https://docs.rs/wgpu/latest/wgpu/struct.FeaturesWGPU.html#associatedconstant.EXPERIMENTAL_RAY_QUERY)).

forge3d already has canonical reservoirs and temporal/spatial passes in `src/path_tracing/restir/`, `src/shaders/pt_restir_*.wgsl`, and `src/path_tracing/hybrid_compute/`. Therefore the shortest sensible move is to connect that work only to an explicit use case. For ordinary daytime maps, cascaded/virtual shadowing plus screen-space or heightfield GI will be cheaper and steadier. For a city with thousands of emissive windows, ReSTIR DI—or the ray-free variant that uses reservoirs to choose which shadow maps receive full resolution—becomes compelling ([ReSTIR-sampled shadow maps 2025](https://research.nvidia.com/labs/rtr/publication/zhang2025many-light/)).

## Neural methods and temporal reconstruction

Online neural radiance caching learns a dynamic approximation while rendering and demonstrated roughly 2.6 ms of cache update/query overhead at 1080p on the authors' tuned GPU implementation ([Müller et al. 2021](https://research.nvidia.com/publication/2021-06_real-time-neural-radiance-caching-path-tracing)). That result relies on a streaming, hardware-efficient neural network implementation; it is evidence that the method works, not that a portable WGSL port will retain the cost. WebGPU offers compute and optional `shader-f16`/subgroups, but portability requires capability checks and fallbacks ([WebGPU optional capabilities](https://gpuweb.github.io/gpuweb/#optional-capabilities)).

Temporal super-resolution is a better first investment. AMD's open FSR2 design reconstructs high-resolution output from low-resolution color using depth, motion vectors, jitter, history, disocclusion tests, and optional reactive/transparency masks; its pipeline is a sequence of compute passes ([official FSR2 source and integration guide](https://github.com/GPUOpen-Effects/FidelityFX-FSR2)). The released shaders target HLSL/Vulkan/DX12, so forge3d cannot drop them into WGSL unchanged, but the algorithmic inputs align with its existing `taa_jitter`, previous view-projection state, and screen-space passes. A small native WGSL TAAU is lower risk than porting the full SDK or building a neural model.

Recommendation: first prove that rendering terrain/PBR/screen-space effects at 67–75% linear resolution and reconstructing them beats native resolution under forge3d's golden-image and timing gates. Keep labels and thin cartographic linework at output resolution.

## Gaussian splatting

3D Gaussian Splatting represents a captured scene with anisotropic translucent primitives and uses visibility-aware splat rasterization; the reference work reports real-time 1080p novel-view synthesis ([Kerbl et al. 2023 and official implementation](https://github.com/graphdeco-inria/gaussian-splatting)). Independent WebGPU implementations show that WGSL compute sorting plus instanced splats is feasible, while also exposing the main cost/complexity: per-view depth ordering, transparency, and potentially large intersection buffers ([WebGPU implementation](https://github.com/Scthe/gaussian-splatting-webgpu)).

For forge3d this should be a **layer**, analogous to a point cloud or 3D Tiles payload. It is strong for a scanned quarry, street, monument, or disaster site. It is not a general replacement for DEM-correct terrain, vector styling, selectable semantic features, deterministic labels, or relightable assets. Training should remain an offline ingestion concern.

## What really exceeds conventional path tracing

If "more powerful" means difficult light transport rather than faster map rendering:

- Bidirectional path tracing connects eye and light subpaths and handles a broader range of hard paths robustly with MIS ([Veach's thesis](https://graphics.stanford.edu/papers/veach_thesis/)).
- Progressive photon mapping converges with bounded memory and is especially effective for caustics and reflections of caustics ([Hachisuka et al. 2008](https://graphics.ucsd.edu/~henrik/papers/progressive_photon_mapping/)).
- Metropolis light transport mutates paths in path space and can be orders of magnitude more efficient for bright indirect light, narrow openings, glossy transport, and caustics ([Veach and Guibas 1997](https://graphics.stanford.edu/papers/metro/)).

Those methods are better estimators for particular transport, not universally better renderers. Their state, synchronization, acceleration-structure, and convergence/debugging costs are poor matches for a portable geospatial package whose dominant light is usually sun/sky.

Differentiable rendering is orthogonal: it differentiates image formation to estimate geometry, reflectance, lighting, or pose from observations. Mitsuba's official examples use it for inverse problems and its high-performance backends are LLVM and CUDA/OptiX, not WebGPU ([Mitsuba inverse-rendering documentation](https://mitsuba.readthedocs.io/en/v3.5.2/src/inverse_rendering_tutorials.html); [Mitsuba 3 implementation](https://github.com/mitsuba-renderer/mitsuba3)). It belongs in an optional reconstruction/calibration workflow, not forge3d's renderer.

## Concrete forge3d roadmap

1. Benchmark and harden the existing clipmap/VT/3D Tiles residency and selection path under fly-throughs; add GPU culling only when CPU/object-count evidence justifies it.
2. Build one minimal WGSL temporal-upscale experiment from the existing TAA history, depth, and motion data. Preserve full-resolution labels/vectors and require both quality and GPU-time wins.
3. Keep raster primary visibility. Reuse heightfield traversal for terrain rays and the existing compute BVH only for bounded hero effects.
4. Gate ReSTIR work on either (a) a night-city many-light benchmark or (b) a MapScene hero/offline integration test. Do not create another ReSTIR abstraction; the repo already has the reservoirs and passes.
5. Add Gaussian splatting only after a real `.ply`/`.splat`/3D Tiles Gaussian dataset and memory/streaming budget are in scope.
6. Do not start neural caching, BDPT/VCM/MLT, or differentiable rendering without a failing benchmark that the simpler stack cannot solve.
