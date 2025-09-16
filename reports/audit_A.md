# Audit — Workstream A: Path Tracing (WebGPU Compute, Offline HQ)

## 1) Scope & Hygiene Summary
- Workstream: A
- Tasks matched: 25 (A1–A25)
- CSV headers: exact schema match
- Hygiene anomalies: Workstream Title comma causes column shifts; Priority/Phase out-of-vocab. See `reports/csv_hygiene_A.md`.

## 2) Readiness Verdicts
- A1 Compute Path Tracer MVP — Present & Wired
- A2 Material Models v1 — Present but Partial
- A3 Triangle Mesh + BVH — Present but Partial
- A4 Light Sampling & MIS — Present but Partial
- A5 Denoiser (A‑trous/SVGF) — Present but Partial
- A6 Dielectric Water (offline) — Absent
- A7 LBVH/SAH Builder & Refit — Present but Partial
- A8 ReSTIR DI — Present but Partial
- A9 PBR Textures in PT — Present but Partial
- A10 SDF Primitives & CSG — Present but Partial
- A11 Participating Media (Single Scatter) — Present but Partial
- A12 Wavefront Path Tracer — Present but Partial
- A13 Path Guiding — Present but Partial
- A14 AOVs & Debug Outputs — Present & Wired
- A15 Progressive/Checkpoint & Tiling — Present & Wired
- A16 QMC & Adaptive Sampler (f16‑ready) — Absent
- A17 Firefly Clamp — Present & Wired
- A18 Ground Plane (PT) — Present & Wired
- A19 Scene Cache for HQ — Present & Wired (CPU path)
- A20 Soft Area Lights Param — Present but Partial
- A21 Ambient Occlusion Integrator (Offline) — Present but Partial
- A22 Instanced Geometry (PT) — Present but Partial
- A23 Hair BSDF + Curve Prims (PT) — Present but Partial
- A24 Anisotropic Microfacet BRDF — Present but Partial
- A25 Object Importance Sampling — Present but Partial

## 3) Evidence Map (selected)
- PT compute wiring: [src/path_tracing/compute.rs](cci:7://file:///Users/mpopovic3/forge3d/src/path_tracing/compute.rs:0:0-0:0) include_str of `../shaders/pt_kernel.wgsl` (≈ lines 55–58, 538–541); AOV bind group layout (≈ lines 155–239)
- Wavefront PT: [src/path_tracing/wavefront/pipeline.rs](cci:7://file:///Users/mpopovic3/forge3d/src/path_tracing/wavefront/pipeline.rs:0:0-0:0) include_str of `pt_{raygen,intersect,shade,scatter,compact}.wgsl` (≈ lines 24–50); scheduler in [src/path_tracing/wavefront/mod.rs](cci:7://file:///Users/mpopovic3/forge3d/src/path_tracing/wavefront/mod.rs:0:0-0:0) (persistent threads dispatch)
- BVH/LBVH: [src/accel/lbvh_gpu.rs](cci:7://file:///Users/mpopovic3/forge3d/src/accel/lbvh_gpu.rs:0:0-0:0) (loads [lbvh_morton.wgsl](cci:7://file:///Users/mpopovic3/forge3d/src/shaders/lbvh_morton.wgsl:0:0-0:0)); [src/accel/sah_cpu.rs](cci:7://file:///Users/mpopovic3/forge3d/src/accel/sah_cpu.rs:0:0-0:0); WGSL kernels: `lbvh_{morton,link}.wgsl`, [bvh_refit.wgsl](cci:7://file:///Users/mpopovic3/forge3d/src/shaders/bvh_refit.wgsl:0:0-0:0), [radix_sort_pairs.wgsl](cci:7://file:///Users/mpopovic3/forge3d/src/shaders/radix_sort_pairs.wgsl:0:0-0:0)
- ReSTIR: [src/path_tracing/restir.rs](cci:7://file:///Users/mpopovic3/forge3d/src/path_tracing/restir.rs:0:0-0:0); WGSL `src/shaders/restir_{reservoir,temporal,spatial}.wgsl`; [docs/api/restir.md](cci:7://file:///Users/mpopovic3/forge3d/docs/api/restir.md:0:0-0:0); [examples/restir_many_lights.py](cci:7://file:///Users/mpopovic3/forge3d/examples/restir_many_lights.py:0:0-0:0)
- SVGF: WGSL `src/shaders/svgf_*.wgsl`; `src/denoise/svgf/*` stubs; [docs/api/svgf.md](cci:7://file:///Users/mpopovic3/forge3d/docs/api/svgf.md:0:0-0:0)
- Firefly clamp: Python handling in [python/forge3d/path_tracing.py](cci:7://file:///Users/mpopovic3/forge3d/python/forge3d/path_tracing.py:0:0-0:0) (luminance clamp), [python/forge3d/path_tracing.pyi](cci:7://file:///Users/mpopovic3/forge3d/python/forge3d/path_tracing.pyi:0:0-0:0); [docs/api/firefly_clamp.md](cci:7://file:///Users/mpopovic3/forge3d/docs/api/firefly_clamp.md:0:0-0:0)
- AOVs: [src/path_tracing/aov.rs](cci:7://file:///Users/mpopovic3/forge3d/src/path_tracing/aov.rs:0:0-0:0); Python [render_aovs()](cci:1://file:///Users/mpopovic3/forge3d/src/path_tracing/compute.rs:519:4-1041:5); [docs/api/aovs.md](cci:7://file:///Users/mpopovic3/forge3d/docs/api/aovs.md:0:0-0:0)

## 4) Blocking Gaps
- Material/texture parameters and anisotropy not fed to PT kernels (A2, A9, A24).
- Mesh BVH traversal not exercised by MVP path; dummy buffers in [compute.rs](cci:7://file:///Users/mpopovic3/forge3d/src/core/async_compute.rs:0:0-0:0) (A3).
- NEE/MIS and ReSTIR not integrated into PT loop (A4, A8).
- SVGF stages scaffolded but not fully wired (A5).
- Wavefront PT lacks an exported entry point that schedules all stages (A12).
- QMC/adaptive sampling not implemented (A16).
- Instances, area lights, and media not connected into PT code path (A22, A20, A11).

## 5) Minimal Change Plan
- See “Minimal Change Plan” below.

## 6) Validation Runbook
- Build: `cargo build --all-features`
- Demos/tests:
  - `python examples/progressive_tiling.py`
  - `python examples/restir_many_lights.py`
- AOVs sanity:
  - `python - <<'PY' ... (render_aovs) ... PY`
- Docs: `sphinx-build -b html docs _build/html`

## Minimal Change Plan (file-level, smallest steps to reach “Present & Wired”)
A2/A9/A24 (Materials/Textures/Aniso)
src/shaders/pt_shade.wgsl
: implement Lambert, GGX metal, dielectric with parameter block; add tangent-frame/αx/αy.
src/path_tracing/compute.rs
: extend scene/material buffers; propagate material params to kernels.
python/forge3d/path_tracing.py
: expose material controls mirroring PT params.
A3/A22 (Mesh + Instances)
src/path_tracing/compute.rs
: replace dummy mesh buffers with actual mesh/BVH buffers; upload via 
src/path_tracing/mesh.rs
.
src/shaders/pt_intersect_mesh.wgsl
: finalize traversal and hit reporting layout; align with host-side structs.
src/path_tracing/accel.rs
: single entry to build LBVH/SAH, return GPU buffers.
A4 (NEE/MIS)
src/shaders/pt_shade.wgsl
: implement next-event estimation; MIS weights (balance or power heuristic).
src/path_tracing/importance.rs
: ensure tags/weights feed into kernel bindings.
A5 (SVGF)
src/denoise/svgf/pipelines.rs
: create wgpu pipelines for variance, reproject, a-trous; wire dispatch from PT output AOVs.
Add test: tests/test_svgf_pipeline.py with small images to validate topology.
A7 (LBVH/Refit)
src/accel/lbvh_gpu.rs
: complete radix pipelines and integrate with lbvh_* WGSL; surface BuildStats.
src/shaders/bvh_refit.wgsl
: verify node flags, complete bottom-up refit; add unit test.
A8 (ReSTIR)
src/path_tracing/restir.rs
: create pipelines for temporal/spatial passes; add scheduling after direct lighting pass.
src/path_tracing/alias_table.rs
: confirm alias tables generation and binding.
A10 (SDF/CSG)
src/shaders/hybrid_kernel.wgsl
: interop between SDF raymarch and BVH traversal; validate hybrid hits.
src/path_tracing/hybrid_compute.rs
: provide sample integration path and emit AOVs.
A11 (Media)
src/shaders/pt_shade.wgsl
: HG phase evaluation and single-scatter estimate; medium params buffer.
A12 (Wavefront)
src/path_tracing/wavefront/mod.rs
: expose a public API entry; launch sequence raygen → intersect → shade → scatter → compact with termination.
Add example: examples/wavefront_cornell.py.
A16 (QMC/Adaptive)
src/shaders/pt_raygen.wgsl
: Sobol/Owen sequence per-pixel; adaptive sampling threshold in uniforms.
A20 (Area lights)
src/shaders/lighting_area.wgsl
: finalize sampling models; connect to PT shading for direct lighting.
A21 (AO offline)
src/post/ambient_occlusion.rs
: finish compute path; add Python hook; add example examples/ao_offline.py.
Validation Runbook
Build
cargo build --all-features
Optional: cargo clippy --all-features
GPU/PT demos and tests
python -c "import forge3d.path_tracing as pt; t=pt.PathTracer(256,256,seed=2); t.render_rgba(spp=1)"
python examples/progressive_tiling.py
python examples/restir_many_lights.py
Denoise and AOVs
python - <<'PY'\nfrom forge3d.path_tracing import render_aovs; a=render_aovs(64,64,None,seed=1); print([k for k in a])\nPY
Docs
sphinx-build -b html docs _build/html


I'll open `task-gpt.xml` to see the instructions it contains so I can execute them step-by-step.

To execute the audit plan, I'll start by inspecting and validating the CSV headers and loading the relevant rows for the specified workstream. I'll also run a repo-wide search for relevant shader/API keywords to gather evidence.



I'll search the repo for evidence related to the Path Tracing workstream (A1–A25): denoiser/SVGF, BVH/LBVH, Firefly clamp, AOVs, and general path-tracing terms. I'll also locate relevant files (WGSL shaders, BVH modules) to cite precise paths and snippets. Running these searches in parallel for speed.

I'll run the task's specified execution commands to validate the CSV headers and do a repo-wide keyword scan for shader/API evidence. Running them now.







I'll gather targeted evidence (file paths, line numbers, snippets) for each A1–A25 task by searching for specific symbols and shaders across the repo. I'll run these ripgrep-style searches in parallel for speed. Then I'll compute the exact count of A workstream rows and a CSV hygiene summary programmatically.





# Audit: Workstream A — Path Tracing (WebGPU Compute, Offline HQ)

Scope: repo-wide audit against roadmap2.csv for Workstream A. I followed the audit-only constraints, produced a hygiene report, evidence-backed readiness verdicts per task, and a minimal change plan. I took into account the prior memory that the repo builds warning-free with cargo build --all-features; this informed the runbook recommendations.

- Workstream: A — Path Tracing (WebGPU Compute, Offline HQ)
- Matched tasks: 25 (A1–A25)
- CSV headers: exact match to schema
- CSV hygiene: anomalies present (see reports/csv_hygiene_A.md below)

## Readiness Verdicts (per task)

- A1 Compute Path Tracer MVP — Present & Wired
  - Evidence: [src/path_tracing/compute.rs](cci:7://file:///Users/mpopovic3/forge3d/src/path_tracing/compute.rs:0:0-0:0) include_str of [pt_kernel.wgsl](cci:7://file:///Users/mpopovic3/forge3d/src/shaders/pt_kernel.wgsl:0:0-0:0) (lines ~55–58); AOV plumbing (bgl4); [docs/api/path_tracing.md](cci:7://file:///Users/mpopovic3/forge3d/docs/api/path_tracing.md:0:0-0:0) “A1: GPU Path Tracer (MVP)”; [src/shaders/pt_kernel.wgsl](cci:7://file:///Users/mpopovic3/forge3d/src/shaders/pt_kernel.wgsl:0:0-0:0)
- A2 Material Models v1 — Present but Partial
  - Evidence: [src/shaders/pbr.wgsl](cci:7://file:///Users/mpopovic3/forge3d/src/shaders/pbr.wgsl:0:0-0:0) (GGX/Lambert/Fresnel in raster); [src/shaders/pt_kernel.wgsl](cci:7://file:///Users/mpopovic3/forge3d/src/shaders/pt_kernel.wgsl:0:0-0:0) contains GGX helpers but no full PT BSDF parameter exposure; no end-to-end PT material parameters
- A3 Triangle Mesh + BVH — Present but Partial
  - Evidence: [src/accel/cpu_bvh.rs](cci:7://file:///Users/mpopovic3/forge3d/src/accel/cpu_bvh.rs:0:0-0:0); [src/accel/lbvh_gpu.rs](cci:7://file:///Users/mpopovic3/forge3d/src/accel/lbvh_gpu.rs:0:0-0:0); [src/shaders/pt_intersect_mesh.wgsl](cci:7://file:///Users/mpopovic3/forge3d/src/shaders/pt_intersect_mesh.wgsl:0:0-0:0); [src/path_tracing/mesh.rs](cci:7://file:///Users/mpopovic3/forge3d/src/path_tracing/mesh.rs:0:0-0:0); [src/path_tracing/compute.rs](cci:7://file:///Users/mpopovic3/forge3d/src/path_tracing/compute.rs:0:0-0:0) binds mesh buffers but uses dummy buffers (not fully wired)
- A4 Light Sampling & MIS — Present but Partial
  - Evidence: [src/path_tracing/importance.rs](cci:7://file:///Users/mpopovic3/forge3d/src/path_tracing/importance.rs:0:0-0:0); MIS mentioned in docs/examples; no clear NEE/MIS integration in PT kernels yet
- A5 Denoiser (A‑trous/SVGF) — Present but Partial
  - Evidence: `src/shaders/svgf_{variance,reproject,atrous,history_reset}.wgsl` (scaffolds); `src/denoise/svgf/*` stubs; [docs/api/svgf.md](cci:7://file:///Users/mpopovic3/forge3d/docs/api/svgf.md:0:0-0:0); Python fallback uses `atrous_denoise`
- A6 Dielectric Water (offline) — Absent
  - Evidence: Only docs stub ([docs/api/water.md](cci:7://file:///Users/mpopovic3/forge3d/docs/api/water.md:0:0-0:0)); no PT integration found
- A7 LBVH/SAH Builder & Refit — Present but Partial
  - Evidence: `src/accel/{lbvh_gpu.rs,sah_cpu.rs,types.rs}`; WGSL `lbvh_{morton,link}.wgsl`, [bvh_refit.wgsl](cci:7://file:///Users/mpopovic3/forge3d/src/shaders/bvh_refit.wgsl:0:0-0:0), [radix_sort_pairs.wgsl](cci:7://file:///Users/mpopovic3/forge3d/src/shaders/radix_sort_pairs.wgsl:0:0-0:0); PT side not fully integrated (dummy buffers in [compute.rs](cci:7://file:///Users/mpopovic3/forge3d/src/core/async_compute.rs:0:0-0:0))
- A8 ReSTIR DI — Present but Partial
  - Evidence: [src/path_tracing/restir.rs](cci:7://file:///Users/mpopovic3/forge3d/src/path_tracing/restir.rs:0:0-0:0); `src/shaders/restir_{reservoir,temporal,spatial}.wgsl`; [docs/api/restir.md](cci:7://file:///Users/mpopovic3/forge3d/docs/api/restir.md:0:0-0:0); [examples/restir_many_lights.py](cci:7://file:///Users/mpopovic3/forge3d/examples/restir_many_lights.py:0:0-0:0) simulates benefits (not actual GPU path)
- A9 PBR Textures in PT — Present but Partial
  - Evidence: [src/shaders/pbr_textured.wgsl](cci:7://file:///Users/mpopovic3/forge3d/src/shaders/pbr_textured.wgsl:0:0-0:0) placeholder; [pt_kernel.wgsl](cci:7://file:///Users/mpopovic3/forge3d/src/shaders/pt_kernel.wgsl:0:0-0:0) minimal shading; no textured lookups in PT path
- A10 SDF Primitives & CSG — Present but Partial
  - Evidence: [src/sdf/hybrid.rs](cci:7://file:///Users/mpopovic3/forge3d/src/sdf/hybrid.rs:0:0-0:0); `src/shaders/hybrid_{kernel,traversal}.wgsl`; [docs/sdf.md](cci:7://file:///Users/mpopovic3/forge3d/docs/sdf.md:0:0-0:0); “hybrid” PT code present but not proven wired to public API
- A11 Participating Media (Single Scatter) — Present but Partial
  - Evidence: [src/shaders/lighting_media.wgsl](cci:7://file:///Users/mpopovic3/forge3d/src/shaders/lighting_media.wgsl:0:0-0:0); [docs/api/participating_media.md](cci:7://file:///Users/mpopovic3/forge3d/docs/api/participating_media.md:0:0-0:0); helpers in Python; no end-to-end PT integration
- A12 Wavefront Path Tracer — Present but Partial
  - Evidence: `src/path_tracing/wavefront/{mod.rs,pipeline.rs,queues.rs}`; WGSL `pt_{raygen,intersect,shade,scatter,compact}.wgsl`; no public API invoking it yet
- A13 Path Guiding — Present but Partial
  - Evidence: [src/path_tracing/guiding.rs](cci:7://file:///Users/mpopovic3/forge3d/src/path_tracing/guiding.rs:0:0-0:0); [src/shaders/pt_guiding.wgsl](cci:7://file:///Users/mpopovic3/forge3d/src/shaders/pt_guiding.wgsl:0:0-0:0); [docs/api/guiding.md](cci:7://file:///Users/mpopovic3/forge3d/docs/api/guiding.md:0:0-0:0)
- A14 AOVs & Debug Outputs — Present & Wired
  - Evidence: [src/path_tracing/aov.rs](cci:7://file:///Users/mpopovic3/forge3d/src/path_tracing/aov.rs:0:0-0:0); AOV bind group in [src/path_tracing/compute.rs](cci:7://file:///Users/mpopovic3/forge3d/src/path_tracing/compute.rs:0:0-0:0) (bgl4 with 7 targets); Python [render_aovs()](cci:1://file:///Users/mpopovic3/forge3d/src/path_tracing/compute.rs:519:4-1041:5); [docs/api/aovs.md](cci:7://file:///Users/mpopovic3/forge3d/docs/api/aovs.md:0:0-0:0)
- A15 Progressive/Checkpoint & Tiling — Present & Wired
  - Evidence: [python/forge3d/path_tracing.py::PathTracer.render_progressive](cci:1://file:///Users/mpopovic3/forge3d/python/forge3d/path_tracing.py:320:4-382:18); [examples/progressive_tiling.py](cci:7://file:///Users/mpopovic3/forge3d/examples/progressive_tiling.py:0:0-0:0)
- A16 QMC & Adaptive Sampler (f16‑ready) — Absent
  - Evidence: no concrete Sobol/Owen/blue-noise/adaptive sampling in PT path
- A17 Firefly Clamp — Present & Wired
  - Evidence: Python API accepts `luminance_clamp`/`firefly_clamp` ([python/forge3d/path_tracing.py](cci:7://file:///Users/mpopovic3/forge3d/python/forge3d/path_tracing.py:0:0-0:0) lines ~183–226); [docs/api/firefly_clamp.md](cci:7://file:///Users/mpopovic3/forge3d/docs/api/firefly_clamp.md:0:0-0:0); stub test present
- A18 Ground Plane (PT) — Present & Wired
  - Evidence: [src/shaders/pt_kernel.wgsl](cci:7://file:///Users/mpopovic3/forge3d/src/shaders/pt_kernel.wgsl:0:0-0:0) mentions minimal ground + env; used by [compute.rs](cci:7://file:///Users/mpopovic3/forge3d/src/core/async_compute.rs:0:0-0:0)
- A19 Scene Cache for HQ — Present & Wired (CPU path)
  - Evidence: [_SceneCache](cci:2://file:///Users/mpopovic3/forge3d/python/forge3d/path_tracing.py:21:0-58:32) + [enable_scene_cache()](cci:1://file:///Users/mpopovic3/forge3d/python/forge3d/path_tracing.py:128:4-131:67) in [python/forge3d/path_tracing.py](cci:7://file:///Users/mpopovic3/forge3d/python/forge3d/path_tracing.py:0:0-0:0); WGSL [pt_cache.wgsl](cci:7://file:///Users/mpopovic3/forge3d/src/shaders/pt_cache.wgsl:0:0-0:0) scaffolding
- A20 Soft Area Lights Param — Present but Partial
  - Evidence: [src/lighting/area_lights.rs](cci:7://file:///Users/mpopovic3/forge3d/src/lighting/area_lights.rs:0:0-0:0); [src/shaders/lighting_area.wgsl](cci:7://file:///Users/mpopovic3/forge3d/src/shaders/lighting_area.wgsl:0:0-0:0); not clearly integrated with PT shaders
- A21 Ambient Occlusion Integrator (Offline) — Present but Partial
  - Evidence: [python/forge3d/ambient_occlusion.py](cci:7://file:///Users/mpopovic3/forge3d/python/forge3d/ambient_occlusion.py:0:0-0:0); [src/post/ambient_occlusion.rs](cci:7://file:///Users/mpopovic3/forge3d/src/post/ambient_occlusion.rs:0:0-0:0); [src/shaders/ao_compute.wgsl](cci:7://file:///Users/mpopovic3/forge3d/src/shaders/ao_compute.wgsl:0:0-0:0); some raster-side AO in PBR/shaders; PT linkage unclear
- A22 Instanced Geometry (PT) — Present but Partial
  - Evidence: PT mesh upload exists; TLAS-style instances not wired through kernels; instancing exists in raster/vector paths
- A23 Hair BSDF + Curve Prims (PT) — Present but Partial
  - Evidence: [src/pbr/hair_bsdf.rs](cci:7://file:///Users/mpopovic3/forge3d/src/pbr/hair_bsdf.rs:0:0-0:0); [python/forge3d/hair.py](cci:7://file:///Users/mpopovic3/forge3d/python/forge3d/hair.py:0:0-0:0); PT kernel integration missing
- A24 Anisotropic Microfacet BRDF — Present but Partial
  - Evidence: Raster PBR supports GGX; explicit anisotropic αx/αy and tangent-frame sampling not evident in PT
- A25 Object Importance Sampling — Present but Partial
  - Evidence: [src/path_tracing/importance.rs](cci:7://file:///Users/mpopovic3/forge3d/src/path_tracing/importance.rs:0:0-0:0); [python/forge3d/importance_sampling.py](cci:7://file:///Users/mpopovic3/forge3d/python/forge3d/importance_sampling.py:0:0-0:0); tagging/MIS weighting not integrated in PT kernels

## Evidence Map (selected pointers)

- PT core wiring: [src/path_tracing/compute.rs](cci:7://file:///Users/mpopovic3/forge3d/src/path_tracing/compute.rs:0:0-0:0) include_str of `../shaders/pt_kernel.wgsl` (around lines 55–58, 538–541), AOV bind group layout (around 155–239)
- Wavefront PT: [src/path_tracing/wavefront/pipeline.rs](cci:7://file:///Users/mpopovic3/forge3d/src/path_tracing/wavefront/pipeline.rs:0:0-0:0) include_str of PT stage shaders; [src/path_tracing/wavefront/mod.rs](cci:7://file:///Users/mpopovic3/forge3d/src/path_tracing/wavefront/mod.rs:0:0-0:0) scheduler loop and persistent threads dispatch
- BVH/LBVH: [src/accel/lbvh_gpu.rs](cci:7://file:///Users/mpopovic3/forge3d/src/accel/lbvh_gpu.rs:0:0-0:0) (loads [lbvh_morton.wgsl](cci:7://file:///Users/mpopovic3/forge3d/src/shaders/lbvh_morton.wgsl:0:0-0:0)), [src/accel/sah_cpu.rs](cci:7://file:///Users/mpopovic3/forge3d/src/accel/sah_cpu.rs:0:0-0:0), WGSL `lbvh_{morton,link}.wgsl`, [bvh_refit.wgsl](cci:7://file:///Users/mpopovic3/forge3d/src/shaders/bvh_refit.wgsl:0:0-0:0), [radix_sort_pairs.wgsl](cci:7://file:///Users/mpopovic3/forge3d/src/shaders/radix_sort_pairs.wgsl:0:0-0:0)
- ReSTIR: [src/path_tracing/restir.rs](cci:7://file:///Users/mpopovic3/forge3d/src/path_tracing/restir.rs:0:0-0:0); WGSL `src/shaders/restir_{reservoir,temporal,spatial}.wgsl`; [examples/restir_many_lights.py](cci:7://file:///Users/mpopovic3/forge3d/examples/restir_many_lights.py:0:0-0:0)
- SVGF: WGSL scaffolds `src/shaders/svgf_*.wgsl`; `src/denoise/svgf/*`; tests `tests/test_svgf_*`
- Firefly clamp: [python/forge3d/path_tracing.py](cci:7://file:///Users/mpopovic3/forge3d/python/forge3d/path_tracing.py:0:0-0:0) (luminance clamp handling), [python/forge3d/path_tracing.pyi](cci:7://file:///Users/mpopovic3/forge3d/python/forge3d/path_tracing.pyi:0:0-0:0), [docs/api/firefly_clamp.md](cci:7://file:///Users/mpopovic3/forge3d/docs/api/firefly_clamp.md:0:0-0:0)
- AOVs: [src/path_tracing/aov.rs](cci:7://file:///Users/mpopovic3/forge3d/src/path_tracing/aov.rs:0:0-0:0), Python [render_aovs()](cci:1://file:///Users/mpopovic3/forge3d/src/path_tracing/compute.rs:519:4-1041:5), [docs/api/aovs.md](cci:7://file:///Users/mpopovic3/forge3d/docs/api/aovs.md:0:0-0:0)
- AO: [src/shaders/ao_compute.wgsl](cci:7://file:///Users/mpopovic3/forge3d/src/shaders/ao_compute.wgsl:0:0-0:0); [src/post/ambient_occlusion.rs](cci:7://file:///Users/mpopovic3/forge3d/src/post/ambient_occlusion.rs:0:0-0:0); [python/forge3d/ambient_occlusion.py](cci:7://file:///Users/mpopovic3/forge3d/python/forge3d/ambient_occlusion.py:0:0-0:0)

## Blocking Gaps (why Partial/Absent)

- PT material system (A2/A9/A24): material parameters not propagated into PT kernels; textured sampling absent in compute path; anisotropic BRDF not implemented in PT.
- Mesh PT (A3): kernel bindings exist but [compute.rs](cci:7://file:///Users/mpopovic3/forge3d/src/core/async_compute.rs:0:0-0:0) uses dummy mesh buffers; traversal kernel not integrated into MVP path.
- Lighting (A4/A8/A20/A11): NEE/MIS not wired in PT; ReSTIR kernels exist but not invoked by PT; area lights and media helpers not piped through PT.
- Denoise (A5): SVGF stages are stubs without full pipeline wiring and tests exercising GPU path.
- Wavefront (A12): orchestration modules and stage shaders exist; no user-facing call path that exercises them end-to-end.
- QMC/Adaptive (A16): missing sampler and f16-ready adaptive policies.
- Instances (A22): TLAS/instances not wired into PT traversal.

## Minimal Change Plan (file-level, smallest steps to reach “Present & Wired”)

- A2/A9/A24 (Materials/Textures/Aniso)
  - [src/shaders/pt_shade.wgsl](cci:7://file:///Users/mpopovic3/forge3d/src/shaders/pt_shade.wgsl:0:0-0:0): implement Lambert, GGX metal, dielectric with parameter block; add tangent-frame/αx/αy.
  - [src/path_tracing/compute.rs](cci:7://file:///Users/mpopovic3/forge3d/src/path_tracing/compute.rs:0:0-0:0): extend scene/material buffers; propagate material params to kernels.
  - [python/forge3d/path_tracing.py](cci:7://file:///Users/mpopovic3/forge3d/python/forge3d/path_tracing.py:0:0-0:0): expose material controls mirroring PT params.
- A3/A22 (Mesh + Instances)
  - [src/path_tracing/compute.rs](cci:7://file:///Users/mpopovic3/forge3d/src/path_tracing/compute.rs:0:0-0:0): replace dummy mesh buffers with actual mesh/BVH buffers; upload via [src/path_tracing/mesh.rs](cci:7://file:///Users/mpopovic3/forge3d/src/path_tracing/mesh.rs:0:0-0:0).
  - [src/shaders/pt_intersect_mesh.wgsl](cci:7://file:///Users/mpopovic3/forge3d/src/shaders/pt_intersect_mesh.wgsl:0:0-0:0): finalize traversal and hit reporting layout; align with host-side structs.
  - [src/path_tracing/accel.rs](cci:7://file:///Users/mpopovic3/forge3d/src/path_tracing/accel.rs:0:0-0:0): single entry to build LBVH/SAH, return GPU buffers.
- A4 (NEE/MIS)
  - [src/shaders/pt_shade.wgsl](cci:7://file:///Users/mpopovic3/forge3d/src/shaders/pt_shade.wgsl:0:0-0:0): implement next-event estimation; MIS weights (balance or power heuristic).
  - [src/path_tracing/importance.rs](cci:7://file:///Users/mpopovic3/forge3d/src/path_tracing/importance.rs:0:0-0:0): ensure tags/weights feed into kernel bindings.
- A5 (SVGF)
  - [src/denoise/svgf/pipelines.rs](cci:7://file:///Users/mpopovic3/forge3d/src/denoise/svgf/pipelines.rs:0:0-0:0): create wgpu pipelines for variance, reproject, a-trous; wire dispatch from PT output AOVs.
  - Add test: `tests/test_svgf_pipeline.py` with small images to validate topology.
- A7 (LBVH/Refit)
  - [src/accel/lbvh_gpu.rs](cci:7://file:///Users/mpopovic3/forge3d/src/accel/lbvh_gpu.rs:0:0-0:0): complete radix pipelines and integrate with `lbvh_*` WGSL; surface `BuildStats`.
  - [src/shaders/bvh_refit.wgsl](cci:7://file:///Users/mpopovic3/forge3d/src/shaders/bvh_refit.wgsl:0:0-0:0): verify node flags, complete bottom-up refit; add unit test.
- A8 (ReSTIR)
  - [src/path_tracing/restir.rs](cci:7://file:///Users/mpopovic3/forge3d/src/path_tracing/restir.rs:0:0-0:0): create pipelines for temporal/spatial passes; add scheduling after direct lighting pass.
  - [src/path_tracing/alias_table.rs](cci:7://file:///Users/mpopovic3/forge3d/src/path_tracing/alias_table.rs:0:0-0:0): confirm alias tables generation and binding.
- A10 (SDF/CSG)
  - [src/shaders/hybrid_kernel.wgsl](cci:7://file:///Users/mpopovic3/forge3d/src/shaders/hybrid_kernel.wgsl:0:0-0:0): interop between SDF raymarch and BVH traversal; validate hybrid hits.
  - [src/path_tracing/hybrid_compute.rs](cci:7://file:///Users/mpopovic3/forge3d/src/path_tracing/hybrid_compute.rs:0:0-0:0): provide sample integration path and emit AOVs.
- A11 (Media)
  - [src/shaders/pt_shade.wgsl](cci:7://file:///Users/mpopovic3/forge3d/src/shaders/pt_shade.wgsl:0:0-0:0): HG phase evaluation and single-scatter estimate; medium params buffer.
- A12 (Wavefront)
  - [src/path_tracing/wavefront/mod.rs](cci:7://file:///Users/mpopovic3/forge3d/src/path_tracing/wavefront/mod.rs:0:0-0:0): expose a public API entry; launch sequence raygen → intersect → shade → scatter → compact with termination.
  - Add example: `examples/wavefront_cornell.py`.
- A16 (QMC/Adaptive)
  - [src/shaders/pt_raygen.wgsl](cci:7://file:///Users/mpopovic3/forge3d/src/shaders/pt_raygen.wgsl:0:0-0:0): Sobol/Owen sequence per-pixel; adaptive sampling threshold in uniforms.
- A20 (Area lights)
  - [src/shaders/lighting_area.wgsl](cci:7://file:///Users/mpopovic3/forge3d/src/shaders/lighting_area.wgsl:0:0-0:0): finalize sampling models; connect to PT shading for direct lighting.
- A21 (AO offline)
  - [src/post/ambient_occlusion.rs](cci:7://file:///Users/mpopovic3/forge3d/src/post/ambient_occlusion.rs:0:0-0:0): finish compute path; add Python hook; add example `examples/ao_offline.py`.

## Validation Runbook

- Build
  - cargo build --all-features
  - Optional: cargo clippy --all-features
- GPU/PT demos and tests
  - python -c "import forge3d.path_tracing as pt; t=pt.PathTracer(256,256,seed=2); t.render_rgba(spp=1)"
  - python examples/progressive_tiling.py
  - python examples/restir_many_lights.py
- Denoise and AOVs
  - python - <<'PY'\nfrom forge3d.path_tracing import render_aovs; a=render_aovs(64,64,None,seed=1); print([k for k in a])\nPY
- Docs
  - sphinx-build -b html docs _build/html
