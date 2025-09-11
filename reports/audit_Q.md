# Audit Report: Workstream Q – Production Features

**Audit Date:** 2025-09-10  
**Workstream ID:** Q  
**Workstream Title:** Production Features  
**Total Tasks:** 5  
**Mode:** Audit-only (no repository modifications)

## Executive Summary

Readiness across Production Features is mixed:
- Present & Wired: 1/5 (Indirect draw support)
- Present but Partial: 2/5 (Post-processing compute pipeline, LOD system)
- Absent: 2/5 (GPU profiling markers, Bloom post-process)

The codebase has strong HDR/tonemap and compute foundations (`async_compute`, mipmap downsample, vector indirect + GPU culling), but lacks GPU timestamp markers and a Bloom effect chain.

## CSV Hygiene Summary (Workstream Q)

- Headers validated against schema: OK
- Rows scanned: 5 (Q1–Q5)  
- Priority values valid: OK  
- Phase values valid: OK (all Beyond MVP)  
- Required fields present: OK  
- See `reports/csv_hygiene_Q.md` for details.

## Task-by-Task Findings

### Q1: Post-processing compute pipeline (Bloom, DOF, SSAO)
- Priority: Medium | Phase: Beyond MVP
- Verdict: Present but Partial

Evidence:
- `src/core/async_compute.rs` — async compute framework, dispatch params, scheduler (e.g., `execute_queued_passes()`), establishing compute groundwork.
- `src/shaders/mipmap_downsample.wgsl` — compute-based downsampler with gamma-aware paths (box filter), demonstrates compute post-processing infrastructure.
- `src/core/tonemap.rs`, `shaders/postprocess_tonemap.wgsl` — post-process pipeline and WGSL shader for tonemapping (render pass), not compute-based but an established post FX stage.

Gaps vs Acceptance Criteria:
- No implemented Bloom (bright-pass + separable blur + add)
- No DOF/SSAO passes or effect chain manager
- No Python API to configure FX chains, no quality parity tests at 1080p

Minimal Change Plan:
- Add WGSL compute shaders:
  - `src/shaders/bloom_brightpass.wgsl`
  - `src/shaders/bloom_blur_h.wgsl`, `src/shaders/bloom_blur_v.wgsl`
  - Optional: `src/shaders/ssao.wgsl`, `src/shaders/dof_separable.wgsl`
- Wire a post-process manager in Rust under `src/core/postfx.rs` using existing `async_compute` where beneficial.
- Python API surface in `python/forge3d/postfx.py` to configure chain and parameters (`strength`, `threshold`, `radius`, etc.).
- Tests: Golden comparisons and perf (<~1–3 ms @1080p for Bloom), quality gates. Docs page `docs/postfx.md`.

---

### Q2: LOD system implementation (discrete/continuous, impostors)
- Priority: High | Phase: Beyond MVP
- Verdict: Present but Partial

Evidence:
- `src/terrain/lod.rs` — Screen-space error LOD implementation (`select_lod_for_tile()`, `calculate_triangle_reduction()`)
- `tests/test_b12_lod.py` — Extensive tests including ≥40% triangle reduction and budget compliance

Gaps vs Acceptance Criteria:
- Impostor generation and continuous LOD transitions not present
- No explicit streaming integration and perf target validation (<16ms LOD updates)

Minimal Change Plan:
- Add impostor path (sprite/quadtree proxies) and optional morphing for continuous LOD.
- Integrate with terrain tiler (`src/terrain/tiling.rs`) and metrics reporting.
- Perf harness measuring LOD update times; golden-based visual checks at multiple zoom levels.

---

### Q3: GPU profiling markers (RenderDoc/NSight/RGP) + timestamp queries
- Priority: High | Phase: Beyond MVP
- Verdict: Absent

Evidence:
- Render passes specify `timestamp_writes: None` (e.g., `src/pipeline/hdr_offscreen.rs:329–341`, `src/core/hdr.rs:286–305`), but no `wgpu::QuerySet` creation or `write_timestamp()` calls found.
- No Python API surface to expose GPU timings/markers.

Minimal Change Plan:
- Implement timestamp queries:
  - Create `wgpu::QuerySet` (timestamp) in core (e.g., `src/core/gpu_timing.rs`)
  - Insert begin/end timestamps around key passes (HDR, tonemap, terrain, vector)
  - Resolve to buffer and surface metrics via Python API (`Renderer.get_gpu_metrics()`)
- Optional pipeline statistics queries gated by feature checks; docs for tool integration (RenderDoc, Nsight, RGP)
- CI sanity to assert non-zero, stable timings with reasonable variance.

---

### Q4: Indirect draw support (GPU-driven rendering)
- Priority: Low | Phase: Beyond MVP
- Verdict: Present & Wired

Evidence:
- `src/vector/indirect.rs` — Complete indirect rendering and GPU/CPU culling pipeline.
  - `draw_indirect(&self, render_pass, draw_count)` calls `render_pass.draw_indirect(...)` (around line ~476)
  - Culling uniforms, instance storage, counters, readback stats
- `src/shaders/culling_compute.wgsl` — Compute kernel generating indirect commands with frustum/distance culling.
- `src/vector/mod.rs` re-exports `IndirectRenderer` and related types for integration.

Notes:
- Strong base for Q4 acceptance. Add integration/perf tests for ≥10× draw reduction on large scenes and feature detection/fallback where unsupported.

---

### Q5: Bloom post-process (bright-pass + separable blur + add)
- Priority: High | Phase: Beyond MVP
- Verdict: Absent

Evidence:
- No `bloom`, `bright-pass`, or blur shaders found.
- HDR/tonemap pipeline exists but no Bloom stage.

Minimal Change Plan:
- Add WGSL shaders for bright-pass and separable blur (H/V). Composite back to HDR/tonemap chain.
- Wire control params (threshold, strength, sigma). Expose via Python API.
- Goldens and SSIM checks; perf target ≤1–3 ms @1080p.

## Blocking Gaps
- GPU profiling markers/timestamps (Q3) are missing — critical for production profiling.
- Bloom effect (Q5) unimplemented — common production post effect.

## Validation Runbook

Build and basic tests:
```bash
# Build Python extension
maturin develop --release

# Run terrain LOD tests (Q2 evidence)
pytest tests/test_b12_lod.py -v

# Run HDR off-screen tests (post pipeline sanity)
pytest tests/test_hdr_offscreen_pipeline.py -v

# Run Rust unit tests (vector indirect, async compute, etc.)
cargo test -q
```

After implementing Q1/Q3/Q5:
```bash
# PostFX bloom tests (new)
pytest tests/test_postfx_bloom.py -v

# GPU timestamp queries (new)
pytest tests/test_gpu_timestamps.py -v

# Performance sanity with FX chain enabled
python python/tools/perf_sanity.py --width 1920 --height 1080 --runs 60 --warmups 5 \
  --json reports/perf_q_fx.json
```

## Dependencies / Cross-links
- Q1 builds upon: `src/core/async_compute.rs`, `src/core/tonemap.rs`, WGSL infra
- Q2 relies on terrain tiling and SSE logic in `src/terrain/lod.rs`
- Q3 relates to future workstream G4 (timestamp queries) — currently absent in code
- Q4 builds upon vector batching and compute culling (H17/H19)
- Q5 extends HDR→tonemap pipeline with additional post stages

## Conclusion
Workstream Q has a solid base (indirect/compute/HDR/tonemap), but to reach production feature readiness it needs GPU timing/markers and a practical post-processing chain starting with Bloom. LOD is functional but lacks impostors/continuous transitions and perf guardrails.
