// PR_BODY_A12.md
// Wavefront Path Tracer (A12) summary for PR
// This file exists to summarize scope, tests, and validation for Task A12.
// RELEVANT FILES:src/path_tracing/wavefront/mod.rs,src/shaders/pt_raygen.wgsl,tests/test_wavefront_*.py,docs/api/wavefront_pt.md

# WS A12: Wavefront Path Tracer (Queue-Based)

## Summary

Implements a wavefront path tracing scaffold alongside the existing mega-kernel.
Adds WGSL stage shaders, Rust wavefront module, Python engine selection, tests (GPU-skipping by default), and docs.

## Scope

- WGSL: `src/shaders/pt_raygen.wgsl`, `pt_intersect.wgsl`, `pt_shade.wgsl`, `pt_scatter.wgsl`, `pt_compact.wgsl`
- Rust: `src/path_tracing/wavefront/{mod.rs,queues.rs,pipeline.rs}`
- Python: `render_rgba(..., engine=TracerEngine.*)` in `python/forge3d/path_tracing.py`
- Tests: `tests/test_wavefront_parity.py`, `tests/test_wavefront_compaction.py` (skipped unless env `FORGE3D_ENABLE_WAVEFRONT_TESTS=1` and GPU present)
- Docs: `docs/api/wavefront_pt.md`; README section on engine selection

## Notes

- CPU fallback remains deterministic when GPU path is unavailable or disabled.
- Native extension path is guarded for empty scenes to avoid wgpu validation errors.
- Full GPU scheduler wiring and performance validation are future work.
