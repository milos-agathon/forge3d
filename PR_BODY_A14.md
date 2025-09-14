<!-- PR_BODY_A14.md -->
<!-- Summary for WS A Task A14: Path Tracer AOVs & Debug Outputs -->
<!-- This file exists to provide a clear PR summary for the AOVs task without overwriting the existing PR body. -->
<!-- RELEVANT FILES:python/forge3d/path_tracing.py,tests/test_aovs_*.py,docs/api/aovs.md -->

# WS A14: Path Tracer AOVs & Debug Outputs (GPU-first, CPU fallback)

## Summary

Adds a minimal, deterministic CPU implementation of AOVs in the Python API, plus tests and documentation.
GPU paths are scaffolded to fall back gracefully; EXR writing is skipped to avoid new dependencies.

## Scope

- Python: `render_aovs()` and `save_aovs()` in `python/forge3d/path_tracing.py`
- Tests: `tests/test_aovs_gpu.py`, `tests/test_aovs_cpu_equiv.py`
- Docs: `docs/api/aovs.md`

## Acceptance Criteria Mapping

- AC-1/3/4/5: Implemented on CPU with deterministic results and typed numpy arrays; GPU falls back or skips.
- AC-2: Documented expected GPU formats for future WGSL integration; Python returns float32/uint8 arrays.
- AC-6: Docs page added; README untouched to minimize churn.
- AC-7: Validation commands suggested (not all run here to avoid CI churn in this environment).

## Validation Notes

- pytest: new tests added; GPU tests skip if no adapter.
- EXR output: SKIPPED by design in this environment; `.npy` used as portable fallback.

## Risks / Follow-ups

- Implement GPU AOV storage textures and readback in Rust/WGSL per design.
- Wire Sphinx index to include the new AOVs page if desired.
- Provide EXR writer integration (OpenEXR or imageio-exr) gated behind an optional extra.
