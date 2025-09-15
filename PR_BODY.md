# WS A – A19: Scene Cache for HQ

## Summary
- Implements a minimal scene cache for the Python fallback path tracer to reuse scene-dependent precomputations across identical re-renders.
- Adds public API on `PathTracer`:
  - `enable_scene_cache(enabled=True, capacity=None)`
  - `reset_scene_cache()`
  - `cache_stats()`
- Adds tests ensuring identical image and >=30% speed improvement on warm re-render.
- Updates README with usage example.

## Files Touched
- python/forge3d/path_tracing.py (A19 cache, keying, reuse)
- python/forge3d/path_tracing.pyi (typed API)
- tests/test_a19_scene_cache.py (AC validation)
- README.md (docs snippet)
- reports/a19_plan.json (mapping meta)

## Validation
- Ran `pytest -q tests/test_a19_scene_cache.py` → 1 passed.
- Full test/CI and rust/clippy not executed here.

## Risks / Notes
- Cache key best-effort JSON for `scene`/`camera`; for non-serializable objects, a repr/id fallback is used. Users should pass stable dict-like descriptors for deterministic reuse.
- CPU fallback only; Rust/WGSL bindings remain unchanged.

