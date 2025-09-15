# PR_BODY.md
# Workstream A — Task A17: Firefly Clamp
# Implements luminance-based clamp in Python path tracer; adds tests and docs.
# RELEVANT FILES:python/forge3d/path_tracing.py,tests/test_a17_firefly_clamp.py,reports/a17_plan.json

## Summary

- Adds optional `luminance_clamp` (alias `firefly_clamp`) to `PathTracer.render_rgba`.
- Scales RGB by `min(1, clamp / L)` where `L` is Rec. 709 luminance.
- Provides focused test asserting ≥10× outlier reduction with <15% mean luminance shift.
- Documents feature in `docs/api/firefly_clamp.md`.
- Plan mapping captured in `reports/a17_plan.json`.

## Files Touched

- python/forge3d/path_tracing.py — add clamp implementation.
- python/forge3d/path_tracing.pyi — expose new optional kwargs in stub.
- tests/test_a17_firefly_clamp.py — acceptance test for A17.
- docs/api/firefly_clamp.md — usage and acceptance notes.
- reports/a17_plan.json — deliverables mapping.

## Validation Run

Commands and results (non-blocking failures are outside A17 scope):

- cargo fmt -- --check → FAILED (unrelated formatting diffs in Rust files).
- cargo clippy --all-targets --all-features -D warnings → SKIPPED (clippy not available or treated as check).
- cargo test -q → FAILED (numerous unrelated Rust errors; A17 does not modify Rust).
- pytest -q (full) → FAILED (legacy smoke tests expecting Renderer class).
- pytest -q tests/test_a17_firefly_clamp.py → PASSED.
- sphinx-build -b html docs _build/html → SKIPPED (sphinx-build not found).
- maturin build --release → FAILED (README.md non-UTF8 blocks build; unrelated to A17 change).
- cmake -S . -B build && cmake --build build → SKIPPED (not required for A17).

## Evidence

- Outlier reduction: test shows ≥10× drop in high-luminance pixels when clamped at 0.6.
- Bias control: mean luminance shift ≤ 15% under clamp per test.

## Risks/Mitigations

- Clamp is opt-in via kwargs; default behavior unchanged.
- Implemented in Python stub; WGSL/Rust integration can follow without API breakage.

## Next Steps

- Optionally thread clamp to GPU path once pt_kernel integrates real radiance.
- Resolve repo-wide Rust formatting/issues separately to restore CI.

```
$ git status -s
```

```
$ git log --oneline -n 50 --decorate --graph --all
```
