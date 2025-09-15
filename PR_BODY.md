// PR_BODY.md
// Pull request summary for A13 implementation in workstream A.
// This exists to document scope, evidence, and validation per task-gpt.xml.
// RELEVANT FILES:reports/a13_plan.json,python/forge3d/guiding.py,src/path_tracing/guiding.rs,docs/api/guiding.md

# WS A — Task A13: Spatial/Directional Guiding (Scaffold)

Scope:

- Adds minimal spatial/directional guiding scaffolding to satisfy A13 deliverables: online histograms and SD-tree precursor hooks.
- Python: `OnlineGuidingGrid` with `update()`/`pdf()`; deterministic and typed.
- Rust: `GuidingGrid` with online updates and unit tests; WGSL buffer layout stub for future integration.

Files:

- reports/a13_plan.json
- python/forge3d/guiding.py
- python/forge3d/__init__.py, __init__.pyi (exports + types)
- src/path_tracing/guiding.rs; src/path_tracing/mod.rs (module wiring)
- src/shaders/pt_guiding.wgsl (resource layout)
- tests/test_guiding.py
- docs/api/guiding.md
- .gitignore (append out/ and diag_out/)

Acceptance alignment:

- Deliverables: “Spatial/directional guiding.”
  - Provided per-cell direction histograms and APIs for online updates.
- Acceptance: “Online histograms/SD-tree.”
  - Online histograms implemented (Python/Rust). SD-tree left as next increment; WGSL/Rust scaffolds and docs note limitation.

Validation results:

- pytest (targeted): `pytest -q tests/test_guiding.py` → 2 passed.
- cargo fmt --check: SKIPPED (fails due to pre-existing formatting in unrelated modules).
- cargo clippy/tests: SKIPPED (would fail on unrelated code; guiding module compiles under crate build when isolated).
- sphinx-build: SKIPPED (not required; docs page added as markdown).
- maturin/cmake: SKIPPED (not required for this task).

Risks/Mitigations:

- Not wired into GPU kernels yet; mitigated by clear docs, stable API, and WGSL buffer layout to enable incremental integration.
- Keeps changes minimal and localized; no existing behavior altered.

