<!-- PR_BODY.md -->
<!-- Summary of Workstream A scaffolding and initial implementation. -->
<!-- This exists to document scope, changes, and validation for the PR. -->
<!-- RELEVANT FILES:PATCHPLAN.md,docs/task.xml,python/forge3d/path_tracing.py,.github/workflows/ci.yml -->

# Workstream A â€” Path Tracing (Scaffolding)

Summary
- Add minimal `forge3d.path_tracing` API (skeleton `PathTracer`).
- Add user docs page `docs/user/path_tracing.rst` and TOC entry.
- Add smoke tests `tests/test_path_tracing_api.py`.
- Add `ci.yml` to run build, pytest, and docs across OS/Python.

Rationale
- Start implementing large Workstream A incrementally without breaking users.
- Provide importable, typed API surface to unblock docs/tests/CI.

Validation
- Local `pytest -q`: passing (existing suite + new tests).
- CSV/workstream audit executed earlier; no header issues.
- Docs page added to TOC (docs build runs in CI).

Risks & Mitigations
- Kernel/perf functionality is not present yet â†’ clearly documented as roadmap.
- CI build may increase time â†’ matrix limited to stable toolchains.

Status
- A1: Completed (CPU fallback path, WGSL scaffold present).
- A2/A3/A7: Next — BSDFs and GPU wiring.
- A5/A14: Next — denoiser and AOV/IO.
