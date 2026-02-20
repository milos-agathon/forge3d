# Rust/Python API Consolidation Implementation Plan (Revised)

Date: 2026-02-19  
Status: implementation-ready  
Replaces: `docs/plans/2026-02-19-api-consolidation-impl-plan.md`

## Objective

Close real Rust/Python API drift with low regression risk, then wire missing runtime behavior with measurable tests.

## Guardrails

1. No speculative API additions without a concrete Python caller.
2. No placeholder tests as acceptance criteria.
3. No panics in Python-facing Rust paths (`unwrap()` forbidden in new code).
4. Stubs/docs must be updated in the same phase as any public API change.
5. Large-risk tasks (point cloud GPU pipeline, COPC LAZ) must include real fixtures or stay deferred.

## Non-goals in this plan

1. Broad deprecation of Rust modules (`export/style/tiles3d/bundle`) without usage evidence.
2. New configuration surfaces that duplicate existing `Scene` controls.
3. Symbol-count inflation as a success metric.

## Phase P0: Contract Stabilization (Low Risk, High Value)

Goal: make existing API contracts consistent and remove confirmed wrapper/native drift.

### P0.1 Baseline + Contract Lock

Files:

- Create: `tests/test_api_contracts.py`
- Create/update: `docs/notes/audit_snapshot.md` (generated baseline)

Work:

1. Generate baseline snapshot with `python scripts/generate_audit_snapshot.py`.
2. Add contract tests for symbols actually used by wrappers:
   - `_forge3d.Scene.render_rgba` exists.
   - `_forge3d.Scene.set_msaa_samples` exists.
   - `_forge3d.mesh_generate_cube_tbn` and `_forge3d.mesh_generate_plane_tbn` behavior (if exported later in P0.4).
3. Record current runtime probes (feature flags and symbol presence) in a single JSON artifact under `docs/notes/`.

Acceptance criteria:

1. Baseline snapshot generated and committed.
2. Contract tests pass on current branch before any functional changes.

### P0.2 Fix Wrapper/Native Callsite Mismatch

Files:

- Modify: `python/forge3d/helpers/offscreen.py`
- Modify: `python/forge3d/viewer.py`
- Modify: `tests/test_api_contracts.py`

Work:

1. Stop relying on nonexistent module-level `_forge3d.render_rgba`.
2. In offscreen helper, prefer `Scene.render_rgba` when `scene` is a native `Scene`; otherwise use existing fallback.
3. Stop relying on nonexistent module-level `_forge3d.set_msaa_samples` as authoritative control path.
4. Keep compatibility behavior explicit and documented.

Acceptance criteria:

1. `tests/test_api_contracts.py` validates offscreen path behavior for:
   - native `Scene` provided,
   - no native scene provided.
2. No warnings/errors caused by calling nonexistent module-level functions.

### P0.3 Register Orphaned PyO3 Classes

Files:

- Modify: `src/lib.rs`
- Modify: `tests/test_api_contracts.py`

Work:

1. Register `Frame` and SDF pyclasses (`PySdfPrimitive`, `PySdfScene`, `PySdfSceneBuilder`) in module init.
2. Add importability and construction tests in Python.

Acceptance criteria:

1. `from forge3d._forge3d import Frame, SdfPrimitive, SdfScene, SdfSceneBuilder` succeeds.
2. `maturin develop --release` succeeds without new warnings introduced by this task.

### P0.4 Resolve Mesh TBN Drift

Files:

- Modify: `src/lib.rs`
- Modify: `tests/test_api_contracts.py`
- Validate existing wrapper behavior: `python/forge3d/mesh.py`

Work:

1. Export mesh TBN functions expected by Python wrapper:
   - `mesh_generate_cube_tbn`
   - `mesh_generate_plane_tbn`
2. Ensure return shape matches wrapper expectations (`vertices`, `indices`, `tbn_data`).
3. Use `?`-style error propagation only; no `unwrap()`.

Acceptance criteria:

1. Existing `python/forge3d/mesh.py` runs on native path when available.
2. Contract test validates keys and basic index counts.

### P0.5 Discoverability Sync for Existing Scene Features

Files:

- Modify: `python/forge3d/__init__.pyi`
- Modify docs under `docs/api/` (feature-specific file)
- Optional: `python/forge3d/__init__.py` export comments

Work:

1. Add type stubs for already-implemented reflection methods on `Scene`.
2. Verify cloud-shadow methods are documented/stubbed consistently.
3. Do not add a duplicate `ReflectionSettings` native class in this phase.

Acceptance criteria:

1. Stubs reflect actual `Scene` methods (no missing public methods discovered in audit).
2. Stub import/type checks pass.

### P0 Exit Gate

All must hold:

1. `maturin develop --release` succeeds.
2. `python -m pytest tests/test_api_contracts.py -v` passes.
3. No task in P0 introduces duplicate API surfaces for reflection/cloud controls.

## Phase P1: Runtime Wiring with Behavioral Proof

Goal: complete partially wired runtime paths with end-to-end verification.

### P1.1 SSGI/SSR Settings Wiring

Files:

- Modify: `src/lib.rs`, `src/core/screen_space_effects.rs`, or owning integration point
- Modify: relevant Python API layer where settings are passed
- Create/modify behavior test file (not registration-only)

Work:

1. Identify authoritative owner for SSGI/SSR settings application.
2. Add explicit path that applies Python-configured settings to `update_settings()`.
3. Add behavior test that proves changed settings affect runtime state or output.

Acceptance criteria:

1. Test demonstrates output/state difference when changing at least one SSGI and one SSR setting.
2. Registration-only tests are not used as sole proof.

### P1.2 Bloom Execute Wiring via Resource Pool

Files:

- Modify: `src/core/bloom.rs`
- Modify/add: postfx chain integration tests

Work:

1. Replace no-op `execute()` with real dispatches using existing `PostFxResourcePool`.
2. Do not create per-frame transient textures inside `execute()`.
3. Respect `config.enabled` passthrough semantics.

Acceptance criteria:

1. Disabled bloom path is bitwise/near-bitwise passthrough on deterministic input.
2. Enabled bloom path changes output on a bright test frame.
3. No per-frame allocation regression introduced by bloom execution path.

### P1.3 Terrain Analysis API Decision (Single Surface)

Files:

- Option A (preferred): Python wrapper-level convenience exposing existing `TerrainSpike` methods
- Option B: module-level `_forge3d` functions, only if justified by callers

Work:

1. Choose one exposure shape and document why.
2. Avoid maintaining duplicate implementations with overlapping behavior.
3. Add tests for numerical sanity and error handling.

Acceptance criteria:

1. One clear public API path documented.
2. Tests cover valid input and at least two invalid-input paths.

### P1 Exit Gate

All must hold:

1. `python -m pytest tests/ -q` passes for touched domains.
2. Runtime-wiring tasks have behavior tests, not just symbol/field tests.
3. No unresolved TODOs remain in modified code paths.

## Phase P2: High-Risk / Large-Scope Items (Gated)

Goal: implement heavier renderer/data-path changes only with fixtures and explicit perf/correctness gates.

### P2.1 Point Cloud GPU Rendering Path

Files:

- Modify: `src/pointcloud/renderer.rs`
- Modify: viewer/render integration path that consumes GPU buffer
- Add integration tests or golden capture harness

Work:

1. Implement complete renderable GPU path, not only `create_gpu_buffer()`.
2. Integrate with actual draw submission path.
3. Add correctness and memory-budget checks.

Acceptance criteria:

1. Viewer or offscreen integration test exercises GPU path end-to-end.
2. CPU fallback remains intact.
3. Memory/perf regressions documented.

### P2.2 COPC LAZ Decompression (Feature-Gated)

Files:

- Modify: `Cargo.toml` (feature + dependency)
- Modify: `src/pointcloud/copc.rs`
- Add fixture tests with real `.copc.laz` sample

Work:

1. Add LAZ decode path under explicit feature (e.g., `copc_laz`).
2. Keep non-feature behavior explicit and safe.
3. Reuse decode logic between raw and decompressed data paths.

Acceptance criteria:

1. Fixture-based test validates decoded point count and sample coordinates.
2. Build passes with and without `copc_laz` feature.

### P2.3 Labels Python Bindings (Scoped MVP)

Files:

- Create/modify: `src/labels/py_bindings.rs`, `src/labels/mod.rs`, `src/lib.rs`
- Add Python tests

Work:

1. Expose minimal, non-conflicting label API mapped to actual Rust `LabelStyle` semantics.
2. Avoid redefining field names that diverge from core types without translation logic.

Acceptance criteria:

1. Bound fields map correctly to underlying Rust model.
2. Tests validate read/write and default behavior.

### P2.4 Deprecation Policy (Separate Decision Track)

Files:

- Design doc update only in this phase unless approved

Work:

1. Gather callsite/usage evidence before applying `#[deprecated]`.
2. Define migration window and warning strategy.
3. Execute deprecation in follow-up phase only after approval.

Acceptance criteria:

1. Signed-off decision record exists.
2. No code-level deprecation annotations merged without approved policy.

### P2 Exit Gate

All must hold:

1. Feature-gated behavior validated both on and off states.
2. Integration tests exist for each merged high-risk task.
3. Rollback instructions documented per task.

## Verification Matrix (All Phases)

1. Build:
   - `maturin develop --release`
2. Contract tests:
   - `python -m pytest tests/test_api_contracts.py -v`
3. Domain tests:
   - run only touched suites per phase, then full suite before phase close.
4. Runtime probe:
   - verify expected symbol presence/absence and method placement.

## Recommended Implementation Order

1. P0.1
2. P0.2
3. P0.3
4. P0.4
5. P0.5
6. P1.1
7. P1.2
8. P1.3
9. P2.* (only after P1 exit gate)

## Deferred from Original Plan

1. New `PyReflectionSettings` class: deferred pending ownership decision.
2. Cloud-shadow fields added to `PyVolumetricSettings`: deferred pending ownership decision.
3. Rust module-wide deprecation annotations: moved to policy track.
4. Placeholder test files as acceptance criteria: removed.
