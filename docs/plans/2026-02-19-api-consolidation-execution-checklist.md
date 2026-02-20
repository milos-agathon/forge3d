# API Consolidation Execution Checklist (Strict)

Use this checklist with `docs/plans/2026-02-19-api-consolidation-impl-plan-revised.md`.

Rule: do not start a later item until the current item passes.

## Global Preconditions

1. Verify tooling.

```powershell
python --version
python -m pip show maturin
cargo --version
```

Pass criteria:

1. All commands exit `0`.

2. Verify clean working state (or record existing dirty state).

```powershell
git status --short
git rev-parse --short HEAD
```

Pass criteria:

1. Current commit hash recorded.

## P0.1 Baseline + Contract Lock

Commands:

```powershell
python scripts/generate_audit_snapshot.py
if (!(Test-Path docs/notes/audit_snapshot.md)) { throw "Missing audit_snapshot.md" }
if (!(Test-Path docs/notes/audit_snapshot.json)) { throw "Missing audit_snapshot.json" }
```

```powershell
python -m pytest tests/test_api_contracts.py -v -x
```

Pass criteria:

1. Snapshot files are generated.
2. `tests/test_api_contracts.py` passes on baseline before functional changes.

Fail criteria:

1. Missing snapshot outputs.
2. Contract test file absent or failing.

## P0.2 Wrapper/Native Callsite Mismatch

Commands:

```powershell
python -m pytest tests/test_api_contracts.py::test_offscreen_prefers_scene_method -v -x
python -m pytest tests/test_api_contracts.py::test_offscreen_fallback_without_scene -v -x
python -m pytest tests/test_api_contracts.py::test_viewer_set_msaa_no_module_level_native_dependency -v -x
```

```powershell
rg -n '_native\.render_rgba|_forge3d\.set_msaa_samples|getattr\(_forge3d' python/forge3d/helpers/offscreen.py python/forge3d/viewer.py
```

Pass criteria:

1. All three tests pass.
2. `rg` returns no matches (exit code `1`).

Fail criteria:

1. Any test failure.
2. Any direct module-level dependency remains in those files.

## P0.3 Register Orphaned PyO3 Classes

Commands:

```powershell
maturin develop --release
python -c "from forge3d._forge3d import Frame, SdfPrimitive, SdfScene, SdfSceneBuilder; print('OK')"
python -m pytest tests/test_api_contracts.py::test_frame_importable tests/test_api_contracts.py::test_sdf_classes_importable -v -x
```

Pass criteria:

1. Build succeeds.
2. Import command succeeds.
3. Both tests pass.

Fail criteria:

1. Any build/import/test failure.

## P0.4 Resolve Mesh TBN Drift

Commands:

```powershell
maturin develop --release
python -c "import forge3d._forge3d as n; assert hasattr(n,'mesh_generate_cube_tbn'); assert hasattr(n,'mesh_generate_plane_tbn'); print('OK')"
python -m pytest tests/test_api_contracts.py::test_mesh_tbn_native_exports tests/test_api_contracts.py::test_mesh_tbn_native_shape_contract -v -x
python -m pytest tests/test_mesh_tbn.py -v -x
```

Pass criteria:

1. Both symbols exist in native module.
2. Contract tests pass.
3. Existing mesh TBN tests pass.

Fail criteria:

1. Missing native symbol.
2. Return shape mismatch.
3. Any test failure.

## P0.5 Discoverability Sync (Stubs + Docs)

Commands:

```powershell
rg -n "enable_reflections|disable_reflections|set_reflection_plane|set_reflection_intensity|set_reflection_fresnel_power|set_reflection_distance_fade|set_reflection_debug_mode|enable_cloud_shadows|disable_cloud_shadows|set_cloud_shadow_intensity|set_cloud_shadow_softness" python/forge3d/__init__.pyi
rg -n "enable_reflections|enable_cloud_shadows" docs/api
python -m pytest tests/test_api_contracts.py::test_scene_stub_matches_runtime -v -x
```

Pass criteria:

1. Stub includes reflection/cloud-shadow Scene methods.
2. Docs mention feature entry points.
3. Stub/runtime consistency test passes.

Fail criteria:

1. Missing methods in stubs.
2. Consistency test fails.

## P0 Exit Gate

Commands:

```powershell
maturin develop --release
python -m pytest tests/test_api_contracts.py -v
python -c "import forge3d._forge3d as n; assert hasattr(n,'Scene'); assert hasattr(n.Scene,'render_rgba'); assert hasattr(n.Scene,'set_msaa_samples'); print('P0 OK')"
```

Pass criteria:

1. All commands succeed.

Fail criteria:

1. Any command fails.

## P1.1 SSGI/SSR Wiring (Behavioral)

Commands:

```powershell
maturin develop --release
python -m pytest tests/test_ssgi_ssr_wiring.py -v -x
```

Pass criteria:

1. Test proves changed SSGI/SSR settings alter runtime state or rendered output.

Fail criteria:

1. Test only checks field mutability or symbol existence.
2. No observable behavior change.

## P1.2 Bloom Execute Wiring (Behavioral)

Commands:

```powershell
maturin develop --release
python -m pytest tests/test_bloom_execute_behavior.py -v -x
rg -n "Bloom is a no-op" src/core/bloom.rs
```

Pass criteria:

1. Bloom behavior tests pass:
   - disabled => passthrough,
   - enabled => output change on bright input.
2. No remaining no-op marker text in bloom execute path (`rg` no match, exit `1`).

Fail criteria:

1. Any behavior test fails.
2. No-op marker still present.

## P1.3 Terrain Analysis Single-Surface Decision

Commands:

```powershell
python -m pytest tests/test_terrain_analysis_api.py -v -x
rg -n "terrain analysis API decision|Option A|Option B" docs/plans/2026-02-19-api-consolidation-impl-plan-revised.md
```

Pass criteria:

1. One public API surface validated by tests.
2. Decision is documented and reflected in tests.

Fail criteria:

1. Duplicate public surfaces introduced without translation rationale.
2. Tests do not cover invalid input paths.

## P1 Exit Gate

Commands:

```powershell
python -m pytest tests/test_api_contracts.py tests/test_ssgi_ssr_wiring.py tests/test_bloom_execute_behavior.py tests/test_terrain_analysis_api.py -v
```

Pass criteria:

1. All touched-domain tests pass.

Fail criteria:

1. Any failure in touched-domain tests.

## P2.1 Point Cloud GPU Path (Integration)

Commands:

```powershell
maturin develop --release
python -m pytest tests/test_pointcloud_gpu_integration.py -v -x
```

Pass criteria:

1. Integration test exercises GPU point-cloud draw path end-to-end.
2. CPU fallback path still passes its tests.

Fail criteria:

1. Only compile/import checks exist.

## P2.2 COPC LAZ Feature-Gated Path

Commands:

```powershell
maturin develop --release
maturin develop --release --features "extension-module,copc_laz,enable-tbn"
python -m pytest tests/test_copc_laz_fixture.py -v -x
```

Pass criteria:

1. Build succeeds with and without `copc_laz`.
2. Fixture test validates decompression correctness.

Fail criteria:

1. No real fixture validation.
2. Either feature mode fails to build.

## P2.3 Labels Binding MVP

Commands:

```powershell
maturin develop --release
python -m pytest tests/test_labels_pybindings.py -v -x
```

Pass criteria:

1. Bound fields match underlying Rust label semantics.
2. Default/read/write behavior verified.

Fail criteria:

1. Field names diverge from Rust model without mapping logic.

## P2.4 Deprecation Policy Gate

Commands:

```powershell
if (!(Test-Path docs/plans/2026-02-19-api-deprecation-policy.md)) { throw "Missing deprecation policy" }
rg -n "approved|sign-off|migration window|warning strategy" docs/plans/2026-02-19-api-deprecation-policy.md
```

Pass criteria:

1. Signed policy exists before any `#[deprecated]` rollout.

Fail criteria:

1. Code deprecations merged without policy sign-off.

## Final Gate

Commands:

```powershell
maturin develop --release
python -m pytest tests/ -q --tb=short
python scripts/generate_audit_snapshot.py
python -c "import forge3d._forge3d as n; print('Native symbols:', len([s for s in dir(n) if not s.startswith('_')]))"
git status --short
```

Pass criteria:

1. Full test suite passes.
2. Snapshot regenerates cleanly.
3. Working tree only contains intended changes.

Fail criteria:

1. Any command fails.
