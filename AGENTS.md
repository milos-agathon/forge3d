# forge3d Agent Guide

## Scope and authority

This file governs the repository unless a deeper `AGENTS.md` applies.

- `AGENTS.md` is the source of truth for durable AI working guidance.
- Code, tests, manifests, and `.github/workflows/ci.yml` are authoritative for product behavior and executable configuration.
- Prefer executable gates over prose. Correct stale in-scope guidance when found.
- Do not mirror new durable guidance into tool-specific instruction files unless explicitly requested.

## Commands

Run focused checks while iterating, then every relevant gate before claiming completion. Report anything not run.

```powershell
# Setup.
python -m pip install -U pip maturin pytest

# Rebuild after Rust or WGSL changes.
maturin develop

# Focused API contracts and canonical Python CI lane.
python -m pytest tests/test_api_contracts.py -v --tb=short
python scripts/ci_pytest_lane.py -v --tb=short

# Rust format and canonical lint.
cargo fmt --check
cargo forge3d-clippy

# CI-equivalent Rust tests; keep synchronized with ci.yml.
cargo test --workspace --features default,async_readback,copc_laz,cog_streaming,gis-remote,geos-topology,weighted-oit,wsI_bigbuf,wsI_double_buf,enable-pbr,enable-tbn,enable-normal-mapping,enable-hdr-offscreen,enable-renderer-config,enable-staging-rings,shader-contract-asserts -- --test-threads=1 --skip gpu_extrusion --skip brdf_tile

# Docs.
make -C docs html
```

`cargo test <filter> --lib` without `--features extension-module` can skip the terrain renderer. Confirm expected test names ran.

## Boundaries

### Always

- Read relevant callers, callees, tests, and signatures before editing; fix shared root causes once.
- Reuse existing helpers and patterns before adding code or dependencies.
- Preserve unrelated working-tree changes and keep the diff scoped.
- Validate inputs at Python/native, file-format, network, and GPU boundaries.
- Update public APIs, stubs, exports, contract lists, tests, and docs together.
- State exactly which checks ran and their results.

### Ask first

- Add or upgrade a production dependency.
- Break public compatibility or serialized formats.
- Update committed goldens, signed certificates, or determinism hashes.
- Weaken, skip, allow-list, or delete a test, capability gate, or safety check.
- Perform destructive, release, deployment, signing, or unrequested external actions.

### Never

- Open, print, copy, modify, or commit credentials, tokens, private keys, `.env` files, or suspected secrets.
- Edit generated products: `target/`, `_build/`, `dist/`, `python/forge3d/_forge3d.pyd`, or `python/forge3d/forge3d.pdb`.
- Claim a check passed unless its successful result was observed.
- Hide unavailable capability behind a placeholder, silent fallback, synthetic success, or misleading certificate.
- Reintroduce deleted legacy postfx/framegraph structures or dead feature flags without a current caller and explicit requirement.

## Stack and repository map

- Rust edition 2021; wgpu 0.19; WGSL shaders.
- Python 3.10+; PyO3 0.21.2; NumPy; maturin >=1.5,<2.0.
- `src/lib.rs`: crate root and native module entry.
- `src/py_module/`, `src/py_functions/`, `src/py_types/`: PyO3 bridge.
- `src/core/`: GPU context, resource tracking, timing, render contracts.
- `src/terrain/`: primary terrain/rendering engine.
- `src/shaders/` and `src/viewer/**/*.wgsl`: shader sources.
- `src/gis/`: native GIS implementation.
- `python/forge3d/`: public Python API, wrappers, and `.pyi` stubs.
- `tests/`: behavior, contract, honesty, reachability, golden gates.
- `.github/workflows/ci.yml`: supported feature matrix and CI authority.

See `README.md`, `CONTRIBUTING.md`, `docs/start/architecture.md`, and `docs/guides/feature_map.md` for architecture and setup.

## Native Python contracts

- Distinguish module functions from instance methods; read the Rust signature before wiring Python.
- Register every `#[pyclass]` and exposed `#[pyfunction]`; match registration and implementation `#[cfg]` gates.
- For public native symbols, update guarded re-exports, `__all__`, `.pyi` stubs, and `EXPECTED_FUNCTIONS`/`EXPECTED_CLASSES`.
- Wheel features belong in `[tool.maturin].features`; directly used optional crates must be direct optional dependencies.
- Prefer thin PyO3 wrappers and existing tuple, dict, constructor-default, and `__repr__` conventions.
- Run `maturin develop` before Python tests to avoid a stale native binary.

## GPU and WGSL contracts

- Synchronize Rust/WGSL bindings, formats, alignment, and storage-array stride; verify byte stride, not field order alone.
- Stay within negotiated adapter limits; dummy bindings cover one full element of their largest represented WGSL type.
- A clean Rust build does not prove WGSL validity. Rebuild and run the focused GPU pipeline.
- Create production GPU resources through tracked helpers in `src/core/resource_tracker.rs`; use Drop guards for early returns.
- Evaluate budgets after relevant allocations; host-visible metrics exclude device-local memory.
- Cache routine-frame bind groups and invalidate them when resource identity changes.
- Negotiate capabilities; never use a default empty device descriptor in production render paths.

## Render honesty and determinism

- `MapScene.render` uses native GPU terrain or raises `MapSceneNativeUnavailable` with structured diagnostics; no CPU placeholder or `allow_placeholder`.
- Depth culling lives only in `MapScene.compile_plan()`; rendering consumes `CompiledScenePlan` without mutating labels.
- Preserve byte-identical manifest/bundle round trips: reject NaN, canonicalize negative zero, normalize optionals at decode, and persist compiled plans.
- GPU render entry points follow the certificate contract or a documented exclusion. Timings come from executed GPU work.
- Preserve explicit scheduler capability fingerprints; probe only when an external renderer supplies none.
- Warm cache indexes avoid per-entry metadata reads, count their own bytes, and fall back on inventory mismatch.
- Cross-backend portability requires an independent physical render with qualifying golden and adapter metadata; otherwise report `ABSENT`.
- Verify committed goldens with `git ls-files`, not working-tree presence.

## Change-specific verification

- Python-only: focused pytest; add the canonical lane for shared behavior or CI accounting.
- Rust: focused Rust test, `cargo fmt --check`, and `cargo forge3d-clippy`.
- Rust/PyO3 API: `maturin develop`, API contracts, focused Python behavior tests.
- WGSL/pipeline: `maturin develop`, shader reachability, focused GPU smoke or golden.
- MapScene/certificate/cache/determinism: named integrity tests and affected golden gate.
- Documentation-only: verify referenced commands/paths and run `git diff --check`.
- CI-only matrices remain reported as not run locally.

## Definition of done

- The requested behavior is implemented at the shared root cause.
- APIs, stubs, exports, serialization, tests, and docs agree.
- Relevant checks passed; actual results and unrun gates are explicit.
- The diff contains only intended changes, with no secrets or generated artifacts.
