# forge3d Codebase Refactor Playbook

This document provides actionable guidance for planning and executing safe, high-quality refactors across Rust, WGSL, and Python in the forge3d codebase. All guidance aligns with `AGENTS.md`.

---

## 1. Core guardrails

| Principle | Rule |
|-----------|------|
| **Rust first** | Rust `src/` is the rendering engine; Python `python/forge3d/` is the validated facade. Keep PyO3 bindings aligned. |
| **Tests define behavior** | Consult `tests/` and `docs/` before changing semantics. Tests are always Python, never Rust. |
| **Memory budget** | Respect the 512 MiB host-visible heap. Reuse buffers/textures; avoid per-frame allocations. |
| **GPU feature flags** | Honor `Cargo.toml` features (`enable-pbr`, `enable-ibl`, `enable-staging-rings`, `weighted-oit`, etc.). |
| **Small steps** | Extract functions, clarify names, remove duplication. Leave every file cleaner. |
| **Stay in sync** | Bind group layouts ↔ WGSL bindings ↔ Python params must match. No speculative hooks (YAGNI). |

---

## 2. Architecture overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Python facade                            │
│   python/forge3d/__init__.py, config.py, render.py, ...        │
├─────────────────────────────────────────────────────────────────┤
│                      PyO3 bindings                              │
│   src/lib.rs (re-exports), #[pyclass] types                    │
├─────────────────────────────────────────────────────────────────┤
│                        Rust core                                │
│   src/core/, src/terrain*, src/path_tracing/, src/passes/, ... │
├─────────────────────────────────────────────────────────────────┤
│                      WGSL shaders                               │
│   src/shaders/*.wgsl, src/shaders/ao/, src/shaders/gbuffer/    │
└─────────────────────────────────────────────────────────────────┘
```

### 2.1 Major domains

| Domain | Rust modules | Python modules | Shaders | Key tests |
|--------|--------------|----------------|---------|-----------|
| **Terrain** | `src/terrain_renderer.rs`, `src/terrain/*` | `python/forge3d/terrain_params.py` | `src/shaders/terrain*.wgsl` | `tests/test_t*.py` |
| **Path tracing** | `src/path_tracing/` | `python/forge3d/path_tracing.py`, `render.py` | `src/shaders/hybrid_kernel.wgsl` | `tests/test_path_tracing*.py` |
| **Screen-space (P5)** | `src/core/screen_space_effects.rs`, `src/p5/*`, `src/passes/*` | `python/forge3d/screen_space_gi.py` | `src/shaders/ssao.wgsl`, `src/shaders/ao/gtao.wgsl` | `tests/test_p5*.py` |
| **Lighting/PBR** | `src/lighting/*`, `src/core/ibl.rs` | `python/forge3d/pbr.py`, `lighting.py`, `shadows.py` | `src/shaders/pbr*.wgsl`, `src/shaders/ibl*.wgsl` | `tests/test_b*.py` |
| **Vector/overlays** | `src/vector/*`, `src/core/overlays.rs`, `src/core/text_overlay.rs` | `python/forge3d/vector.py` | `src/shaders/vector*.wgsl` | `tests/test_vector*.py` |
| **Memory** | `src/core/memory_tracker.rs`, `src/core/virtual_texture.rs` | `python/forge3d/mem.py`, `memory.py`, `streaming.py` | — | `tests/test_memory*.py` |

### 2.2 Sync pairs (must change together)

| Rust | Python | WGSL | Notes |
|------|--------|------|-------|
| `src/render/params.rs` | `python/forge3d/config.py` | — | Renderer config structs |
| `src/terrain_renderer.rs` bind groups | `python/forge3d/terrain_params.py` | `src/shaders/terrain*.wgsl` @group/@binding | Terrain pipeline |
| `src/core/screen_space_effects.rs` | `python/forge3d/screen_space_gi.py` | `src/shaders/ssao.wgsl`, `src/shaders/ao/gtao.wgsl` | AO/SSGI/SSR |
| `src/lighting/*` uniforms | `python/forge3d/lighting.py` | `src/shaders/pbr*.wgsl` light buffer | PBR lighting |
| `src/lib.rs` PyO3 exports | `python/forge3d/__init__.py` imports | — | Public API surface |

---

## 3. Cargo features reference

Key features in `Cargo.toml` that affect compilation and behavior:

| Feature | Purpose |
|---------|---------|
| `enable-pbr` | PBR material system |
| `enable-ibl` | Image-based lighting |
| `enable-renderer-config` | Extended renderer configuration |
| `enable-staging-rings` | Staging buffer ring allocator |
| `weighted-oit` | Order-independent transparency |

Always build with `--all-features` for full coverage: `cargo check --all-features`

---

## 4. Shader inventory

| Shader file | Purpose | Rust consumer |
|-------------|---------|---------------|
| `src/shaders/ssao.wgsl` | Screen-space ambient occlusion | `src/core/screen_space_effects.rs` |
| `src/shaders/ao/gtao.wgsl` | Ground-truth AO variant | `src/passes/ao.rs` |
| `src/shaders/gbuffer/common.wgsl` | G-buffer packing utilities | Multiple passes |
| `src/shaders/bloom_blur_h.wgsl` | Horizontal bloom blur | `src/core/bloom.rs` |
| `src/shaders/bloom_blur_v.wgsl` | Vertical bloom blur | `src/core/bloom.rs` |
| `src/shaders/tonemap*.wgsl` | Tonemapping (ACES, etc.) | `src/core/tonemap.rs` |
| `src/shaders/terrain*.wgsl` | Terrain PBR + POM | `src/terrain_renderer.rs` |
| `src/shaders/hybrid_kernel.wgsl` | Path tracing kernel | `src/path_tracing/` |
| `src/shaders/ibl*.wgsl` | IBL prefilter/irradiance | `src/core/ibl.rs` |

When modifying a shader:
1. Check the `@group` / `@binding` indices match the Rust bind group layout.
2. Verify uniform struct layouts match (alignment, field order).
3. Run the domain's Python tests to confirm output stability.

---

## 5. Safe-by-default workflow
1. **Snapshot the workspace**
   - `pwd`
   - `git status --porcelain`
   - `git rev-parse --show-toplevel`
   - `git rev-parse HEAD`

2. **Read the intent before editing**
   - Re-read `AGENTS.md` rules.
   - Open relevant docs/tests (e.g., `docs/`, `tests/`) to understand expected behavior and invariants.

3. **Scope the change**
   - State the goal, impacted domains (Rust core, Python facade, WGSL), and constraints (memory, feature flags).
   - Identify the exact files and bindings that must stay in sync (e.g., `src/render/params.rs` ↔ `python/forge3d/config.py`, WGSL bind layouts ↔ Rust bind groups).

4. **Design the refactor**
   - Prefer extraction over in-place complexity.
   - Keep functions small and single-responsibility; avoid flag arguments.
   - Maintain the stepdown rule in files: high-level first, details below.
   - Keep public APIs stable unless the change demands otherwise; use expand/contract when altering public surfaces.

5. **Code with guardrails**
   - Use `apply_patch`-style minimal diffs.
   - Align names with intent; avoid ambiguous or cryptic identifiers.
   - Keep terrain, GI, and memory bindings consistent across Rust/WGSL/Python.
   - For third-party boundaries, use thin adapters; isolate vendor-specific calls.

6. **Validate early and often**
   - Prefer targeted, fast checks: `cargo check -q`, `cargo fmt -- --check`, `cargo clippy` (warnings as appropriate), `python -m compileall -q .`, `pytest -q tests/<scope>`.
   - For rendering changes, run the smallest relevant Python test or example; keep golden outputs stable unless intentionally updating them.
   - If behavior changes, add/adjust Python tests; do not add Rust tests for behavior coverage.

7. **Document what changed**
   - Update nearby docs/schemas/config comments when behavior or expectations change.
   - Keep doc updates concise and aligned with actual code paths.

8. **Finish clean**
   - Rerun `git status --porcelain` to ensure only intended files changed.
   - Summarize the change, risks, and validation steps in your PR/commit message.

---

## 6. Domain-specific refactor guidance

### 6.1 Terrain

**Files:** `src/terrain_renderer.rs`, `src/terrain/*`, `python/forge3d/terrain_params.py`, `src/shaders/terrain*.wgsl`

**Key invariants:**
- Heightmap bind group layout must match shader `@group(0)` bindings.
- POM (parallax occlusion mapping) parameters in Python must mirror Rust defaults.
- MSAA selection flows from Python config to Rust pipeline.

**Validation:**
```bash
pytest -q tests/test_terrain*.py tests/test_t*.py
```

### 6.2 Path tracing

**Files:** `src/path_tracing/`, `python/forge3d/path_tracing.py`, `python/forge3d/render.py`, `src/shaders/hybrid_kernel.wgsl`

**Key invariants:**
- CPU fallback (`PathTracer`) must produce deterministic output under fixed seed.
- GPU path is opportunistic; tests tolerate CPU-only environments.
- AOV export shapes/types must match test expectations.

**Validation:**
```bash
pytest -q tests/test_path_tracing*.py tests/test_raytrace*.py
```

### 6.3 Screen-space effects (P5: AO/SSGI/SSR)

**Files:** `src/core/screen_space_effects.rs`, `src/p5/*`, `src/passes/*`, `python/forge3d/screen_space_gi.py`, `src/shaders/ssao.wgsl`, `src/shaders/ao/gtao.wgsl`

**Key invariants:**
- AO/SSGI/SSR can be enabled/disabled independently via `PyScreenSpaceGI`.
- GI composition logic stays in designated shader (`src/shaders/gi/composite.wgsl` if present).
- Energy and component-isolation constraints enforced by P5 tests.

**Validation:**
```bash
pytest -q tests/test_p5*.py
```

### 6.4 Lighting / PBR / IBL

**Files:** `src/lighting/*`, `src/core/ibl.rs`, `python/forge3d/pbr.py`, `python/forge3d/lighting.py`, `python/forge3d/shadows.py`, `src/shaders/pbr*.wgsl`, `src/shaders/ibl*.wgsl`

**Key invariants:**
- Light buffer struct layout in Rust must match WGSL.
- IBL prefilter/irradiance compute passes use `Rgba16Float` format.
- Shadow map resolution and cascade count flow from Python config.

**Validation:**
```bash
pytest -q tests/test_b*.py tests/test_lighting*.py
```

### 6.5 Vector / overlays

**Files:** `src/vector/*`, `src/core/overlays.rs`, `src/core/text_overlay.rs`, `python/forge3d/vector.py`, `src/shaders/vector*.wgsl`

**Key invariants:**
- OIT (order-independent transparency) requires `weighted-oit` feature.
- Text overlay font atlas size and glyph metrics must stay consistent.

**Validation:**
```bash
pytest -q tests/test_vector*.py
```

### 6.6 Memory / streaming

**Files:** `src/core/memory_tracker.rs`, `src/core/virtual_texture.rs`, `python/forge3d/mem.py`, `python/forge3d/memory.py`, `python/forge3d/streaming.py`

**Key invariants:**
- 512 MiB host-visible budget is enforced; see `docs/memory_budget.rst`.
- Staging ring allocator requires `enable-staging-rings` feature.
- `memory_metrics()` and `budget_remaining()` must reflect actual GPU allocations.

**Validation:**
```bash
pytest -q tests/test_memory*.py
```

---

## 7. Refactor checklists

### Bindings & API sync
- When updating renderer/config params: change Rust structs/enums, Python mirrors, default values, and any serialization/deserialization.
- When touching WGSL layouts: update Rust bind group layouts, descriptor set creation, and any Python-side size/shape assumptions.

### Memory & performance
- Reuse GPU resources where possible; avoid per-frame allocations.
- Confirm texture formats and sizes respect the memory budget; prefer reuse over duplication.
- Avoid O(n) per-frame work in hot paths; consider precomputation or caching if safe.

### Testing discipline
- Add regression tests in Python for changed behavior.
- Use deterministic seeds/fixtures where possible.
- Keep tests fast and isolated; avoid shared mutable fixtures.

### Naming and structure
- Make intent obvious through names.
- Group related code; maintain vertical whitespace for readability.
- Extract polymorphism/strategies when type-conditionals spread; avoid premature abstractions.

### WGSL shader changes
- [ ] Verify `@group` / `@binding` indices match Rust bind group layouts.
- [ ] Check uniform struct alignment (16-byte for mat4, 4-byte for f32, etc.).
- [ ] Confirm sampler/texture pairs are consistent.
- [ ] Run domain Python tests to validate output.

### PyO3 binding changes
- [ ] Update `#[pyclass]` / `#[pymethods]` in `src/lib.rs` or relevant module.
- [ ] Mirror changes in `python/forge3d/__init__.py` imports.
- [ ] Add/adjust Python tests for new behavior.
- [ ] Rebuild with `maturin develop` and re-test.

---

## 8. Risk assessment framework

| Risk level | Criteria | Mitigation |
|------------|----------|------------|
| **Low** | Single file, no API change, no shader change | Run targeted tests |
| **Medium** | Multiple files in one domain, minor API change | Run domain tests + smoke test |
| **High** | Cross-domain, API breaking, shader layout change | Full test suite + manual validation |
| **Critical** | Memory system, core bindings, feature flags | Full suite + golden image comparison |

Before merging high/critical changes:
1. Run `pytest -q tests/` (full suite).
2. Verify `cargo build --all-features` succeeds.
3. Spot-check an example (e.g., `python examples/terrain_demo.py`).

---

## 9. Command palette

### Rust
```bash
# Check compilation (fast)
cargo check -q --all-features

# Lint
cargo fmt -- --check
cargo clippy --all-features

# Build release
cargo build --release --all-features

# Run Rust tests (if any)
cargo test --all-features
```

### Python
```bash
# Syntax check
python -m compileall -q python/

# Import smoke test
python -c "import forge3d; print(forge3d.__version__)"

# Run all tests
pytest -q tests/

# Run domain tests
pytest -q tests/test_terrain*.py
pytest -q tests/test_p5*.py
pytest -q tests/test_b*.py

# Run with verbose output
pytest -v tests/test_api.py
```

### Git hygiene
```bash
# Pre-flight check
git status --porcelain
git diff --stat

# Verify no unintended changes
git diff HEAD -- .
```

---

## 10. Test categories

| Category | Pattern | Purpose |
|----------|---------|---------|
| **Smoke** | `tests/smoke_test.py`, `tests/test_api.py` | Minimal API contracts |
| **Terrain** | `tests/test_t*.py`, `tests/test_terrain*.py` | Terrain rendering |
| **Path tracing** | `tests/test_path_tracing*.py`, `tests/test_raytrace*.py` | Path tracer correctness |
| **Screen-space** | `tests/test_p5*.py` | AO/SSGI/SSR (P5 workstream) |
| **Lighting/PBR** | `tests/test_b*.py`, `tests/test_lighting*.py` | PBR, shadows, IBL |
| **Vector** | `tests/test_vector*.py` | OIT, overlays, text |
| **Memory** | `tests/test_memory*.py` | Budget, streaming |
| **Workstream** | `tests/test_workstream_*.py` | Feature workstreams |
| **Performance** | `tests/perf/*.py` | Benchmarks (optional) |
| **Golden** | `tests/golden/`, `golden_images.rs` | Pixel-perfect regression |

When adding tests:
- Place in the appropriate category file or create a new `test_<domain>_<feature>.py`.
- Use deterministic seeds/fixtures.
- Prefer fast unit tests over slow integration tests.

---

## 11. Change log template (for PRs/commits)

```markdown
## Summary
<!-- What changed and why -->

## Behavior
<!-- What users/tests observe; note any intentional diffs -->

## Risk/Surface
<!-- Rust modules, Python APIs, WGSL shaders touched -->

## Validation
<!-- Commands/tests/examples run and results -->

## Follow-ups
<!-- Any deferred cleanups (with issue/plan) -->
```

---

## 12. Definition of done

A refactor is complete when:

- [ ] All impacted sync pairs (Rust ↔ Python ↔ WGSL) are updated consistently.
- [ ] `cargo check --all-features` passes.
- [ ] `cargo fmt -- --check` passes.
- [ ] `python -m compileall -q python/` passes.
- [ ] Domain-specific Python tests pass.
- [ ] No unintended file changes (`git status --porcelain`).
- [ ] Docs updated if behavior/API changed.
- [ ] Change log entry written (for significant changes).

---

## 13. Final reminder

Code is the design. Keep it expressive, small, and test-backed.

Respect forge3d's layering:
```
Python facade → PyO3 bindings → Rust core → WGSL shaders
```

Honor the 512 MiB memory budget and GPU feature flags on every change.