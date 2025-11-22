# forge3d – Agent Orientation Guide

This document is for **AI coding agents** working inside the `forge3d` repository. It gives you a fast, *code-grounded* overview so you can navigate, modify, and debug the project effectively without breaking tests or architecture assumptions.

If you only remember a few things, remember:

- **Rust crate first, Python package second** – Rust `src/` is the rendering engine, Python `python/forge3d/` is a high-level, well-validated facade.
- **Tests and docs define behavior** – Always consult `tests/` and `docs/` before changing semantics or signatures.
- **Memory, GPU features, and QA are strict** – Don’t ignore the memory budget, feature flags, or acceptance tests.

---

## 1. High-level Architecture

### 1.1 Core idea

forge3d is a **Rust + wgpu/WebGPU renderer** with **PyO3 bindings** and a rich **Python API**. It targets:

- **Headless deterministic rendering** (PNG ↔ NumPy)
- **Terrain rendering** (DEM/heightmaps, PBR+POM, colormaps)
- **Path tracing** (GPU-oriented design with CPU fallbacks)
- **Screen-space GI & postfx** (AO, SSGI, SSR, bloom, tonemap)
- **Vector graphics & overlays** (OIT points/lines/polygons, text)

Execution paths:

- **Rust core** (`src/`) implements GPU pipelines, memory systems, and low-level rendering.
- **PyO3 extension** (`forge3d._forge3d`) exposes selected Rust types to Python.
- **Python package** (`python/forge3d/`) layers high-level APIs, validation, fallbacks, and integrations.
- **Tests** (`tests/`) are exhaustive and define many invariants (quality, performance envelopes, API contracts).

### 1.2 Major domains

- **Terrain & raster** – `src/terrain*`, `src/terrain_renderer.rs`, `python/forge3d/terrain_params.py`, examples.
- **Path tracing** – `src/path_tracing/`, `python/forge3d/path_tracing.py`, `python/forge3d/render.py` (raytrace mesh API).
- **Lighting / PBR / shadows / IBL** – `src/lighting`, `src/core/*` (clouds, shadows, ibl, dof, etc.), `python/forge3d/pbr.py`, `python/forge3d/shadows.py`, `python/forge3d/lighting.py`.
- **Screen-space effects (P5)** – `src/core/screen_space_effects.rs`, `src/p5/*`, `src/passes/*`, `src/shaders/ssao.wgsl` and GI/SSR-related shaders, Python helpers in `python/forge3d/screen_space_gi.py`.
- **Vector & overlays** – `src/vector/*`, `src/core/overlays.rs`, `src/core/text_overlay.rs`, `python/forge3d/vector.py`.
- **Memory & streaming** – `src/core/memory_tracker.rs`, `src/core/virtual_texture.rs`, `python/forge3d/mem.py`, `python/forge3d/memory.py`, `python/forge3d/streaming.py`, docs `docs/memory_budget.rst`.

---

## 2. Directory Map (Agent-centric)

### 2.1 Root

- `Cargo.toml` – Rust crate definition, **features** crucial for behavior (`enable-pbr`, `enable-ibl`, `enable-renderer-config`, `enable-staging-rings`, `weighted-oit`, etc.).
- `pyproject.toml` – Python packaging via `maturin`, ABI3 config, optional-deps groups.
- `README.md` – Short intro and quickstart; good sanity check for triangle/terrain.
- `prompt.md` – A high-level **task prompt** used for some workstreams (e.g., P5.4 GI composition). Treat it as a spec, not as code.
- `AGENTS.md` – This file.

### 2.2 Rust core – `src/`

- `src/lib.rs`
  - Crate root, PyO3 bindings, and module exports.
  - Re-exports many **core types** for Python (e.g. `RendererConfig`, `RendererGiMode`, `IBLRenderer`, `CloudRenderer`, etc.).
  - Defines PyO3 classes like `Frame`, `PyScreenSpaceGI` (Python GI manager), and utility functions for vector OIT.
- `src/core/mod.rs`
  - High-level **engine subsystems**: framegraph, GPU timing, postfx, bloom, memory/virtual textures, tonemap, matrix stack, scene graph, async compute, envmap/ibl, PBR material, shadows, reflections, clouds, ground plane, water surface, soft light radius, text, render bundles, screen-space effects (`gbuffer`, `screen_space_effects`).
- `src/terrain_renderer.rs`
  - `#[pyclass] TerrainRenderer` – PBR+POM terrain pipeline.
  - Bind group layouts for heightmap, materials, colormap, overlay, IBL; MSAA selection; light buffer integration.
- Other key modules:
  - `src/terrain/*` – Heightmap terrain pipeline implementation.
  - `src/lighting/*` – Lighting types and BRDF integration used by terrain+PBR.
  - `src/path_tracing/mod.rs` and submodules – GPU path tracing infrastructure (Megakernel/Wavefront).
  - `src/scene/mod.rs` – `Scene` PyO3 class (terrain scene, SSAO resources, toggles for reflections/DOF/clouds/etc.).
  - `src/render/mod.rs` + `src/render/params.rs` – Renderer configuration types used on both Rust + Python side.
  - `src/viewer/mod.rs` – Interactive viewer loop and integration with `ScreenSpaceEffectsManager`.
  - `src/passes/*`, `src/p5/*`, `src/shaders/*` – GI passes, SSR/SSGI/AO/tonemap, etc.

### 2.3 Python package – `python/forge3d/`

- `__init__.py`
  - **Public Python entrypoint**, layered:
    - Top section: imports `_native`, `_gpu`, memory facade `mem`, colormaps, terrain params, presets.
    - Re-exports native types when the extension module exists (`Scene`, `TerrainRenderer`, `IBL`, lighting/atmosphere, etc.).
    - Public helpers: memory metrics (`memory_metrics`, `budget_remaining`, `utilization_ratio`, `override_memory_limit`), GPU helpers (`enumerate_adapters`, `device_probe`, `has_gpu`, `get_device`), vector OIT demo wrappers, `composite_rgba_over`, matrix-stack QA helper (`c9_push_pop_roundtrip`).
  - Second section: **high-level rendering facade**:
    - Imports `RendererConfig` and config helpers.
    - Imports `PathTracer`, `make_camera`, rendering helpers from `.render`.
    - Imports `PbrMaterial`, `textures`, `geometry`, `io`, SDF wrappers, offscreen helpers, IPython display, frame dumper.
    - Defines **fallback `Renderer` class** used in many tests: triangle rendering, config/preset plumbing, lighting/shading config caching.
    - Exposes `render_triangle_rgba`, `render_triangle_png`, `numpy_to_png`, `png_to_numpy`, DEM stats/normalize, `open_viewer`, sampler utilities.
- `config.py`
  - **RendererConfig** tree for lighting/shading/shadows/GI/atmosphere and normalization of BRDF/techniques/GI modes.
  - Mirrors and validates Rust `src/render/params.rs`. Changes here must stay in sync with Rust side.
- `render.py`
  - High-level **rayshader-like APIs**:
    - `render_raytrace_mesh` – ingest mesh (OBJ or numpy), build BVH, attempt GPU path tracing via native `_pt_render_gpu_mesh`, fallback to CPU `PathTracer`.
    - DEM / vector ingestion helpers (rasterio, geopandas, shapely), palette resolution, camera autoframing, AOV export.
- `path_tracing.py`
  - Deterministic **CPU fallback path tracer** and AOV generator.
  - Used heavily in tests for conformance, firefly clamp behavior, AOV shapes/types.
- `mem.py` / `_memory.py`
  - Python facade over native memory tracker, exposing `MEMORY_LIMIT_BYTES`, `memory_metrics`, and budget helpers.
- `_gpu.py` / `_native.py`
  - GPU adapter detection, device probe, and fallback `MockDevice` when native not available.
- Other important submodules: `pbr.py`, `shadows.py`, `lighting.py`, `screen_space_gi.py`, `vector.py`, `postfx.py`, `streaming.py`, `terrain_params.py`, `memory.py`, `tiles/`, etc.

### 2.4 Tests – `tests/`

- Thousands of unit/integration tests across Rust and Python.
- Key patterns:
  - `test_api.py`, `smoke_test.py` – minimal API contracts for `Renderer` and triangle rendering.
  - Workstream-specific suites: `test_b*` (lighting/postfx), `test_p*` (PBR, GI, P5), `test_t*` (terrain), `test_f*` (geometry/mesh ops), `test_m*` (media/sky), `test_workstream_*`.
  - Rust tests under `tests/*.rs` and golden image harnesses (`golden_images.rs`, `scripts/generate_golden_images.py`).

### 2.5 Docs – `docs/`

- `index.rst` – documentation index, highlights **Core**, **Advanced Features**, **Integrations**, **Examples**, **Troubleshooting**.
- `quickstart.rst` – minimal Python usage path (triangle, terrain, vector graphics, GPU detection).
- `api_reference.rst` – Sphinx API docs; good map of *intended* top-level Python API.
- `memory_budget.rst` – authoritative source on **512 MiB host-visible memory budget** and patterns.
- `docs/api/*.md` / `docs/user/*.md` – deeper feature-specific docs.

### 2.6 Examples – `examples/`

- Python examples – galleries, terrain demos, SSGI demo, raytrace demos.
- Rust examples – interactive viewer, P5 SSR/SSGI tools, GI ablation harnesses.
- Useful for:
  - Sanity-checking rendering after changes.
  - Understanding how Py and Rust pieces are expected to compose.

---

## 3. Main Workflows & Data Flow

### 3.1 Basic Python triangle

1. User imports `forge3d`:
   - `python/forge3d/__init__.py` initializes shims and re-exports native objects if available.
2. `Renderer(width, height)` creates Python fallback renderer (`RendererConfig` + state cached in Python).
3. `render_triangle_rgba()` synthesizes an RGBA gradient triangle image entirely in Python; used for A1.4 acceptance tests.
4. `numpy_to_png` writes the PNG (Pillow or raw bytes fallback).

**Impact for agents:**
- Don’t break the fallback `Renderer` semantics: tests assert shape, dtype, non-empty output.

### 3.2 Terrain rendering

Typical path:

- Python side:
  - User uses higher-level API (`examples/terrain_demo.py` or `Scene` + `TerrainRenderer` in Python).
  - Terrain params built in Python (`terrain_params.py`), serialized to native configs.
- Rust side:
  - `TerrainRenderer` (PyO3 class) orchestrates heightmap upload, texture layout, PBR+POM shader, IBL. See `src/terrain_renderer.rs` + terrain shaders.

**Agent note:** when touching terrain:

- Keep Rust `TerrainRenderer` bind group layouts in sync with WGSL shader resource bindings.
- Ensure `terrain_render_params` and Python `terrain_params.py` stay aligned.

### 3.3 Path tracing and raytrace mesh

- Python `render_raytrace_mesh`:
  - Loads mesh (`forge3d.io.load_obj` or numpy/mesh dict inputs).
  - Validates via `forge3d.mesh` helpers and builds BVH.
  - Attempts GPU rendering via native `_pt_render_gpu_mesh`; falls back to CPU `PathTracer.render_rgba`.
  - Optionally writes AOVs, returns final RGBA + metadata.

**Agent note:**

- Maintain deterministic output where tests expect it (path_tracing CPU fallback is synthetic but stable under seed).
- GPU path is opportunistic; tests are usually written to tolerate CPU-only environments.

### 3.4 Screen-space effects (P5): AO / SSGI / SSR

- Manager: `core::screen_space_effects::ScreenSpaceEffectsManager` (Rust).
- Python binding: `PyScreenSpaceGI` in `src/lib.rs` with methods `enable_ssao`, `enable_ssgi`, `enable_ssr`, `disable`, `resize`, `execute`.
- AO path example: `Scene` holds `SsaoResources`, creates compute pipelines from `shaders/ssao.wgsl`, dispatches SSAO + composite into color buffer.

For more advanced GI work (e.g. P5.4 described in `prompt.md`):

- GI composition logic should live in GI-specific WGSL (e.g. `shaders/gi/composite.wgsl`).
- Orchestration is in `src/passes/gi.rs` and integrated into viewer/examples.
- Tests enforce energy and component-isolation constraints via P5-specific suites in `tests/`.

**Agent note:** when editing GI:

- Don’t smear AO/SSGI/SSR semantics across files; keep composition in designated shader and wiring in `passes/gi.rs`.
- Always re-read P5 prompt and the relevant tests (e.g. `tests/test_p5_screen_space_effects.py`, `tests/test_p53_ssr_status.py`) before structural changes.

### 3.5 Interactive viewer

- Python `open_viewer` delegates to native `open_viewer` in `_forge3d`.
- Rust `viewer` module sets up winit loop, GPU device, Scene/GBuffer, and integrates screen-space effects and overlays.

**Agent note:**
- Viewer code is sensitive to event loop, device lifetime, and pipeline ordering; avoid large refactors unless guided by a clear spec and backed by tests/examples.

---

## 4. Build, Test, and CI Expectations

### 4.1 Local builds

- **Rust only**:
  - `cargo check --workspace --all-features`
  - `cargo test --workspace --all-features -- --test-threads=1`
- **Python extension via maturin** (from repo root):
  - `pip install -U maturin`
  - `maturin develop --release`

### 4.2 Python tests

- Install built wheel or `maturin develop` first.
- Run Python tests:
  - `pytest tests/ -v --tb=short`

### 4.3 CI snapshot

From `.github/workflows/ci.yml`:

- Rust: `cargo check`, `cargo test --workspace --all-features -- --test-threads=1`, `cargo clippy` (warnings as errors, but allowed to fail in CI).
- Wheels: `maturin build` on Windows/Linux/macOS.
- Python tests on matrix of OS/Python versions.
- Golden images, shader param tests, example sanity runs.

**Agent note:**

- Keep Rust code generally **Clippy-clean** (warnings matter, even if CI allows some slack).
- Never introduce Python dependencies not reflected in `pyproject.toml` optional groups.

---

## 5. Design & API Conventions

### 5.1 Python API policy

See `python/forge3d/api_policy.md`:

- **Core module** (`import forge3d as f3d`):
  - Only exports stable, often-used symbols: `Renderer`, `Scene`, utility functions (`numpy_to_png`, `png_to_numpy`, DEM helpers, GPU helpers, vector helpers).
- **Specialized functionality** lives in submodules (`forge3d.pbr`, `forge3d.shadows`, `forge3d.path_tracing`, etc.).
- **Stability levels**:
  - Stable: exported in `__all__`, tests + docs, semver guarantees.
  - Experimental: submodules that may change but are documented.
  - Internal: not exported; no stability guarantees.

**Agent rule:**
- Don’t add new public top-level names casually. If you need a new user-facing feature, consider submodule placement and update API docs/tests accordingly.

### 5.2 Memory budget

From `docs/memory_budget.rst` and `python/forge3d/mem.py`:

- Default **host-visible budget**: **512 MiB**.
- Memory tracker distinguishes host-visible vs GPU-only allocations; only host-visible counts against budget.
- Exposed metrics: `memory_metrics()`, `budget_remaining()`, `utilization_ratio()`, `override_memory_limit()`.

**Agent rule:**

- Any new buffers/textures that are host-visible must be accounted for in memory tracking.
- Prefer reusing buffers / ring buffers where possible.

### 5.3 Feature detection and fallbacks

- Many GPU features are **optional** (e.g. weighted OIT, shadows, some PBR/IBL modes).
- Python and Rust sides have **feature-detection utilities**:
  - `forge3d.has_gpu()`, `forge3d.enumerate_adapters()`, `forge3d.device_probe()`.
  - Checks in shaders and Rust for optional extensions.

**Agent rule:**

- Never assume GPU or advanced features are always present; preserve or extend the existing detection/guard patterns.

---

## 6. Debugging Strategy for Agents

When debugging or modifying behavior, prefer this order:

1. **Find the contract**:
   - Look for tests: `grep` test name or feature in `tests/`.
   - Read relevant docs in `docs/`.
   - Inspect Python facade (often easier than diving straight into Rust).
2. **Locate the Rust core**:
   - Use `src/lib.rs` exports to locate types.
   - For PBR/lighting: `src/core/` and `src/lighting/`.
   - For GI/SSR/SSGI: `src/core/screen_space_effects.rs`, `src/passes/*`, `src/shaders/*`.
3. **Trace data flow**:
   - For terrain: `Scene` → `TerrainRenderer` → WGSL.
   - For path tracing: Python `render_raytrace_mesh` → native GPU path / CPU `PathTracer`.
   - For GI: GBuffer → AO/SSGI/SSR intermediates → final composite.
4. **Check memory & GPU environment**:
   - Use `forge3d.memory_metrics()` and `forge3d.has_gpu()` in tests or debugging snippets.
5. **Prefer small, localized changes**:
   - Modify a single module/shader at a time.
   - If new data is needed by shaders, propagate via minimal new fields and keep struct layout compatible.

Common failure modes:

- **Shape/dtype mismatches** between Python and Rust (e.g. NumPy arrays not C-contiguous, wrong dtype). Many tests exist explicitly to catch this.
- **Feature gate mismatches** – forgetting to enable a Cargo feature needed by a code path, or assuming it is always enabled.
- **Breaking energy/quality invariants** – especially around PBR, GI, and tonemapping.
- **Exceeding memory budget** – large textures or readback buffers without budget checks.

---

## 7. How to Safely Extend forge3d (Agent Checklist)

When you implement a new feature or change behavior:

1. **Identify scope**
   - Is this a Python-only helper? A Rust pipeline change? A shader tweak? A new example?
2. **Align with existing patterns**
   - Follow existing naming, module placement, and config patterns (e.g. `RendererConfig` for renderer options, `TerrainRenderParams` for terrain).
3. **Wire both sides if needed**
   - For new GPU features:
     - Rust struct + device/pipeline code.
     - WGSL shader changes.
     - PyO3 bindings in `src/lib.rs` or relevant module.
     - Python wrappers / validation.
4. **Update tests and docs (if public API)**
   - Add or extend tests under `tests/`.
   - If user-facing, update `docs/` and/or `api_policy.md`.
5. **Respect CI constraints**
   - Keep `cargo test --workspace --all-features` and `pytest tests/` passing.
   - Don’t add heavy new dependencies unless absolutely necessary.

---

## 8. Quick Pointers by Task Type

- **You’re asked to change GI / SSR / SSGI**
  - Start from `prompt.md` (if P5-related), `src/core/screen_space_effects.rs`, `src/passes/*`, and shaders under `src/shaders/` (especially `ssao.wgsl`, `gi/*`, `ssr/*`).
  - Mirror any new GI controls into Python config only if explicitly required.
- **You’re asked to modify terrain appearance**
  - Look first at `src/terrain_renderer.rs`, terrain shaders, and `python/forge3d/terrain_params.py`.
- **You’re asked to adjust Python API behavior**
  - Check `python/forge3d/__init__.py`, `api_policy.md`, and tests like `test_api.py`, `test_renderer_config.py`, `test_presets.py`.
- **You’re asked to optimize memory or streaming**
  - Inspect `src/core/memory_tracker.rs`, `src/core/virtual_texture.rs`, and Python `mem.py`, `memory.py`, `streaming.py`.

---

This document is intentionally high-level but code-grounded. When in doubt, prefer:

- Reading tests and docs over making assumptions.
- Adding small, focused changes over broad refactors.
- Preserving GPU feature and memory constraints.

If a future task prompt (like `prompt.md`) conflicts with this file, treat the prompt + tests as the authoritative spec and use this guide only as orientation.
