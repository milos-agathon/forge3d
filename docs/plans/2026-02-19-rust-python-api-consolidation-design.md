# Forge3D Rust/Python API Consolidation Design

Date: 2026-02-19
Inputs: `docs/notes/audit_2026-02-17.md`, `docs/rust-python-exposure-deep-dive-2026-02-17.md`, session deep-dive audit (2026-02-19)

## 1. Problem Statement

The forge3d codebase has ~500 Rust source files across 40 modules, but only 33 files (8%) carry PyO3 annotations. The 2026-02-17 audit and today's deep-dive confirm four categories of debt:

1. **Exposure gaps:** Production-quality Rust features (labels, cloud shadows, planar reflections, terrain analysis) have zero Python API.
2. **API drift:** Python wrappers reference 9 native symbols that don't exist (`create_restir_di`, `mesh_generate_cube_tbn`, etc.).
3. **Dead/duplicate code:** Four Rust modules (SVG export, Mapbox style, 3D Tiles, bundle) are reimplemented in Python; the Rust code is unused.
4. **Incomplete implementations:** Bloom execute is a no-op, HDR draw is unwired, point cloud has no GPU path, COPC LAZ returns zeroed data.

## 2. Scope

This design covers three workstreams:

- **W1 — Wire existing Rust to Python** (exposure gaps + API drift)
- **W2 — Consolidate duplicates** (pick one implementation per feature, deprecate the other)
- **W3 — Complete partial implementations** (finish what's started)

Out of scope: new feature development, viewer IPC protocol changes, documentation site redesign.

## 3. Codebase State Summary

### 3.1 Rust Module Exposure Levels (from audit)

| Level | Count | Modules |
|-------|-------|---------|
| **L3** (full native API) | 7 | scene, terrain, geometry, lighting, camera, io, vector |
| **L2** (partial exposure) | 10 | core, render, geo, animation, picking, viewer, uv, colormap, import, converters |
| **L1** (gated/fallback-heavy) | 4 | mesh, offscreen, path_tracing, sdf, shadows |
| **L0** (no Python API) | 19 | accel, cli, external_image, formats, loaders, labels, p5, passes, pipeline, pointcloud, tiles3d, bundle, style, export, renderer, util, math, textures |

### 3.2 Python Wrapper Categories

| Category | Count | Examples |
|----------|-------|---------|
| Pure Python (no Rust) | 19 | tiles3d, map_plate, legend, scale_bar, export, style, bundle, denoise |
| Thick wrapper (validation + native) | 9 | terrain_params, lighting, mesh, geometry, io, path_tracing, crs, buildings, vector |
| Thin wrapper (re-export) | 4 | _native, animation, cog, terrain_pbr_pom |
| Orchestration | 4 | __init__, render, terrain_demo, viewer |

### 3.3 Confirmed API Drift (9 missing native symbols)

| Symbol | Referenced in | Status |
|--------|--------------|--------|
| `create_restir_di` | `lighting.py:221` | No Rust export |
| `restir_set_lights` | `lighting.py:291` | No Rust export |
| `restir_clear_lights` | `lighting.py:299` | No Rust export |
| `restir_sample_light` | `lighting.py:314` | No Rust export |
| `restir_render_frame` | `lighting.py:367` | No Rust export |
| `restir_get_statistics` | `lighting.py:420` | No Rust export |
| `mesh_generate_cube_tbn` | `mesh.py:56` | No Rust export |
| `mesh_generate_plane_tbn` | `mesh.py:136` | No Rust export |
| `CogDataset` | `cog.py:31` | Feature-gated (`cog_streaming`) |

### 3.4 Unregistered PyO3 Classes

| Class | Declared at | Status |
|-------|------------|--------|
| `Frame` | `src/lib.rs:71` | `#[pyclass]` but no `m.add_class` |
| `SdfPrimitive` | `src/sdf/py.rs:10` | `#[pyclass]` but no `m.add_class` |
| `SdfScene` | `src/sdf/py.rs:108` | `#[pyclass]` but no `m.add_class` |
| `SdfSceneBuilder` | `src/sdf/py.rs:155` | `#[pyclass]` but no `m.add_class` |

### 3.5 Duplicate Implementations (Rust vs Python)

| Feature | Rust (LOC) | Python (LOC) | Currently Used |
|---------|-----------|-------------|---------------|
| SVG export | `src/export/` (994) | `python/forge3d/export.py` (626) | Python |
| Mapbox style | `src/style/` (1,918) | `style.py` + `style_expressions.py` (1,708) | Python |
| 3D Tiles | `src/tiles3d/` (1,940) | `python/forge3d/tiles3d.py` (404) | Python |
| Bundle format | `src/bundle/` (182) | `python/forge3d/bundle.py` (360) | Python |
| SDF raymarcher | `src/sdf/` (1,965) | `python/forge3d/sdf.py` (555) | Python (default) |

### 3.6 Incomplete Rust Implementations

| Module | File | Issue | Severity |
|--------|------|-------|----------|
| Bloom | `src/core/bloom.rs` | `execute()` is a no-op | Medium |
| HDR | `src/core/hdr.rs` | Render pass opened but no draw call | Medium |
| Indirect draw | `src/vector/indirect.rs` | Encoder never submitted; draw ignores count | Medium |
| GPU LBVH refit | `src/accel/lbvh_gpu.rs` | Internal node refit dispatches 1 workgroup only | Low |
| TLAS instancing | `src/accel/instancing.rs` | Pure stub, no GPU buffers | Low |
| SDF hybrid GPU | `src/sdf/hybrid.rs` | GPU upload fills buffers with empty data | Low |
| Point cloud GPU | `src/pointcloud/renderer.rs` | CPU-only, no wgpu buffer/pipeline | Medium |
| COPC LAZ | `src/pointcloud/copc.rs` | Returns zeroed placeholder | High |
| LTC LUT | `src/core/ltc_area_lights.rs` | Polynomial approximation, not BRDF-fitted | Low |
| SSGI/SSR wire-up | `src/passes/ssgi.rs`, `ssr.rs` | Py settings exist but never forwarded to GPU | Medium |

## 4. Design: W1 — Wire Existing Rust to Python

### 4.1 High-Priority Exposure (new PyO3 bindings)

#### 4.1.1 Terrain Analysis (`src/terrain/analysis.rs`)

Add `#[pyfunction]` wrappers:

```rust
// src/terrain/analysis.rs or new py_bindings section
#[pyfunction]
fn compute_slope_aspect_py(heights: PyReadonlyArray2<f32>, cell_size: f32) -> (Py<PyArray2<f32>>, Py<PyArray2<f32>>)

#[pyfunction]
fn extract_contours_py(heights: PyReadonlyArray2<f32>, levels: Vec<f32>, cell_size: f32) -> Vec<Vec<(f32, f32)>>
```

Register in `src/lib.rs` init block. These are standard GIS functions that users expect to call directly.

#### 4.1.2 Labels System (`src/labels/`)

Create `src/labels/py_bindings.rs` exposing:

```rust
#[pyclass]
struct PyLabelManager { inner: LabelManager }

#[pyclass]
struct PyLabelLayer { inner: LabelLayerConfig }

#[pyclass]
struct PyLabelStyle { /* fields with get/set */ }
```

This enables headless label rendering without the interactive viewer.

#### 4.1.3 Cloud Shadows (`src/core/cloud_shadows/`)

Add fields to existing `PyVolumetricSettings` in `src/lighting/py_bindings.rs`:

```rust
// Extend PyVolumetricSettings
pub cloud_shadows_enabled: bool,
pub cloud_shadows_quality: String,  // "low"|"medium"|"high"|"ultra"
pub cloud_shadow_density: f32,
pub cloud_shadow_speed: f32,
```

#### 4.1.4 Planar Reflections (`src/core/reflections.rs`)

Add `PyReflectionSettings` to `src/lighting/py_bindings.rs`:

```rust
#[pyclass]
struct PyReflectionSettings {
    pub enabled: bool,
    pub quality: String,  // "low"|"medium"|"high"|"ultra"
    pub plane_y: f32,
    pub fresnel_power: f32,
}
```

### 4.2 Fix API Drift (resolve 9 missing symbols)

#### 4.2.1 ReSTIR symbols (6 missing)

Two options:
- **Option A (recommended):** Remove dead Python references, document that ReSTIR is internal to the wavefront path tracer and not independently callable. The Python `RestirDI` class stays as a configuration holder that feeds into the path tracer.
- **Option B:** Implement 6 `#[pyfunction]` wrappers delegating to `src/path_tracing/restir/system.rs`.

Recommendation: **Option A.** ReSTIR is a path tracer sub-component, not an independent API. Clean the dead references.

#### 4.2.2 Mesh TBN symbols (2 missing)

Two options:
- **Option A (recommended):** Add `#[pyfunction]` for `mesh_generate_cube_tbn` and `mesh_generate_plane_tbn` wrapping the existing `src/mesh/tbn.rs` functions. These are simple geometry utilities with clear value.
- **Option B:** Remove Python references and rely on pure-Python TBN fallback.

Recommendation: **Option A.** Small effort, genuine utility.

#### 4.2.3 CogDataset (feature-gated)

No code change needed. Document that `cog_streaming` feature must be enabled in `pyproject.toml` wheel build features. Add a clear error message in `python/forge3d/cog.py` when the feature is absent.

### 4.3 Register Orphaned PyO3 Classes

Add to `src/lib.rs` init block:

```rust
m.add_class::<Frame>()?;
m.add_class::<crate::sdf::py::PySdfPrimitive>()?;
m.add_class::<crate::sdf::py::PySdfScene>()?;
m.add_class::<crate::sdf::py::PySdfSceneBuilder>()?;
```

### 4.4 Fix Method Placement Mismatch

Add module-level forwarding functions in `src/lib.rs`:

```rust
#[pyfunction]
fn render_rgba(py: Python, scene: &Scene, ...) -> PyResult<PyObject> {
    scene.render_rgba(py, ...)
}

#[pyfunction]
fn set_msaa_samples(scene: &Scene, samples: u32) -> PyResult<()> {
    scene.set_msaa_samples(samples)
}
```

This satisfies wrapper expectations at `python/forge3d/helpers/offscreen.py:41` and `python/forge3d/viewer.py:42`.

## 5. Design: W2 — Consolidate Duplicates

### 5.1 Decision Matrix

| Feature | Keep | Deprecate | Rationale |
|---------|------|-----------|-----------|
| SVG export | **Python** | Rust `src/export/` | Python SVG is mature, tested, no GPU needed |
| Mapbox style | **Python** | Rust `src/style/` | Python style pipeline is complete with expression evaluator |
| 3D Tiles | **Python** | Rust `src/tiles3d/` | Python tileset parser is simpler and feature-complete for current needs |
| Bundle | **Python** | Rust `src/bundle/` | Python bundle with SHA256 checksums is sufficient |
| SDF raymarcher | **Both** | Neither | Keep Python default + native opt-in (current design is correct) |

### 5.2 Deprecation Process

For each Rust module being deprecated:
1. Add `#[deprecated(note = "Use Python forge3d.{module} instead")]` to public items
2. Add `// DEPRECATED: Python implementation is canonical` header comment
3. Do NOT delete code yet — it may serve as reference for future GPU acceleration
4. Remove from `pub mod` in `src/lib.rs` only after one release cycle

## 6. Design: W3 — Complete Partial Implementations

### 6.1 Priority Completions

#### 6.1.1 Bloom (`src/core/bloom.rs`) — Wire `execute()`

The pipelines (`bloom_brightpass.wgsl`, `bloom_blur_h.wgsl`, `bloom_blur_v.wgsl`) are already created. The `execute()` method needs:
1. Create intermediate textures from the post-fx resource pool
2. Dispatch brightpass compute → horizontal blur → vertical blur
3. Return output texture view

Estimated: ~50 LOC change in `execute()`.

#### 6.1.2 SSGI/SSR Settings Wire-up

`PySSGISettings` and `PySSRSettings` exist but their values never reach `SsgiRenderer::update_settings()` / `SsrRenderer::update_settings()`. Wire them through the `TerrainRenderer` or `Scene` render path.

#### 6.1.3 Point Cloud GPU Path

`src/pointcloud/renderer.rs` produces `Vec<f32>` but creates no wgpu resources. Add:
1. Vertex buffer creation from point data
2. Simple point render pipeline (position + color, `PrimitiveTopology::PointList`)
3. Draw call in viewer render loop

#### 6.1.4 COPC LAZ Decompression

Integrate `laz-rs` crate for LAZ chunk decompression in `src/pointcloud/copc.rs:decode_laz_chunk()`. This is the critical missing piece blocking real COPC file support.

### 6.2 Deferred (not in this plan)

- TLAS instancing GPU buffers — low priority, CPU instancing works
- SDF hybrid GPU upload — Python SDF is the primary path
- LTC LUT proper BRDF fitting — simplified version is adequate
- GPU LBVH refit workgroup count — edge case for very large scenes
- Indirect draw encoder submission — vector batching works without GPU indirect

## 7. Deliverables Summary

| # | Workstream | Item | Files Touched | Est. LOC |
|---|-----------|------|---------------|----------|
| 1 | W1 | Terrain analysis PyO3 wrappers | `src/terrain/analysis.rs`, `src/lib.rs` | ~60 |
| 2 | W1 | Labels PyO3 bindings | `src/labels/py_bindings.rs` (new), `src/lib.rs` | ~200 |
| 3 | W1 | Cloud shadow settings | `src/lighting/py_bindings.rs` | ~40 |
| 4 | W1 | Planar reflection settings | `src/lighting/py_bindings.rs` | ~60 |
| 5 | W1 | Clean ReSTIR dead refs | `python/forge3d/lighting.py` | ~-30 |
| 6 | W1 | Add mesh TBN exports | `src/lib.rs` | ~30 |
| 7 | W1 | Register orphaned pyclasses | `src/lib.rs` | ~4 |
| 8 | W1 | Module-level forwarding fns | `src/lib.rs` | ~20 |
| 9 | W2 | Deprecate 4 Rust modules | `src/export/`, `src/style/`, `src/tiles3d/`, `src/bundle/` | ~20 (annotations) |
| 10 | W3 | Wire bloom execute | `src/core/bloom.rs` | ~50 |
| 11 | W3 | Wire SSGI/SSR settings | `src/core/` or `src/terrain/renderer.rs` | ~40 |
| 12 | W3 | Point cloud GPU path | `src/pointcloud/renderer.rs` | ~150 |
| 13 | W3 | COPC LAZ decompression | `src/pointcloud/copc.rs`, `Cargo.toml` | ~80 |

**Total estimated: ~720 LOC net change across 13 items.**

## 8. Acceptance Criteria

Each item must satisfy:

1. **Build:** `maturin develop --release` produces zero new warnings
2. **Tests:** All existing 916 tests pass; new items add at least one test each
3. **Runtime:** `python -c "import forge3d; print(len(dir(forge3d._forge3d)))"` reports increased symbol count
4. **Docs:** Updated docstrings for all new PyO3 functions/classes

### Per-item acceptance:

| # | Acceptance |
|---|-----------|
| 1 | `forge3d.compute_slope_aspect(heights, cell_size)` returns (slope, aspect) arrays |
| 2 | `forge3d.LabelManager` constructable from Python |
| 3 | `TerrainRenderParams` accepts `cloud_shadows_enabled` |
| 4 | `TerrainRenderParams` accepts `reflection_quality` |
| 5 | `python/forge3d/lighting.py` has no `_forge3d.create_restir_di` references |
| 6 | `forge3d._forge3d.mesh_generate_cube_tbn()` callable |
| 7 | `forge3d.Frame`, `forge3d.SdfPrimitive` importable |
| 8 | `forge3d._forge3d.render_rgba()` callable as module function |
| 9 | Deprecated Rust modules have `#[deprecated]` attribute |
| 10 | Bloom effect produces non-identity output in test render |
| 11 | `PySSGISettings.radius` change affects rendered pixel values |
| 12 | Point cloud renders visible points in viewer |
| 13 | COPC file loads with non-zero vertex positions |

## 9. Risks

| Risk | Mitigation |
|------|-----------|
| Labels PyO3 surface is large | Start with read-only config; defer label editing API |
| Bloom resource pool dependency | Use dedicated textures instead of shared pool |
| `laz-rs` crate compatibility | Verify it builds on all 3 target platforms before integrating |
| SSGI/SSR wire-up may affect render output | Add regression test comparing baseline renders before/after |

## 10. Implementation Order

Recommended sequence (dependencies noted):

1. Items 7, 8 (register orphans + forwarding) — unblocks wrapper consistency
2. Items 5, 6 (clean drift + add TBN) — fixes known mismatches
3. Item 1 (terrain analysis) — standalone, no dependencies
4. Items 3, 4 (cloud shadow + reflection settings) — extends existing lighting bindings
5. Item 2 (labels) — largest W1 item, can be parallelized
6. Item 9 (deprecation annotations) — W2, standalone
7. Items 10, 11 (bloom + SSGI/SSR) — W3, core rendering fixes
8. Items 12, 13 (point cloud GPU + COPC LAZ) — W3, can be parallelized
