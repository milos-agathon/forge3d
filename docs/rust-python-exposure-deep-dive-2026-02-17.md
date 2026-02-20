# Forge3D Rust Feature Implementation vs Python Frontend Exposure

Date: 2026-02-17  
Repository: `D:\forge3d`  
Scope: implementation-level audit of Rust feature surfaces and their exposure to Python frontend (`forge3d` + `forge3d._forge3d`)

## 1) Executive Summary

- Rust implementation breadth is large: `38` top-level Rust modules declared in `src/lib.rs` (`src/lib.rs:1089` to `src/lib.rs:1127`) plus inline modules `math`/`textures` (`src/lib.rs:1047`, `src/lib.rs:1114`).
- Python native exposure is concentrated in selected domains (scene, geometry, lighting, terrain, camera, io), not a 1:1 mirror of all Rust modules.
- Native binding registration is centralized in `_forge3d` init (`src/lib.rs:4431`) plus terrain sub-registrations:
  - clipmap: `src/terrain/clipmap/py_bindings.rs:193`
  - COG: `src/terrain/cog/py_bindings.rs:155` (feature-gated)
- Current runtime extension (`forge3d._forge3d`) exports `134` public symbols; only `32` are visible at `forge3d` package root.
- Feature gates materially affect exposure:
  - `weighted-oit`: available in current build (`True`)
  - `proj`: bound but unavailable in current build (`False`)
  - `enable-gpu-instancing`: bound/stubbed but unavailable in current build (`False`)
  - `cog_streaming`: not enabled in current build (`CogDataset` absent)
- There are explicit Python wrapper expectations for native symbols that are currently missing (ReSTIR functions, mesh TBN native functions, COG native class in this build).

---

## 2) Methodology

Audit combined static and runtime analysis:

1. Rust module inventory from `src/lib.rs`.
2. Binding surface extraction from `m.add_function(...)` / `m.add_class::<...>()`.
3. Feature flag inspection from `Cargo.toml` and wheel build features from `pyproject.toml`.
4. Python wrapper symbol usage scanning (`_forge3d.<symbol>`, `getattr(..., "symbol")`, `hasattr(...)`).
5. Runtime introspection of built extension:
   - symbol inventory (`dir(forge3d._forge3d)`)
   - class API depth (`Scene`, `TerrainRenderer`, `TerrainRenderParams`)
   - feature probes (`is_weighted_oit_available`, `proj_available`, `gpu_instancing_available_py`).

---

## 3) Build-Time Feature Gates and Exposure Impact

### 3.1 Cargo features

Defined in `Cargo.toml:14` onward:

- `extension-module` (`Cargo.toml:18`)
- `weighted-oit` (`Cargo.toml:22`)
- `enable-gpu-instancing` (`Cargo.toml:36`)
- `images` (`Cargo.toml:43`)
- `cog_streaming` (`Cargo.toml:45`)
- `proj` (`Cargo.toml:47`)

### 3.2 Wheel build feature set

Maturin wheel config enables:

- `["extension-module", "weighted-oit"]` at `pyproject.toml:94`

Implication:

- Python extension bindings are built.
- Weighted OIT path is built.
- `cog_streaming`, `proj`, `enable-gpu-instancing` require explicit opt-in builds.

### 3.3 Feature-gated binding points

- COG registration is gated:
  - `#[cfg(feature = "cog_streaming")]` at `src/lib.rs:4687`
  - call to register at `src/lib.rs:4688`
- GPU instancing advanced bindings gated:
  - `#[cfg(all(feature = "enable-gpu-instancing"))]` block at `src/lib.rs:4545`
  - gated function exports at `src/lib.rs:4548` and `src/lib.rs:4552`
- PROJ bindings exported unconditionally under extension-module:
  - registration calls at `src/lib.rs:4741` and `src/lib.rs:4742`
  - actual implementations in `src/geo/reproject.rs:159` and `src/geo/reproject.rs:167`

---

## 4) Binding Architecture and Public Surfaces

### 4.1 Native module (`forge3d._forge3d`)

- Init function: `src/lib.rs:4431`
- Binding registration count (source scan):
  - ~`105` `m.add_function(...)` entries
  - `34` `m.add_class::<...>()` entries including clipmap/cog registration units

### 4.2 Python package root (`forge3d`)

- Selective native re-export loop starts at `python/forge3d/__init__.py:52`.
- Root `__all__` is declared at `python/forge3d/__init__.py:346`.
- Only a subset of native symbols are surfaced at root (runtime measured: `32` native symbols available at root).

### 4.3 Current runtime API depth (built extension)

- Native public symbol count: `134`.
- Root public symbol count: `158`.
- Native symbols present at root: `32`.
- Native-only (not at root): `102`.
- Class API sizes:
  - `Scene`: `199` public attrs/methods
  - `TerrainRenderer`: `5`
  - `TerrainRenderParams`: `37`

Key method placement:

- `Scene.render_rgba`: `src/scene/mod.rs:1721`
- `Scene.set_msaa_samples`: `src/scene/mod.rs:2145`

---

## 5) Rust Module Inventory (Every Module) and Exposure Level

Legend:

- `L3`: directly exposed as stable native Python API (class/functions)
- `L2`: partially exposed or indirectly exposed
- `L1`: feature-gated / fallback-heavy / partially wired
- `L0`: no meaningful direct native exposure to Python

| Rust module | Declared at | Impl LOC | PyO3 markers (pyfn+pyclass+pymethods) | Exposure | Notes |
|---|---:|---:|---:|---|---|
| `accel` | `src/lib.rs:1089` | 2403 | 0 | L0 | Internal acceleration structures, no direct bindings. |
| `camera` | `src/lib.rs:1090` | 492 | 10 | L3 | Native camera math API exported (`camera_*` functions). |
| `cli` | `src/lib.rs:1091` | 1126 | 0 | L0 | CLI-only concerns. |
| `colormap` | `src/lib.rs:1092` | 351 | 2 | L2 | `Colormap1D` class exposed. |
| `converters` | `src/lib.rs:1093` | 107 | 1 | L2 | `converters_multipolygonz_to_obj_py` exported. |
| `core` | `src/lib.rs:1094` | 29919 | 4 | L2 | Exposes `Session`, `OverlayLayer`, `ScreenSpaceGI`; most core internals unbound. |
| `external_image` | `src/lib.rs:1095` | 383 | 0 | L0 | No direct native Python API. |
| `formats` | `src/lib.rs:1096` | 290 | 0 | L0 | No direct native Python API. |
| `geo` | `src/lib.rs:1097` | 195 | 0 | L2 | `proj_available` + `reproject_coords` exported; depends on `proj` feature availability. |
| `geometry` | `src/lib.rs:1098` | 3020 | 29 | L3 | Broad geometry function surface. |
| `import` | `src/lib.rs:1099` | 1427 | 2 | L2 | OSM/cityjson helpers exported as functions. |
| `io` | `src/lib.rs:1100` | 846 | 4 | L3 | OBJ/STL/GLTF import/export bindings. |
| `lighting` | `src/lib.rs:1101` | 4733 | 26 | L3 | Large class-based lighting settings + sun APIs. |
| `loaders` | `src/lib.rs:1102` | 512 | 0 | L0 | Internal. |
| `mesh` | `src/lib.rs:1103` | 610 | 0 | L1 | Python wrapper expects native TBN funcs not exported in current build. |
| `offscreen` | `src/lib.rs:1104` | 1517 | 0 | L1 | Helpers expect module-level native `render_rgba`; falls back to Python path. |
| `path_tracing` | `src/lib.rs:1105` | 7306 | 0 | L1 | Native hooks limited; frontend mostly Python orchestration. |
| `pipeline` | `src/lib.rs:1106` | 2437 | 0 | L0 | Internal pipeline graph/render plumbing. |
| `render` | `src/lib.rs:1107` | 3482 | 7 | L2 | Includes material set + instancing functions (some gated). |
| `renderer` | `src/lib.rs:1108` | 106 | 0 | L0 | Internal. |
| `scene` | `src/lib.rs:1109` | 5990 | 1 | L3 | `Scene` class is major native API surface. |
| `sdf` | `src/lib.rs:1110` | 1965 | 3 | L1 | `hybrid_render` exists; SDF pyclasses declared but not class-registered. |
| `shadows` | `src/lib.rs:1111` | 2010 | 0 | L1 | CSM controls exposed from `lib.rs`; shadows module internals mostly unbound. |
| `terrain` | `src/lib.rs:1112` | 17846 | 15 | L3 | Terrain renderer/params/spikes + clipmap (+ COG when enabled). |
| `uv` | `src/lib.rs:1113` | 95 | 2 | L2 | Planar/spherical unwrap functions exposed. |
| `labels` | `src/lib.rs:1115` | 3395 | 0 | L0 | No direct native API exposure. |
| `p5` | `src/lib.rs:1116` | 1409 | 0 | L0 | Internal workstream module. |
| `passes` | `src/lib.rs:1117` | 447 | 0 | L0 | Internal render passes. |
| `util` | `src/lib.rs:1118` | 735 | 0 | L0 | Internal helpers. |
| `vector` | `src/lib.rs:1119` | 7041 | 9 | L3 | Vector add/extrude/OIT/pick APIs exported. |
| `picking` | `src/lib.rs:1120` | 3412 | 0 | L2 | Picking result classes exposed via `lib.rs`, not direct per-module API. |
| `viewer` | `src/lib.rs:1121` | 26019 | 0 | L2 | Open-viewer entrypoints exposed; most viewer engine internals unbound. |
| `animation` | `src/lib.rs:1122` | 403 | 2 | L2 | Camera animation classes exported. |
| `tiles3d` | `src/lib.rs:1123` | 1940 | 0 | L0 | Native module exists, frontend largely Python-level implementation. |
| `pointcloud` | `src/lib.rs:1124` | 1067 | 0 | L0 | Native module exists, frontend largely Python-level implementation. |
| `bundle` | `src/lib.rs:1125` | 182 | 0 | L0 | Exposed in frontend as Python workflows, not native-bound. |
| `style` | `src/lib.rs:1126` | 1918 | 0 | L0 | Python style parser pipeline is pure Python-facing. |
| `export` | `src/lib.rs:1127` | 994 | 0 | L0 | Python export utilities dominate frontend API. |
| `math` (inline) | `src/lib.rs:1047` | inline | n/a | L0 | Inline helper module, not standalone native API. |
| `textures` (inline, empty) | `src/lib.rs:1114` | empty | n/a | L0 | Placeholder module. |

---

## 6) High-Fidelity Exposure Findings by Feature Area

### 6.1 Scene and rendering control

- `Scene` class is exported (`src/lib.rs:4703`) and is the dominant Python-native class.
- Includes:
  - `render_rgba` method: `src/scene/mod.rs:1721`
  - `set_msaa_samples` method: `src/scene/mod.rs:2145`
- Python helper code currently checks for module-level natives:
  - `render_rgba` in `python/forge3d/helpers/offscreen.py:41`
  - `set_msaa_samples` in `python/forge3d/viewer.py:42`
- Runtime confirms both symbols are class methods, not module-level functions:
  - `_forge3d.render_rgba`: absent
  - `_forge3d.Scene.render_rgba`: present
  - `_forge3d.set_msaa_samples`: absent
  - `_forge3d.Scene.set_msaa_samples`: present

Assessment: exposure is present but ergonomics mismatch exists between wrapper expectations and where APIs are mounted.

### 6.2 Terrain stack

- Core terrain classes exported:
  - `TerrainRenderParams`: `src/lib.rs:4699`
  - `TerrainRenderer`: `src/lib.rs:4700`
  - `TerrainSpike`: `src/lib.rs:4705`
- Clipmap bound through dedicated registration:
  - `register_clipmap_bindings`: called `src/lib.rs:4685`
  - class/function registration in `src/terrain/clipmap/py_bindings.rs:194` to `src/terrain/clipmap/py_bindings.rs:197`
- COG native class is feature-gated:
  - registration gating at `src/lib.rs:4687`
  - class definition in `src/terrain/cog/py_bindings.rs:11`
- Python wrapper gracefully falls back:
  - `python/forge3d/cog.py:30` checks for `_forge3d.CogDataset`

Assessment: terrain is one of the strongest native surfaces; COG exposure depends on build flags.

### 6.3 Lighting and shadows

- Lighting setting classes and sun ephemeris API are directly exported:
  - classes registered at `src/lib.rs:4707` to `src/lib.rs:4720`
  - functions `sun_position` / `sun_position_utc` at `src/lib.rs:4721` to `src/lib.rs:4722`
- CSM controls exported as module-level functions (`src/lib.rs:4447` to `src/lib.rs:4454`).
- Python wrapper expects additional optional native hooks:
  - `set_exposure_scale` via `getattr` in `python/forge3d/lighting.py:58`
  - ReSTIR function family in `python/forge3d/lighting.py:220`, `python/forge3d/lighting.py:291`, `python/forge3d/lighting.py:299`, `python/forge3d/lighting.py:314`, `python/forge3d/lighting.py:367`, `python/forge3d/lighting.py:420`
- Source scan found no Rust exports for ReSTIR symbols in `src/`.

Assessment: core lighting exposure is robust; advanced ReSTIR path is wrapper-defined but currently unbacked natively.

### 6.4 Geometry and instancing

- Geometry has one of the broadest function surfaces from `lib.rs` registration block (`src/lib.rs:4464` onward).
- Instancing exposure tiers:
  - always exported:
    - `geometry_instance_mesh_py`
    - `gpu_instancing_available_py`
    - `geometry_instance_mesh_gpu_stub_py`
    (`src/lib.rs:4533` to `src/lib.rs:4544`)
  - feature-gated exports:
    - `geometry_instance_mesh_gpu_py`
    - `geometry_instance_mesh_gpu_render_py`
    (`src/lib.rs:4545` to `src/lib.rs:4553`)
- Python wrapper checks dynamically for `geometry_instance_mesh_gpu_render_py` at `python/forge3d/geometry.py:336`.
- Runtime flag `gpu_instancing_available_py()` is currently `False`.

Assessment: API is intentionally dual-path (stub + gated real GPU), current build runs non-GPU-instancing path.

### 6.5 CRS / reprojection

- Rust binds:
  - `proj_available` (`src/geo/reproject.rs:159`)
  - `reproject_coords` (`src/geo/reproject.rs:167`)
- Both are registered in module init (`src/lib.rs:4741`, `src/lib.rs:4742`).
- Python CRS frontend imports these in `python/forge3d/crs.py:51` and `python/forge3d/crs.py:52`.
- Runtime: `proj_available()` is currently `False`, so frontend should use non-native paths when needed.

Assessment: exposure is correct; runtime capability depends on build feature/environment.

### 6.6 SDF

- Rust contains SDF Python-facing wrappers with `pyclass` attrs:
  - `SdfPrimitive` in `src/sdf/py.rs:10`
  - `SdfScene` in `src/sdf/py.rs:108`
  - `SdfSceneBuilder` in `src/sdf/py.rs:155`
- `hybrid_render` accepts `PySdfScene` extraction path in `src/lib.rs:2867`.
- However, `_forge3d` class registration block has no `m.add_class::<...Sdf...>()` entries.

Assessment: implementation exists, but direct Python class exposure for SDF builders/primitives is incomplete.

---

## 7) Runtime Symbol Diff: Native vs Python Wrapper Expectations

### 7.1 Wrapper-referenced symbols absent in current native module

Detected `_forge3d.<symbol>` references with no runtime symbol:

1. `CogDataset` -> `python/forge3d/cog.py:31`
2. `create_restir_di` -> `python/forge3d/lighting.py:221`
3. `restir_set_lights` -> `python/forge3d/lighting.py:291`
4. `restir_clear_lights` -> `python/forge3d/lighting.py:299`
5. `restir_sample_light` -> `python/forge3d/lighting.py:314`
6. `restir_render_frame` -> `python/forge3d/lighting.py:367`
7. `restir_get_statistics` -> `python/forge3d/lighting.py:420`
8. `mesh_generate_cube_tbn` -> `python/forge3d/mesh.py:56`
9. `mesh_generate_plane_tbn` -> `python/forge3d/mesh.py:136`

### 7.2 Dynamic optional lookups (`getattr`) that miss in current build

- `set_exposure_scale` at `python/forge3d/lighting.py:58` -> absent
- `set_msaa_samples` at `python/forge3d/viewer.py:42` -> absent as module function (available as `Scene` method)
- `geometry_instance_mesh_gpu_render_py` at `python/forge3d/geometry.py:336` -> absent (feature-gated)

### 7.3 Wrappers that are fully satisfied by current native exports

- `python/forge3d/geometry.py`: 21/21 referenced symbols present.
- `python/forge3d/io.py`: 6/6 present.

---

## 8) Python Frontend Ownership Model (Native-backed vs Pure Python)

Native-importing wrapper modules (directly import `_forge3d`):

- `python/forge3d/geometry.py`
- `python/forge3d/io.py`
- `python/forge3d/lighting.py`
- `python/forge3d/mesh.py`
- `python/forge3d/cog.py`
- `python/forge3d/render.py`
- `python/forge3d/sdf.py`
- `python/forge3d/viewer.py`

Predominantly pure-Python modules (no direct native import), including major user-facing feature groups:

- `python/forge3d/tiles3d.py`
- `python/forge3d/pointcloud.py`
- `python/forge3d/bundle.py`
- `python/forge3d/style.py`
- `python/forge3d/export.py`
- `python/forge3d/buildings.py`
- `python/forge3d/vector.py`
- `python/forge3d/path_tracing.py`
- plus many utility/UI/config modules.

Interpretation:

- Frontend architecture is hybrid: selective native acceleration over broad Python orchestration.
- Several Rust modules with substantial implementation are currently non-exposed or only indirectly used by Python.

---

## 9) Precision Notes on Registered/Unregistered PyClasses

- `Frame` is declared as PyClass:
  - declaration at `src/lib.rs:71`
  - methods at `src/lib.rs:677`
  - but no visible `m.add_class::<Frame>()` in registration block around `src/lib.rs:4696` onward.
- `SdfPrimitive`, `SdfScene`, `SdfSceneBuilder` are declared with `pyclass` attrs in `src/sdf/py.rs` but not added in `_forge3d` init.
- `CogDataset` class is implemented and registered conditionally:
  - class at `src/terrain/cog/py_bindings.rs:11`
  - registration function at `src/terrain/cog/py_bindings.rs:155`
  - init wiring gated by `cog_streaming` at `src/lib.rs:4687`

---

## 10) Overall Assessment

### 10.1 Strengths

- Strong native exposure for core rendering workflows: scene, terrain, lighting, geometry, io.
- Clear feature-gated architecture for optional capabilities (COG, PROJ, GPU instancing).
- Python wrappers generally include graceful fallbacks when native symbols are missing.

### 10.2 Exposure Risks / Gaps

1. API drift between Python wrappers and native exports:
   - ReSTIR symbols referenced but not exported.
   - mesh TBN native symbols referenced but not exported.
2. Method placement mismatch:
   - wrappers probe module-level `render_rgba` / `set_msaa_samples`, while current native implementation places these on `Scene`.
3. Partially wired native feature classes:
   - SDF classes are implemented but not registered into Python module.
4. Root package under-exposes native module breadth:
   - only a minority of native public symbols appear at `forge3d` root.

### 10.3 Exposure Maturity Summary

- High maturity (`L3`): `scene`, `terrain`, `geometry`, `lighting`, `camera`, `io`, `vector`.
- Medium (`L2`): `core`, `render`, `geo`, `animation`, `picking`, `viewer`, `uv`, `colormap`, `import`, `converters`.
- Low/none (`L1/L0`): `tiles3d`, `pointcloud`, `bundle`, `style`, `export`, `labels`, `pipeline`, `accel`, `shadows` internals, and other internal modules.

---

## 11) Reproducibility Snapshot (Exact Checks Run)

Representative checks used:

1. Rust module declarations:
   - `rg -n "^(pub\\s+)?mod\\s+[A-Za-z0-9_]+\\s*;" src/lib.rs`
2. Binding registration:
   - `rg -n "m\\.add_(function|class)" src/lib.rs src/terrain/clipmap/py_bindings.rs src/terrain/cog/py_bindings.rs`
3. Feature config:
   - `rg -n "^\\[features\\]|^extension-module\\s*=|^weighted-oit\\s*=|^cog_streaming\\s*=|^proj\\s*=|^enable-gpu-instancing\\s*=|^images\\s*=" Cargo.toml`
   - `rg -n "features\\s*=\\s*\\[" pyproject.toml`
4. Wrapper expectation scan:
   - `_forge3d.<symbol>` and `getattr(_forge3d, "...")` across `python/forge3d`.
5. Runtime introspection:
   - `import forge3d, forge3d._forge3d as native`
   - `dir(native)`, feature flags, class attr counts.

---

## 12) Key Anchors (Quick Reference)

- Rust module list: `src/lib.rs:1089` to `src/lib.rs:1127`
- Native module init: `src/lib.rs:4431`
- COG gating in init: `src/lib.rs:4687`
- GPU instancing gating in init: `src/lib.rs:4545`
- PROJ registration: `src/lib.rs:4741` and `src/lib.rs:4742`
- Scene method anchors:
  - `src/scene/mod.rs:1721`
  - `src/scene/mod.rs:2145`
- Python native re-export loop: `python/forge3d/__init__.py:52`
- Python `__all__`: `python/forge3d/__init__.py:346`
- Wrapper callsites for major mismatches:
  - `python/forge3d/lighting.py:220`
  - `python/forge3d/mesh.py:56`
  - `python/forge3d/cog.py:31`
  - `python/forge3d/helpers/offscreen.py:41`
  - `python/forge3d/viewer.py:42`

