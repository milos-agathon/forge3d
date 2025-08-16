# Changelog
All notable changes to this project will be documented in this file.

This project adheres to [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and follows SemVer (pre-1.0 may include breaking changes).

## 0.0.9 — T4.1 Scene integration
- Added `scene` module with `Scene` Py API (camera, height upload, render to PNG).
- Reused T3 terrain pipeline and kept bind groups cached.
- Docs: README usage snippet; ROADMAP updated.

## [0.0.8] — 2025-08-16
### Added
- **Workstream T3 — Terrain Shaders & Pipeline.**
- `TerrainPipeline` in `src/terrain/pipeline.rs` with:
  - Bind group layouts: (0) Globals UBO, (1) height **R32Float** + **NonFiltering** sampler, (2) LUT texture + Filtering sampler.
  - Vertex layout: `position.xy` and `uv` as two `Float32x2` attributes.
  - sRGB color target (recommended): `Rgba8UnormSrgb`.
- Python-facing spike `TerrainSpike` for offscreen rendering and PNG output.
- `ColormapLUT` supporting runtime format selection; defaults to sRGB, can force UNORM via `VF_FORCE_LUT_UNORM`.

### Changed
- Cached pipeline and bind groups now used in the render pass (no runtime re-creation).
- Documentation updates:
  - Exact single-line docstring for `build_grid_xyuv` clarifying `[x, z, u, v]` layout.
  - Local comment explaining **NonFiltering (nearest)** requirement for `R32Float` height textures.

### Fixed
- Verified uniform block layout (176 bytes, std140-compatible) and WGPU clip-space projection via tests.

## [0.0.7] — 2025-08-15
### Added
- Completed **Workstream T2 — Uniforms, Camera, and Lighting**.
- New Rust module `src/camera.rs` with:
  - `camera_look_at()`, `camera_perspective(clip_space={'wgpu','gl'})`, `camera_view_proj()` exposed to Python (PyO3).
  - Precise parameter validation and exact error messages.
  - NumPy-friendly outputs: C-contiguous, `float32`, shape `(4,4)`.
- Terrain uniforms:
  - `TerrainUniforms` struct (std140-compatible, **176 bytes**, 16-byte aligned).
  - `Globals` container and `TerrainSpike::debug_uniforms_f32()` to inspect 44-float UBO layout.
- Terrain camera integration:
  - `TerrainSpike::set_camera_look_at(...)` computes aspect from framebuffer and updates UBO.
  - Default projection switched to **WGPU clip space** via `camera::perspective_wgpu()`.

### Changed
- `build_view_matrices()` now uses WGPU depth range [0,1] (was GL [-1,1]).
- GL→WGPU depth conversion refactored to `gl_to_wgpu()` helper.

### Tests
- Rust: unit test guarantees `TerrainUniforms` size and alignment.
- Rust: unit test verifies default projection is WGPU clip space.
- Python: `tests/test_camera.py` (~20 tests) covering camera math, validation, and TerrainSpike integration.

### Docs
- README updated to document WGPU clip-space default and camera API examples.

## [0.0.6] - 2025-08-15
### Workstream T1 — CPU Mesh & GPU Resources
**Status:** Complete

### Added
- **CPU grid mesh generator**
  - New `terrain::mesh::{make_grid, GridMesh, GridVertex, Indices}` with CCW winding and centered origin.
  - Python API `_vulkan_forge.grid_generate(nx, nz, spacing, origin)` returning NumPy arrays:
    - `XY: (N,2) float32`, `UV: (N,2) float32`, `indices: (M,) uint32`.
  - Validation of shapes/dtypes; zero-copy where possible.
- **Height texture upload (R32Float)**
  - `Renderer.upload_height_r32f()` with proper 256-byte `bytes_per_row` padding.
  - Debug helpers: `debug_read_height_patch()` and `read_full_height_texture()`.
- **Colormap LUT system**
  - Central registry `src/colormap/mod.rs` with embedded 256×1 PNG assets: `viridis`, `magma`, `terrain`.
  - Unconditional Python/Rust discovery: `colormap_supported()`.
  - Runtime texture format selection:
    - Prefer `Rgba8UnormSrgb`; fallback to `Rgba8Unorm` with CPU sRGB→linear conversion.
    - Env toggle `VF_FORCE_LUT_UNORM=1` to exercise fallback path.
  - `TerrainSpike` integration (feature-gated): bind group layout `(0=UBO, 1=texture, 2=sampler)`, linear-filtered LUT sampling.
  - `TerrainSpike.debug_lut_format()` for inspection.
- **Docs & API polish**
  - Expanded README sections (T11, T1.2, T1.3).
  - Rich docstrings in `python/vulkan_forge/__init__.py`.

### Changed
- Removed stale `grid` module; Python keeps a `generate_grid` alias to the new `grid_generate` for compatibility.
- `TerrainSpike` now seeds lighting from the computed light vector.
- WGSL shader cleaned up and aligned with binding layout.

### Fixed
- WGSL parsing errors (commas vs semicolons) and linear-space lighting correctness.

### Tests
- `tests/test_grid_generate.py`: shapes, dtypes, UV corners, CCW winding, u16/u32 index switch, large grids.
- `tests/test_colormap.py`: registry/discovery, format fallback (including `VF_FORCE_LUT_UNORM`), shader sanity.

### Compatibility
- No breaking Python API changes; `TerrainSpike` remains feature-gated.
- Rust callers should migrate imports from `crate::grid` → `crate::terrain::mesh`.
  - Python alias preserves old name: `generate_grid = grid_generate`.


## \[0.0.5] - 2025-08-08

### Added

* **T33 – Colormap LUT & assets:** Embedded 256×1 PNG LUTs (`viridis`, `magma`, `terrain`) and a central registry `colormap.rs` exposing `SUPPORTED` and `resolve_bytes()`. LUTs are sampled in the fragment shader for height-mapped color.
* **A2 – Terrain spike renderer:** `TerrainSpike(width, height, grid=128, colormap='viridis')` headless renderer with off-screen target and `render_png(path)` for test coverage.
* **T2.2 – Sun direction & tonemap:** Uniforms now carry `sun_dir` and `exposure`; shader computes diffuse `N·L` and applies Reinhard tonemap for perceptually non-flat output. Python helpers `set_sun(elevation_deg, azimuth_deg)` and `set_exposure(exposure)` (gated by `terrain_spike` feature).
* **T1.1 – Grid index/vertex generator:** CPU grid mesh (positions + normals) for the spike terrain; Python wrapper `grid_generate(nx, nz, spacing=(dx,dy), origin='center')` returning NumPy arrays.

### Changed

* **Uniform layout:** `TerrainUniforms` repacked to std140-compatible **176 bytes**:
  `view(64) + proj(64) + sun_exposure(16) + spacing_h_exag_pad(16) + _pad_tail(16)`.
  Matches WGSL reflection and avoids validation errors.
* **Shader pipeline:** `terrain.wgsl` updated to consume the new uniform layout, sample the LUT, apply diffuse lighting, and tonemap; bindings: `@group(0) @binding(0)=UBO`, `1=LUT texture_2d`, `2=Sampler`.
* **Colormap selection:** Strict, case-sensitive names validated against `SUPPORTED`; shared error text across Rust/Python to keep tests deterministic.

### Fixed

* **wgpu validation panic** “buffer size 164, shader expects 176”: corrected by the new UBO layout.
* **WGSL parse error** (“expected ',', found ';'”): struct fields now comma-separated; shader module creation no longer fails.
* **wgpu 0.19 API mismatch**: `ImageDataLayout.{bytes_per_row,rows_per_image}` now `Option<u32>`—converted `NonZeroU32` via `.into()` at all call sites.
* **Colormap ‘magma’ rejected**: registry and asset mapping added; constructor accepts `"magma"`.
* **Uniform PNG output (\~710B)**: shader now maps height→LUT and lights scene; PNG sizes comfortably exceed the test threshold.

### Technical Notes

* Off-screen color target `Rgba8UnormSrgb` with 256-byte row alignment for copies; readback performs CPU unpadding.
* Validation layers remain enabled in Debug; any device/shader error is a test failure.
* Paths kept zero-copy for NumPy interop; no unnecessary heap churn during readbacks.

## [0.0.4] - 2025-08-05
### Added
- **T0.1 – Public API & validation:** `Renderer.add_terrain(heightmap, spacing, exaggeration, colormap)` with robust NumPy array validation; accepts `float32`/`float64` with shape `(H, W)` and C‑contiguous requirement; clear `PyRuntimeError` for invalid inputs.
- **T0.2 – DEM statistics & normalization:** Automatic `h_min`/`h_max` computation from heightmap with optional percentile clamping; `Renderer.set_height_range(min, max)` override method for custom height ranges.
- **T1.1 – Grid index/vertex generator:** CPU mesh generation in `terrain/mesh.rs` with `make_grid(W, H, dx, dy)` producing indexed triangle grids; vertex attributes include world‑space `position.xy` and `uv` coordinates for height sampling; automatic `u16`/`u32` index format selection based on vertex count.
- **T1.2 – Height texture upload:** GPU height texture creation with `R32Float` format and `TEXTURE_BINDING | COPY_DST` usage; 256‑byte row alignment handling for cross‑platform compatibility; linear clamp sampler configuration.
- **T1.3 – Colormap LUT texture:** Built‑in terrain colormaps (`viridis`, `magma`, `terrain`) as 256×1 `RGBA8UnormSrgb` textures; height‑to‑color mapping with `h_min`/`h_max` normalization uniforms; CPU reference implementation for unit testing.

### Changed
- Enhanced `Renderer` constructor to support terrain rendering pipeline initialization.
- Terrain metadata storage including `dx`, `dy`, `h_min`, `h_max`, `exaggeration`, and colormap selection.

### Fixed
- Proper error handling for unsupported heightmap dtypes and shapes.
- Memory‑efficient reuse of vertex/index buffers for terrain mesh generation.

### Technical Notes
- Grid generation optimized for 1024×1024 heightmaps with sub‑40ms performance target.
- Cross‑platform texture upload with proper row padding handled automatically.
- Colormap validation ensures known scalar inputs map to expected palette colors.

## [0.0.3] - 2025-08-01
### Added
- **A1.9 – Device diagnostics & failure modes:** Rust PyO3 APIs `enumerate_adapters()` and `device_probe(backend)`; Python CLI `python/tools/device_diagnostics.py` that writes JSON and classifies outcomes.
- **A1.10 – Performance sanity:** `python/tools/perf_sanity.py` measuring init/steady timings with JSON/CSV output; optional budget/baseline enforcement via `VF_ENFORCE_PERF=1`.
- **A1.8 – CI matrix & artifacts:** New workflow `.github/workflows/ci.yml` running pytest, determinism harness, and (Win/macOS) cross-backend runner with uploaded artifacts.
- **Docs (A1.11):** Quickstart, Tools, Testing, CI, and Troubleshooting sections updated.

### Changed
- README guidance for Python 3.13 builds using `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1`.

### Fixed
- Robust import paths for the compiled module in tools; improved error messages.


## [0.0.2] - 2025-07-31
### Added
- **A1.4 – Off-screen target & readback**: persistent RGBA8 UNORM SRGB color target (`RENDER_ATTACHMENT | COPY_SRC`) and persistent readback buffer with 256-byte row alignment + CPU unpadding.
- **A1.5 – Python API surface**: public package `vulkan_forge` with:
  - `Renderer(width, height)`
  - `render_triangle_rgba(width, height) -> (H,W,4) uint8`
  - `render_triangle_png(path, width, height) -> None`
  - `__version__` metadata
- Legacy alias **`vshade`** re-exports the public API for compatibility.

### Changed
- PyO3 text signatures and docstrings for `__init__`, `render_triangle_rgba`, `render_triangle_png`, and `info`.
- Robust import in `vulkan_forge/__init__.py` (supports top-level or package-internal `_vulkan_forge`).

### Fixed
- Import symbol mismatch by standardizing the module to `#[pymodule] fn _vulkan_forge(...)`.
- Deterministic pipeline preserved (blend=None, CLEAR_COLOR, CCW + back-cull, fixed viewport/scissor).

### Notes
- For Python 3.13 + PyO3 0.21, build with `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1`.
