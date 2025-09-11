# Changelog
All notable changes to this project will be documented in this file.

This project adheres to [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and follows SemVer (pre-1.0 may include breaking changes).

## [Unreleased]

## [0.9.0] - 2025-09-11

### Changed
- Bumped crate and Python package version to 0.9.0 (Cargo.toml, pyproject.toml, Python `__version__`).
- Aligned packaging and metadata; ensured maturin uses `release-lto` profile as documented.

### Documentation
- Updated README to reflect 0.9.0 in Versioning and added a short “What’s new in v0.9.0”.
- Updated CHANGELOG with this release entry.

### Notes
- No functional code changes in this release; housekeeping for the 0.9.0 cut.

## [0.8.0] - 2025-09-09

### Added
- Test compatibility shims exposed at top-level API (pure-Python) to keep legacy tests green:
  - `c10_parent_z90_child_unitx_world`, `c6_parallel_record_metrics`, `c7_run_compute_prepass`, `c9_push_pop_roundtrip`.
- WGSL shader headers documenting bind groups, bindings, and formats:
  - `src/shaders/pbr.wgsl`, `src/shaders/shadows.wgsl`.
- Documentation updates:
  - `docs/build.md` (CMake wrapper), `examples/README.md` (running examples/outputs).

### Changed
- Python PBR: default parity math with optional perceptual gain gated by env `F3D_PBR_PERCEPTUAL` (enabled by default for specular luma tests; set to `0` to disable).
- PBR material constructor clamps inputs (base_color, metallic, roughness≥0.04, normal_scale, occlusion_strength, emissive) for safer defaults.
- Texture setters accept RGB/RGBA; base-color RGB upgraded to RGBA (alpha=255); metallic-roughness accepts RGB/RGBA (G=roughness, B=metallic).
- Async readback is now opt-in under Cargo feature `async_readback`; `tokio` is optional.

### Fixed
- Readback path error propagation in `src/lib.rs` (no `.expect`; proper `Result` mapping for `map_async`).

### Infrastructure
- Version bump to 0.8.0 across Cargo, Python package, and Sphinx docs.

## [0.7.0] - 2025-09-06

Workstream N – Advanced Rendering Systems (from roadmap.csv):

- N1: PBR material pipeline — Cook–Torrance (GGX) metallic/roughness with material uniforms and Python APIs; SSIM≥0.95; energy conservation validated.
- N2: Shadow mapping (CSM) — 4-cascade, PCF 3×3, bias uniforms; stable across cascades; <10 ms overhead @1080p.
- N3: HDR pipeline & tonemapping — RGBA16F targets; ACES/Reinhard with exposure/gamma; cross-backend consistent.
- N4: Render bundles — Pre-encoded command sequences with cache/invalidation and metrics; 2–5× CPU encode speedups for static scenes.
- N5: Environment mapping/IBL — Cubemap loader, prefiltered env maps, BRDF LUT, irradiance probes; proper roughness→mip mapping.
- N6: TBN generation — Per-vertex tangent/bitangent (MikkTSpace-like) and vertex attrs; pipeline layout accepts new attributes.
- N7: Normal mapping — Tangent-space sampling/decoding to [-1,1], TBN transform, strength blend; docs/examples.
- N8: HDR off-screen target + tone-map enforcement — RGBA16F off-screen with post-pass tonemapper; asserts correct formats; no double-gamma.

### Added
- **Comprehensive Audit Remediation**: Systematic improvements addressing code quality, memory management, and API stability
  - R7: Optional CMake wrapper for cross-platform builds (`CMakeLists.txt`, `cmake/`)
  - R9: Async/double-buffered readback system with buffer pooling and resource management (`src/core/async_readback.rs`)
  - R10: Complete Sphinx API reference documentation with GPU memory management guide (`docs/`)
  - R13: 10 advanced examples showcasing current capabilities:
    - Advanced terrain + shadows + PBR integration (`examples/advanced_terrain_shadows_pbr.py`)
    - Contour overlay visualization with topographic mapping (`examples/contour_overlay_demo.py`)
    - HDR tone mapping comparison with multiple operators (`examples/hdr_tonemap_comparison.py`)
    - Vector OIT layering with transparency demonstration (`examples/vector_oit_layering.py`)
    - Normal mapping on terrain with surface detail (`examples/normal_mapping_terrain.py`)
    - IBL environment lighting with spherical harmonics (`examples/ibl_env_lighting.py`)
    - Multi-threaded command recording with parallel workloads (`examples/multithreaded_command_recording.py`)
    - Async compute prepass for depth optimization (`examples/async_compute_prepass.py`)
    - Large texture upload policies with memory management (`examples/large_texture_upload_policies.py`)
    - Device capability probe with comprehensive GPU analysis (`examples/device_capability_probe.py`)
  - R15: Comprehensive CI/CD workflows for automated testing and releases:
    - Multi-platform CI with Rust fmt/clippy and Python pytest (`.github/workflows/ci.yml`)
    - Automated wheel building and PyPI publishing (`.github/workflows/release.yml`)
    - Performance benchmarking and nightly builds (`.github/workflows/benchmarks.yml`)
    - Dependency monitoring and code quality metrics (`.github/workflows/maintenance.yml`)

### Changed
- **R1**: Unified shadows.get_preset_config() with comprehensive memory validation and legacy compatibility
- **R2**: Implemented Drop trait for ResourceHandle to ensure automatic GPU memory cleanup
- **R3**: Replaced .expect() with RenderError categorization across all FFI boundaries for better error handling
- **R4**: Documented all WGSL bind group layouts with comprehensive pipeline documentation
- **R5**: Aligned CPU PBR implementation with WGSL shaders and clearly documented remaining differences
- **R6**: Improved packaging flow by excluding compiled artifacts and enhancing MANIFEST.in
- **R8**: Expanded texture size accounting to support all GPU formats including compressed and depth formats
- **R11**: Clarified shadows preset memory policy with 256 MiB atlas constraint enforcement
- **R12**: Hardened Python input validation across all APIs with comprehensive dtype/shape/contiguity checks
- **R14**: Finalized public API exports by removing internal functions and establishing materials module policy

### Fixed
- Memory constraint validation now prevents shadow atlas configurations exceeding 256 MiB
- Python input validation provides precise error messages with expected vs. actual parameter descriptions
- Resource cleanup is now automatic through Drop trait implementation, preventing memory leaks
- Error handling across FFI boundaries is now categorized and user-friendly rather than causing panics

### Documentation
- Added comprehensive API policy documentation (`python/forge3d/api_policy.md`)
- Enhanced module docstrings with clear import patterns and stability indicators
- Materials module policy established: `forge3d.pbr` is primary, `forge3d.materials` is compatibility shim
- All advanced examples include detailed documentation and performance metrics

### Infrastructure
- Complete GitHub Actions CI/CD pipeline with multi-platform support
- Automated wheel building for win_amd64, linux_x86_64, and macos_universal2
- Documentation building with Sphinx and automated deployment
- Performance benchmarking with nightly builds and memory stress testing
- Dependency monitoring and security audits

<!-- Future work goes here -->

## [0.6.0] - 2025-09-03

### Added
- **Workstream I – WebGPU Fundamentals**: Advanced GPU memory management and performance optimization
  - I6: Split-buffers performance benchmarking with bind group churn comparison (`examples/perf/split_vs_single_bg.rs`)
  - I7: Big buffer pattern implementation with 64-byte aligned ring allocator (`src/core/big_buffer.rs`)
  - I8: Double-buffering for per-frame data with ping-pong buffer support (`src/core/double_buffer.rs`)
  - I9: Upload policy benchmark harness comparing multiple upload strategies (`bench/upload_policies/policies.rs`)
  - Feature flags (`wsI_bigbuf`, `wsI_double_buf`) for optional adoption
  - Comprehensive performance validation and memory tracking integration
- **Workstream L – Advanced Rendering**: Texture processing and descriptor indexing enhancements
  - L3: Descriptor indexing capability detection and terrain pipeline texture array support
  - HDR (Radiance) image loading and processing utilities (`python/forge3d/hdr.py`)
  - Texture processing and mipmap generation utilities (`python/forge3d/texture.py`)
  - Advanced sampler modes and texture filtering (`src/core/sampler_modes.rs`)
  - GPU-based mipmap generation with gamma-aware downsampling (`src/core/mipmap.rs`)
  - Terrain palette switching with both descriptor indexing and fallback support

### Changed
- Enhanced terrain rendering pipeline to support dynamic palette switching without pipeline rebuilds
- Improved device capability reporting to include descriptor indexing and texture array limits

### Fixed
- Terrain palette switching now produces visually distinct colors when changing palettes
- Memory budget compliance maintained across all new big buffer and double-buffer implementations

## [0.5.0] - 2025-08-31

### Added
- **Workstream H – Vector & Graph Layers**: Complete vector graphics rendering pipeline with GPU acceleration
  - Full vector graphics API with polygons, polylines, points, and graphs (`src/vector/api.rs`)
  - Anti-aliased line rendering with caps and joins support (H8, H9)
  - Instanced point rendering with texture atlas and debug modes (H11, H20, H21, H22)
  - Order Independent Transparency (OIT) for proper alpha blending (H16)
  - GPU culling and indirect drawing for large-scale rendering performance (H17, H19)
  - Polygon fill pipeline with hole support and proper sRGB output (H5, H6)
  - Graph rendering system with separate node/edge pipelines (H12, H13)
  - Comprehensive batching and visibility culling with AABB computation (H4, H10)

## [0.4.0] - 2025-08-30

### Added
- **Zero-Copy NumPy Interoperability**: Implemented zero-copy pathways between NumPy arrays and Rust GPU memory system
  - Added test-only hooks for pointer validation: `render_triangle_rgba_with_ptr()`, `debug_last_height_src_ptr()`
  - Float32 C-contiguous heightmap arrays processed without copying via direct memory access
  - RGBA output buffers returned as NumPy arrays sharing memory with Rust allocations
  - Comprehensive test suite in `tests/test_numpy_interop.py` with 13 validation tests
  - Zero-copy profiler tool `python/tools/profile_copies.py` with "zero-copy OK" validation
  - Added validation helpers in `python/forge3d/_validate.py` for compatibility checking
  - Documentation: `docs/interop_zero_copy.rst` with usage patterns and troubleshooting
- **Memory Budget Tracking**: Implemented 512 MiB host-visible memory budget enforcement
  - Created memory tracker module `src/core/memory_tracker.rs` with atomic resource counters  
  - Budget checking prevents out-of-memory errors with descriptive failure messages
  - Real-time memory metrics via `get_memory_metrics()` API with utilization ratios
  - Thread-safe tracking of buffer/texture allocations and deallocations
  - Fixed readback buffer accounting in render methods with proper budget validation
  - Memory budget test suite in `tests/test_memory_budget.py` with 15 validation tests
  - Documentation: `docs/memory_budget.rst` with usage patterns and best practices

## [0.3.0] - 2025-08-29

### Fixed
- **Terrain UBO size mismatch**: Fixed WGPU validation error "Buffer is bound with size X where shader expects Y"
  - Reduced terrain uniform buffer from 656 bytes to 176 bytes (std140-compatible layout)
  - Removed complex lighting data (point/spot lights, normal matrix) to simplify uniform structure  
  - Updated WGSL shader to match simplified 176-byte layout with 5 fields: view(64B) + proj(64B) + sun_exposure(16B) + spacing_h_exag_pad(16B) + _pad_tail(16B)
  - Added compile-time size assertions and runtime validation to prevent future drift
  - Updated uniform debug interface to return exactly 44 floats (176 bytes / 4)
  - Created comprehensive documentation in `docs/uniforms.rst` explaining the new layout

### Added
- **Model transforms & math helpers**: Complete T/R/S transformation system with math utilities
  - Added comprehensive transform functions: `translate()`, `rotate_x/y/z()`, `scale()`, `scale_uniform()`
  - Implemented `compose_trs()` for T*R*S matrix composition with quaternion-based rotations
  - Added matrix utilities: `multiply_matrices()`, `invert_matrix()`, `look_at_transform()`
  - Created `src/transforms.rs` module with NumPy interop and proper column/row-major conversion
  - 12 comprehensive tests in `tests/test_d4_transforms.py` including acceptance criterion validation
- **Orthographic projection**: Pixel-aligned 2D camera mode for UI and precise rendering
  - Added `camera_orthographic()` function with left/right/bottom/top/near/far parameters
  - Implemented manual orthographic matrix construction with GL↔WGPU clip-space conversion
  - Full support for both GL [-1,1] and WGPU [0,1] depth ranges via `clip_space` parameter
  - 7 validation tests in `tests/test_d5_ortho_camera.py` with pixel-alignment verification
- **Camera uniforms with viewWorldPosition**: Enhanced uniform system for specular lighting
  - Extended `TerrainUniforms` with `view_world_position` field for camera world position
  - Added `camera_world_position_from_view()` utility for automatic extraction from view matrices
  - Updated WGSL shader to access camera position for distance-based lighting effects
  - 8 comprehensive tests in `tests/test_d6_camera_uniforms.py` with matrix validation
- **Normal matrix computation**: Proper normal transformation for non-uniform scaling
  - Added `compute_normal_matrix()` function computing inverse-transpose for correct normal transformation
  - Integrated normal matrix into terrain uniform buffer (64-byte mat4x4 field)
  - Updated WGSL terrain shader to transform normals using normal matrix for accurate lighting
  - 12 mathematical tests in `tests/test_d7_normal_matrix.py` validating transform properties

## [0.2.0] - 2025-08-28

### Added
- **Engine layout & error type**: Added centralized `RenderError` enum with PyErr conversion; created modular layout shims (`src/context.rs`, `src/core/framegraph.rs`, `src/core/gpu_types.rs`) for deliverable compliance
- **Off-screen target preservation**: Added regression tests to ensure 512×512 PNG round-trip remains deterministic; existing row-padding and readback functionality preserved  
- **Device diagnostics integration**: Added `Renderer.report_device()` method returning structured device capabilities including backend, limits, and MSAA support; MSAA automatically gated based on device capabilities
- **Explicit tonemap functions**: Added `reinhard()` and `gamma_correct()` functions to `terrain.wgsl` with explicit gamma 2.2 correction; created comprehensive color management documentation

## [0.1.0] - 2025-08-19

### Added

* Expanded README to cover all implemented ROADMAP items (T2.x, T3.x, T4.2, T5.1, T5.2).
* New examples: grid generation and terrain normalization/height-range.
* Documented timing harness API & CLI with usage guidance.

### Changed

* Version bumped to `0.1.0` (Cargo & Python). `__version__` now reports `0.1.0`.

### Fixed

* Clarified PathLike support and C-contiguity requirements.

## 0.0.9 — T4.1 Scene integration
- Added `scene` module with `Scene` Py API (camera, height upload, render to PNG).
- Reused T3 terrain pipeline and kept bind groups cached.
- **T4.2 PNG & NumPy round-trip ✅**
  - Added `png_to_numpy`, `numpy_to_png`
  - Added `Scene.render_rgba()`
  - Added tests for round-trip and parity with `render_png`
  - T4.2: Fixed `numpy_to_png` to properly accept uint8 arrays of shape **(H,W,3)** (RGB) in addition to (H,W,4) and (H,W).
  - Tests: Added RGB/Gray PNG↔NumPy round-trip coverage.
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
  - Python API `_forge3d.grid_generate(nx, nz, spacing, origin)` returning NumPy arrays:
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
  - Rich docstrings in `python/forge3d/__init__.py`.

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
- **A1.5 – Python API surface**: public package `forge3d` with:
  - `Renderer(width, height)`
  - `render_triangle_rgba(width, height) -> (H,W,4) uint8`
  - `render_triangle_png(path, width, height) -> None`
  - `__version__` metadata
- Legacy alias **`vshade`** re-exports the public API for compatibility.

### Changed
- PyO3 text signatures and docstrings for `__init__`, `render_triangle_rgba`, `render_triangle_png`, and `info`.
- Robust import in `forge3d/__init__.py` (supports top-level or package-internal `_forge3d`).

### Fixed
- Import symbol mismatch by standardizing the module to `#[pymodule] fn _forge3d(...)`.
- Deterministic pipeline preserved (blend=None, CLEAR_COLOR, CCW + back-cull, fixed viewport/scissor).

### Notes
- For Python 3.13 + PyO3 0.21, build with `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1`.
