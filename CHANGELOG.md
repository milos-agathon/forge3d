# Changelog

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
