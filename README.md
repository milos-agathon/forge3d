<div align="center">
  <a href="./">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="assets/logo-2000-dark.png">
      <img src="assets/logo-2000.png"
           alt="forge3d logo"
           width="224"
           height="224"
           decoding="async"
           loading="eager">
    </picture>
  </a>
</div>

# forge3d

Headless GPU rendering + PNG↔NumPy utilities (Rust + PyO3 + wgpu).

## Installation

```bash
# from source
pip install -U maturin
maturin develop --release
# or via wheel (if provided)
# pip install forge3d
```

## Quick Start (< 10 minutes)

New to forge3d? Get a terrain rendering example working in under 10 minutes:

1. **Install prerequisites**: Ensure you have Python ≥3.8 installed
2. **Install maturin**: `pip install -U maturin`
3. **Build forge3d**: `maturin develop --release`
4. **Run terrain example**: `python examples/terrain_single_tile.py`
5. **Verify output**: Check that `terrain_single_tile.png` was created and shows a shaded terrain

Expected runtime: ~1-2 minutes for the script at default 128×128 resolution. The PNG will be saved in your current directory and shows a procedural terrain with the viridis colormap.

```bash
# Complete quick start sequence
pip install -U maturin
maturin develop --release
python examples/terrain_single_tile.py
# -> Creates terrain_single_tile.png (512×512 image)
```

## Platform requirements

Runs anywhere wgpu supports: Vulkan / Metal / DX12 / GL (and Browser WebGPU for diagnostics). A discrete or integrated GPU is recommended. Examples/tests that need a GPU will skip if no compatible adapter is found.

## What's new in v0.14.0

- Workstream V – Datashader Interop:
  - V1: Datashader adapter for zero-copy RGBA overlays, alignment validation, and simple conversion helpers.
  - V2: Performance + fidelity tests over Z0/Z4/Z8/Z12 with SSIM vs goldens and frame-time/memory metrics; CI workflow uploads artifacts and guards regressions.
  - Demo: `examples/datashader_overlay_demo.py` produces `examples/output/datashader_overlay_demo.png` and prints `OK`.
  - Optional dependency handling: functions skip gracefully when Datashader isn’t installed.

Docs:
- `docs/user/datashader_interop.rst` (in ToC)

## What's new in v0.13.0

- Workstream U – Basemaps & Tiles:
  - U1: XYZ/WMTS tile fetch client with on-disk cache, offline mode, and conditional GET handling; mosaic composition to RGBA via Pillow.
  - U2: Attribution overlay utility with DPI-aware text/logo rendering and TL/TR/BL/BR presets.
  - U3: Cartopy interop example (Agg backend) with correct extent matching and attribution.
  - Provider policy compliance: polite `User-Agent` and `Referer` supported (env: `FORGE3D_TILE_USER_AGENT`, `FORGE3D_TILE_REFERER`); OSM attribution uses “© OpenStreetMap contributors”.

Examples:
- `examples/xyz_tile_compose_demo.py` → `reports/u1_tiles.png`
- `examples/cartopy_overlay.py` → `reports/u3_cartopy.png`

Docs:
- `docs/tiles/xyz_wmts.md`, `docs/integration/cartopy.md`

## What's new in v0.12.0

- Workstream S — Raster IO & Streaming:
  - S1: RasterIO windowed reads + block iterator (`windowed_read`, `block_iterator`) with window/out_shape parity.
  - S2: Nodata/mask → alpha propagation (`extract_masks`, RGBA alpha synthesis) with color fidelity.
  - S3: CRS normalization via WarpedVRT + pyproj (`WarpedVRTWrapper`, `reproject_window`, `get_crs_info`).
  - S6: Overview selection (`select_overview_level`, `windowed_read_with_overview`) to reduce bytes at low zoom.
  - S4: xarray/rioxarray DataArray ingestion (`ingest_dataarray`) preserving dims and CRS/transform.
  - S5: Dask-chunked ingestion (`ingest_dask_array`, streaming materialization) with memory guardrails.
  - Demos: `examples/raster_window_demo.py`, `examples/mask_to_alpha_demo.py`, `examples/reproject_window_demo.py`, `examples/overview_selection_demo.py`, `examples/xarray_ingest_demo.py`, `examples/dask_ingest_demo.py`.

## What's new in v0.11.0

- Workstream R — Matplotlib & Array Interop:
  - R1: Matplotlib colormap interop and linear Normalize parity (accepts names and `Colormap` objects; SSIM ≥ 0.999 on ramp; optional dependency handling)
  - R3: Normalization presets (LogNorm, PowerNorm, BoundaryNorm) mapped with ≤1e-7 parity
  - R4: Display helpers `imshow_rgba` with correct extent/DPI handling and zero-copy for `uint8` inputs
  - Demos: `examples/mpl_cmap_demo.py`, `examples/mpl_norms_demo.py`, `examples/mpl_imshow_demo.py`

## What's new in v0.10.0

- Workstream Q deliverables (initial cut):
  - Q1: Post-Processing effect chain (Python API enable/disable/list, presets) with HDR pipeline and tonemap integration.
  - Q5: Bloom passes scaffolded and wired — bright-pass + separable blur (H/V) + composite into HDR chain output prior to tonemap.
  - Q3: GPU profiling surfaces present (timestamp markers/types), Python `gpu_metrics` accessors and indirect culling metric exposure.
  - Q2: LOD impostors scaffolding and sweep demo; basic triangle-reduction performance test added.
  - Demos/artifacts: `examples/postfx_chain_demo.py`, `examples/bloom_demo.py`, `examples/lod_impostors_demo.py` write outputs under `reports/`.
  - Docs: PostFX page added and Sphinx build enabled.

The comprehensive API below provides access to all these features.

## Quickstart: PNG ↔ NumPy

```python
from pathlib import Path
import numpy as np
import forge3d as f3d

# RGB -> PNG (PathLike supported)
rgb = (np.linspace(0,255,64, dtype=np.uint8)[None, :, None] * np.ones((64,1,3), np.uint8)).copy(order="C")
f3d.numpy_to_png(Path("out_rgb.png"), rgb)

# PNG -> RGBA ndarray (H, W, 4) uint8, C-contiguous
arr = f3d.png_to_numpy("out_rgb.png")
assert arr.shape == (64, 64, 4)
```

**Notes**

* Accepts `(H,W,4)` RGBA, `(H,W,3)` RGB, or `(H,W)` grayscale, dtype=uint8, **C-contiguous**.
* PathLike is supported anywhere a path is accepted.

## Curated Python API (top-level)

* I/O: `png_to_numpy(path)`, `numpy_to_png(path, array)`
* Rendering: `Renderer(width, height)` → `render_triangle_png(path)`, `render_triangle_rgba()`
* Scene: `Scene(width, height, grid=..., colormap='viridis')`:

  * `set_height_from_r32f(height_float32_HxW)`
  * `set_camera_look_at(eye, target, up, fovy_deg, znear, zfar)`
  * `render_png(path)`, `render_rgba()`
* Terrain utils: `terrain_stats()`, `set_height_range(min, max)`, `normalize_terrain(mode, range=None, eps=1e-8)`,
  `upload_height_r32f()`, `debug_read_height_patch(x,y,w,h)`, `read_full_height_texture()`
* Vector graphics: `add_polygons(...)`, `add_lines(...)`, `add_points(...)`, `add_graph(...)`
* Colormaps: `colormap_supported()`
* Mesh: `grid_generate(nx, nz, spacing=(sx,sz), origin='center')`
* Camera helpers: `camera_look_at(...)`, `camera_perspective(...)`, `camera_view_proj(...)`
* Transform utilities: `translate(...)`, `rotate_x/y/z(...)`, `scale(...)`, `scale_uniform(...)`
* Diagnostics: `enumerate_adapters()`, `device_probe(backend=None)`, `get_memory_metrics()`
* Timing: `run_benchmark(operation, width, height, iters, ...)` and CLI `vf-bench`
* Introspection: `__version__` (sourced from the Rust crate)

## Renderer: triangle demo

```python
from pathlib import Path
import forge3d as f3d

r = f3d.Renderer(256, 256)
r.render_triangle_png(Path("triangle.png"))     # file
tri = r.render_triangle_rgba()                  # np.ndarray (H,W,4) uint8
```

## Scene: terrain workflow (PNG/NumPy parity)

```python
import numpy as np
from pathlib import Path
import forge3d as f3d

H, W = 64, 64
# simple synthetic slope for demo
h32 = (np.linspace(0,1,W, dtype=np.float32)[None,:] * np.ones((H,1), np.float32)).copy(order="C")

s = f3d.Scene(256, 256, grid=128, colormap="viridis")
s.set_height_from_r32f(h32)
s.set_camera_look_at((3,2,3), (0,0,0), (0,1,0), fovy_deg=45, znear=0.1, zfar=100.0)
s.render_png(Path("terrain.png"))
rgba = s.render_rgba()  # byte-for-byte pixel parity with the PNG write
```

## Terrain utilities

```python
r = f3d.Renderer(256, 256)
r.add_terrain(height_data, spacing=(1.0, 1.0), exaggeration=1.0, colormap="viridis")
mn, mx, mean, std = r.terrain_stats()
r.set_height_range(mn, mx)                # override normalization
r.normalize_terrain("minmax", (0.0, 1.0)) # or zscore
r.upload_height_r32f()                    # send to GPU
patch = r.debug_read_height_patch(0, 0, 16, 16)
full = r.read_full_height_texture()
```

## Colormaps

```python
from forge3d import colormap_supported
print(colormap_supported())  # list of valid names
```

## Mesh & grid

```python
xy, uv, indices = f3d.grid_generate(64, 64, spacing=(1.0, 1.0), origin="center")
```

## Camera helpers

```python
vp = f3d.camera_view_proj(
    eye=(3,2,3), target=(0,0,0), up=(0,1,0), fovy_deg=45.0, aspect=1.0, znear=0.1, zfar=100.0
)
```

## Diagnostics

```python
import forge3d as f3d
print(f3d.enumerate_adapters())
print(f3d.device_probe())  # {'status': 'ok', ...} or diagnostic info
```

## Timing harness (T5.2)

Python:

```python
from forge3d import run_benchmark
res = run_benchmark("renderer_rgba", width=512, height=512, iterations=50)
print(res["stats"], res["throughput"])
```

CLI:

```bash
vf-bench --op renderer_rgba --width 512 --height 512 --iterations 50 --json bench.json
```

## Synthetic DEMs & tests (T5.1)

* The test suite includes **deterministic synthetic height fields** (planes, ramps, sinusoidal) to validate terrain processing and image parity.
* Tests are **GPU-aware** and will skip if no compatible adapter is present.
* Run all tests:

  ```bash
  pytest -q
  ```

## Development

```bash
# build
maturin develop --release

# run tests
pytest -q
```

## Troubleshooting

* Ensure a supported backend is available (Vulkan/Metal/DX12/GL).
* You can constrain backends via environment variables (e.g., `WGPU_BACKENDS=VULKAN`) if needed.
* Arrays for PNG write and terrain upload must be **C-contiguous**.

## Versioning

`forge3d.__version__` mirrors the Rust crate version (`env!("CARGO_PKG_VERSION")`), now **0.10.0**.

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## PostFX & Renderer Toggle

- Control PostFX at runtime from Python:

```python
import forge3d as f3d
import forge3d.postfx as postfx

r = f3d.Renderer(512, 512)

# Enable PostFX for this renderer (Python helper works even without rebuilding)
postfx.set_renderer_postfx_enabled(r, True)
assert postfx.is_renderer_postfx_enabled(r)

# Configure effects
postfx.enable("bloom", threshold=1.0, strength=0.6)
postfx.enable("tonemap", exposure=1.1, gamma=2.2)

# Render as usual (renderer will apply the PostFX chain when enabled)
r.render_triangle_png("triangle_postfx.png")
```

## A1: GPU Path Tracer (MVP)

Use the minimal path tracing API for a single-sphere scene.

```python
from forge3d.path_tracing import PathTracer, make_sphere, make_camera

tracer = PathTracer(1, 1)
scene = [make_sphere(center=(0,0,-3), radius=1.0, albedo=(0.8,0.2,0.2))]
cam   = make_camera(origin=(0,0,0), look_at=(0,0,-1), up=(0,1,0), fov_y=45.0, aspect=1.0)
img   = tracer.render_rgba(64,64,scene,cam,seed=123,frames=1,use_gpu=True)  # (64,64,4) uint8
```

- Native toggle methods (available after rebuilding the extension):
  - `Renderer.set_postfx_enabled(True|False)`
  - `Renderer.is_postfx_enabled()`

## Wavefront PT (Queue-Based)

You can select a wavefront engine from Python for parity testing and experimentation.

```python
from forge3d.path_tracing import render_rgba, TracerEngine

img_wave = render_rgba(128, 128, scene, cam, seed=7, frames=1,
                       use_gpu=True, engine=TracerEngine.WAVEFRONT)
img_mega = render_rgba(128, 128, scene, cam, seed=7, frames=1,
                       use_gpu=True, engine=TracerEngine.MEGAKERNEL)
```

If the GPU wavefront path isn’t available, calls gracefully fall back to the deterministic CPU implementation.
See `docs/api/wavefront_pt.md` for details.
