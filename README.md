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

## Platform requirements

Runs anywhere wgpu supports: Vulkan / Metal / DX12 / GL (and Browser WebGPU for diagnostics). A discrete or integrated GPU is recommended. Examples/tests that need a GPU will skip if no compatible adapter is found.

## Features by ROADMAP (what's implemented so far)

* **T2.x Infrastructure & exports**

  * Curated Python surface with stable wrappers and `__version__` re-export.
* **T3.x Geometry & scene prep**

  * Grid generation primitives used by terrain rendering.
* **T4.2 Packaging & Core Unification**

  * Shared GPU context (`gpu.rs`) and alignment helper (`align_copy_bpr()`).
  * Uniform PathLike support for PNG I/O and render-to-PNG.
  * Docstrings clarifying PNG↔NumPy parity & contiguity.
  * Typing stubs + `py.typed`.
* **T5.1 Synthetic DEM tests**

  * Deterministic height fields for validation; GPU-aware test strategy.
* **T5.2 Timing harness**

  * API: `run_benchmark(...)`
  * CLI: `vf-bench` for reproducible measurements (no perf pass/fail).

Each item below links back to these features.

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
* Colormaps: `colormap_supported()`
* Mesh: `grid_generate(nx, nz, spacing=(sx,sz), origin='center')`
* Camera helpers: `camera_look_at(...)`, `camera_perspective(...)`, `camera_view_proj(...)`
* Diagnostics: `enumerate_adapters()`, `device_probe(backend=None)`
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

`forge3d.__version__` mirrors the Rust crate version (`env!("CARGO_PKG_VERSION")`), now **0.1.0**.

## Changelog

See [CHANGELOG.md](CHANGELOG.md).