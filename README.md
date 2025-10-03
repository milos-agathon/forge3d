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

Headless GPU rendering + PNGâ†”NumPy utilities (Rust + PyO3 + wgpu).

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

1. **Install prerequisites**: Ensure you have Python 3.8 installed
2. **Install maturin**: `pip install -U maturin`
3. **Build forge3d**: `maturin develop --release`

```bash
# Complete quick start sequence
pip install -U maturin
maturin develop --release
python examples/terrain_single_tile.py
# -> Creates terrain_single_tile.png (512Ã—512 image)
```

## Platform requirements

Runs anywhere wgpu supports: Vulkan / Metal / DX12 / GL (and Browser WebGPU for diagnostics). A discrete or integrated GPU is recommended. Examples/tests that need a GPU will skip if no compatible adapter is found.

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