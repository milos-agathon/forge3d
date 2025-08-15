<!-- A1.11-BEGIN:readme -->
# vulkan-forge

Headless, deterministic triangle renderer built on **wgpu** with a **PyO3** Python API.  
Status: pre-0.1 (research/prototyping). Latest release: **0.0.7**.

## Quickstart (from source)

> Requires Rust (stable), Python 3.10–3.13, and a working GPU runtime.  
> Python 3.13 with PyO3 0.21 needs `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1`.

```bash
# 1) Create & activate a venv
python -m venv .venv

# PowerShell (Windows)
.\.venv\Scripts\Activate.ps1
# or Git Bash (Windows)
source .venv/Scripts/activate
# or Unix
source .venv/bin/activate

# 2) Build and install the extension in editable mode
python -m pip install -U pip maturin numpy
# If using Python 3.13:
#   PowerShell: $Env:PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
#   bash/cmd  : export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
maturin develop --release
```

Render a demo PNG:

```bash
cd python
python -m examples.triangle
# => writes ./triangle.png (gradient triangle on white background)
```

Or from Python:

```python
from vulkan_forge import Renderer, render_triangle_rgba, render_triangle_png

r = Renderer(512, 512)
print(r.info())                      # e.g., "Renderer 512x512, format=Rgba8UnormSrgb"
arr = render_triangle_rgba(256, 256) # (H,W,4) uint8 tightly packed
render_triangle_png("triangle.png", 512, 512)
```

> Legacy compatibility: `from vshade import Renderer` is a re-export of `vulkan_forge.Renderer`.

<!-- T02-BEGIN:api -->
### DEM normalization

```python
# Override auto min/max when outliers skew the range
renderer.set_height_range(min_elev_m, max_elev_m)
```

Auto-computed range uses a robust 1–99 percentile clamp to ignore outliers.

<!-- T02-END:api -->

<!-- T01-BEGIN:api -->
<!-- T22-BEGIN:api -->
### Lighting & Tonemap

```python
renderer.set_sun(elevation_deg=35.0, azimuth_deg=120.0)
renderer.set_exposure(1.1)  # > 0
```

**Policy:** Do lighting in **linear**. If the render target is **sRGB** (e.g., `Rgba8UnormSrgb`), **do not** apply manual gamma—hardware performs the sRGB encode. A simple **Reinhard** tonemap is applied in linear. Use manual gamma only when writing to a **linear** target.
<!-- T22-END:api -->

<!-- T11-BEGIN:mesh-notes -->
### Grid generator (T1.1)

```python
from vulkan_forge import grid_generate
import numpy as np

# Generate 4x3 grid with spacing (2.0, 1.0)
xy, uv, indices = grid_generate(nx=4, nz=3, spacing=(2.0, 1.0), origin="center")
# Returns:
# - xy: (12, 2) float32 array of world XY positions  
# - uv: (12, 2) float32 array of texture coordinates [0,1]
# - indices: (36,) uint32 array of triangle indices (CCW winding)

print(f"Grid vertices: {xy.shape} {xy.dtype}")       # (12, 2) float32
print(f"UV coords: {uv.shape} {uv.dtype}")          # (12, 2) float32  
print(f"Triangle indices: {indices.shape} {indices.dtype}")  # (36,) uint32

# Create heightmap and render terrain
heightmap = np.random.rand(128, 128).astype(np.float32) * 100.0
from vulkan_forge import Renderer
r = Renderer(512, 384)
r.add_terrain(heightmap, spacing=(1.0, 1.0), exaggeration=2.0, colormap="terrain")
r.render_triangle_png("my_terrain.png")
```

**Validation errors:**

The `grid_generate` function validates all parameters and raises `ValueError` with specific messages:

- `"nx must be >= 2 (got: {nx})"` - Grid dimensions too small
- `"nz must be >= 2 (got: {nz})"` - Grid dimensions too small  
- `"spacing values must be > 0 (got: {spacing})"` - Invalid spacing
- `"origin must be 'center' (got: '{origin}')"` - Unsupported origin mode
- `"spacing must be a 2-tuple (got: {type(spacing).__name__})"` - Wrong spacing type

- Grid is centered at origin in world XY (Z comes from height texture)
- UVs cover `[0,1]×[0,1]`; indices are CCW triangles; always uint32 dtype
<!-- T11-END:mesh-notes -->

<!-- T32-BEGIN:lighting -->
### Terrain lighting (WGSL, fragment)

- Normals from **forward differences** of the R32F height texture with `spacing=(dx,dy)` and `exaggeration` (see **Globals**), then **Lambert + small ambient**, color via 256×1 LUT with height normalization `[h_min,h_max]`.
- **Linear workflow**: lighting and tonemap in linear; write to **`Rgba8UnormSrgb`** so the hardware performs sRGB encoding (avoid double-gamma).

**Binding expectation (terrain path):**
- `@group(0)`: `Globals` uniform buffer
- `@group(1)`: height texture (R32F) + sampler
- `@group(2)`: LUT texture (RGBA8UnormSrgb) + sampler
<!-- T32-END:lighting -->

## Public API (Python)

```python
from vulkan_forge import Renderer, render_triangle_rgba, render_triangle_png, make_terrain

# Core triangle
arr = render_triangle_rgba(256, 256)              # (H,W,4) uint8
render_triangle_png("triangle.png", 256, 256)     # writes PNG

# Optional terrain (requires cargo feature 'terrain_spike')
# t = make_terrain(512, 384, grid=128)
# t.render_png("terrain.png")
```

Arguments are validated and raise `ValueError` with actionable messages if invalid.

<!-- T01-END:api -->

<!-- T01-BEGIN:add_terrain-doc -->
### Terrain upload (T0.1)

```python
from vulkan_forge import Renderer
import numpy as np

Z = np.random.rand(128, 128).astype("float32")     # (H,W), C-contiguous
r = Renderer(512, 512)
r.add_terrain(Z, spacing=(1.0, 1.0), exaggeration=1.0, colormap="viridis")
r.render_triangle_png("terrain_overlay.png")       # temporary writer
```

### Height texture upload (T1.2)

```python
from vulkan_forge import Renderer
import numpy as np

# Upload heightmap to GPU as R32Float texture with linear clamp sampler
Z = np.random.rand(128, 128).astype("float32")
r = Renderer(512, 512)
r.add_terrain(Z, spacing=(1.0, 1.0), exaggeration=1.0, colormap="viridis")
r.upload_height_r32f()                             # Create GPU texture
readback = r.read_full_height_texture()            # Roundtrip validation
```

Height texture is created as R32Float with usages `TEXTURE_BINDING | COPY_DST | COPY_SRC` and linear clamp sampler (Nearest/Nearest/Nearest). 256-byte row alignment is handled internally during transfer for robust upload of arbitrary (W,H) float heightmaps.

### Colormap LUT system (T1.3)

```python
from vulkan_forge import TerrainSpike, colormap_supported

# Discover available colormaps
print(colormap_supported())  # ["viridis","magma","terrain"]

# Create terrain renderer with specific colormap  
terrain = TerrainSpike(512, 384, grid=64, colormap="viridis")
terrain.render_png("output.png")

# Supported colormap names (case-sensitive)
terrain_magma = TerrainSpike(512, 384, grid=64, colormap="magma")
terrain_terrain = TerrainSpike(512, 384, grid=64, colormap="terrain")

# Default to viridis if not specified
terrain_default = TerrainSpike(512, 384, grid=64)  # Uses "viridis"
```

The colormap system uses 256×1 RGBA8 lookup textures (LUT) with embedded PNG assets and a central registry. Each LUT is sampled in the fragment shader to map normalized height values to colors. The supported colormap names are maintained in a central `SUPPORTED` list for consistent validation.

**Features:**
- Central `crate::colormap` registry with embedded 256×1 PNG assets
- GPU LUT texture (**RGBA8UnormSrgb** preferred) with linear clamp sampler  
- Proper `bytes_per_row` and `rows_per_image` handling for texture upload
- **Terrain FS bind groups (matches T3.2/T32):**  
  **group(0)** = `Globals` UBO, **group(1)** = *height* `R32F` texture **+ sampler**, **group(2)** = *LUT* `RGBA8UnormSrgb` **+ sampler**
- WGSL shader sampling with minimal lighting and colormap application
- Strict case-sensitive validation against central SUPPORTED list
- Debug toggle `VF_FORCE_LUT_UNORM=1` forces UNORM fallback for CI coverage

<!-- T01-END:add_terrain-doc -->

### T2.1 Camera & Uniforms

Robust camera math module with Python API providing right-handed, Y-up, -Z forward camera conventions (standard GL-style look-at).

```python
from vulkan_forge import camera_look_at, camera_perspective, camera_view_proj, TerrainSpike
import numpy as np

# Compute view matrix using standard conventions
eye = (0.0, 2.0, 5.0)      # Camera position
target = (0.0, 0.0, 0.0)   # Look-at target  
up = (0.0, 1.0, 0.0)       # Up vector (Y-up)
view = camera_look_at(eye, target, up)  # Returns (4,4) float32 array

# Compute projection matrix with clip space options
fovy_deg = 45.0            # Field of view in degrees
aspect = 16.0 / 9.0        # Width / height ratio
znear, zfar = 0.1, 100.0   # Near and far plane distances

proj_wgpu = camera_perspective(fovy_deg, aspect, znear, zfar, clip_space="wgpu")  # 0..1 Z (default)
proj_gl = camera_perspective(fovy_deg, aspect, znear, zfar, clip_space="gl")      # -1..1 Z

# Combined view-projection matrix
view_proj = camera_view_proj(eye, target, up, fovy_deg, aspect, znear, zfar, clip_space="wgpu")

print(f"View matrix shape: {view.shape}, dtype: {view.dtype}")        # (4, 4), float32
print(f"All arrays are C-contiguous: {view.flags.c_contiguous}")      # True
```

**TerrainSpike Integration:**

```python
# Set camera parameters on TerrainSpike for GPU rendering
terrain = TerrainSpike(512, 384, grid=128)

# Update camera and render
terrain.set_camera_look_at(
    eye=(1.0, 3.0, 4.0), 
    target=(0.0, 0.0, 0.0), 
    up=(0.0, 1.0, 0.0),
    fovy_deg=60.0, 
    znear=0.1, 
    zfar=100.0
)
terrain.render_png("camera_view.png")

# Debug: inspect current uniform buffer contents
uniforms = terrain.debug_uniforms_f32()  # Returns flat array of 44 floats (176 bytes)
view_matrix = uniforms[:16].reshape(4, 4, order='F')  # Extract view (column-major)
proj_matrix = uniforms[16:32].reshape(4, 4, order='F')  # Extract projection
```

**Coordinate System & Clip Spaces:**

- **View**: Right-handed, Y-up, forward -Z (glam::Mat4::look_at_rh)
- **Projection**: 
  - `clip_space="gl"`: Standard OpenGL [-1,1] Z range
  - `clip_space="wgpu"`: WGPU/Vulkan/Metal [0,1] Z range (default)
  - Conversion: `proj_wgpu = gl_to_wgpu() * proj_gl` where gl_to_wgpu() applies GL→WGPU depth remap
  - Internally, `TerrainSpike` uses WGPU clip space by default

**Parameter Validation:**

All functions validate inputs with precise error messages:
- `"fovy_deg must be finite and in (0, 180)"`
- `"znear must be finite and > 0"`  
- `"zfar must be finite and > znear"`
- `"aspect must be finite and > 0"`
- `"eye/target/up components must be finite"`
- `"up vector must not be colinear with view direction"`
- `"clip_space must be 'wgpu' or 'gl'"`

## Tools (CLI)

All tools live under `python/tools` and write JSON artifacts for CI.

### Determinism harness

Ensures repeated renders are byte-identical (raw RGBA).

```bash
python python/tools/determinism_harness.py --width 128 --height 128 --runs 5 --png --out-dir determinism_artifacts
# Prints JSON; writes determinism_artifacts/determinism_report.json (+ triangle.png)
```

### Cross-backend runner

Spawns a fresh Python process per backend; validates within-backend determinism; optional cross-backend compare.

```bash
# Windows/macOS example
python python/tools/backends_runner.py --runs 2 --png --out-dir backends_artifacts
```

### Device diagnostics

Enumerates adapters and probes device creation per backend.

```bash
python python/tools/device_diagnostics.py --json diag_out/device_diagnostics.json --summary
```

### Performance sanity

Times cold init and steady-state renders; optional budget/baseline enforcement.

```bash
# CI-safe (no enforcement)
python python/tools/perf_sanity.py --width 96 --height 96 --runs 20 --warmups 3 --json perf_out/perf_report.json
# Enforce budgets:
#   VF_ENFORCE_PERF=1 python python/tools/perf_sanity.py --baseline perf_out/perf_report.json
```

## Testing

```bash
python -m pip install -U pytest
# Build the extension first
maturin develop --release
pytest -q
```

Optional tests are gated by env:

* `VF_TEST_BACKENDS=1` for cross-backend test
* `VF_TEST_PERF=1` for performance test

**Terrain FS tests:**  
Set `VF_ENABLE_TERRAIN_TESTS=1` to enable the east↔west directional-lighting flip test once the terrain pipeline is wired in **T3.3**. By default this test is skipped in CI.

## CI

Matrix workflow: `.github/workflows/ci.yml`

* OS: Windows, Ubuntu, macOS × Python: 3.10–3.13
* Runs pytest, determinism harness (artifacts), and cross-backend runner on Windows/macOS.

## Troubleshooting

* **ImportError: No module named `_vulkan_forge`**
  Activate the same venv you used for `maturin develop`. Re-run:

  ```bash
  python -m pip install -U pip maturin
  maturin develop --release
  ```

* **Python 3.13 build errors (PyO3 0.21)**
  Set: `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1`

* **No suitable GPU adapter / unsupported backend**
  Try another backend or run the cross-backend runner to discover a working one.

## Versioning

* Current version: **0.0.3**
* See `CHANGELOG.md` for details.

<!-- T02-BEGIN:readme-dem -->
### DEM statistics & normalization

```python
import numpy as np
from vulkan_forge import dem_stats, dem_normalize, Renderer

Z = np.random.rand(256,256).astype(np.float32) * 1000.0
print("stats:", dem_stats(Z))
Z01 = dem_normalize(Z, mode="minmax", out_range=(0.0,1.0))

r = Renderer(800, 600)
r.add_terrain(Z, spacing=(1.0,1.0), exaggeration=1.0, colormap="viridis")
print("renderer stats:", r.terrain_stats())
r.normalize_terrain("zscore")
```
<!-- T02-END:readme-dem -->

## License

MIT (see `LICENSE`).

<!-- A1.11-END:readme -->
