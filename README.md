<!-- A1.11-BEGIN:readme -->
# vulkan-forge

Headless, deterministic triangle renderer built on **wgpu** with a **PyO3** Python API.  
Status: pre-0.1 (research/prototyping). Latest release: **0.0.3**.

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

<!-- T01-BEGIN:api -->
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

<!-- T01-END:add_terrain-doc -->

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

## License

MIT (see `LICENSE`).

<!-- A1.11-END:readme -->
