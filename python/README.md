<!-- A1.11-BEGIN:python-readme -->
# forge3d (Python)

Public Python API for the forge3d extension.

## Install (dev)
```bash
python -m venv .venv
source .venv/Scripts/activate    # Windows Git Bash
# or .\.venv\Scripts\Activate.ps1 (PowerShell)
python -m pip install -U pip maturin numpy
export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1  # only needed on Python 3.13
maturin develop --release
```

## Usage
```python
from forge3d import Renderer, render_triangle_rgba, render_triangle_png
r = Renderer(256, 256)
arr = render_triangle_rgba(256, 256)
render_triangle_png("triangle.png", 256, 256)
```

See also CLI tools in `python/tools/`:
- `determinism_harness.py`
- `backends_runner.py`
- `device_diagnostics.py`
- `perf_sanity.py`
<!-- A1.11-END:python-readme -->