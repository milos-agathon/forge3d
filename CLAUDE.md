# CLAUDE Knowledge: Repository Overview

> This file helps AI assistants quickly understand and navigate the repo. Keep it accurate and succinct.

## Overview

forge3d is a headless GPU rendering library with Python bindings, providing PNG↔NumPy utilities, terrain rendering, vector graphics, and comprehensive GPU-accelerated visualization capabilities. Built with Rust + PyO3 + wgpu for cross-platform compatibility.

## Quick Facts
- **Languages**: Rust (core), Python (bindings), WGSL (shaders)
- **Build**: cargo (workspace), maturin (Python bindings), pytest (tests)
- **OS/Arch**: win_amd64 • linux_x86_64 • macos_universal2
- **GPU Constraint**: ≤ 512 MiB host-visible heap (documented where applicable)
- **Entry Points**: `forge3d` Python module, `policies` binary, examples

## Repo Map
| Path | Purpose | Key Files |
|------|---------|-----------|
| src/ | Main Rust source code with core rendering, terrain, vector graphics | `lib.rs`, `renderer.rs`, `terrain/mod.rs`, `shaders/*.wgsl` |
| python/ | Python package with PyO3 bindings and tools | `forge3d/__init__.py`, `tools/device_diagnostics.py` |
| tests/ | Comprehensive test suite with pytest configuration | `conftest.py`, `test_*.py` |
| examples/ | Python example scripts demonstrating API usage | `terrain_single_tile.py`, `triangle_png.py` |
| docs/ | Sphinx documentation with RST files | `conf.py`, `index.rst` |
| assets/ | Static assets including colormaps and logos | `colormaps/*.png`, `logo-*.png` |
| data/ | Color palette data files | `*.rgba` palette files |

## Build & Test

### Rust
```bash
# workspace build
cargo build --workspace
cargo test --workspace
cargo build --release
cargo doc
```

### Python (PyO3 via maturin)
```bash
pip install -U maturin
maturin develop --release
pytest -q
```

### Complete Setup
```bash
# Complete quick start sequence  
pip install -U maturin
maturin develop --release
python examples/terrain_single_tile.py
# -> Creates terrain_single_tile.png (512×512 image)
```

## Runbook

* **Terrain Demo**: `python examples/terrain_single_tile.py` - Creates procedural terrain with viridis colormap
* **Triangle Demo**: `python examples/triangle_png.py` - Basic rendering test
* **PNG/NumPy Demo**: `python examples/png_numpy_roundtrip.py` - Image I/O utilities
* **Diagnostics**: `python examples/diagnostics.py` - GPU adapter detection
* **Performance Testing**: `python examples/run_bench.py` or CLI `vf-bench`

## CI/CD

* Workflows location: `git/hooks/` (git hooks)
* CI scripts: `ci/run_benches.sh`
* No GitHub Actions workflows detected

## Architecture Sketch

* **Core**: Rust library with wgpu-based GPU rendering pipeline
* **Bindings**: PyO3 integration exposing Rust functionality to Python
* **Data Flow**: NumPy arrays ↔ Rust (zero-copy) ↔ GPU textures/buffers
* **Rendering Pipeline**: Vertex/fragment shaders (WGSL) → GPU compute → PNG/RGBA output
* **Components**:
  - **Renderer**: Basic triangle/terrain rendering
  - **Scene**: High-level terrain visualization with camera controls  
  - **Vector Graphics**: Polygons, lines, points with anti-aliasing
  - **Terrain**: DEM processing, LOD, tiling, analysis
  - **Colormaps**: Built-in color schemes (viridis, magma, terrain)

## Feature Flags & Profiles

**Cargo Features**:
- `default = ["extension-module"]`
- `extension-module`: PyO3 Python bindings
- `terrain_spike`: Terrain-specific functionality
- `weighted-oit`: Order Independent Transparency
- `wsI_bigbuf`: Big buffer workstream feature
- `wsI_double_buf`: Double buffering workstream feature

**Environment Variables**:
- `WGPU_BACKENDS`: Constrain GPU backends (VULKAN, METAL, DX12, GL)
- `VF_ENABLE_TERRAIN_TESTS`: Enable terrain-specific test suite
- `RUST_LOG`: Logging level control

**Profiles**:
- Release build: `codegen-units = 1`, `lto = "thin"` for optimization

## Troubleshooting

* **Missing native module**: run `maturin develop --release` then retry `pytest -q`
* **Windows build tools**: install MSVC Build Tools for Rust
* **GPU compatibility**: Ensure Vulkan/Metal/DX12/GL backend available
* **Memory issues**: Check GPU has sufficient host-visible memory (≤512 MiB budget)
* **Array contiguity**: PNG write and terrain upload arrays must be C-contiguous
* **Backend selection**: Use `WGPU_BACKENDS=VULKAN` to force specific GPU backend

## Docs & Roadmap

* **Docs**: `docs/` (Sphinx-based with RST files)
* **Build docs**: `cd docs && make html`
* **Roadmap**: `roadmap.csv`  
* **Audits**: `reports/audit_I.md`
* **Changelog**: `CHANGELOG.md`

## Glossary

* **DEM**: Digital Elevation Model (height field data)
* **OIT**: Order Independent Transparency 
* **WGSL**: WebGPU Shading Language
* **LOD**: Level of Detail
* **PyO3**: Python bindings for Rust
* **maturin**: Build tool for Rust-Python projects
* **wgpu**: Cross-platform GPU abstraction library
* **Host-visible memory**: GPU memory accessible from CPU (budget: ≤512 MiB)