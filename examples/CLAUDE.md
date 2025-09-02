# CLAUDE Knowledge: examples/

## Purpose

Demonstration scripts showcasing forge3d API capabilities including terrain rendering, PNG/NumPy utilities, vector graphics, diagnostics, and performance benchmarking. All examples are self-contained with minimal dependencies.

## Public API / Entry Points

* **Terrain rendering**: `terrain_single_tile.py`, `scene_terrain_demo.py`
* **Basic rendering**: `triangle_png.py`  
* **PNG/NumPy utilities**: `png_numpy_roundtrip.py`
* **System diagnostics**: `diagnostics.py`
* **Grid generation**: `grid_generate_demo.py`
* **Benchmarking**: `run_bench.py`
* **Performance testing**: `perf/split_vs_single_bg.py` (Rust), `perf/split_vs_single_bg.rs`

## Build & Test (component-scoped)

```bash
# Quick start - terrain example (< 2 minutes)
python examples/terrain_single_tile.py
# -> Creates terrain_single_tile.png

# Other examples
python examples/triangle_png.py
python examples/png_numpy_roundtrip.py
python examples/diagnostics.py
python examples/grid_generate_demo.py
python examples/run_bench.py

# Performance examples
cd examples/perf
python split_vs_single_bg.py
# or run the Rust version:
# cargo run --example split_vs_single_bg --release
```

## Important Files

* `_import_shim.py` - Shared utility for repository-local imports
* `terrain_single_tile.py` - **Primary demo**: procedural terrain with viridis colormap
* `scene_terrain_demo.py` - Advanced terrain scene with camera controls
* `triangle_png.py` - Basic GPU rendering test
* `png_numpy_roundtrip.py` - Image I/O demonstration
* `diagnostics.py` - GPU adapter detection and system info
* `grid_generate_demo.py` - Mesh generation utilities
* `run_bench.py` - Performance benchmarking tool
* `perf/` - Performance testing utilities (Python + Rust)
* `terrain_normalize_demo.py` - DEM normalization workflows

## Dependencies

* **Internal**: forge3d package with all core functionality
* **External**: 
  - numpy (array operations)
  - pathlib (file operations, built-in)
  - time (performance timing, built-in)

## Gotchas & Limits

* **GPU requirement**: Most examples require compatible GPU adapter
* **Import shim**: Examples use `_import_shim.py` for running from repository root
* **Output files**: Examples create PNG files in current working directory
* **Deterministic seeds**: Examples use fixed seeds for reproducible output
* **Error handling**: Examples gracefully handle missing GPU with informative messages
* **Memory budget**: Respect ≤512 MiB host-visible memory constraint

## Common Tasks

* **Add new example**: Follow existing patterns with import shim and error handling
* **Test example**: Ensure it works both from repo root and examples/ directory
* **Debug GPU issues**: Run `diagnostics.py` first to check adapter availability
* **Performance testing**: Use `run_bench.py` for timing analysis
* **Visual output**: Examples create PNG files that can be inspected manually

## Ownership / TODOs

* **Coverage**: Examples demonstrate all major API surfaces
* **Documentation**: Each example includes docstring explaining purpose and expected output
* **Platform compatibility**: Examples work across Windows, Linux, macOS
* **CI integration**: Key examples run in automated testing to ensure they don't break