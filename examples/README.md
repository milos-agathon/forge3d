Examples
Examples (Python)

Overview
- All examples are exposed as Python entry points in this directory to keep user-facing workflows in Python.
- GPU-backed examples are gated by environment variables and fall back to build-only dry runs in CI without GPUs.

Workstream M (plot_py Adapters)
- M1: Matplotlib Adapter (Image)
  - File: `examples/m1_mpl_image_demo.py`
  - Usage:
    - `python examples/m1_mpl_image_demo.py --dpi 150 --out reports/m1_mpl_image.png`
    - Options: `--axes` (tight bbox rasterization), `--heightmap` (save luminance→height PNG)
  - Optional deps: `pip install matplotlib pillow`

- M2: Matplotlib Adapter (Data)
  - File: `examples/m2_mpl_data_demo.py`
  - Usage:
    - `python examples/m2_mpl_data_demo.py --outdir reports/m2_mpl_data_demo`
    - Options: `--save-obj` (requires native extension), `--dpi 150`
  - Optional deps: `pip install matplotlib`

- M3: GeoPandas Adapter
  - File: `examples/m3_geopandas_demo.py`
  - Usage:
    - `python examples/m3_geopandas_demo.py --outdir reports/m3_geopandas_demo --save-obj`
  - Optional deps: `pip install geopandas shapely pyproj pillow`

- M4: Rasterio/Xarray Adapter
  - File: `examples/m4_raster_xarray_demo.py`
  - Usage:
    - GeoTIFF path: `python examples/m4_raster_xarray_demo.py --geotiff path/to.tif --out reports/m4_raster_xarray.png`
    - Synthetic xarray: `python examples/m4_raster_xarray_demo.py --synthetic --out reports/m4_raster_xarray.png`
  - Optional deps: `pip install rasterio xarray rioxarray pillow`

- M5: Cartopy Integration
  - File: `examples/m5_cartopy_demo.py`
  - Usage:
    - `python examples/m5_cartopy_demo.py --dpi 150 --out reports/m5_cartopy.png`
  - Optional deps: `pip install cartopy matplotlib pillow`

- M6: Seaborn/Plotly Convenience
  - File: `examples/m6_charts_demo.py`
  - Usage:
    - Plotly: `python examples/m6_charts_demo.py --backend plotly --width 800 --height 600 --out reports/m6_charts.png`
    - Seaborn: `python examples/m6_charts_demo.py --backend seaborn --dpi 150 --out reports/m6_charts.png`
  - Optional deps: `pip install plotly kaleido pillow seaborn matplotlib`

GPU Gating
- Set these env vars to 1 to enable GPU runs:
  - FORGE3D_RUN_WAVEFRONT: enables wavefront path tracer examples
  - FORGE3D_CI_GPU: enables GPU perf benchmark execution
- Without these, examples perform a cargo build-only dry run and skip GPU execution.

Wavefront Path Tracer
- File: `examples/wavefront_instances.py`
- Usage:
  - Baseline (build-only without GPU env):
    `python examples/wavefront_instances.py`
  - With hair (and GPU enabled):
    `FORGE3D_RUN_WAVEFRONT=1 python examples/wavefront_instances.py --hair-demo --hair-width=0.05 --hair-mat=1`
- Useful flags:
  - `--restir`, `--restir-debug`, `--restir-spatial`
  - `--swap-materials`, `--skinny-blas1`, `--camera-jitter=<f>`, `--force-blas=<0|1>`
  - `--dump-aov-depth=<path>`, `--dump-aov-albedo=<path>`, `--dump-aov-normal=<path>`, `--dump-aov-with-header`
  - `--medium-enable`, `--medium-g=<f>`, `--medium-sigma-t=<f>`, `--medium-density=<f>`
  - `--compute-ao`, `--ao-samples=<u>`, `--ao-intensity=<f>`, `--ao-bias=<f>`
  - Hair: `--hair-demo`, `--hair-width=<f>`, `--hair-mat=<0|1>`

Hair Demo (Focused)
- File: `examples/hair_demo.py`
- Usage:
  - Build-only dry run (no GPU):
    `python examples/hair_demo.py --dry-run`
  - Run with GPU:
    `FORGE3D_RUN_WAVEFRONT=1 python examples/hair_demo.py --enable-hair --hair-width=0.04 --hair-mat=0`
- Flags: `--enable-hair`, `--hair-width=<f>`, `--hair-mat=<0|1>`, `--output=<png>`, `--restir`, `--restir-debug`, `--restir-spatial`, `--dry-run`.

Performance Benchmark (I6)
- File: `examples/perf/split_vs_single_bg.py`
- Usage:
  - Build-only (no GPU):
    `python examples/perf/split_vs_single_bg.py --frames 10 --objects 100 --out artifacts/perf/ci.csv`
  - With GPU:
    `FORGE3D_CI_GPU=1 python examples/perf/split_vs_single_bg.py --frames 300 --objects 2000 --out artifacts/perf/I6.csv`
- Validates CSV structure and prints split vs single bind group performance.

CMake Integration
- File: `examples/cmake_integration.py`
- Usage:
  - Configure and build:
    `python examples/cmake_integration.py --build-dir build/cmake_integration`
  - Use installed package flow:
    `python examples/cmake_integration.py --no-subdir`
  - Run C++ demo:
    `python examples/cmake_integration.py --run-cpp`

CI
- A workflow `/.github/workflows/python_examples.yml` runs these Python examples in CI.
  - Default job: build-only dry runs for GPU examples.
  - Optional GPU job: set repository/env `ENABLE_GPU=1` or run manually via `workflow_dispatch` input `enable_gpu=true`.

Notes
- Outputs are written to `out/` or `artifacts/` and are gitignored.

