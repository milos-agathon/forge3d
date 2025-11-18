# forge3d Project Overview

`forge3d` is a headless GPU rendering library primarily written in Rust, leveraging `wgpu` for cross-platform GPU access, and providing Python bindings via `PyO3`. It focuses on 3D rendering, scientific visualization (especially for geospatial data like Digital Elevation Models - DEMs), and GPU-accelerated computing.

## Key Features

*   **Rendering:** Physically-Based Rendering (PBR), Image-Based Lighting (IBL), shadow mapping, Screen-Space Global Illumination (SSGI), Screen-Space Reflections (SSR).
*   **Geometry:** Signed Distance Fields (SDFs) for primitive and constructive solid geometry (CSG) operations.
*   **Vector Graphics:** Rendering of points, lines, and polygons with advanced techniques like Weighted Order-Independent Transparency (OIT) and picking.
*   **Scientific Visualization:** Strong support for terrain rendering from DEMs, custom colormaps, and integration with `numpy` for data handling.
*   **GPU Utilization:** Built on `wgpu` for efficient, cross-platform GPU access (Vulkan, Metal, DX12, OpenGL backends).
*   **Python Integration:** Comprehensive Python APIs with type annotations, enabling easy scripting and integration into data science workflows.
*   **Interactive Viewer:** Provides an interactive windowed viewer for real-time 3D exploration.

## Technologies Used

*   **Rust:** Core rendering engine and low-level GPU logic.
*   **PyO3:** Python bindings for Rust code.
*   **wgpu:** Cross-platform GPU API for rendering.
*   **Python:** High-level API, scripting, and scientific computing integration.
*   **NumPy:** Essential for data exchange and manipulation between Python and Rust.
*   **Maturin:** Build system for Rust-Python interoperability.
*   **WGSL:** WebGPU Shading Language for GPU shaders.

## Building and Running

### Building from Source

To build the `forge3d` Python extension from source, ensure you have Rust and Python installed, then run:

```bash
pip install -U maturin
maturin develop --release
```

### Running Examples

The `examples/` directory contains various scripts demonstrating `forge3d`'s capabilities.

**Terrain Rendering Example:**

```bash
python examples/geopandas_demo.py --output-size 1200 900 --lighting-type "blinn-phong" --lighting-intensity 1 --lighting-azimuth 315 --lighting-elevation 315 --shadow-intensity 1 --contrast-pct 0.5 --gamma 0.5 --camera-theta 90
```

**Interactive Viewer:**

```python
import forge3d as f3d
f3d.open_viewer(width=1280, height=720, title="My Scene")
```

### Testing

The project uses `pytest` for its test suite. To run tests:

```bash
pytest
```

## Development Conventions

*   **Language Priority:** Rust for performance-critical components and GPU interaction; Python for user-facing APIs, data processing, and scripting.
*   **Interoperability:** Seamless integration between Rust and Python through `PyO3` and `maturin`.
*   **GPU API Abstraction:** Uses `wgpu` to provide a unified API across different native GPU backends.
*   **Documentation:** Utilizes Sphinx with `autodoc` for Python API reference generated from `.rst` files.
*   **Data Handling:** Relies heavily on `numpy` arrays for efficient data transfer and numerical operations in Python.
*   **Memory Management:** Adheres to an explicit 512 MiB host-visible GPU memory budget, with tools and APIs for tracking and enforcing it.
*   **Error Handling:** Employs a hierarchy of custom exceptions (e.g., `RenderError`, `MemoryError`) for robust error reporting.
*   **Configuration:** Runtime behavior can be configured via environment variables (e.g., `WGPU_BACKENDS` for backend selection, `RUST_LOG` for logging).
*   **Code Style:** Rust code follows idiomatic Rust practices; Python code uses type hints for clarity and maintainability.
