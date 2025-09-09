Examples Guide
==============

This guide covers the comprehensive examples included with forge3d, organized from basic to advanced.

Basic Examples
--------------

**triangle_png.py**
Simple triangle rendering demonstration:

.. code-block:: bash

    python examples/triangle_png.py

Creates a basic colored triangle and saves it as a PNG file.

**png_numpy_roundtrip.py**
Image I/O utilities demonstration:

.. code-block:: bash

    python examples/png_numpy_roundtrip.py

Shows how to load PNG files to NumPy arrays and save NumPy arrays as PNG files.

**terrain_single_tile.py**
Basic terrain rendering with colormaps:

.. code-block:: bash

    python examples/terrain_single_tile.py

Generates a synthetic terrain and renders it with the viridis colormap.

Diagnostic Examples
-------------------

**diagnostics.py**
GPU detection and system information:

.. code-block:: bash

    python examples/diagnostics.py

Provides detailed information about your GPU adapters and system capabilities.

**device_capability_probe.py**
Comprehensive GPU analysis and capability testing:

.. code-block:: bash

    python examples/device_capability_probe.py

Advanced example that tests all major forge3d features and generates a detailed capability report.

Advanced Examples (R13)
------------------------

These examples demonstrate sophisticated rendering techniques and were developed as part of the audit remediation:

**Terrain + Shadows + PBR Integration**

.. code-block:: bash

    python examples/advanced_terrain_shadows_pbr.py

Features:
- Procedural terrain generation with multiple noise octaves
- PBR material system with metallic-roughness workflow
- High-quality shadow mapping with cascade optimization
- Realistic lighting and material properties

**Contour Overlay Demonstration**

.. code-block:: bash

    python examples/contour_overlay_demo.py

Features:
- Topographic contour line generation from height data
- Vector graphics integration with terrain rendering
- Major and minor contour styling
- Supports both scipy-based and simplified contour algorithms

**HDR Tone Mapping Comparison**

.. code-block:: bash

    python examples/hdr_tonemap_comparison.py

Features:
- Wide dynamic range scene generation
- Multiple tone mapping operators (ACES, Reinhard, exposure, filmic)
- Side-by-side comparison rendering
- HDR visualization with false color mapping

**Vector Order-Independent Transparency**

.. code-block:: bash

    python examples/vector_oit_layering.py

Features:
- Multiple transparent vector layers
- Order-independent transparency techniques
- Complex overlay scenarios with depth sorting
- Performance comparison between rendering approaches

**Normal Mapping on Terrain**

.. code-block:: bash

    python examples/normal_mapping_terrain.py

Features:
- Surface normal calculation from height fields
- Normal map generation and visualization
- Lighting with and without normal mapping comparison
- Surface roughness and curvature analysis

**IBL Environment Lighting**

.. code-block:: bash

    python examples/ibl_env_lighting.py

Features:
- Procedural HDRI environment map generation
- Spherical harmonics coefficient calculation
- Image-based lighting with diffuse approximation
- Environment lighting comparison with test geometry

**Multi-threaded Command Recording**

.. code-block:: bash

    python examples/multithreaded_command_recording.py

Features:
- Parallel rendering task execution
- ThreadPoolExecutor and producer-consumer patterns
- Performance comparison between threading approaches
- Thread utilization analysis and metrics

**Async Compute Prepass**

.. code-block:: bash

    python examples/async_compute_prepass.py

Features:
- Depth buffer precomputation for early-Z optimization
- Object culling and overdraw reduction
- Performance comparison with and without prepass
- Memory efficiency analysis

**Large Texture Upload Policies**

.. code-block:: bash

    python examples/large_texture_upload_policies.py

Features:
- Multiple texture upload strategies (naive, tiled, mipmap, streaming)
- Memory constraint enforcement (512 MiB budget)
- LRU eviction and streaming policies
- Performance and memory utilization comparison

Performance Examples
--------------------

**run_bench.py**
Benchmarking utilities:

.. code-block:: bash

    python examples/run_bench.py

Provides timing analysis and performance metrics for various operations.

**Performance Monitoring**
Many examples include built-in performance monitoring:

.. code-block:: python

    # Examples generate metrics files
    cat out/example_metrics.json

These JSON files contain detailed performance and configuration data for reproducible analysis.

Running Examples
----------------

**Prerequisites:**

.. code-block:: bash

    # Ensure forge3d is installed
    pip install forge3d
    
    # Or build from source
    maturin develop --release

**Output Directory:**
Most examples create an ``out/`` directory with:

- Generated images (PNG format)
- Metrics files (JSON format) 
- Performance data and analysis results

**GPU Requirements:**
Some advanced examples work better with GPU acceleration:

.. code-block:: python

    import forge3d as f3d
    if f3d.has_gpu():
        print("GPU acceleration available")
    else:
        print("Running in fallback mode")

**Memory Considerations:**
Examples respect the 512 MiB GPU memory budget and include fallbacks for memory-constrained environments.

Understanding Example Output
----------------------------

**Image Files:**
All examples produce PNG images with standard naming conventions:

- ``example_name.png`` - Main output
- ``example_name_comparison.png`` - Side-by-side comparisons
- ``example_name_analysis.png`` - Visualizations and charts

**Metrics Files:**
JSON files with consistent structure:

.. code-block:: json

    {
        "configuration": {
            "size": [800, 600],
            "parameters": {...}
        },
        "performance": {
            "render_time_ms": 42.5,
            "memory_used_mb": 128.3
        },
        "outputs": {
            "main_image": "/path/to/output.png"
        }
    }

**Console Output:**
Examples provide detailed logging:

- Configuration parameters
- Performance metrics
- Feature availability
- Error handling and fallbacks

Example Development Guidelines
------------------------------

When creating new examples:

1. **Self-contained**: No external data dependencies
2. **Deterministic**: Use fixed seeds for reproducible output  
3. **Documented**: Clear docstrings and inline comments
4. **Robust**: Handle missing features gracefully
5. **Metrics**: Generate JSON metrics for analysis
6. **Performance**: Include timing and memory usage data

**Template Structure:**

.. code-block:: python

    #!/usr/bin/env python3
    """
    Example Name: Brief Description
    
    Detailed description of what this example demonstrates.
    """
    
    import numpy as np
    import sys
    from pathlib import Path
    
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    def main():
        print("Example Name")
        print("=" * 20)
        
        out_dir = Path(__file__).parent.parent / "out"
        out_dir.mkdir(exist_ok=True)
        
        try:
            import forge3d as f3d
            # Example implementation
            return 0
        except Exception as e:
            print(f"Error: {e}")
            return 1
    
    if __name__ == "__main__":
        sys.exit(main())

For more details on any specific example, run it with detailed output or check the source code in the ``examples/`` directory.