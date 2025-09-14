Installation Guide
==================

This guide covers different ways to install forge3d depending on your needs.

Quick Installation
------------------

The easiest way to install forge3d is via pip:

.. code-block:: bash

    pip install forge3d

This will install pre-built wheels for:

* **Windows**: win_amd64
* **Linux**: linux_x86_64
* **macOS**: macos_universal2 (Intel + Apple Silicon)

System Requirements
-------------------

**Minimum Requirements:**

* Python 3.10 or later
* 4 GB RAM
* Graphics card with Vulkan 1.2, DirectX 12, or Metal support

**Supported Platforms:**

* Windows 10/11 (x64)
* Linux (x86_64) with glibc 2.17+
* macOS 10.15+ (Intel and Apple Silicon)

**GPU Support:**

* **Primary**: Vulkan 1.2 compatible graphics cards
* **Secondary**: DirectX 12 (Windows), Metal (macOS), OpenGL 4.5+ (fallback)
* **Fallback**: Software rendering (limited functionality)

Installation from Source
-------------------------

For development or if pre-built wheels aren't available:

**Prerequisites:**

.. code-block:: bash

    # Install Rust toolchain
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    source ~/.cargo/env
    
    # Install maturin (Rust-Python build tool)
    pip install maturin[patchelf]

**Build and Install:**

.. code-block:: bash

    git clone https://github.com/anthropics/forge3d.git
    cd forge3d
    
    # Development build (faster, includes debug info)
    maturin develop
    
    # Release build (optimized, slower to build)
    maturin develop --release

**Alternative: CMake Build**

For integration with CMake-based projects:

.. code-block:: bash

    mkdir build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    cmake --build .
    
    # Install Python package
    cd ..
    pip install -e .

Virtual Environment Setup
--------------------------

**Using venv:**

.. code-block:: bash

    python -m venv forge3d-env
    
    # Windows
    forge3d-env\Scripts\activate
    
    # Linux/macOS
    source forge3d-env/bin/activate
    
    pip install forge3d

**Using conda:**

.. code-block:: bash

    conda create -n forge3d python=3.11
    conda activate forge3d
    pip install forge3d

Verification
------------

Test your installation:

.. code-block:: python

    import forge3d as f3d
    
    # Check version
    print(f"forge3d version: {f3d.__version__}")
    
    # Check GPU availability
    print(f"GPU available: {f3d.has_gpu()}")
    
    # List adapters
    adapters = f3d.enumerate_adapters()
    print(f"Graphics adapters: {len(adapters)}")
    for i, adapter in enumerate(adapters):
        print(f"  {i}: {adapter}")
    
    # Basic functionality test
    try:
        renderer = f3d.Renderer(64, 64)
        image = renderer.render_triangle_rgba()
        print("✓ Basic rendering works")
    except Exception as e:
        print(f"✗ Rendering failed: {e}")

Troubleshooting
---------------

**"No GPU acceleration available"**

* Verify your graphics drivers are up to date
* Check that Vulkan/DirectX 12/Metal is supported
* Try setting ``WGPU_BACKENDS`` environment variable:

.. code-block:: bash

    # Force specific backend
    export WGPU_BACKENDS=VULKAN    # Linux
    set WGPU_BACKENDS=VULKAN       # Windows CMD
    $env:WGPU_BACKENDS="VULKAN"    # Windows PowerShell

**Import errors on Linux**

.. code-block:: bash

    # Install system dependencies
    sudo apt-get install libegl1-mesa-dev libgl1-mesa-dev libxcb1-dev

**Compilation errors from source**

* Ensure Rust toolchain is up to date: ``rustup update``
* Check that you have Python development headers: ``sudo apt-get install python3-dev``
* On Windows, install Microsoft C++ Build Tools

**Performance issues**

* Ensure you're using the release build: ``maturin develop --release``
* Check that GPU acceleration is enabled: ``f3d.has_gpu()``
* Consider the 512 MiB GPU memory budget for large datasets

Development Setup
-----------------

For contributors and advanced users:

.. code-block:: bash

    git clone https://github.com/anthropics/forge3d.git
    cd forge3d
    
    # Install development dependencies
    pip install -r requirements-dev.txt
    
    # Install pre-commit hooks
    pre-commit install
    
    # Run tests
    pytest -v
    
    # Build documentation
    cd docs
    make html

**Environment Variables for Development:**

* ``VF_ENABLE_TERRAIN_TESTS=1`` - Enable GPU-dependent tests
* ``RUST_LOG=debug`` - Enable detailed logging
* ``WGPU_BACKENDS=VULKAN`` - Force specific GPU backend

Docker Installation
-------------------

For containerized environments:

.. code-block:: dockerfile

    FROM python:3.11-slim
    
    # Install system dependencies
    RUN apt-get update && apt-get install -y \
        curl \
        build-essential \
        libegl1-mesa-dev \
        libgl1-mesa-dev \
        libxcb1-dev \
        && rm -rf /var/lib/apt/lists/*
    
    # Install forge3d
    RUN pip install forge3d
    
    # Test installation
    RUN python -c "import forge3d; print('forge3d:', forge3d.__version__)"

.. note::
   GPU acceleration may not be available in all Docker environments.
   Consider using nvidia-docker or similar for GPU support.