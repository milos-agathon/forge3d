Zero-Copy NumPy Interoperability
====================================

This document describes forge3d's zero-copy NumPy interoperability features, which minimize memory copies when transferring data between Python and the Rust graphics core.

Overview
--------

Zero-copy interoperability allows NumPy arrays to share memory directly with forge3d's Rust implementation, eliminating unnecessary data copies that can impact performance and memory usage. This is particularly important for:

- **RGBA output**: Render results returned as NumPy arrays
- **Height input**: Heightmap data passed from NumPy to terrain rendering  
- **Large datasets**: Where copying would be expensive

Guaranteed Zero-Copy Pathways
------------------------------

RGBA Output
~~~~~~~~~~~

When rendering operations return RGBA data, forge3d uses zero-copy pathways where possible::

    import forge3d as f3d
    import numpy as np

    # Method 1: Via Renderer instance
    renderer = f3d.Renderer(512, 512)
    rgba_array = renderer.render_triangle_rgba()  # Zero-copy when possible

    # Method 2: Via module function  
    rgba_array = f3d.render_triangle_rgba(512, 512)  # Zero-copy when possible

The returned RGBA arrays have the following properties:

- **Shape**: ``(height, width, 4)`` with RGBA channels
- **Dtype**: ``numpy.uint8`` 
- **Memory layout**: C-contiguous (row-major)
- **Zero-copy**: When backing buffer allows direct access

Height Input
~~~~~~~~~~~~

For terrain heightmaps, forge3d can accept NumPy arrays with zero-copy transfer when they meet specific requirements::

    import numpy as np
    import forge3d as f3d
    
    # Create heightmap (zero-copy compatible)
    heightmap = np.random.rand(256, 256).astype(np.float32)
    
    # Ensure C-contiguous layout
    if not heightmap.flags['C_CONTIGUOUS']:
        heightmap = np.ascontiguousarray(heightmap)
    
    renderer = f3d.Renderer(512, 512)
    renderer.add_terrain(heightmap, spacing=(1.0, 1.0), exaggeration=2.0, colormap="viridis")

Zero-Copy Requirements for Height Input
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To ensure zero-copy transfer for heightmap data:

**Required Properties**:

- **Dtype**: ``numpy.float32`` (preferred) or ``numpy.float64`` 
- **Layout**: C-contiguous (``arr.flags['C_CONTIGUOUS']`` must be ``True``)
- **Dimensions**: 2D array with shape ``(height, width)``

**Performance Notes**:

- ``float32`` arrays use direct zero-copy transfer
- ``float64`` arrays are converted to ``float32`` (involves a copy)
- Non-contiguous arrays are converted to contiguous (involves a copy)

Validation and Diagnostics
---------------------------

Validation Tools
~~~~~~~~~~~~~~~~

forge3d provides validation utilities in the ``forge3d._validate`` module::

    from forge3d._validate import (
        ptr, is_c_contiguous, validate_zero_copy_path, 
        check_zero_copy_compatibility
    )
    
    # Check array properties
    heightmap = np.random.rand(128, 128).astype(np.float32)
    
    # Get memory pointer
    memory_ptr = ptr(heightmap)
    print(f"Array data pointer: 0x{memory_ptr:x}")
    
    # Check contiguity 
    contiguous = is_c_contiguous(heightmap)
    print(f"Is C-contiguous: {contiguous}")
    
    # Check zero-copy compatibility
    compat = check_zero_copy_compatibility(heightmap, "heightmap")
    if not compat['compatible']:
        print(f"Issues: {compat['issues']}")

Profiling Tool
~~~~~~~~~~~~~~

The ``python/tools/profile_copies.py`` tool measures and validates zero-copy behavior::

    # Run zero-copy profiler
    python python/tools/profile_copies.py --render-size 1024x1024 --terrain-size 512x512
    
    # Expected output on success:
    # ✓ zero-copy OK

The profiler validates both RGBA output and height input zero-copy pathways by comparing memory pointers between NumPy arrays and their Rust backing stores.

Test Validation
~~~~~~~~~~~~~~~

Zero-copy behavior is verified through comprehensive tests in ``tests/test_numpy_interop.py`` using pointer equality assertions::

    # RGBA output validation
    rgba_array, rust_ptr = renderer.render_triangle_rgba_with_ptr()
    numpy_ptr = rgba_array.ctypes.data
    assert numpy_ptr == rust_ptr  # Pointer equality = zero-copy confirmed
    
    # Height input validation  
    renderer.add_terrain(heightmap, ...)
    captured_ptr = renderer.debug_last_height_src_ptr()
    input_ptr = heightmap.ctypes.data
    assert input_ptr == captured_ptr  # Zero-copy confirmed

Best Practices
--------------

For Optimal Zero-Copy Performance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Use float32 for heightmaps**::

    # Preferred - direct zero-copy
    heightmap = np.random.rand(256, 256).astype(np.float32)

2. **Ensure C-contiguous layout**::

    # Check and fix contiguity
    if not heightmap.flags['C_CONTIGUOUS']:
        heightmap = np.ascontiguousarray(heightmap)

3. **Pre-allocate when possible**::

    # Create arrays with the right properties from the start
    heightmap = np.zeros((height, width), dtype=np.float32, order='C')

4. **Validate critical paths**::

    from forge3d._validate import check_zero_copy_compatibility
    
    compat = check_zero_copy_compatibility(heightmap, "heightmap")
    if not compat['compatible']:
        print(f"Zero-copy issues: {compat['issues']}")

Error Handling
~~~~~~~~~~~~~~

Common zero-copy violations and their solutions:

**Non-contiguous arrays**::

    # Error: "array must be C-contiguous"
    # Solution:
    heightmap = np.ascontiguousarray(heightmap)

**Wrong dtype**::

    # Error: "dtype must be float32 or float64"  
    # Solution:
    heightmap = heightmap.astype(np.float32)

**Wrong dimensions**::

    # Error: "heightmap must be 2-D"
    # Solution: reshape or use correct array
    heightmap = heightmap.reshape((height, width))

Performance Impact
------------------

Zero-copy pathways provide significant benefits:

**Memory Efficiency**:

- Eliminates duplicate copies of large arrays
- Reduces peak memory usage during operations
- Enables processing of larger datasets

**Performance Benefits**:

- Faster transfer times (no copying overhead)
- Reduced memory bandwidth usage
- Lower latency for frequent operations

**Scaling Characteristics**:

- Benefits increase with array size
- Critical for real-time applications  
- Enables efficient streaming of large terrain data

Implementation Notes
--------------------

Technical Details
~~~~~~~~~~~~~~~~~

The zero-copy implementation uses:

- **PyO3 integration**: Direct access to NumPy array data pointers
- **Memory layout validation**: Ensures compatible strides and alignment
- **Rust slice views**: Zero-copy access to contiguous array data
- **GPU buffer mapping**: Direct CPU access to GPU-allocated memory where supported

Platform Considerations
~~~~~~~~~~~~~~~~~~~~~~~

Zero-copy behavior may vary across:

- **Operating systems**: Windows, Linux, macOS  
- **GPU backends**: Vulkan, DirectX 12, Metal
- **Device types**: Integrated vs discrete GPUs
- **Driver versions**: May affect memory mapping capabilities

The profiling tool helps identify platform-specific behavior and validate zero-copy paths on target systems.