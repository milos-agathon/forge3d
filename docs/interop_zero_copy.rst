Zero-Copy NumPy Interoperability
=================================

forge3d provides zero-copy interoperability between NumPy arrays and the underlying Rust GPU memory system, enabling efficient data transfer without unnecessary copying.

Overview
--------

Zero-copy operations allow NumPy arrays to share memory directly with Rust-allocated buffers, eliminating the performance overhead of data copying between Python and Rust. This is particularly important for:

- Large terrain heightmaps (float32 arrays)
- High-resolution RGBA output buffers (uint8 arrays) 
- Real-time applications where memory bandwidth is critical

Supported Zero-Copy Pathways
-----------------------------

**1. RGBA Output (Rust → NumPy)**

When rendering operations produce RGBA data, forge3d returns NumPy arrays that directly reference the underlying Rust-allocated buffer:

.. code-block:: python

    import forge3d as f3d
    
    renderer = f3d.Renderer(512, 512)
    rgba = renderer.render_triangle_rgba()  # Zero-copy NumPy array
    
    # The NumPy array shares memory with Rust buffer
    assert rgba.flags['C_CONTIGUOUS'] == True
    assert rgba.dtype == np.uint8
    assert rgba.shape == (512, 512, 4)

**2. Height Input (NumPy → Rust)**

Float32 heightmap arrays that are C-contiguous can be processed without copying:

.. code-block:: python

    import numpy as np
    import forge3d as f3d
    
    # Create C-contiguous float32 heightmap
    heights = np.random.rand(256, 256).astype(np.float32)
    assert heights.flags['C_CONTIGUOUS'] == True
    
    renderer = f3d.Renderer(512, 512)
    renderer.add_terrain(heights, spacing=(1.0, 1.0), exaggeration=2.0)
    # Zero-copy: Rust directly accesses NumPy's data buffer

Requirements for Zero-Copy
---------------------------

For zero-copy operations to work, arrays must meet these requirements:

**Array Properties:**
- **C-contiguous layout**: ``array.flags['C_CONTIGUOUS'] == True``
- **Correct dtype**: ``float32`` for heightmaps, ``uint8`` for RGBA
- **Valid data pointer**: Non-null memory address

**Height Input Requirements:**
- 2D NumPy array with shape ``(height, width)``
- ``dtype`` must be ``float32`` (preferred) or ``float64`` (converted)
- Must be C-contiguous (row-major layout)

.. code-block:: python

    # Ensure zero-copy compatibility
    heights = np.ascontiguousarray(heights, dtype=np.float32)

Validation and Debugging
-------------------------

forge3d provides validation helpers to check zero-copy compatibility:

.. code-block:: python

    from forge3d._validate import check_zero_copy_compatibility, ptr
    
    # Check if array is zero-copy compatible
    result = check_zero_copy_compatibility(heights, "heightmap")
    if not result['compatible']:
        print(f"Issues: {result['issues']}")
        
    # Get memory pointer for debugging
    data_ptr = ptr(heights)
    print(f"NumPy data pointer: 0x{data_ptr:x}")

**Profiling Tool:**

Use the built-in profiler to validate zero-copy pathways:

.. code-block:: bash

    python python/tools/profile_copies.py --render-size 512x512

This tool validates pointer equality between NumPy and Rust buffers and prints "zero-copy OK" when validation succeeds.

Performance Benefits
--------------------

Zero-copy operations provide significant performance benefits:

- **Memory efficiency**: No duplicate allocations
- **Bandwidth savings**: No memcpy overhead  
- **Lower latency**: Direct memory access
- **Scalability**: Benefits increase with array size

For a 4K RGBA buffer (4096×4096×4 = 64MB), zero-copy eliminates:
- 64MB of additional memory allocation
- ~100ms of copy time (typical bandwidth)
- GPU↔CPU sync overhead

Troubleshooting
---------------

**Common Issues:**

1. **Non-contiguous arrays**: Use ``np.ascontiguousarray()``
2. **Wrong dtype**: Convert to ``float32`` for heightmaps
3. **Array views**: Create a copy if working with sliced data

.. code-block:: python

    # Fix common issues
    if not heights.flags['C_CONTIGUOUS']:
        heights = np.ascontiguousarray(heights)
        
    if heights.dtype != np.float32:
        heights = heights.astype(np.float32)

**Validation Failures:**

If zero-copy validation fails, check:
- Array contiguity with ``array.flags``
- Data type compatibility
- Memory alignment requirements

**Debug Output:**

Enable verbose output to see pointer information:

.. code-block:: bash

    python python/tools/profile_copies.py --render-size 512x512 --verbose

This shows detailed memory pointer information and validation results.