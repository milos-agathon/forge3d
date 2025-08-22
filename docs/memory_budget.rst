Memory Budget Management
========================

This document describes forge3d's memory budget tracking and enforcement system, which ensures efficient GPU memory usage and prevents excessive memory consumption.

Overview
--------

forge3d implements a comprehensive memory budget system that:

- **Tracks all GPU resource allocations** (buffers and textures)
- **Enforces a 512 MiB limit** on host-visible memory usage
- **Provides detailed metrics** for monitoring and debugging
- **Prevents out-of-memory scenarios** before they occur

The system focuses primarily on **host-visible memory** as this is typically the most constrained resource in GPU environments.

Budget Limits
-------------

Default Budget: 512 MiB
~~~~~~~~~~~~~~~~~~~~~~~~

forge3d enforces a default budget limit of **512 MiB (536,870,912 bytes)** for host-visible memory allocations. This limit is:

- **Conservative and widely compatible** across different GPU types
- **Focused on host-visible resources** (CPU-accessible GPU memory)
- **Automatically enforced** at allocation time
- **Configurable at the global level** (though not exposed in Python API)

What Counts Towards the Budget
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Resource Type Budget Accounting
   :header-rows: 1
   :widths: 25 10 65

   * - Resource Type
     - Counted
     - Notes
   * - Readback buffers
     - Yes
     - COPY_DST + MAP_READ usage (host-visible)
   * - Color target textures
     - No
     - GPU-only resources
   * - Height textures
     - No
     - GPU-only resources  
   * - Vertex/index buffers
     - No
     - GPU-only resources
   * - Temporary readback buffers
     - Yes
     - Created and freed during operations
   * - Debug buffers
     - Yes
     - Test-only buffers with MAP_READ usage

Budget Enforcement
~~~~~~~~~~~~~~~~~~

The memory budget is enforced at allocation time:

1. **Before allocation**: Check if requested size would exceed budget
2. **During allocation**: Track the new resource in global registry
3. **After deallocation**: Update tracking to reflect freed resources

When the budget would be exceeded, operations fail with detailed error messages containing:

- Current memory usage
- Requested allocation size  
- Budget limit
- Suggested actions

Memory Tracking API
-------------------

Getting Memory Metrics
~~~~~~~~~~~~~~~~~~~~~~~

The ``Renderer.get_memory_metrics()`` method provides comprehensive memory usage information::

    import forge3d as f3d
    
    renderer = f3d.Renderer(512, 512)
    metrics = renderer.get_memory_metrics()
    
    print(f"Buffers: {metrics['buffer_count']} ({metrics['buffer_bytes']} bytes)")
    print(f"Textures: {metrics['texture_count']} ({metrics['texture_bytes']} bytes)")
    print(f"Host-visible: {metrics['host_visible_bytes']} bytes")
    print(f"Total: {metrics['total_bytes']} bytes")
    print(f"Budget limit: {metrics['limit_bytes']} bytes")
    print(f"Within budget: {metrics['within_budget']}")
    print(f"Utilization: {metrics['utilization_ratio']:.1%}")

Metrics Structure
~~~~~~~~~~~~~~~~~

The returned metrics dictionary contains:

.. list-table:: Memory Metrics Fields
   :header-rows: 1
   :widths: 25 15 60

   * - Field
     - Type
     - Description
   * - ``buffer_count``
     - ``int``
     - Number of allocated buffers
   * - ``texture_count``
     - ``int``
     - Number of allocated textures
   * - ``buffer_bytes``
     - ``int``
     - Total bytes in buffer allocations
   * - ``texture_bytes``
     - ``int``
     - Total bytes in texture allocations
   * - ``host_visible_bytes``
     - ``int``
     - Bytes that count against budget limit
   * - ``total_bytes``
     - ``int``
     - Sum of buffer_bytes + texture_bytes
   * - ``limit_bytes``
     - ``int``
     - Budget limit (536,870,912 bytes)
   * - ``within_budget``
     - ``bool``
     - True if host_visible_bytes ≤ limit_bytes
   * - ``utilization_ratio``
     - ``float``
     - host_visible_bytes / limit_bytes

Global Tracking
~~~~~~~~~~~~~~~

Memory tracking is global across all renderer instances::

    # Multiple renderers share the same global tracker
    r1 = f3d.Renderer(256, 256)
    r2 = f3d.Renderer(512, 512)
    
    # Both show the same global totals
    metrics1 = r1.get_memory_metrics()
    metrics2 = r2.get_memory_metrics()
    
    assert metrics1['total_bytes'] == metrics2['total_bytes']

This ensures accurate tracking when multiple renderers or operations are active simultaneously.

Budget Enforcement Examples
---------------------------

Normal Usage Within Budget
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Small to moderate operations stay within the 512 MiB budget::

    import forge3d as f3d
    import numpy as np
    
    # Create renderer and add terrain
    renderer = f3d.Renderer(1024, 1024)
    heightmap = np.random.rand(256, 256).astype(np.float32)
    renderer.add_terrain(heightmap, spacing=(1.0, 1.0), exaggeration=2.0, colormap="viridis")
    renderer.upload_height_r32f()
    
    # Render and readback
    rgba = renderer.render_triangle_rgba()
    heights_back = renderer.read_full_height_texture()
    
    # Check budget status
    metrics = renderer.get_memory_metrics()
    print(f"Memory usage: {metrics['host_visible_bytes']:,} / {metrics['limit_bytes']:,} bytes")
    print(f"Within budget: {metrics['within_budget']}")

Budget Exceeded Scenarios
~~~~~~~~~~~~~~~~~~~~~~~~~

Very large operations may exceed the budget and raise errors::

    import forge3d as f3d
    
    try:
        # Try to create a very large render target
        huge_renderer = f3d.Renderer(20000, 20000)
        rgba = huge_renderer.render_triangle_rgba()
        
    except RuntimeError as e:
        print(f"Budget exceeded: {e}")
        # Error message includes current, requested, and limit information

Error Message Format
~~~~~~~~~~~~~~~~~~~~

Budget exceeded errors provide detailed information::

    RuntimeError: Memory budget exceeded: current 450,560,000 bytes + requested 134,217,728 bytes 
    would exceed limit of 536,870,912 bytes. Consider reducing render size or freeing resources.

The error message contains:

- **Current usage**: Currently allocated host-visible memory
- **Requested size**: Size of the failed allocation
- **Budget limit**: The 512 MiB limit  
- **Actionable advice**: Suggestions for resolution

Best Practices
--------------

Efficient Memory Usage
~~~~~~~~~~~~~~~~~~~~~~

1. **Choose appropriate render sizes**::

    # Good: Reasonable size for most use cases
    renderer = f3d.Renderer(2048, 2048)  # ~16 MB readback buffer
    
    # Caution: Very large sizes may exceed budget  
    renderer = f3d.Renderer(8192, 8192)  # ~256 MB readback buffer

2. **Monitor memory usage in long-running applications**::

    def render_with_monitoring(renderer):
        metrics = renderer.get_memory_metrics()
        if metrics['utilization_ratio'] > 0.8:
            print("Warning: High memory utilization")
        
        rgba = renderer.render_triangle_rgba()
        return rgba

3. **Consider multiple smaller operations vs single large operations**::

    # Instead of one huge render:
    # huge_rgba = renderer.render_triangle_rgba(16384, 16384)
    
    # Consider tiling:
    tiles = []
    tile_size = 2048
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            tile = render_tile(x, y, tile_size, tile_size)
            tiles.append(tile)

Resource Cleanup
~~~~~~~~~~~~~~~~

Resources are automatically cleaned up, but understanding the lifecycle helps::

    # Readback buffers are reused when possible
    r = f3d.Renderer(1024, 1024)
    
    # First render allocates readback buffer
    rgba1 = r.render_triangle_rgba()  # Allocates ~4MB buffer
    
    # Second render reuses same buffer
    rgba2 = r.render_triangle_rgba()  # Reuses existing buffer
    
    # Larger render may reallocate
    r2 = f3d.Renderer(2048, 2048)
    rgba3 = r2.render_triangle_rgba()  # May allocate ~16MB buffer

Testing Memory Budget
---------------------

Test Infrastructure
~~~~~~~~~~~~~~~~~~~

The ``tests/test_memory_budget.py`` module provides comprehensive testing::

    # Run memory budget tests
    pytest tests/test_memory_budget.py -v
    
    # Run specific budget enforcement tests
    pytest tests/test_memory_budget.py::TestMemoryBudgetEnforcement -v

The tests validate:

- Correct metrics structure and types
- Budget limit enforcement  
- Error message formatting
- Allocation tracking accuracy
- Multi-renderer global tracking

Debugging Memory Issues
~~~~~~~~~~~~~~~~~~~~~~~

When encountering memory budget issues:

1. **Check current usage**::

    metrics = renderer.get_memory_metrics()
    print(f"Host-visible usage: {metrics['host_visible_bytes']:,} bytes")
    print(f"Utilization: {metrics['utilization_ratio']:.1%}")

2. **Identify large allocations**::

    # Large readback buffers are the most common issue
    render_area = width * height
    expected_readback = render_area * 4  # RGBA bytes
    print(f"Expected readback size: {expected_readback:,} bytes")

3. **Consider alternatives**::

    # Reduce render size
    # Use streaming/tiling approaches
    # Process in multiple passes

Implementation Details
----------------------

Tracking Mechanism
~~~~~~~~~~~~~~~~~~

The memory tracker uses:

- **Atomic counters** for thread-safe tracking
- **Resource registry** with allocation metadata
- **Usage classification** (host-visible vs GPU-only)
- **Global singleton** for cross-renderer consistency

Host-Visible Detection
~~~~~~~~~~~~~~~~~~~~~~

Buffer usage flags are analyzed to determine host-visibility:

.. code-block:: rust

   // Host-visible usages (count against budget)
   wgpu::BufferUsages::MAP_READ      // CPU-readable
   wgpu::BufferUsages::MAP_WRITE     // CPU-writable
   
   // GPU-only usages (don't count against budget)
   wgpu::BufferUsages::VERTEX        // GPU vertex data
   wgpu::BufferUsages::INDEX         // GPU index data
   wgpu::BufferUsages::UNIFORM       // GPU uniform data

Allocation Lifecycle
~~~~~~~~~~~~~~~~~~~~

1. **Pre-allocation check**: Verify budget compliance
2. **Resource creation**: Create GPU resource via wgpu
3. **Tracking registration**: Record in global tracker
4. **Usage monitoring**: Track through resource lifecycle
5. **Deallocation cleanup**: Remove from tracker on drop

Platform Considerations
~~~~~~~~~~~~~~~~~~~~~~~

Memory behavior may vary across:

- **GPU types**: Integrated vs discrete GPUs have different memory architectures
- **Operating systems**: Memory mapping and allocation strategies differ
- **Drivers**: GPU driver versions affect memory management
- **Backend APIs**: Vulkan, DirectX 12, Metal have different memory models

The 512 MiB limit is conservative enough to work across all supported platforms while providing sufficient space for most use cases.