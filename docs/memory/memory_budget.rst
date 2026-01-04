Memory Budget Management
========================

forge3d provides comprehensive GPU memory budget tracking and enforcement to prevent out-of-memory errors and ensure predictable resource usage.
Overview
--------

The memory budget system tracks all GPU resource allocations and enforces a configurable limit on host-visible memory usage. This prevents applications from exhausting system memory while providing detailed metrics for monitoring and optimization.

**Default Budget**: 512 MiB of host-visible GPU memory

Key Features
------------

- **Automatic tracking**: All buffer and texture allocations are monitored
- **Budget enforcement**: Operations fail gracefully when budget would be exceeded
- **Detailed metrics**: Real-time usage statistics and utilization ratios
- **Host-visible focus**: Tracks memory that impacts system resources
- **Thread-safe**: Atomic counters support concurrent operations

Getting Memory Metrics
-----------------------

Access current memory usage and budget information:

.. code-block:: python

    import forge3d as f3d
    
    # Get current memory metrics
    metrics = f3d.get_memory_metrics()
    
    print(f"Budget limit: {metrics['budget_limit_mb']:.1f} MB")
    print(f"Host-visible used: {metrics['host_visible_mb']:.1f} MB") 
    print(f"Utilization: {metrics['utilization_ratio']:.1%}")
    print(f"Buffer count: {metrics['buffer_count']}")
    print(f"Texture count: {metrics['texture_count']}")

**Sample Output:**

.. code-block::

    Budget limit: 512.0 MB
    Host-visible used: 24.3 MB
    Utilization: 4.7%
    Buffer count: 3
    Texture count: 2

Metrics Structure
-----------------

The memory metrics dictionary contains:

.. code-block:: python

    {
        'budget_limit_mb': 512.0,           # Maximum allowed (MB)
        'host_visible_mb': 24.3,            # Currently allocated (MB)
        'gpu_only_mb': 8.1,                 # GPU-only memory (MB)
        'utilization_ratio': 0.047,         # host_visible / budget_limit
        'buffer_count': 3,                  # Number of allocated buffers
        'texture_count': 2,                 # Number of allocated textures
        'buffer_bytes': 25485312,           # Total buffer bytes
        'texture_bytes': 8388608            # Total texture bytes
    }

Budget Enforcement
------------------

When operations would exceed the memory budget, they fail with descriptive errors:

.. code-block:: python

    import forge3d as f3d
    
    try:
        # This might exceed budget
        renderer = f3d.Renderer(8192, 8192)  # Large render target
    except RuntimeError as e:
        if "budget" in str(e):
            print(f"Budget exceeded: {e}")
            # Handle gracefully or reduce size
            renderer = f3d.Renderer(4096, 4096)  # Smaller fallback

**Error Format:**

.. code-block::

    RuntimeError: Memory budget exceeded: requested 268.4 MB for readback buffer resize would exceed limit of 512.0 MB (currently using 245.2 MB)

Budget Tracking Scope
----------------------

The memory budget tracks these allocation types:

**Host-Visible Memory (counted against budget):**
- Readback buffers (``MAP_READ`` usage)
- Staging buffers for uploads
- CPU-accessible resources

**GPU-Only Memory (informational only):**
- Render targets and textures
- Vertex/index buffers  
- GPU-only resources

Only host-visible allocations count against the 512 MiB budget limit, as these directly impact system memory availability.

Usage Patterns
---------------

**Monitoring During Operations:**

.. code-block:: python

    import forge3d as f3d
    
    renderer = f3d.Renderer(1024, 1024)
    print("After renderer:", f3d.get_memory_metrics()['host_visible_mb'])
    
    # Add terrain data
    heights = np.random.rand(512, 512).astype(np.float32)  
    renderer.add_terrain(heights, (1.0, 1.0), 2.0)
    print("After terrain:", f3d.get_memory_metrics()['host_visible_mb'])
    
    # Render operations
    rgba = renderer.render_triangle_rgba()
    print("After render:", f3d.get_memory_metrics()['host_visible_mb'])

**Progressive Size Testing:**

.. code-block:: python

    def find_max_render_size():
        """Find maximum render size within budget."""
        for size in [512, 1024, 2048, 4096, 8192]:
            try:
                renderer = f3d.Renderer(size, size)
                rgba = renderer.render_triangle_rgba()
                print(f"Size {size}x{size}: OK")
                del renderer  # Free memory
            except RuntimeError as e:
                if "budget" in str(e):
                    print(f"Size {size}x{size}: Budget exceeded")
                    break
        return size // 2  # Last successful size

Memory Lifecycle
----------------

**Allocation Tracking:**

- Buffer/texture creation increments counters
- Memory usage is tracked in real-time
- Budget checks occur before allocation

**Deallocation:**

- Resources are freed when Python objects are deleted
- Memory tracking decrements automatically
- Budget space becomes available immediately

**Buffer Reuse:**

- Renderers reuse readback buffers when possible
- Only resizes trigger new allocations
- Old buffers are properly deallocated before replacement

Best Practices
---------------

**1. Monitor Usage Patterns:**

.. code-block:: python

    # Check usage before large operations
    metrics = f3d.get_memory_metrics()
    if metrics['utilization_ratio'] > 0.8:  # Above 80%
        print("Warning: High memory usage")

**2. Handle Budget Errors Gracefully:**

.. code-block:: python

    try:
        renderer = f3d.Renderer(target_width, target_height)
    except RuntimeError as e:
        if "budget" in str(e):
            # Fallback to smaller size
            target_width = int(target_width * 0.7)
            target_height = int(target_height * 0.7)
            renderer = f3d.Renderer(target_width, target_height)

**3. Explicit Memory Management:**

.. code-block:: python

    # Free resources explicitly when done
    del renderer
    
    # Or use context managers for automatic cleanup
    with f3d.Renderer(1024, 1024) as renderer:
        rgba = renderer.render_triangle_rgba()
    # Automatically freed

**4. Size Estimation:**

For planning resource usage:

- **RGBA buffer**: ``width × height × 4 bytes``
- **Height texture**: ``width × height × 4 bytes`` (R32F format)
- **Readback buffer**: Uses 256-byte aligned rows (padded)

Troubleshooting
---------------

**Budget Exceeded Errors:**

1. Check current usage with ``get_memory_metrics()``
2. Reduce render target sizes
3. Free unused renderers with ``del``
4. Consider processing in smaller batches

**Memory Leaks:**

If memory usage grows unexpectedly:

1. Ensure renderer objects are deleted
2. Check for retained references to RGBA arrays
3. Monitor metrics over time to identify leaks

**Platform Differences:**

Budget enforcement may vary by GPU driver and available system memory. The 512 MiB limit is conservative to work across platforms.
