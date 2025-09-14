Memory Management
=================

forge3d provides advanced memory management systems designed for efficient GPU resource allocation and usage. The memory management framework consists of four main components:

**Workstream O: Resource & Memory Management**

* **O1 - Staging Buffer Rings**: Efficient GPU↔CPU data transfer with fence synchronization
* **O2 - GPU Memory Pools**: Size-bucket allocation with reference counting and defragmentation
* **O3 - Compressed Texture Pipeline**: KTX2 support with format detection and quality optimization
* **O4 - Virtual Texture Streaming**: Large texture support with on-demand tile loading

All memory management systems respect the 512 MiB host-visible GPU memory constraint and are designed for optimal performance across Windows, Linux, and macOS platforms.

.. toctree::
   :maxdepth: 2
   :caption: Core Memory Systems:

   staging_rings
   memory_pools
   compressed_textures
   virtual_texturing

Overview
--------

**Memory Budget Management**

forge3d strictly enforces a 512 MiB host-visible memory budget to ensure compatibility across diverse GPU hardware:

* **Conservative approach**: Works on integrated GPUs with limited memory
* **Deterministic behavior**: Predictable memory usage patterns
* **Cross-platform compatibility**: Consistent behavior across all supported platforms
* **Memory tracking**: Real-time monitoring and statistics collection

**Integration Architecture**

The memory systems are designed to work together:

* **Staging rings** handle temporary upload/download operations
* **Memory pools** manage long-lived GPU resources with efficient allocation
* **Compressed textures** reduce memory footprint by 30-70%
* **Virtual textures** enable very large textures through streaming

**Performance Characteristics**

* **Staging Rings (O1)**: < 100ms upload latency for 64MB transfers
* **Memory Pools (O2)**: ≥ 50% reduction in allocation overhead, < 5% fragmentation
* **Compressed Textures (O3)**: 30-70% memory reduction, PSNR > 35 dB quality
* **Virtual Textures (O4)**: Support for textures up to 16K×16K with on-demand loading

Quick Start
-----------

**Basic Memory System Setup**::

    import forge3d.memory as memory
    
    # Initialize all memory systems
    result = memory.init_memory_system(
        staging_memory_mb=64,      # O1: Staging buffer rings
        pool_memory_mb=128,        # O2: GPU memory pools
        compressed_textures=True,   # O3: Compressed texture pipeline
        virtual_textures=True      # O4: Virtual texture streaming
    )
    
    if result['success']:
        print("Memory systems initialized successfully")
        print(f"Total memory allocated: {result['total_memory_mb']} MB")

**Staging Buffer Operations**::

    # Upload data with automatic staging
    upload_id = memory.stage_upload(data, target_buffer)
    if memory.is_upload_complete(upload_id):
        print("Upload completed successfully")

**Memory Pool Allocation**::

    # Allocate from size-appropriate bucket
    block = memory.allocate_from_pool(1024)  # 1KB allocation
    if block:
        print(f"Allocated block {block['id']} of size {block['size']}")

**Compressed Texture Loading**::

    import forge3d.compressed as compressed
    
    # Load compressed texture with automatic format detection
    texture = compressed.load_texture("asset.ktx2", quality="high")
    print(f"Texture loaded with {texture.compression_ratio:.1f}:1 compression")

**Virtual Texture Streaming**::

    import forge3d.streaming as streaming
    
    # Create virtual texture system
    vt_system = streaming.VirtualTextureSystem(device, max_memory_mb=256)
    
    # Load large texture for streaming
    texture = vt_system.load_texture("world_texture_8192x8192.ktx2")
    
    # Update based on camera position
    result = vt_system.update_streaming(camera_pos, view_matrix, proj_matrix)
    print(f"Loaded {result['tiles_loaded']} new tiles")

Memory Monitoring
-----------------

**Real-time Statistics**::

    # Get comprehensive memory statistics
    stats = memory.get_system_stats()
    
    print("Memory System Status:")
    print(f"  Staging rings: {stats['staging']['active_transfers']} active transfers")
    print(f"  Memory pools: {stats['pools']['fragmentation_ratio']:.1%} fragmentation")
    print(f"  Compressed textures: {stats['compressed']['memory_saved_mb']} MB saved")
    print(f"  Virtual textures: {stats['virtual']['cache_hit_rate']:.1f}% cache hit rate")

**Performance Monitoring**::

    # Monitor performance characteristics
    perf = memory.get_performance_metrics()
    
    print("Performance Metrics:")
    print(f"  Upload bandwidth: {perf['upload_bandwidth_mbps']:.1f} MB/s")
    print(f"  Allocation latency: {perf['allocation_latency_us']:.1f} μs")
    print(f"  Texture decode time: {perf['decode_time_ms']:.1f} ms")
    print(f"  Streaming update time: {perf['streaming_update_ms']:.1f} ms")

Advanced Features
-----------------

**Memory Pool Defragmentation**::

    # Monitor fragmentation and trigger defrag when needed
    pool_stats = memory.pool_stats()
    if pool_stats['fragmentation_ratio'] > 0.25:
        defrag_result = memory.pool_manager.defragment()
        print(f"Defragmentation completed: {defrag_result['blocks_moved']} blocks moved")

**Texture Quality Optimization**::

    # Configure compression quality vs performance trade-offs
    compressed.set_quality_settings(
        compression_speed="normal",  # fast, normal, high
        target_psnr=40.0,           # Quality target in dB
        memory_priority="balanced"   # memory, balanced, quality
    )

**Virtual Texture Prefetching**::

    # Prefetch tiles based on predicted camera movement
    camera_velocity = get_camera_velocity()
    predicted_pos = camera_pos + camera_velocity * 0.5  # 0.5 second prediction
    
    vt_system.prefetch_region(
        texture,
        region_center=predicted_pos,
        radius=1000,  # World units
        mip_level=0
    )

Best Practices
--------------

**Memory Budget Management**

1. **Monitor total usage**: Keep combined memory usage under 512 MiB
2. **Use appropriate systems**: Choose the right tool for each use case
3. **Profile regularly**: Track memory usage patterns in your application
4. **Plan for peaks**: Account for temporary memory spikes during operations

**Performance Optimization**

1. **Batch operations**: Group related memory operations together
2. **Reuse resources**: Leverage pooling and caching systems
3. **Async patterns**: Use non-blocking operations where possible
4. **Monitor bottlenecks**: Identify and address performance constraints

**Error Handling**

1. **Check return values**: Always validate memory operation results
2. **Handle exhaustion gracefully**: Plan for out-of-memory scenarios
3. **Clean up resources**: Ensure proper resource lifecycle management
4. **Use debugging tools**: Enable detailed logging when troubleshooting

Integration Examples
--------------------

See the ``examples/`` directory for complete demonstrations:

* ``staging_rings_demo.py`` - O1 staging buffer operations
* ``memory_pools_demo.py`` - O2 GPU memory pool management
* ``compressed_texture_demo.py`` - O3 compressed texture pipeline
* ``virtual_texture_demo.py`` - O4 virtual texture streaming
* ``memory_integration_demo.py`` - Combined memory system usage

For detailed API documentation, see the individual component pages.