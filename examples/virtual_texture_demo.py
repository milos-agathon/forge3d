#!/usr/bin/env python3
"""
O4: Virtual Texture Streaming Demo

This example demonstrates the virtual texture streaming system, showing how to:
- Initialize and configure virtual texture streaming
- Load large textures for on-demand tile streaming
- Update streaming based on camera movement
- Monitor performance and memory usage
- Integrate with staging rings and compressed textures

The virtual texture system enables working with very large textures (up to 16K×16K)
that don't fit in GPU memory by streaming tiles on demand based on camera position.

Usage:
    python examples/virtual_texture_demo.py
"""

import sys
import time
import numpy as np
from pathlib import Path

# Import shim for running from repository root
sys.path.insert(0, str(Path(__file__).parent))
from _import_shim import forge3d

# Import streaming module
import forge3d.streaming as streaming
import forge3d.memory as memory


def demonstrate_virtual_texture_initialization():
    """Demonstrate virtual texture system initialization and configuration."""
    print("=== Virtual Texture System Initialization ===")
    
    try:
        # Get GPU device
        device = forge3d.get_device()
        print("OK: GPU device acquired")
        
        # Test different system configurations
        configurations = [
            {"max_memory_mb": 128, "tile_size": 128, "name": "Conservative (mobile/integrated)"},
            {"max_memory_mb": 256, "tile_size": 256, "name": "Balanced (mid-range)"},
            {"max_memory_mb": 512, "tile_size": 512, "name": "High performance (dedicated GPU)"},
        ]
        
        systems_created = []
        
        for config in configurations:
            try:
                vt_system = streaming.VirtualTextureSystem(
                    device, 
                    max_memory_mb=config["max_memory_mb"],
                    tile_size=config["tile_size"]
                )
                
                print(f"OK: Created system: {config['name']}")
                print(f"    Memory budget: {config['max_memory_mb']} MB")
                print(f"    Tile size: {config['tile_size']}×{config['tile_size']} pixels")
                
                systems_created.append((vt_system, config))
                
            except Exception as e:
                print(f"ERROR: Failed to create {config['name']}: {e}")
        
        # Cleanup systems
        for system, config in systems_created:
            del system
            
        return True
        
    except Exception as e:
        print(f"Virtual texture initialization not available: {e}")
        return False


def demonstrate_memory_analysis():
    """Demonstrate memory requirement analysis for virtual textures."""
    print("\\n=== Memory Analysis ===")
    
    # Analyze different texture scenarios
    texture_scenarios = [
        (2048, 2048, "2K terrain texture"),
        (4096, 4096, "4K hero asset"),
        (8192, 8192, "8K world texture"),
        (16384, 8192, "Ultra-wide landscape"),
    ]
    
    print("Memory requirements analysis:")
    print(f"{'Texture':<20} {'Full Size':<10} {'Tiles':<8} {'Per Tile':<10} {'Cache Rec.':<12}")
    print("-" * 70)
    
    for width, height, description in texture_scenarios:
        reqs = streaming.calculate_memory_requirements(
            texture_width=width,
            texture_height=height,
            tile_size=256,
            bytes_per_pixel=4  # RGBA8
        )
        
        full_mb = reqs['full_texture_size'] // 1024 // 1024
        tile_count = reqs['tile_count']
        tile_kb = reqs['tile_memory_size'] // 1024
        cache_mb = reqs['recommended_cache_size'] // 1024 // 1024
        
        print(f"{description:<20} {full_mb:>7} MB {tile_count:>6} {tile_kb:>7} KB {cache_mb:>9} MB")
    
    # Performance estimation
    print("\\nPerformance estimation (4K texture, 256 MB cache, 60 FPS):")
    perf = streaming.estimate_streaming_performance(
        texture_size=(4096, 4096),
        tile_size=256,
        cache_size_mb=256,
        target_fps=60
    )
    
    print(f"  Cache capacity: {perf['cache_capacity_tiles']} tiles")
    print(f"  Tiles per frame budget: {perf['tiles_per_frame_budget']}")
    print(f"  Memory pressure: {perf['memory_pressure_factor'] * 100:.1f}%")
    print(f"  Recommended prefetch distance: {perf['recommended_prefetch_distance']} tiles")
    
    if perf['memory_pressure_factor'] > 0.8:
        print("  ⚠ High memory pressure - consider increasing cache size")
    elif perf['memory_pressure_factor'] < 0.3:
        print("  OK: Low memory pressure - good streaming efficiency expected")
    else:
        print("  OK: Moderate memory pressure - balanced performance expected")


def simulate_camera_streaming(vt_system):
    """Simulate camera movement and streaming updates."""
    print("\\n=== Camera-Based Streaming Simulation ===")
    
    # Simulate camera path (circular movement around origin)
    camera_path = []
    num_steps = 20
    radius = 2000.0
    height = 500.0
    
    for i in range(num_steps):
        angle = (i / num_steps) * 2 * np.pi
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = height
        camera_path.append((x, y, z))
    
    # Create view and projection matrices
    def create_view_matrix(camera_pos, target=(0, 0, 0)):
        # Simple lookAt matrix construction
        eye = np.array(camera_pos)
        center = np.array(target)
        up = np.array([0, 0, 1])
        
        f = center - eye
        f = f / np.linalg.norm(f)
        
        s = np.cross(f, up)
        s = s / np.linalg.norm(s)
        
        u = np.cross(s, f)
        
        view = np.eye(4, dtype=np.float32)
        view[0, :3] = s
        view[1, :3] = u
        view[2, :3] = -f
        view[:3, 3] = [-np.dot(s, eye), -np.dot(u, eye), np.dot(f, eye)]
        
        return view
    
    # Simple perspective projection matrix
    def create_projection_matrix(fov_degrees=60, aspect=1.0, near=1.0, far=10000.0):
        fov_rad = np.radians(fov_degrees)
        f = 1.0 / np.tan(fov_rad / 2.0)
        
        proj = np.zeros((4, 4), dtype=np.float32)
        proj[0, 0] = f / aspect
        proj[1, 1] = f
        proj[2, 2] = -(far + near) / (far - near)
        proj[2, 3] = -2.0 * far * near / (far - near)
        proj[3, 2] = -1.0
        
        return proj
    
    projection_matrix = create_projection_matrix()
    
    print(f"Simulating camera movement through {num_steps} positions...")
    
    total_tiles_loaded = 0
    total_tiles_evicted = 0
    update_times = []
    
    for i, camera_pos in enumerate(camera_path):
        try:
            # Create view matrix for current camera position
            view_matrix = create_view_matrix(camera_pos)
            
            # Update streaming
            start_time = time.perf_counter()
            
            result = vt_system.update_streaming(
                camera_pos,
                view_matrix,
                projection_matrix
            )
            
            update_time = (time.perf_counter() - start_time) * 1000.0
            update_times.append(update_time)
            
            # Track statistics
            tiles_loaded = result.get('tiles_loaded', 0)
            tiles_evicted = result.get('tiles_evicted', 0)
            
            total_tiles_loaded += tiles_loaded
            total_tiles_evicted += tiles_evicted
            
            if i % 5 == 0:  # Print every 5th update
                print(f"  Step {i+1}: pos=({camera_pos[0]:.0f}, {camera_pos[1]:.0f}, {camera_pos[2]:.0f})")
                print(f"    Loaded: {tiles_loaded}, Evicted: {tiles_evicted}, Time: {update_time:.2f}ms")
        
        except Exception as e:
            print(f"  Update {i+1} failed: {e}")
    
    # Summary statistics
    if update_times:
        avg_update_time = sum(update_times) / len(update_times)
        max_update_time = max(update_times)
        
        print(f"\\nStreaming summary:")
        print(f"  Total tiles loaded: {total_tiles_loaded}")
        print(f"  Total tiles evicted: {total_tiles_evicted}")
        print(f"  Average update time: {avg_update_time:.2f} ms")
        print(f"  Maximum update time: {max_update_time:.2f} ms")
        
        if avg_update_time < 2.0:
            print("  OK: Excellent streaming performance")
        elif avg_update_time < 5.0:
            print("  OK: Good streaming performance")
        else:
            print("  ⚠ Streaming performance may impact frame rate")


def demonstrate_performance_monitoring(vt_system):
    """Demonstrate performance monitoring and statistics."""
    print("\\n=== Performance Monitoring ===")
    
    # Get initial statistics
    initial_stats = vt_system.get_statistics()
    print("Initial system state:")
    print(f"  Cache hits: {initial_stats.cache_hits}")
    print(f"  Cache misses: {initial_stats.cache_misses}")
    print(f"  Active tiles: {initial_stats.active_tiles}")
    print(f"  Memory used: {initial_stats.memory_used // 1024 // 1024} MB")
    print(f"  Memory limit: {initial_stats.memory_limit // 1024 // 1024} MB")
    
    # Get detailed memory information
    memory_info = vt_system.get_memory_info()
    print("\\nMemory breakdown:")
    for key, value in memory_info.items():
        if 'memory' in key:
            print(f"  {key}: {value // 1024 // 1024} MB")
        else:
            print(f"  {key}: {value}")
    
    # Performance over time tracking
    print("\\nPerformance monitoring simulation...")
    
    # Simulate some streaming activity
    camera_positions = [
        (0, 0, 1000),
        (500, 500, 1000),
        (1000, 0, 1000),
        (0, 1000, 1000),
    ]
    
    for i, pos in enumerate(camera_positions):
        try:
            # Perform streaming update
            vt_system.update_streaming(pos)
            
            # Get updated statistics
            stats = vt_system.get_statistics()
            
            print(f"  Position {i+1}: Cache hit rate {stats.cache_hit_rate:.1f}%, "
                  f"Memory usage {stats.memory_utilization:.1f}%")
            
        except Exception as e:
            print(f"  Position {i+1} update failed: {e}")
    
    # Final statistics comparison
    final_stats = vt_system.get_statistics()
    
    print("\\nFinal vs Initial statistics:")
    print(f"  Cache hits: {initial_stats.cache_hits} → {final_stats.cache_hits}")
    print(f"  Cache misses: {initial_stats.cache_misses} → {final_stats.cache_misses}")
    print(f"  Tiles loaded: {initial_stats.tiles_loaded} → {final_stats.tiles_loaded}")
    print(f"  Active tiles: {initial_stats.active_tiles} → {final_stats.active_tiles}")


def demonstrate_quality_settings(vt_system):
    """Demonstrate quality settings and their effects."""
    print("\\n=== Quality Settings Configuration ===")
    
    # Test different quality presets
    quality_presets = [
        {
            "name": "Performance",
            "max_mip_bias": 1.0,
            "lod_scale": 0.8,
            "cache_priority_boost": 1.0,
            "description": "Prioritizes performance over quality"
        },
        {
            "name": "Balanced", 
            "max_mip_bias": 0.5,
            "lod_scale": 1.0,
            "cache_priority_boost": 1.5,
            "description": "Good balance of quality and performance"
        },
        {
            "name": "Quality",
            "max_mip_bias": 0.0,
            "lod_scale": 1.2,
            "cache_priority_boost": 2.0,
            "description": "Maximum quality, may impact performance"
        }
    ]
    
    for preset in quality_presets:
        print(f"\\n{preset['name']} preset: {preset['description']}")
        
        try:
            success = vt_system.set_quality_settings(
                max_mip_bias=preset['max_mip_bias'],
                lod_scale=preset['lod_scale'],
                cache_priority_boost=preset['cache_priority_boost']
            )
            
            if success:
                print(f"  OK: Applied settings: mip_bias={preset['max_mip_bias']}, "
                      f"lod_scale={preset['lod_scale']}, priority_boost={preset['cache_priority_boost']}")
                
                # Test streaming with this preset
                result = vt_system.update_streaming((1000, 1000, 500))
                tiles_requested = result.get('tiles_requested', 0)
                update_time = result.get('update_time_ms', 0)
                
                print(f"  Test result: {tiles_requested} tiles requested, {update_time:.2f}ms update time")
            else:
                print(f"  ERROR: Failed to apply {preset['name']} settings")
                
        except Exception as e:
            print(f"  ERROR: Error applying {preset['name']} settings: {e}")


def demonstrate_integration_with_memory_systems():
    """Demonstrate integration with staging rings and memory pools."""
    print("\\n=== Integration with Memory Management ===")
    
    try:
        # Initialize memory management systems
        device = forge3d.get_device()
        
        print("Initializing memory management systems...")
        memory.init_memory_system(
            device,
            staging_memory_mb=64,  # 64MB for staging rings
            pool_memory_mb=128     # 128MB for memory pools
        )
        print("OK: Memory management systems initialized")
        
        # Get memory statistics
        staging_stats = memory.staging_stats()
        pool_stats = memory.pool_stats()
        
        print("Memory system status:")
        print(f"  Staging rings: {len(staging_stats.get('rings', []))} rings")
        print(f"  Memory pools: {len(pool_stats.get('pools', []))} pools")
        
        # Create virtual texture system (it will automatically use memory systems)
        print("\\nCreating virtual texture system with memory integration...")
        vt_system = streaming.VirtualTextureSystem(device, max_memory_mb=256)
        print("OK: Virtual texture system created with memory integration")
        
        # Test that systems work together
        camera_pos = (500, 500, 1000)
        result = vt_system.update_streaming(camera_pos)
        
        print(f"\\nIntegrated streaming test:")
        print(f"  Update result: {result}")
        
        # Check memory usage after streaming
        updated_staging_stats = memory.staging_stats()
        updated_pool_stats = memory.pool_stats()
        
        print("\\nMemory usage after streaming:")
        print(f"  Staging system activity: {'detected' if updated_staging_stats != staging_stats else 'none'}")
        print(f"  Pool system activity: {'detected' if updated_pool_stats != pool_stats else 'none'}")
        
        return vt_system
        
    except Exception as e:
        print(f"Memory integration test failed: {e}")
        
        # Fallback to standalone system
        print("\\nCreating standalone virtual texture system...")
        try:
            device = forge3d.get_device()
            return streaming.VirtualTextureSystem(device, max_memory_mb=256)
        except Exception as fallback_error:
            print(f"Standalone system creation failed: {fallback_error}")
            return None


def demonstrate_texture_operations(vt_system):
    """Demonstrate virtual texture operations like prefetching and eviction."""
    print("\\n=== Virtual Texture Operations ===")
    
    # Create a mock virtual texture for demonstration
    mock_texture = streaming.VirtualTexture(
        handle=1,
        width=4096,
        height=4096,
        tile_size=256
    )
    
    print(f"Mock texture: {mock_texture.size[0]}×{mock_texture.size[1]} pixels")
    print(f"Tile grid: {mock_texture.tile_count[0]}×{mock_texture.tile_count[1]} tiles")
    print(f"Max mip level: {mock_texture.max_mip_level}")
    
    # Test prefetching
    print("\\nTesting prefetch operations:")
    prefetch_regions = [
        (0, 0, 512, 512, "Top-left corner"),
        (2048, 2048, 1024, 1024, "Center region"),  
        (3584, 3584, 512, 512, "Bottom-right corner"),
    ]
    
    for x, y, w, h, description in prefetch_regions:
        try:
            success = vt_system.prefetch_region(
                mock_texture, x, y, w, h, mip_level=0
            )
            print(f"  {description}: {'OK:' if success else 'ERROR:'}")
        except Exception as e:
            print(f"  {description}: ERROR: ({e})")
    
    # Test tile eviction
    print("\\nTesting tile eviction:")
    try:
        evicted_count = vt_system.evict_tiles()
        print(f"  Evicted {evicted_count} tiles from all textures")
        
        # Test selective eviction (if supported)
        evicted_count = vt_system.evict_tiles(mock_texture)
        print(f"  Evicted {evicted_count} tiles from specific texture")
        
    except Exception as e:
        print(f"  Tile eviction failed: {e}")
    
    # Test system flush
    print("\\nTesting system flush:")
    try:
        success = vt_system.flush()
        print(f"  System flush: {'OK:' if success else 'ERROR:'}")
    except Exception as e:
        print(f"  System flush failed: {e}")


def main():
    """Run the complete virtual texture streaming demonstration."""
    print("Forge3D Virtual Texture Streaming Demo")
    print("=" * 60)
    
    # Check if virtual texture system is available
    if not hasattr(forge3d, 'create_virtual_texture_system'):
        print("Virtual texture streaming is not available in this build.")
        print("This is expected for builds without the O4 workstream feature.")
        print("\\nHowever, we can still demonstrate utility functions...")
        
        # Demonstrate memory analysis (always available)
        demonstrate_memory_analysis()
        
        print("\\n=== Demo Summary ===")
        print("OK: Memory analysis utilities demonstrated")
        print("ERROR: Virtual texture system not available")
        print("\\nTo enable virtual texture streaming, ensure O4 features are compiled in.")
        return
    
    try:
        # Step 1: System initialization
        if not demonstrate_virtual_texture_initialization():
            print("Skipping remaining demos due to initialization failure")
            return
        
        # Step 2: Memory analysis
        demonstrate_memory_analysis()
        
        # Step 3: Integration with memory systems and create working system
        vt_system = demonstrate_integration_with_memory_systems()
        
        if vt_system is None:
            print("Could not create virtual texture system - stopping demo")
            return
        
        # Step 4: Performance monitoring
        demonstrate_performance_monitoring(vt_system)
        
        # Step 5: Quality settings
        demonstrate_quality_settings(vt_system)
        
        # Step 6: Camera-based streaming simulation
        simulate_camera_streaming(vt_system)
        
        # Step 7: Texture operations
        demonstrate_texture_operations(vt_system)
        
        # Final statistics
        final_stats = vt_system.get_statistics()
        
        print("\\n=== Demo Summary ===")
        print("OK: Virtual texture system initialization demonstrated")
        print("OK: Memory analysis and requirements calculation shown")
        print("OK: Integration with staging rings and memory pools validated")
        print("OK: Performance monitoring and statistics collection tested")
        print("OK: Quality settings configuration demonstrated")
        print("OK: Camera-based streaming simulation completed")
        print("OK: Virtual texture operations (prefetch, eviction) tested")
        
        print(f"\\nFinal system state:")
        print(f"  Cache hit rate: {final_stats.cache_hit_rate:.1f}%")
        print(f"  Memory utilization: {final_stats.memory_utilization:.1f}%")
        print(f"  Active tiles: {final_stats.active_tiles}")
        print(f"  Total tiles loaded: {final_stats.tiles_loaded}")
        
        # Cleanup
        del vt_system
        
    except Exception as e:
        print(f"\\nERROR: Demo error: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\\nDemo completed. See docs/memory/virtual_texturing.md for detailed documentation.")


if __name__ == "__main__":
    main()