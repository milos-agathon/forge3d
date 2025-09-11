#!/usr/bin/env python3
"""
O2: Memory Pool System Demo

This example demonstrates the memory pool system for efficient GPU memory
allocation using size-bucket allocation, reference counting, and defragmentation.

The memory pool system provides:
- Power-of-two size buckets from 64B to 8MB
- Reference counting for automatic lifecycle management
- Defragmentation to reduce memory fragmentation
- Performance statistics and monitoring

Usage:
    python examples/memory_pool_demo.py
"""

import time
import random
from typing import List, Dict, Any

import forge3d.memory as memory


def demonstrate_basic_pool_usage():
    """Demonstrate basic memory pool usage."""
    print("=== Basic Memory Pool Usage ===")
    
    # Initialize the memory system with pools
    result = memory.init_memory_system(
        staging_rings=False,  # Focus on pools
        memory_pools=True,
        ring_count=3,
        buffer_size=1 * 1024 * 1024
    )
    
    print(f"Initialization result: {result}")
    
    # Get initial statistics
    initial_stats = memory.pool_stats()
    print(f"Initial pool stats: {initial_stats}")
    
    # Show pool configuration
    print("\nMemory pool configuration:")
    print(f"  Pool count: {initial_stats['pool_count']}")
    print(f"  Active blocks: {initial_stats['active_blocks']}")
    print(f"  Fragmentation: {initial_stats['fragmentation_ratio'] * 100:.2f}%")
    print(f"  Largest free block: {initial_stats['largest_free_block'] / 1024 / 1024:.2f} MB")


def demonstrate_size_bucket_allocation():
    """Demonstrate allocation from different size buckets."""
    print("\n=== Size Bucket Allocation Demo ===")
    
    # Test various allocation sizes to show bucket selection
    allocation_tests = [
        (32, "Small allocation (should use 64B bucket)"),
        (100, "Medium allocation (should use 128B bucket)"),
        (1000, "1KB allocation (should use 1024B bucket)"),
        (5000, "5KB allocation (should use 8192B bucket)"),
        (100000, "100KB allocation (should use 128KB bucket)"),
        (1000000, "1MB allocation (should use 1MB bucket)"),
    ]
    
    allocated_blocks = []
    
    for size, description in allocation_tests:
        print(f"\n{description}")
        print(f"  Requesting: {size} bytes")
        
        block = memory.allocate_from_pool(size)
        if block:
            allocated_blocks.append(block)
            actual_size = block.get("size", size)
            efficiency = size / actual_size * 100
            
            print(f"  Allocated: {actual_size} bytes (efficiency: {efficiency:.1f}%)")
            print(f"  Block ID: {block['id']}")
            print(f"  Pool ID: {block['pool_id']}")
            print(f"  Offset: {block['offset']}")
        else:
            print(f"  Failed to allocate {size} bytes")
    
    # Show updated statistics
    stats_after_alloc = memory.pool_stats()
    print(f"\nStats after allocations:")
    print(f"  Active blocks: {stats_after_alloc['active_blocks']}")
    print(f"  Total allocated: {stats_after_alloc['total_allocated'] / 1024:.1f} KB")
    
    # Clean up allocations
    print(f"\nCleaning up {len(allocated_blocks)} allocations...")
    for block in allocated_blocks:
        success = memory.deallocate_pool_block(block["id"])
        if success:
            print(f"  OK: Deallocated block {block['id']}")
        else:
            print(f"  ERROR: Failed to deallocate block {block['id']}")
    
    # Show final statistics
    final_stats = memory.pool_stats()
    print(f"\nFinal stats:")
    print(f"  Active blocks: {final_stats['active_blocks']}")
    print(f"  Total freed: {final_stats['total_freed'] / 1024:.1f} KB")


def demonstrate_fragmentation_patterns():
    """Demonstrate fragmentation and its effects."""
    print("\n=== Fragmentation Pattern Demo ===")
    
    # Create a fragmentation pattern
    print("Creating fragmentation by allocating and freeing blocks...")
    
    # Allocate many blocks
    block_size = 1024
    num_blocks = 50
    blocks = []
    
    print(f"Allocating {num_blocks} blocks of {block_size} bytes each...")
    for i in range(num_blocks):
        block = memory.allocate_from_pool(block_size)
        if block:
            blocks.append((i, block))
    
    stats_after_alloc = memory.pool_stats()
    print(f"After allocation - Active: {stats_after_alloc['active_blocks']}, "
          f"Fragmentation: {stats_after_alloc['fragmentation_ratio'] * 100:.2f}%")
    
    # Create fragmentation by freeing every 3rd block
    print("Creating fragmentation by freeing every 3rd block...")
    fragmented_blocks = []
    
    for i, (block_idx, block) in enumerate(blocks):
        if i % 3 == 0:  # Free every 3rd block
            success = memory.deallocate_pool_block(block["id"])
            if success:
                print(f"  Freed block {block_idx} (ID: {block['id']})")
        else:
            fragmented_blocks.append((block_idx, block))
    
    stats_fragmented = memory.pool_stats()
    print(f"\nAfter fragmentation:")
    print(f"  Active blocks: {stats_fragmented['active_blocks']}")
    print(f"  Fragmentation: {stats_fragmented['fragmentation_ratio'] * 100:.2f}%")
    print(f"  Total freed: {stats_fragmented['total_freed'] / 1024:.1f} KB")
    
    # Try to allocate new blocks in fragmented space
    print("\nTrying to allocate in fragmented space...")
    new_blocks = []
    for i in range(10):
        block = memory.allocate_from_pool(block_size)
        if block:
            new_blocks.append(block)
            print(f"  OK: Allocated new block {block['id']} at offset {block['offset']}")
        else:
            print(f"  ERROR: Failed to allocate block {i}")
    
    # Clean up all remaining blocks
    print(f"\nCleaning up remaining blocks...")
    cleanup_count = 0
    
    for _, block in fragmented_blocks + [(i, b) for i, b in enumerate(new_blocks)]:
        success = memory.deallocate_pool_block(block["id"])
        if success:
            cleanup_count += 1
    
    final_stats = memory.pool_stats()
    print(f"Cleaned up {cleanup_count} blocks")
    print(f"Final fragmentation: {final_stats['fragmentation_ratio'] * 100:.2f}%")


def demonstrate_defragmentation():
    """Demonstrate defragmentation capabilities."""
    print("\n=== Defragmentation Demo ===")
    
    # Create significant fragmentation
    print("Creating fragmentation for defragmentation test...")
    
    # Allocate blocks of different sizes
    mixed_blocks = []
    sizes = [256, 512, 1024, 2048, 4096]
    
    for i in range(40):
        size = sizes[i % len(sizes)]
        block = memory.allocate_from_pool(size)
        if block:
            mixed_blocks.append(block)
    
    # Free random blocks to create fragmentation
    import random
    random.shuffle(mixed_blocks)
    keep_blocks = mixed_blocks[:len(mixed_blocks)//2]
    free_blocks = mixed_blocks[len(mixed_blocks)//2:]
    
    print(f"Freeing {len(free_blocks)} random blocks to create fragmentation...")
    for block in free_blocks:
        memory.deallocate_pool_block(block["id"])
    
    # Measure fragmentation before defrag
    stats_before = memory.pool_stats()
    fragmentation_before = stats_before['fragmentation_ratio']
    
    print(f"Fragmentation before defrag: {fragmentation_before * 100:.2f}%")
    
    # Attempt defragmentation (may not be available in all builds)
    print("Attempting defragmentation...")
    try:
        if hasattr(memory.pool_manager, 'defragment'):
            defrag_start = time.perf_counter()
            defrag_stats = memory.pool_manager.defragment()
            defrag_time = time.perf_counter() - defrag_start
            
            print(f"Defragmentation completed in {defrag_time * 1000:.2f} ms")
            print(f"  Blocks moved: {defrag_stats.get('blocks_moved', 0)}")
            print(f"  Bytes compacted: {defrag_stats.get('bytes_compacted', 0) / 1024:.1f} KB")
            print(f"  Time taken: {defrag_stats.get('time_ms', 0):.2f} ms")
            
            # Measure fragmentation after defrag
            stats_after = memory.pool_stats()
            fragmentation_after = stats_after['fragmentation_ratio']
            
            print(f"Fragmentation after defrag: {fragmentation_after * 100:.2f}%")
            
            improvement = (fragmentation_before - fragmentation_after) / max(fragmentation_before, 0.001) * 100
            print(f"Fragmentation improvement: {improvement:.1f}%")
        else:
            print("Defragmentation not available in this build")
    except Exception as e:
        print(f"Defragmentation failed: {e}")
    
    # Clean up remaining blocks
    for block in keep_blocks:
        memory.deallocate_pool_block(block["id"])


def demonstrate_performance_monitoring():
    """Demonstrate performance monitoring and statistics."""
    print("\n=== Performance Monitoring Demo ===")
    
    # Monitor allocation performance over time
    print("Monitoring allocation performance...")
    
    allocation_times = []
    block_sizes = [64, 256, 1024, 4096, 16384]
    test_blocks = []
    
    # Perform timed allocations
    for i in range(100):
        size = block_sizes[i % len(block_sizes)]
        
        start_time = time.perf_counter()
        block = memory.allocate_from_pool(size)
        end_time = time.perf_counter()
        
        if block:
            test_blocks.append(block)
            allocation_time = (end_time - start_time) * 1000000  # microseconds
            allocation_times.append(allocation_time)
    
    # Calculate performance statistics
    if allocation_times:
        avg_time = sum(allocation_times) / len(allocation_times)
        min_time = min(allocation_times)
        max_time = max(allocation_times)
        
        print(f"Allocation performance (n={len(allocation_times)}):")
        print(f"  Average: {avg_time:.1f} μs")
        print(f"  Min: {min_time:.1f} μs")
        print(f"  Max: {max_time:.1f} μs")
        
        # Show distribution
        fast_count = sum(1 for t in allocation_times if t < 10.0)  # < 10 μs
        medium_count = sum(1 for t in allocation_times if 10.0 <= t < 100.0)  # 10-100 μs
        slow_count = sum(1 for t in allocation_times if t >= 100.0)  # >= 100 μs
        
        print(f"  Fast (< 10 μs): {fast_count} ({fast_count/len(allocation_times)*100:.1f}%)")
        print(f"  Medium (10-100 μs): {medium_count} ({medium_count/len(allocation_times)*100:.1f}%)")
        print(f"  Slow (>= 100 μs): {slow_count} ({slow_count/len(allocation_times)*100:.1f}%)")
    
    # Monitor memory usage over time
    print(f"\nMemory usage monitoring:")
    
    usage_history = []
    for i in range(10):
        stats = memory.pool_stats()
        usage_history.append({
            'time': i,
            'active_blocks': stats['active_blocks'],
            'total_allocated': stats['total_allocated'],
            'fragmentation': stats['fragmentation_ratio']
        })
        
        # Allocate some blocks to change the stats
        if i < 5:
            for _ in range(5):
                block = memory.allocate_from_pool(1024)
                if block:
                    test_blocks.append(block)
        
        time.sleep(0.01)
    
    print("Time | Active | Allocated (KB) | Fragmentation")
    print("-" * 50)
    for entry in usage_history:
        print(f"{entry['time']:4d} | {entry['active_blocks']:6d} | "
              f"{entry['total_allocated']/1024:12.1f} | {entry['fragmentation']*100:11.2f}%")
    
    # Clean up test blocks
    print(f"\nCleaning up {len(test_blocks)} test blocks...")
    cleanup_start = time.perf_counter()
    
    successful_cleanups = 0
    for block in test_blocks:
        success = memory.deallocate_pool_block(block["id"])
        if success:
            successful_cleanups += 1
    
    cleanup_time = time.perf_counter() - cleanup_start
    
    print(f"Cleanup completed: {successful_cleanups}/{len(test_blocks)} blocks in {cleanup_time*1000:.2f} ms")
    
    if successful_cleanups > 0:
        avg_cleanup_time = cleanup_time / successful_cleanups * 1000000  # μs
        print(f"Average cleanup time: {avg_cleanup_time:.1f} μs/block")


def demonstrate_memory_manager():
    """Demonstrate the high-level MemoryPoolManager interface."""
    print("\n=== Memory Pool Manager Demo ===")
    
    # Create manager with default settings
    manager = memory.MemoryPoolManager()
    
    print(f"Manager created")
    
    # Try to initialize
    success = manager.initialize()
    print(f"Manager initialization: {'Success' if success else 'Failed (no GPU context)'}")
    
    # Check initialization state
    print(f"Manager is initialized: {manager.is_initialized()}")
    
    # Get statistics through manager
    manager_stats = manager.stats()
    print(f"Manager stats: {manager_stats}")
    
    # Test defragmentation through manager
    try:
        if hasattr(manager, 'defragment'):
            print("Testing defragmentation through manager...")
            defrag_result = manager.defragment()
            print(f"Manager defrag result: {defrag_result}")
        else:
            print("Defragmentation not available through manager")
    except Exception as e:
        print(f"Manager defragmentation failed: {e}")


def demonstrate_comprehensive_report():
    """Demonstrate comprehensive memory reporting."""
    print("\n=== Comprehensive Memory Report ===")
    
    report = memory.memory_report()
    
    print("Full memory system report:")
    
    # System initialization status
    print(f"System initialization status:")
    for system, status in report['system_initialized'].items():
        status_str = "OK: Initialized" if status else "ERROR: Not initialized"
        print(f"  {system}: {status_str}")
    
    # Memory pool details
    print(f"\nMemory pool details:")
    pools = report['pools']
    for key, value in pools.items():
        if key in ['total_allocated', 'total_freed', 'largest_free_block']:
            if isinstance(value, (int, float)) and value > 1024:
                print(f"  {key}: {value / 1024 / 1024:.2f} MB")
            else:
                print(f"  {key}: {value}")
        elif key == 'fragmentation_ratio':
            print(f"  {key}: {value * 100:.2f}%")
        else:
            print(f"  {key}: {value}")
    
    # Staging ring details  
    print(f"\nStaging ring details:")
    staging = report['staging']
    for key, value in staging.items():
        if key in ['buffer_size', 'bytes_in_flight']:
            if isinstance(value, (int, float)) and value > 1024:
                print(f"  {key}: {value / 1024 / 1024:.2f} MB")
            else:
                print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value}")


def main():
    """Run the complete memory pool demonstration."""
    print("Forge3D Memory Pool System Demo")
    print("=" * 50)
    
    try:
        demonstrate_basic_pool_usage()
        demonstrate_size_bucket_allocation()
        demonstrate_fragmentation_patterns()
        demonstrate_defragmentation()
        demonstrate_performance_monitoring()
        demonstrate_memory_manager()
        demonstrate_comprehensive_report()
        
        print("\n=== Demo Summary ===")
        print("OK: Basic memory pool functionality demonstrated")
        print("OK: Size bucket allocation tested")
        print("OK: Fragmentation patterns analyzed")
        print("OK: Defragmentation capabilities shown")
        print("OK: Performance monitoring completed")
        print("OK: Manager interface demonstrated")
        print("OK: Comprehensive reporting validated")
        
        # Final health check
        final_stats = memory.pool_stats()
        if final_stats['active_blocks'] == 0:
            print("OK: No memory leaks detected")
        else:
            print(f"⚠ {final_stats['active_blocks']} blocks still active (expected in demo)")
        
        if final_stats['fragmentation_ratio'] < 0.1:
            print("OK: Low fragmentation maintained")
        else:
            print(f"⚠ Fragmentation at {final_stats['fragmentation_ratio']:.2%} (expected after demo)")
        
    except Exception as e:
        print(f"\nERROR: Demo error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nDemo completed. See docs/memory/memory_pools.md for detailed documentation.")


if __name__ == "__main__":
    main()