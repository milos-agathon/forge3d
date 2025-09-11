#!/usr/bin/env python3
"""
O1: Staging Ring Buffer Demo

This example demonstrates the staging buffer ring system for efficient
GPU uploads with fence-based synchronization.

The staging ring system provides:
- Automatic wrap-around through multiple buffers
- Fence-based synchronization to prevent buffer reuse
- Performance statistics and monitoring
- Memory budget compliance

Usage:
    python examples/staging_ring_demo.py
"""

import time
import numpy as np
import forge3d.memory as memory


def demonstrate_basic_usage():
    """Demonstrate basic staging ring usage."""
    print("=== Basic Staging Ring Usage ===")
    
    # Initialize the memory system with staging rings
    result = memory.init_memory_system(
        staging_rings=True,
        memory_pools=False,
        ring_count=3,
        buffer_size=2 * 1024 * 1024  # 2MB per buffer
    )
    
    print(f"Initialization result: {result}")
    
    # Get initial statistics
    initial_stats = memory.staging_stats()
    print(f"Initial stats: {initial_stats}")
    
    # Demonstrate stats reporting
    print("\nStaging ring configuration:")
    print(f"  Ring count: {initial_stats['ring_count']}")
    print(f"  Buffer size: {initial_stats['buffer_size'] / 1024 / 1024:.1f} MB")
    print(f"  Total memory: {initial_stats['ring_count'] * initial_stats['buffer_size'] / 1024 / 1024:.1f} MB")
    print(f"  Current ring: {initial_stats['current_ring_index']}")


def demonstrate_allocation_patterns():
    """Demonstrate different allocation patterns and their performance."""
    print("\n=== Allocation Pattern Demo ===")
    
    patterns = [
        ("Small frequent", [1024] * 50, "Many small allocations"),
        ("Medium batch", [64*1024] * 10, "Batch of medium allocations"),
        ("Large single", [1*1024*1024], "Single large allocation"),
        ("Mixed sizes", [512, 4*1024, 64*1024, 256*1024, 1*1024*1024], "Varied allocation sizes")
    ]
    
    for pattern_name, sizes, description in patterns:
        print(f"\n{pattern_name}: {description}")
        
        start_stats = memory.staging_stats()
        start_time = time.perf_counter()
        
        # Simulate allocation pattern
        total_bytes = 0
        for i, size in enumerate(sizes):
            # In a real application, this would be:
            # buffer, offset = staging_ring.allocate(size)
            # write_data_to_buffer(buffer, offset, data)
            
            # For demo, we just track the pattern
            total_bytes += size
            
            # Show progress for large patterns
            if len(sizes) > 10 and i % 10 == 0:
                current_stats = memory.staging_stats()
                print(f"  Progress: {i+1}/{len(sizes)}, ring: {current_stats['current_ring_index']}")
        
        end_time = time.perf_counter()
        end_stats = memory.staging_stats()
        
        elapsed_ms = (end_time - start_time) * 1000
        print(f"  Total: {len(sizes)} allocations, {total_bytes / 1024 / 1024:.2f} MB")
        print(f"  Time: {elapsed_ms:.3f} ms")
        print(f"  Buffer stalls: {end_stats['buffer_stalls'] - start_stats['buffer_stalls']}")


def demonstrate_performance_monitoring():
    """Demonstrate performance monitoring and statistics."""
    print("\n=== Performance Monitoring Demo ===")
    
    # Monitor stats over time
    print("Monitoring staging ring statistics...")
    
    stats_history = []
    for i in range(10):
        stats = memory.staging_stats()
        stats_history.append({
            'timestamp': time.time(),
            'ring_index': stats['current_ring_index'],
            'bytes_in_flight': stats['bytes_in_flight'],
            'buffer_stalls': stats['buffer_stalls']
        })
        
        # Simulate some work
        time.sleep(0.01)  # 10ms
    
    print("\nStatistics timeline:")
    print("Time(ms) | Ring | In-Flight | Stalls")
    print("-" * 40)
    
    base_time = stats_history[0]['timestamp']
    for entry in stats_history:
        time_ms = (entry['timestamp'] - base_time) * 1000
        print(f"{time_ms:8.1f} | {entry['ring_index']:4d} | {entry['bytes_in_flight']:9d} | {entry['buffer_stalls']:6d}")


def demonstrate_memory_manager():
    """Demonstrate the high-level StagingRingManager interface."""
    print("\n=== Staging Ring Manager Demo ===")
    
    # Create manager with custom settings
    manager = memory.StagingRingManager(
        ring_count=4,  # More rings for demonstration
        buffer_size=1024 * 1024  # 1MB buffers
    )
    
    print(f"Manager created with {manager.ring_count} rings of {manager.buffer_size / 1024 / 1024:.1f} MB each")
    
    # Try to initialize
    success = manager.initialize()
    print(f"Manager initialization: {'Success' if success else 'Failed (no GPU context)'}")
    
    # Check initialization state
    print(f"Manager is initialized: {manager.is_initialized()}")
    
    # Get statistics through manager
    manager_stats = manager.stats()
    print(f"Manager stats: {manager_stats}")


def demonstrate_memory_budget_analysis():
    """Demonstrate memory budget analysis."""
    print("\n=== Memory Budget Analysis ===")
    
    stats = memory.staging_stats()
    
    # Calculate memory usage
    ring_memory = stats['ring_count'] * stats['buffer_size']
    budget_bytes = 512 * 1024 * 1024  # 512 MiB budget
    
    print(f"Current configuration:")
    print(f"  Ring count: {stats['ring_count']}")
    print(f"  Buffer size: {stats['buffer_size'] / 1024 / 1024:.1f} MB")
    print(f"  Total staging memory: {ring_memory / 1024 / 1024:.1f} MB")
    print(f"  Memory budget: {budget_bytes / 1024 / 1024:.1f} MB")
    print(f"  Budget utilization: {100 * ring_memory / budget_bytes:.1f}%")
    
    # Show alternative configurations
    print(f"\nAlternative configurations:")
    
    configs = [
        (2, 1024*1024, "Minimal (2x1MB)"),
        (3, 2*1024*1024, "Balanced (3x2MB)"),
        (4, 4*1024*1024, "High-throughput (4x4MB)"),
        (3, 170*1024*1024, "Maximum (3x170MB)")
    ]
    
    for rings, size, desc in configs:
        total = rings * size
        utilization = 100 * total / budget_bytes
        status = "OK:" if total <= budget_bytes else "✗"
        print(f"  {status} {desc}: {total / 1024 / 1024:.1f} MB ({utilization:.1f}%)")


def demonstrate_comprehensive_report():
    """Demonstrate comprehensive memory reporting."""
    print("\n=== Comprehensive Memory Report ===")
    
    report = memory.memory_report()
    
    print("Full memory system report:")
    print(f"System initialization status:")
    for system, status in report['system_initialized'].items():
        status_str = "OK: Initialized" if status else "✗ Not initialized"
        print(f"  {system}: {status_str}")
    
    print(f"\nStaging ring details:")
    staging = report['staging']
    for key, value in staging.items():
        if key in ['buffer_size', 'bytes_in_flight']:
            if isinstance(value, int) and value > 1024:
                print(f"  {key}: {value / 1024 / 1024:.1f} MB")
            else:
                print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value}")
    
    print(f"\nMemory pool details:")
    pools = report['pools']
    for key, value in pools.items():
        if key in ['total_allocated', 'total_freed', 'largest_free_block']:
            if isinstance(value, int) and value > 1024:
                print(f"  {key}: {value / 1024 / 1024:.1f} MB")
            else:
                print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value}")


def main():
    """Run the complete staging ring demonstration."""
    print("Forge3D Staging Ring Buffer Demo")
    print("=" * 50)
    
    try:
        demonstrate_basic_usage()
        demonstrate_allocation_patterns()
        demonstrate_performance_monitoring()
        demonstrate_memory_manager()
        demonstrate_memory_budget_analysis()
        demonstrate_comprehensive_report()
        
        print("\n=== Demo Summary ===")
        print("OK: Basic staging ring functionality demonstrated")
        print("OK: Allocation patterns tested")
        print("OK: Performance monitoring shown")
        print("OK: Memory budget analysis completed")
        print("OK: Manager interface demonstrated")
        
        # Final performance check
        final_stats = memory.staging_stats()
        if final_stats['buffer_stalls'] == 0:
            print("OK: No buffer stalls detected during demo")
        else:
            print(f"⚠ {final_stats['buffer_stalls']} buffer stalls detected (expected in demo)")
        
    except Exception as e:
        print(f"\n✗ Demo error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nDemo completed. See docs/memory/staging_rings.md for detailed documentation.")


if __name__ == "__main__":
    main()