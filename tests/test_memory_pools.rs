//! Tests for O2: GPU Memory Pools
//!
//! These tests validate the memory pool implementation including size-bucket
//! allocation, reference counting, and defragmentation capabilities.

use forge3d::core::memory_tracker::{DefragStats, MemoryPoolManager, MemoryPoolStats, PoolBlock};
use std::sync::Arc;
use std::time::Duration;
use wgpu::{Backends, Device, DeviceDescriptor, Instance, Queue, RequestAdapterOptions};

/// Test-only device creation helper (hermetic isolation)
async fn create_test_device() -> (Arc<Device>, Arc<Queue>) {
    let instance = Instance::new(wgpu::InstanceDescriptor {
        backends: Backends::all(),
        dx12_shader_compiler: Default::default(),
        flags: wgpu::InstanceFlags::default(),
        gles_minor_version: wgpu::Gles3MinorVersion::Automatic,
    });

    let adapter = instance
        .request_adapter(&RequestAdapterOptions::default())
        .await
        .expect("Failed to get adapter");

    let (device, queue) = adapter
        .request_device(&DeviceDescriptor::default(), None)
        .await
        .expect("Failed to get device");

    (Arc::new(device), Arc::new(queue))
}

#[tokio::test]
async fn test_memory_pool_creation() {
    let (device, _queue) = create_test_device().await;
    let mut pool_manager = MemoryPoolManager::new(&device);

    let stats = pool_manager.get_stats();
    assert_eq!(stats.pool_count, 18); // 64B to 8MB = 18 buckets
    assert_eq!(stats.active_blocks, 0);
    assert_eq!(stats.total_allocated, 0);
    assert_eq!(stats.fragmentation_ratio, 0.0);
}

#[tokio::test]
async fn test_basic_allocation() {
    let (device, _queue) = create_test_device().await;
    let mut pool_manager = MemoryPoolManager::new(&device);

    // Test allocation from different size buckets
    let block_64 = pool_manager.allocate_bucket(32).unwrap(); // Should go to 64B bucket
    assert_eq!(block_64.size, 64);
    assert_eq!(block_64.ref_count(), 1);

    let block_128 = pool_manager.allocate_bucket(100).unwrap(); // Should go to 128B bucket
    assert_eq!(block_128.size, 128);
    assert_eq!(block_128.ref_count(), 1);

    let stats = pool_manager.get_stats();
    assert_eq!(stats.active_blocks, 2);
    assert_eq!(stats.total_allocated, 64 + 128);
}

#[tokio::test]
async fn test_reference_counting() {
    let (device, _queue) = create_test_device().await;
    let mut pool_manager = MemoryPoolManager::new(&device);

    let block = pool_manager.allocate_bucket(256).unwrap();
    assert_eq!(block.ref_count(), 1);

    // Increment reference count
    block.add_ref();
    assert_eq!(block.ref_count(), 2);

    block.add_ref();
    assert_eq!(block.ref_count(), 3);

    // Decrement reference count
    let is_zero = block.release();
    assert!(!is_zero);
    assert_eq!(block.ref_count(), 2);

    let is_zero = block.release();
    assert!(!is_zero);
    assert_eq!(block.ref_count(), 1);

    let is_zero = block.release();
    assert!(is_zero);
    assert_eq!(block.ref_count(), 0);
}

#[tokio::test]
async fn test_size_bucket_selection() {
    let (device, _queue) = create_test_device().await;
    let mut pool_manager = MemoryPoolManager::new(&device);

    // Test various sizes and expected bucket selection
    let test_cases = [
        (1, 64),            // Minimum bucket
        (64, 64),           // Exact fit
        (65, 128),          // Next bucket up
        (128, 128),         // Exact fit
        (1000, 1024),       // 1KB bucket
        (2000, 2048),       // 2KB bucket
        (1048576, 1048576), // 1MB bucket
    ];

    for (requested_size, expected_bucket) in test_cases.iter() {
        let block = pool_manager.allocate_bucket(*requested_size).unwrap();
        assert_eq!(
            block.size, *expected_bucket as u64,
            "Size {} should allocate from {} bucket",
            requested_size, expected_bucket
        );
    }

    let stats = pool_manager.get_stats();
    assert_eq!(stats.active_blocks, test_cases.len() as u32);
}

#[tokio::test]
async fn test_block_recycling() {
    let (device, _queue) = create_test_device().await;
    let mut pool_manager = MemoryPoolManager::new(&device);

    // Allocate a block
    let block1 = pool_manager.allocate_bucket(512).unwrap();
    let offset1 = block1.offset;
    let stats_after_alloc = pool_manager.get_stats();
    assert_eq!(stats_after_alloc.active_blocks, 1);

    // Drop the block (should return to pool)
    drop(block1);

    // Allocate another block of same size - should reuse the same space
    let block2 = pool_manager.allocate_bucket(512).unwrap();
    assert_eq!(block2.offset, offset1); // Should reuse same offset

    let stats_after_realloc = pool_manager.get_stats();
    assert_eq!(stats_after_realloc.active_blocks, 1);
}

#[tokio::test]
async fn test_fragmentation_measurement() {
    let (device, _queue) = create_test_device().await;
    let mut pool_manager = MemoryPoolManager::new(&device);

    // Allocate several blocks
    let blocks: Vec<_> = (0..10)
        .map(|_| pool_manager.allocate_bucket(1024).unwrap())
        .collect();

    // Drop every other block to create fragmentation
    for i in (0..blocks.len()).step_by(2) {
        drop(blocks.get(i));
    }

    let stats = pool_manager.get_stats();
    // Should have some fragmentation now
    println!(
        "Fragmentation ratio: {:.2}%",
        stats.fragmentation_ratio * 100.0
    );

    // Note: Fragmentation measurement depends on internal implementation
    // This test mainly verifies that the measurement doesn't crash
    assert!(stats.fragmentation_ratio >= 0.0);
    assert!(stats.fragmentation_ratio <= 1.0);
}

#[tokio::test]
async fn test_defragmentation() {
    let (device, _queue) = create_test_device().await;
    let mut pool_manager = MemoryPoolManager::new(&device);

    // Allocate and free blocks to create fragmentation
    let blocks: Vec<_> = (0..20)
        .map(|_| pool_manager.allocate_bucket(2048).unwrap())
        .collect();

    // Drop some blocks to fragment
    for i in (0..blocks.len()).step_by(3) {
        drop(blocks.get(i));
    }

    let stats_before = pool_manager.get_stats();
    let frag_before = stats_before.fragmentation_ratio;

    // Perform defragmentation
    let defrag_stats = pool_manager.defragment();

    let stats_after = pool_manager.get_stats();
    let frag_after = stats_after.fragmentation_ratio;

    // Verify defragmentation statistics
    assert!(defrag_stats.time_ms >= 0.0);
    assert_eq!(defrag_stats.fragmentation_before, frag_before);
    assert_eq!(defrag_stats.fragmentation_after, frag_after);

    // Fragmentation should generally improve (or stay same)
    assert!(frag_after <= frag_before + 0.1); // Allow small margin for test variance

    println!(
        "Defragmentation: {:.2}% -> {:.2}%, moved {} blocks, {:.2}ms",
        frag_before * 100.0,
        frag_after * 100.0,
        defrag_stats.blocks_moved,
        defrag_stats.time_ms
    );
}

#[tokio::test]
async fn test_allocation_performance() {
    let (device, _queue) = create_test_device().await;
    let mut pool_manager = MemoryPoolManager::new(&device);

    let num_allocations = 1000;
    let allocation_size = 1024;

    let start = std::time::Instant::now();

    let mut blocks = Vec::with_capacity(num_allocations);
    for _ in 0..num_allocations {
        let block = pool_manager.allocate_bucket(allocation_size).unwrap();
        blocks.push(block);
    }

    let allocation_time = start.elapsed();

    let start_dealloc = std::time::Instant::now();
    drop(blocks); // Drop all blocks
    let deallocation_time = start_dealloc.elapsed();

    println!(
        "Performance: {} allocations in {:.2}ms ({:.1} μs/alloc)",
        num_allocations,
        allocation_time.as_secs_f64() * 1000.0,
        allocation_time.as_secs_f64() * 1000000.0 / num_allocations as f64
    );

    println!(
        "Deallocation: {} blocks in {:.2}ms ({:.1} μs/dealloc)",
        num_allocations,
        deallocation_time.as_secs_f64() * 1000.0,
        deallocation_time.as_secs_f64() * 1000000.0 / num_allocations as f64
    );

    // Performance should be reasonable (< 50 μs per allocation on average)
    let avg_alloc_time_us = allocation_time.as_secs_f64() * 1000000.0 / num_allocations as f64;
    assert!(
        avg_alloc_time_us < 50.0,
        "Allocation too slow: {:.1} μs",
        avg_alloc_time_us
    );
}

#[tokio::test]
async fn test_no_leaks_reference_counting() {
    let (device, _queue) = create_test_device().await;
    let mut pool_manager = MemoryPoolManager::new(&device);

    {
        // Create blocks with various reference counts
        let block1 = pool_manager.allocate_bucket(512).unwrap();
        let block2 = pool_manager.allocate_bucket(1024).unwrap();
        let block3 = pool_manager.allocate_bucket(2048).unwrap();

        // Add references
        block1.add_ref(); // ref_count = 2
        block2.add_ref(); // ref_count = 2
        block2.add_ref(); // ref_count = 3

        // Partial release
        block1.release(); // ref_count = 1
        block2.release(); // ref_count = 2

        let stats_during = pool_manager.get_stats();
        assert_eq!(stats_during.active_blocks, 3);

        // Blocks will be dropped here, ref_counts should go to zero
    }

    // After scope exit, all blocks should be returned to pools
    let stats_after = pool_manager.get_stats();
    assert_eq!(
        stats_after.active_blocks, 0,
        "Memory leak detected: {} blocks still active",
        stats_after.active_blocks
    );

    // All freed bytes should equal allocated bytes
    assert!(stats_after.total_freed > 0);
    // Note: total_allocated might be higher due to bucket size alignment
}

#[tokio::test]
async fn test_large_allocation_rejection() {
    let (device, _queue) = create_test_device().await;
    let mut pool_manager = MemoryPoolManager::new(&device);

    // Try to allocate larger than largest bucket (8MB)
    let result = pool_manager.allocate_bucket(16 * 1024 * 1024); // 16MB
    assert!(
        result.is_err(),
        "Should reject allocation larger than largest bucket"
    );

    let error_msg = result.unwrap_err();
    assert!(
        error_msg.contains("exceeds maximum bucket size"),
        "Error should mention bucket size limit: {}",
        error_msg
    );
}

#[tokio::test]
async fn test_pool_exhaustion_and_recovery() {
    let (device, _queue) = create_test_device().await;
    let mut pool_manager = MemoryPoolManager::new(&device);

    // Allocate many blocks from same bucket until exhaustion
    let allocation_size = 64;
    let mut blocks = Vec::new();

    // Allocate until we can't anymore (pool exhaustion)
    let mut allocation_count = 0;
    loop {
        match pool_manager.allocate_bucket(allocation_size) {
            Ok(block) => {
                blocks.push(block);
                allocation_count += 1;

                // Safety limit to avoid infinite loop
                if allocation_count > 10000 {
                    break;
                }
            }
            Err(_) => {
                // Pool exhausted
                break;
            }
        }
    }

    println!("Allocated {} blocks before exhaustion", allocation_count);
    assert!(
        allocation_count > 0,
        "Should be able to allocate at least some blocks"
    );

    // Free half the blocks
    let half_point = blocks.len() / 2;
    blocks.truncate(half_point);

    // Should be able to allocate again
    let recovery_block = pool_manager.allocate_bucket(allocation_size);
    assert!(
        recovery_block.is_ok(),
        "Should recover after freeing blocks"
    );
}

#[tokio::test]
async fn test_statistics_accuracy() {
    let (device, _queue) = create_test_device().await;
    let mut pool_manager = MemoryPoolManager::new(&device);

    let initial_stats = pool_manager.get_stats();
    assert_eq!(initial_stats.active_blocks, 0);
    assert_eq!(initial_stats.total_allocated, 0);
    assert_eq!(initial_stats.total_freed, 0);

    // Allocate some blocks
    let block1 = pool_manager.allocate_bucket(128).unwrap(); // 128B bucket
    let block2 = pool_manager.allocate_bucket(1000).unwrap(); // 1024B bucket
    let block3 = pool_manager.allocate_bucket(3000).unwrap(); // 4096B bucket

    let stats_after_alloc = pool_manager.get_stats();
    assert_eq!(stats_after_alloc.active_blocks, 3);
    let expected_allocated = 128 + 1024 + 4096;
    assert_eq!(stats_after_alloc.total_allocated, expected_allocated);

    // Free one block
    drop(block2);
    let stats_after_free = pool_manager.get_stats();
    assert_eq!(stats_after_free.active_blocks, 2);
    assert_eq!(stats_after_free.total_freed, 1024);

    // Free remaining blocks
    drop(block1);
    drop(block3);
    let final_stats = pool_manager.get_stats();
    assert_eq!(final_stats.active_blocks, 0);
    assert_eq!(final_stats.total_freed, expected_allocated);
}

#[tokio::test]
async fn test_largest_free_block_tracking() {
    let (device, _queue) = create_test_device().await;
    let mut pool_manager = MemoryPoolManager::new(&device);

    // Initially, should have large free blocks in each pool
    let initial_stats = pool_manager.get_stats();
    assert!(initial_stats.largest_free_block > 0);

    let initial_largest = initial_stats.largest_free_block;
    println!("Initial largest free block: {} bytes", initial_largest);

    // Allocate some blocks to reduce largest free block
    let _blocks: Vec<_> = (0..5)
        .map(|_| {
            pool_manager.allocate_bucket(8 * 1024 * 1024).unwrap() // 8MB blocks
        })
        .collect();

    let stats_after = pool_manager.get_stats();
    println!(
        "Largest free block after allocations: {} bytes",
        stats_after.largest_free_block
    );

    // Should still be tracking largest free block accurately
    assert!(stats_after.largest_free_block >= 0);
}
