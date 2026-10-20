use super::*;
use crate::core::resource_tracker::ledger_snapshot;
use crate::core::texture_readback::tests::{abort_next_mapping, gpu, pause_next_mapping, upload};

#[tokio::test]
async fn test_async_readback_config() {
    let config = AsyncReadbackConfig::default();
    assert!(config.double_buffered);
    assert!(config.pre_allocate);
    assert_eq!(config.max_pending_ops, 4);
}

#[test]
fn test_copy_alignment() {
    assert_eq!(align_copy_bpr(100), 256);
    assert_eq!(align_copy_bpr(256), 256);
    assert_eq!(align_copy_bpr(300), 512);
}

#[test]
fn pending_reservation_is_atomic_and_released_on_drop() {
    let count = Arc::new(Mutex::new(0));
    let barrier = std::sync::Barrier::new(2);
    std::thread::scope(|scope| {
        let count = &count;
        let barrier = &barrier;
        scope.spawn(move || {
            let reservation = PendingOperation::reserve(count, 1).unwrap();
            barrier.wait();
            barrier.wait();
            drop(reservation);
        });
        barrier.wait();
        assert!(PendingOperation::reserve(count, 1).is_err());
        barrier.wait();
    });
    assert_eq!(*count.lock().unwrap(), 0);
    drop(PendingOperation::reserve(&count, 1).unwrap());
    assert_eq!(*count.lock().unwrap(), 0);
}

// A current-thread runtime makes queued-work assertions deterministic: submitting
// a readback does not yield, so the worker cannot run before these assertions.
#[tokio::test]
async fn consecutive_and_overlapping_readbacks_reuse_exclusive_leases() {
    let Some(context) = gpu() else { return };
    let config = AsyncReadbackConfig::default();
    let limit = config.max_pending_ops;
    let manager =
        AsyncReadbackManager::new(context.device.clone(), context.queue.clone(), config).unwrap();
    let mut pending = Vec::new();
    for index in 0..limit {
        let expected = vec![(index + 1) as u8; 7 * 3 * 4];
        let texture = upload(
            &context.device,
            &context.queue,
            7,
            3,
            wgpu::TextureFormat::Rgba8Unorm,
            &expected,
        );
        let mut handle = manager
            .readback_texture_async(&texture, 7, 3)
            .await
            .unwrap();
        assert!(handle.try_get().unwrap().is_none());
        assert_eq!(handle.expected_size(), 256 * 3);
        pending.push((handle, expected));
    }
    let stats = manager.get_stats();
    assert_eq!(stats.pending_operations, limit);
    assert_eq!(stats.in_use_buffers, limit);
    assert_eq!(stats.available_buffers, 0);
    let texture = upload(
        &context.device,
        &context.queue,
        7,
        3,
        wgpu::TextureFormat::Rgba8Unorm,
        &[91; 84],
    );
    assert!(manager
        .readback_texture_async(&texture, 7, 3)
        .await
        .is_err());
    for (handle, expected) in pending {
        assert_eq!(handle.wait().await.unwrap(), expected);
    }
    assert_eq!(manager.get_stats().pending_operations, 0);
    assert_eq!(manager.get_stats().in_use_buffers, 0);
    assert_eq!(manager.get_stats().available_buffers, limit);

    // More consecutive operations than the configured pending limit catches
    // both a leaked reservation and failure to return buffers to the pool.
    for _ in 0..=limit {
        let handle = manager
            .readback_texture_async(&texture, 7, 3)
            .await
            .unwrap();
        assert_eq!(handle.wait().await.unwrap(), vec![91; 84]);
        assert_eq!(manager.get_stats().pending_operations, 0);
    }
    assert_eq!(manager.get_stats().total_buffers, limit);
    // A smaller transfer uses only its own mapped range in a larger pool buffer.
    let small = upload(
        &context.device,
        &context.queue,
        3,
        1,
        wgpu::TextureFormat::Bgra8UnormSrgb,
        &[13; 12],
    );
    assert_eq!(
        manager
            .readback_texture_async(&small, 3, 1)
            .await
            .unwrap()
            .wait()
            .await
            .unwrap(),
        vec![13; 12]
    );
    assert_eq!(manager.get_stats().total_buffers, limit);
    // Works within a current-thread runtime without block_in_place panicking.
    assert_eq!(
        manager.readback_texture_sync(&small, 3, 1).unwrap(),
        vec![13; 12]
    );
    assert_eq!(manager.get_stats().pending_operations, 0);
    assert_eq!(manager.get_stats().in_use_buffers, 0);
}

#[tokio::test]
async fn dropped_handles_queue_failure_and_shutdown_release_operations() {
    let Some(context) = gpu() else { return };
    let texture = upload(
        &context.device,
        &context.queue,
        7,
        3,
        wgpu::TextureFormat::Rgba8Unorm,
        &[23; 84],
    );
    let mut manager = AsyncReadbackManager::new(
        context.device.clone(),
        context.queue.clone(),
        AsyncReadbackConfig::default(),
    )
    .unwrap();
    drop(
        manager
            .readback_texture_async(&texture, 7, 3)
            .await
            .unwrap(),
    );
    // A subsequent result proves the worker processed the discarded operation.
    assert_eq!(
        manager
            .readback_texture_async(&texture, 7, 3)
            .await
            .unwrap()
            .wait()
            .await
            .unwrap(),
        vec![23; 84]
    );
    assert_eq!(manager.get_stats().pending_operations, 0);
    assert_eq!(manager.get_stats().in_use_buffers, 0);
    assert!(manager
        .readback_texture_async(&texture, 0, 3)
        .await
        .is_err());
    assert_eq!(manager.get_stats().pending_operations, 0);

    let (closed_tx, closed_rx) = mpsc::unbounded_channel();
    drop(closed_rx);
    manager.worker_tx = closed_tx;
    assert!(manager
        .readback_texture_async(&texture, 7, 3)
        .await
        .is_err());
    assert_eq!(manager.get_stats().pending_operations, 0);
    assert_eq!(manager.get_stats().in_use_buffers, 0);
    assert_eq!(
        manager.readback_texture_sync(&texture, 7, 3).unwrap(),
        vec![23; 84]
    );
    drop(manager);

    let before = ledger_snapshot().by_label.get("async-readback-0").copied();
    let manager = AsyncReadbackManager::new(
        context.device.clone(),
        context.queue.clone(),
        AsyncReadbackConfig::default(),
    )
    .unwrap();
    let handle = manager
        .readback_texture_async(&texture, 7, 3)
        .await
        .unwrap();
    drop(manager);
    assert_eq!(handle.wait().await.unwrap(), vec![23; 84]);
    assert_eq!(
        ledger_snapshot().by_label.get("async-readback-0").copied(),
        before
    );
}

#[tokio::test]
async fn temporary_buffers_and_discarded_worker_tasks_release_leases() {
    let Some(context) = gpu() else { return };
    let config = AsyncReadbackConfig {
        pre_allocate: false,
        double_buffered: false,
        ..Default::default()
    };
    let mut manager =
        AsyncReadbackManager::new(context.device.clone(), context.queue.clone(), config).unwrap();
    let texture = upload(
        &context.device,
        &context.queue,
        7,
        3,
        wgpu::TextureFormat::Rgba8Unorm,
        &[17; 84],
    );
    let first = manager
        .readback_texture_async(&texture, 7, 3)
        .await
        .unwrap();
    let second = manager
        .readback_texture_async(&texture, 7, 3)
        .await
        .unwrap();
    assert_eq!(manager.get_stats().total_buffers, 1);
    assert_eq!(manager.get_stats().in_use_buffers, 1);
    assert!(!manager.get_stats().double_buffered);
    assert_eq!(first.wait().await.unwrap(), vec![17; 84]);
    assert_eq!(second.wait().await.unwrap(), vec![17; 84]);
    assert_eq!(manager.get_stats().pending_operations, 0);
    assert!(!ledger_snapshot()
        .by_label
        .contains_key("async_readback.temp_readback"));

    // Simulate worker shutdown dropping queued work before it starts mapping.
    let (worker_tx, mut worker_rx) = mpsc::unbounded_channel();
    manager.worker_tx = worker_tx;
    let mut handle = manager
        .readback_texture_async(&texture, 7, 3)
        .await
        .unwrap();
    drop(worker_rx.try_recv().unwrap());
    assert!(handle.try_get().is_err());
    assert_eq!(manager.get_stats().pending_operations, 0);
    assert_eq!(manager.get_stats().in_use_buffers, 0);
    assert_eq!(
        manager.readback_texture_sync(&texture, 7, 3).unwrap(),
        vec![17; 84]
    );
}

#[test]
fn manager_sync_readback_does_not_enter_a_runtime() {
    let Some(context) = gpu() else { return };
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let manager = runtime.block_on(async {
        AsyncReadbackManager::new(
            context.device.clone(),
            context.queue.clone(),
            AsyncReadbackConfig::default(),
        )
        .unwrap()
    });
    let texture = upload(
        &context.device,
        &context.queue,
        7,
        3,
        wgpu::TextureFormat::Rgba8Unorm,
        &[41; 84],
    );
    assert_eq!(
        manager.readback_texture_sync(&texture, 7, 3).unwrap(),
        vec![41; 84]
    );
    assert_eq!(manager.get_stats().pending_operations, 0);
}

#[tokio::test]
async fn worker_reports_real_mapping_failure_without_gpu_errors_and_remains_usable() {
    let Some(context) = gpu() else { return };
    let texture = upload(
        &context.device,
        &context.queue,
        7,
        3,
        wgpu::TextureFormat::Rgba8Unorm,
        &[67; 84],
    );
    let manager = AsyncReadbackManager::new(
        context.device.clone(),
        context.queue.clone(),
        AsyncReadbackConfig::default(),
    )
    .unwrap();
    {
        let _abort = abort_next_mapping();
        let error = manager
            .readback_texture_async(&texture, 7, 3)
            .await
            .unwrap()
            .wait()
            .await
            .unwrap_err();
        assert!(
            error.to_string().contains("Buffer mapping failed"),
            "{error}"
        );
    }
    assert_eq!(manager.get_stats().pending_operations, 0);
    assert_eq!(manager.get_stats().in_use_buffers, 0);
    assert_eq!(manager.get_stats().available_buffers, 1);
    context.assert_clean();
    assert_eq!(
        manager
            .readback_texture_async(&texture, 7, 3)
            .await
            .unwrap()
            .wait()
            .await
            .unwrap(),
        vec![67; 84]
    );
    assert_eq!(manager.get_stats().total_buffers, 1);
    context.assert_clean();
}

#[test]
fn actual_runtime_shutdown_during_mapping_releases_resources_without_gpu_errors() {
    let Some(context) = gpu() else { return };
    let texture = upload(
        &context.device,
        &context.queue,
        7,
        3,
        wgpu::TextureFormat::Rgba8Unorm,
        &[73; 84],
    );
    let before = ledger_snapshot().by_label.get("async-readback-0").copied();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    let (_pause, started, resume) = pause_next_mapping();
    let (manager, mut handle) = runtime.block_on(async {
        let manager = AsyncReadbackManager::new(
            context.device.clone(),
            context.queue.clone(),
            AsyncReadbackConfig::default(),
        )
        .unwrap();
        let handle = manager
            .readback_texture_async(&texture, 7, 3)
            .await
            .unwrap();
        // The test gate suspends the real worker after map_async accepted the
        // request and before any device poll can deliver its callback.
        started.await.unwrap();
        assert_eq!(manager.get_stats().pending_operations, 1);
        assert_eq!(manager.get_stats().in_use_buffers, 1);
        eprintln!("Shutting down a real Tokio runtime with a mapping request pending");
        (manager, handle)
    });
    drop(runtime);
    drop(resume);
    assert_eq!(manager.get_stats().pending_operations, 0);
    assert!(handle.try_get().is_err());
    // Polling the device retires GPU work even though the async runtime is gone.
    context.device.poll(wgpu::Maintain::Wait);
    assert_eq!(manager.get_stats().in_use_buffers, 0);
    assert_eq!(manager.get_stats().available_buffers, 1);
    context.assert_clean();
    assert_eq!(
        manager.readback_texture_sync(&texture, 7, 3).unwrap(),
        vec![73; 84]
    );
    drop(manager);
    assert_eq!(
        ledger_snapshot().by_label.get("async-readback-0").copied(),
        before
    );
    context.assert_clean();
}
