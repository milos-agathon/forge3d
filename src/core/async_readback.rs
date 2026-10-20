//! Async texture readback scheduling with reusable staging leases.
//! The transfer module owns allocation, copying, mapping and cleanup.

use super::error::RenderError;
use super::texture_readback::{pool::StagingPool, ReadbackRequest, TextureReadback};
use std::sync::{Arc, Mutex};
use tokio::sync::{mpsc, oneshot};
use wgpu::{Device, Queue, Texture};

/// Configuration for async readback operations.
#[derive(Debug, Clone)]
pub struct AsyncReadbackConfig {
    /// Report the caller's requested double-buffering mode.
    pub double_buffered: bool,
    /// Retain newly allocated buffers in the pool (otherwise retain only the first).
    pub pre_allocate: bool,
    /// Maximum number of pending readback operations.
    pub max_pending_ops: usize,
}

impl Default for AsyncReadbackConfig {
    fn default() -> Self {
        Self {
            double_buffered: true,
            pre_allocate: true,
            max_pending_ops: 4,
        }
    }
}

/// Handle for a pending async readback operation.
pub struct AsyncReadbackHandle {
    receiver: oneshot::Receiver<Result<Vec<u8>, RenderError>>,
    size: u64,
}

impl AsyncReadbackHandle {
    /// Wait for completion. Dropping the handle discards the result; the worker
    /// still finishes the transfer and releases its staging lease.
    pub async fn wait(self) -> Result<Vec<u8>, RenderError> {
        self.receiver
            .await
            .map_err(|_| RenderError::readback("Readback operation was cancelled"))?
    }

    /// Try to get the result if available (non-blocking).
    pub fn try_get(&mut self) -> Result<Option<Vec<u8>>, RenderError> {
        match self.receiver.try_recv() {
            Ok(result) => result.map(Some),
            Err(oneshot::error::TryRecvError::Empty) => Ok(None),
            Err(oneshot::error::TryRecvError::Closed) => {
                Err(RenderError::readback("Readback operation was cancelled"))
            }
        }
    }

    /// Size of the padded GPU transfer, preserving the existing handle contract.
    pub fn expected_size(&self) -> u64 {
        self.size
    }
}

/// A reservation is taken atomically before submitting any GPU work. Its Drop
/// covers normal completion, worker shutdown, queue failure and cancellation.
struct PendingOperation {
    count: Arc<Mutex<usize>>,
}

impl PendingOperation {
    fn reserve(count: &Arc<Mutex<usize>>, limit: usize) -> Result<Self, RenderError> {
        let mut pending = count.lock().unwrap();
        if *pending >= limit {
            return Err(RenderError::readback(format!(
                "Too many pending readback operations ({}/{})",
                *pending, limit,
            )));
        }
        *pending += 1;
        Ok(Self {
            count: count.clone(),
        })
    }
}

impl Drop for PendingOperation {
    fn drop(&mut self) {
        *self.count.lock().unwrap() -= 1;
    }
}

/// Async readback manager. Construction requires an active Tokio runtime;
/// synchronous reads themselves use the shared blocking transfer path.
pub struct AsyncReadbackManager {
    device: Arc<Device>,
    queue: Arc<Queue>,
    config: AsyncReadbackConfig,
    buffers: StagingPool,
    pending_ops: Arc<Mutex<usize>>,
    worker_tx: mpsc::UnboundedSender<ReadbackTask>,
}

struct ReadbackTask {
    transfer: TextureReadback,
    operation: PendingOperation,
    sender: oneshot::Sender<Result<Vec<u8>, RenderError>>,
}

impl AsyncReadbackManager {
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        config: AsyncReadbackConfig,
    ) -> Result<Self, RenderError> {
        let runtime = tokio::runtime::Handle::try_current()
            .map_err(|_| RenderError::readback("Async readback requires a Tokio runtime"))?;
        let (worker_tx, mut worker_rx) = mpsc::unbounded_channel::<ReadbackTask>();
        let device_worker = device.clone();
        runtime.spawn(async move {
            while let Some(task) = worker_rx.recv().await {
                let ReadbackTask {
                    transfer,
                    operation,
                    sender,
                } = task;
                let result = transfer.read_async(&device_worker).await;
                // All GPU and scheduling resources are released before the
                // result wakes the caller, including a caller dropping its handle.
                drop(operation);
                let _ = sender.send(result);
            }
        });
        Ok(Self {
            device,
            queue,
            config,
            buffers: StagingPool::default(),
            pending_ops: Arc::new(Mutex::new(0)),
            worker_tx,
        })
    }

    fn prepare(
        &self,
        texture: &Texture,
        width: u32,
        height: u32,
    ) -> Result<(TextureReadback, PendingOperation, u64), RenderError> {
        let operation = PendingOperation::reserve(&self.pending_ops, self.config.max_pending_ops)?;
        let request =
            ReadbackRequest::new(texture, width, height, wgpu::TextureFormat::Rgba8Unorm)?;
        let size = request.layout.transfer_bytes;
        let staging = self
            .buffers
            .acquire(&self.device, size, self.config.pre_allocate)?;
        let transfer = staging.submit(&self.device, &self.queue, texture, request);
        Ok((transfer, operation, size))
    }

    /// Start an async readback operation.
    pub async fn readback_texture_async(
        &self,
        texture: &Texture,
        width: u32,
        height: u32,
    ) -> Result<AsyncReadbackHandle, RenderError> {
        let (transfer, operation, size) = self.prepare(texture, width, height)?;
        let (sender, receiver) = oneshot::channel();
        self.worker_tx
            .send(ReadbackTask {
                transfer,
                operation,
                sender,
            })
            .map_err(|_| RenderError::readback("Worker task queue closed"))?;
        Ok(AsyncReadbackHandle { receiver, size })
    }

    /// Blocking readback; no runtime entry, block_in_place or async worker wait.
    pub fn readback_texture_sync(
        &self,
        texture: &Texture,
        width: u32,
        height: u32,
    ) -> Result<Vec<u8>, RenderError> {
        let (transfer, _operation, _) = self.prepare(texture, width, height)?;
        transfer.read_sync(&self.device)
    }

    pub fn get_stats(&self) -> AsyncReadbackStats {
        let (total_buffers, in_use_buffers, total_buffer_memory) = self.buffers.stats();
        AsyncReadbackStats {
            pending_operations: *self.pending_ops.lock().unwrap(),
            total_buffers,
            in_use_buffers,
            available_buffers: total_buffers - in_use_buffers,
            total_buffer_memory,
            double_buffered: self.config.double_buffered,
        }
    }
}

/// Statistics for async readback operations.
#[derive(Debug, Clone)]
pub struct AsyncReadbackStats {
    pub pending_operations: usize,
    pub total_buffers: usize,
    pub in_use_buffers: usize,
    pub available_buffers: usize,
    pub total_buffer_memory: u64,
    pub double_buffered: bool,
}

#[cfg(test)]
fn align_copy_bpr(unpadded: u32) -> u32 {
    super::texture_layout::align_copy_row(unpadded).expect("Texture row exceeds u32")
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "async_readback_tests.rs"]
mod tests;
