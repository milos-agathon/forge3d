//! Async and double-buffered readback system
//!
//! Provides asynchronous texture readback with optional double-buffering
//! for improved performance in scenarios with frequent readbacks.

use super::error::RenderError;
use crate::core::memory_tracker::global_tracker;
use std::num::NonZeroU32;
use std::sync::{Arc, Mutex};
use tokio::sync::{mpsc, oneshot};
use wgpu::{Buffer, BufferDescriptor, BufferUsages, Device, Queue, Texture};

/// Configuration for async readback operations
#[derive(Debug, Clone)]
pub struct AsyncReadbackConfig {
    /// Enable double-buffering for overlapped readbacks
    pub double_buffered: bool,
    /// Pre-allocate buffers for better performance
    pub pre_allocate: bool,
    /// Maximum number of pending readback operations
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

/// Handle for a pending async readback operation
pub struct AsyncReadbackHandle {
    receiver: oneshot::Receiver<Result<Vec<u8>, RenderError>>,
    size: u64,
}

impl AsyncReadbackHandle {
    /// Wait for the readback to complete and get the result
    pub async fn wait(self) -> Result<Vec<u8>, RenderError> {
        self.receiver
            .await
            .map_err(|_| RenderError::Readback("Readback operation was cancelled".to_string()))?
    }

    /// Try to get the result if available (non-blocking)
    pub fn try_get(&mut self) -> Result<Option<Vec<u8>>, RenderError> {
        match self.receiver.try_recv() {
            Ok(result) => result.map(Some),
            Err(oneshot::error::TryRecvError::Empty) => Ok(None),
            Err(oneshot::error::TryRecvError::Closed) => Err(RenderError::Readback(
                "Readback operation was cancelled".to_string(),
            )),
        }
    }

    /// Get the expected size of the readback data
    pub fn expected_size(&self) -> u64 {
        self.size
    }
}

/// Internal readback buffer state
struct ReadbackBuffer {
    buffer: Arc<Buffer>,
    size: u64,
    in_use: bool,
}

impl ReadbackBuffer {
    fn new(device: &Device, size: u64, label: &str) -> Self {
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some(label),
            size,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Track allocation
        global_tracker().track_buffer_allocation(size, true); // host-visible

        Self {
            buffer: Arc::new(buffer),
            size,
            in_use: false,
        }
    }
}

impl Drop for ReadbackBuffer {
    fn drop(&mut self) {
        // Free allocation tracking
        global_tracker().free_buffer_allocation(self.size, true);
    }
}

/// Async readback manager with double-buffering support
pub struct AsyncReadbackManager {
    device: Arc<Device>,
    queue: Arc<Queue>,
    config: AsyncReadbackConfig,
    buffers: Mutex<Vec<ReadbackBuffer>>,
    pending_ops: Arc<Mutex<usize>>,
    worker_tx: mpsc::UnboundedSender<ReadbackTask>,
}

/// Internal task for the readback worker
struct ReadbackTask {
    buffer: Arc<Buffer>,
    #[allow(dead_code)]
    width: u32,
    height: u32,
    padded_bpr: u32,
    row_bytes: u32,
    sender: oneshot::Sender<Result<Vec<u8>, RenderError>>,
}

impl AsyncReadbackManager {
    /// Create a new async readback manager
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        config: AsyncReadbackConfig,
    ) -> Result<Self, RenderError> {
        let (worker_tx, mut worker_rx) = mpsc::unbounded_channel::<ReadbackTask>();

        // Spawn worker task for processing readbacks
        let device_worker = device.clone();
        tokio::spawn(async move {
            while let Some(task) = worker_rx.recv().await {
                Self::process_readback_task(device_worker.clone(), task).await;
            }
        });

        Ok(Self {
            device,
            queue,
            config,
            buffers: Mutex::new(Vec::new()),
            pending_ops: Arc::new(Mutex::new(0)),
            worker_tx,
        })
    }

    /// Start an async readback operation
    pub async fn readback_texture_async(
        &self,
        texture: &Texture,
        width: u32,
        height: u32,
    ) -> Result<AsyncReadbackHandle, RenderError> {
        // Check if we're at the limit for pending operations
        {
            let pending_count = *self.pending_ops.lock().unwrap();
            if pending_count >= self.config.max_pending_ops {
                return Err(RenderError::Readback(format!(
                    "Too many pending readback operations ({}/{})",
                    pending_count, self.config.max_pending_ops
                )));
            }
        }

        let row_bytes = width * 4; // Assume RGBA8
        let padded_bpr = align_copy_bpr(row_bytes);
        let buffer_size = (padded_bpr * height) as u64;

        // Get or create a readback buffer
        let buffer = self.get_readback_buffer(buffer_size)?;

        // Submit copy command
        self.submit_copy_command(texture, buffer.as_ref(), width, height, padded_bpr)?;

        // Create channel for result
        let (sender, receiver) = oneshot::channel();

        // Submit task to worker
        let task = ReadbackTask {
            buffer: buffer.clone(),
            width,
            height,
            padded_bpr,
            row_bytes,
            sender,
        };

        self.worker_tx
            .send(task)
            .map_err(|_| RenderError::Readback("Worker task queue closed".to_string()))?;

        // Increment pending operations counter
        *self.pending_ops.lock().unwrap() += 1;

        Ok(AsyncReadbackHandle {
            receiver,
            size: buffer_size,
        })
    }

    /// Synchronous readback (fallback for compatibility)
    pub fn readback_texture_sync(
        &self,
        texture: &Texture,
        width: u32,
        height: u32,
    ) -> Result<Vec<u8>, RenderError> {
        // Use tokio runtime to run async version synchronously
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                let handle = self.readback_texture_async(texture, width, height).await?;
                handle.wait().await
            })
        })
    }

    /// Get an available readback buffer or create a new one
    fn get_readback_buffer(&self, required_size: u64) -> Result<Arc<Buffer>, RenderError> {
        let mut buffers = self.buffers.lock().unwrap();

        // Try to find an available buffer of sufficient size
        for buffer_state in buffers.iter_mut() {
            if !buffer_state.in_use && buffer_state.size >= required_size {
                buffer_state.in_use = true;
                // Return a reference to the buffer (we'll manage the in_use flag separately)
                return Ok(buffer_state.buffer.clone());
            }
        }

        // Create new buffer if pre-allocation is enabled or no suitable buffer found
        if self.config.pre_allocate || buffers.is_empty() {
            let buffer_state = ReadbackBuffer::new(
                &self.device,
                required_size,
                &format!("async-readback-{}", buffers.len()),
            );

            let buffer = buffer_state.buffer.clone();
            buffers.push(buffer_state);
            Ok(buffer)
        } else {
            // Create temporary buffer (not managed by the pool)
            let buffer = self.device.create_buffer(&BufferDescriptor {
                label: Some("temp-readback"),
                size: required_size,
                usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            global_tracker().track_buffer_allocation(required_size, true);
            Ok(Arc::new(buffer))
        }
    }

    /// Submit the texture-to-buffer copy command
    fn submit_copy_command(
        &self,
        texture: &Texture,
        buffer: &Buffer,
        width: u32,
        height: u32,
        padded_bpr: u32,
    ) -> Result<(), RenderError> {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("async-readback-copy"),
            });

        let copy_src = wgpu::ImageCopyTexture {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        };

        let copy_dst = wgpu::ImageCopyBuffer {
            buffer,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(
                    NonZeroU32::new(padded_bpr)
                        .ok_or_else(|| {
                            RenderError::Upload("bytes_per_row cannot be zero".to_string())
                        })?
                        .into(),
                ),
                rows_per_image: Some(
                    NonZeroU32::new(height)
                        .ok_or_else(|| RenderError::Upload("height cannot be zero".to_string()))?
                        .into(),
                ),
            },
        };

        let extent = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        encoder.copy_texture_to_buffer(copy_src, copy_dst, extent);

        self.queue.submit([encoder.finish()]);

        Ok(())
    }

    /// Process a readback task asynchronously
    async fn process_readback_task(device: Arc<Device>, task: ReadbackTask) {
        let result = Self::execute_readback_task(device, &task).await;
        let _ = task.sender.send(result);
    }

    /// Execute the actual readback operation
    async fn execute_readback_task(
        device: Arc<Device>,
        task: &ReadbackTask,
    ) -> Result<Vec<u8>, RenderError> {
        let slice = task.buffer.slice(..);

        // Create channel for map_async callback
        let (tx, mut rx) = oneshot::channel();

        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        // Poll device until mapping is complete
        loop {
            device.poll(wgpu::Maintain::Poll);

            // Check if mapping completed
            if let Ok(map_result) = rx.try_recv() {
                map_result
                    .map_err(|e| RenderError::Readback(format!("MapAsync failed: {:?}", e)))?;
                break;
            }

            // Yield to avoid busy waiting
            tokio::task::yield_now().await;
        }

        // Extract data
        let data = slice.get_mapped_range();
        let mut out = vec![0u8; (task.row_bytes * task.height) as usize];

        let src_stride = task.padded_bpr as usize;
        let dst_stride = task.row_bytes as usize;

        for y in 0..(task.height as usize) {
            let src_off = y * src_stride;
            let dst_off = y * dst_stride;
            out[dst_off..dst_off + dst_stride]
                .copy_from_slice(&data[src_off..src_off + dst_stride]);
        }

        drop(data);
        task.buffer.unmap();

        Ok(out)
    }

    /// Get statistics about the readback manager
    pub fn get_stats(&self) -> AsyncReadbackStats {
        let buffers = self.buffers.lock().unwrap();
        let pending_ops = *self.pending_ops.lock().unwrap();

        let total_buffers = buffers.len();
        let in_use_buffers = buffers.iter().filter(|b| b.in_use).count();
        let total_buffer_memory = buffers.iter().map(|b| b.size).sum();

        AsyncReadbackStats {
            pending_operations: pending_ops,
            total_buffers,
            in_use_buffers,
            available_buffers: total_buffers - in_use_buffers,
            total_buffer_memory,
            double_buffered: self.config.double_buffered,
        }
    }
}

/// Statistics for async readback operations
#[derive(Debug, Clone)]
pub struct AsyncReadbackStats {
    pub pending_operations: usize,
    pub total_buffers: usize,
    pub in_use_buffers: usize,
    pub available_buffers: usize,
    pub total_buffer_memory: u64,
    pub double_buffered: bool,
}

/// Align bytes-per-row for copy operations (256-byte alignment requirement)
fn align_copy_bpr(unpadded: u32) -> u32 {
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    ((unpadded + align - 1) / align) * align
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_async_readback_config() {
        let config = AsyncReadbackConfig::default();
        assert!(config.double_buffered);
        assert!(config.pre_allocate);
        assert_eq!(config.max_pending_ops, 4);
    }

    #[test]
    fn test_copy_alignment() {
        assert_eq!(align_copy_bpr(100), 256); // Aligns to 256
        assert_eq!(align_copy_bpr(256), 256); // Already aligned
        assert_eq!(align_copy_bpr(300), 512); // Next 256-byte boundary
    }
}
