//! Multi-threaded command recording system
//!
//! Provides utilities for recording GPU commands across multiple threads,
//! improving parallelization and GPU utilization.

use crate::error::{RenderError, RenderResult};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::Instant;
use wgpu::*;

/// Handle for a recorded command buffer from a thread
#[derive(Debug)]
pub struct ThreadCommandBuffer {
    /// The recorded command buffer
    pub buffer: CommandBuffer,
    /// Thread ID that recorded this buffer
    pub thread_id: std::thread::ThreadId,
    /// Time taken to record (for profiling)
    pub record_time_ms: f32,
    /// Number of commands recorded (estimate)
    pub command_count: usize,
}

/// Configuration for multi-threaded recording
#[derive(Debug, Clone)]
pub struct MultiThreadConfig {
    /// Number of worker threads to use (0 = auto-detect)
    pub thread_count: usize,
    /// Maximum time to wait for thread completion (milliseconds)
    pub timeout_ms: u64,
    /// Whether to enable timing/profiling
    pub enable_profiling: bool,
    /// Label prefix for command buffers
    pub label_prefix: String,
}

impl Default for MultiThreadConfig {
    fn default() -> Self {
        Self {
            thread_count: 0,  // Auto-detect
            timeout_ms: 5000, // 5 second timeout
            enable_profiling: false,
            label_prefix: "mt_cmd".to_string(),
        }
    }
}

/// Task for a worker thread to execute
pub trait CommandTask: Send + Sync {
    /// Error type for task execution
    type Error: std::error::Error + Send + 'static;

    /// Execute the task, recording commands into the encoder
    fn execute(
        &self,
        encoder: &mut CommandEncoder,
        device: &Device,
        queue: &Queue,
    ) -> Result<usize, Self::Error>;

    /// Get a descriptive name for this task (for debugging/profiling)
    fn name(&self) -> &str;
}

/// Multi-threaded command recorder
pub struct MultiThreadRecorder {
    /// GPU device reference
    device: Arc<Device>,
    /// GPU queue reference
    queue: Arc<Queue>,
    /// Configuration
    config: MultiThreadConfig,
    /// Thread pool for reuse
    thread_pool: Option<ThreadPool>,
}

/// Simple thread pool implementation
struct ThreadPool {
    workers: Vec<Worker>,
    sender: mpsc::Sender<Job>,
}

struct Worker {
    #[allow(dead_code)]
    id: usize,
    handle: Option<thread::JoinHandle<()>>,
}

type Job = Box<dyn FnOnce() + Send + 'static>;

impl ThreadPool {
    fn new(size: usize) -> ThreadPool {
        assert!(size > 0);

        let (sender, receiver) = mpsc::channel();
        let receiver = Arc::new(Mutex::new(receiver));

        let mut workers = Vec::with_capacity(size);

        for id in 0..size {
            workers.push(Worker::new(id, Arc::clone(&receiver)));
        }

        ThreadPool { workers, sender }
    }

    fn execute<F>(&self, f: F) -> Result<(), mpsc::SendError<Job>>
    where
        F: FnOnce() + Send + 'static,
    {
        let job = Box::new(f);
        self.sender.send(job)
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        // Close the channel
        drop(self.sender.clone());

        // Wait for all workers to finish
        for worker in &mut self.workers {
            if let Some(handle) = worker.handle.take() {
                handle
                    .join()
                    .unwrap_or_else(|_| eprintln!("Worker thread panicked"));
            }
        }
    }
}

impl Worker {
    fn new(id: usize, receiver: Arc<Mutex<mpsc::Receiver<Job>>>) -> Worker {
        let handle = thread::spawn(move || loop {
            let receiver = receiver.lock().unwrap();
            match receiver.recv() {
                Ok(job) => {
                    drop(receiver); // Release lock before executing
                    job();
                }
                Err(_) => break, // Channel closed
            }
        });

        Worker {
            id,
            handle: Some(handle),
        }
    }
}

impl MultiThreadRecorder {
    /// Create a new multi-threaded recorder
    pub fn new(device: Arc<Device>, queue: Arc<Queue>, config: MultiThreadConfig) -> Self {
        let thread_count = if config.thread_count == 0 {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
        } else {
            config.thread_count
        };

        let mut recorder_config = config;
        recorder_config.thread_count = thread_count;

        Self {
            device,
            queue,
            config: recorder_config,
            thread_pool: None,
        }
    }

    /// Initialize the thread pool (optional - will be created on first use)
    pub fn initialize_pool(&mut self) -> RenderResult<()> {
        if self.thread_pool.is_none() {
            self.thread_pool = Some(ThreadPool::new(self.config.thread_count));
        }
        Ok(())
    }

    /// Record commands using multiple threads
    pub fn record_parallel<T: CommandTask + 'static>(
        &mut self,
        tasks: Vec<Arc<T>>,
    ) -> Result<Vec<ThreadCommandBuffer>, Box<dyn std::error::Error + Send>> {
        if tasks.is_empty() {
            return Ok(Vec::new());
        }

        // Ensure thread pool is initialized
        self.initialize_pool()
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send>)?;

        let start_time = if self.config.enable_profiling {
            Some(Instant::now())
        } else {
            None
        };

        // Channel for collecting results
        let (result_sender, result_receiver) =
            mpsc::channel::<Result<ThreadCommandBuffer, Box<dyn std::error::Error + Send>>>();

        // Spawn tasks across threads
        let mut active_tasks = 0;
        for (task_index, task) in tasks.into_iter().enumerate() {
            let device = Arc::clone(&self.device);
            let queue = Arc::clone(&self.queue);
            let config = self.config.clone();
            let sender = result_sender.clone();

            // Use thread pool or spawn directly
            let job = move || {
                let result = Self::record_task_on_thread(task, device, queue, config, task_index);
                let _ = sender.send(result);
            };

            if let Some(ref pool) = self.thread_pool {
                pool.execute(job).map_err(|_| {
                    Box::new(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        "Failed to submit task to thread pool",
                    )) as Box<dyn std::error::Error + Send>
                })?;
            } else {
                thread::spawn(job);
            }

            active_tasks += 1;
        }

        // Drop the sender so we know when all threads are done
        drop(result_sender);

        // Collect results
        let mut results = Vec::new();
        let timeout = std::time::Duration::from_millis(self.config.timeout_ms);
        let start_wait = Instant::now();

        while active_tasks > 0 {
            match result_receiver.recv_timeout(timeout) {
                Ok(result) => {
                    results.push(result?);
                    active_tasks -= 1;
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    if start_wait.elapsed() > timeout {
                        return Err(Box::new(std::io::Error::new(
                            std::io::ErrorKind::TimedOut,
                            "Timeout waiting for multi-threaded command recording",
                        )));
                    }
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }

        if let Some(start) = start_time {
            let total_time = start.elapsed().as_millis();
            println!(
                "Multi-thread recording completed in {}ms with {} tasks",
                total_time,
                results.len()
            );
        }

        Ok(results)
    }

    /// Submit recorded command buffers to the GPU queue
    pub fn submit_buffers(&self, buffers: Vec<ThreadCommandBuffer>) -> RenderResult<()> {
        if buffers.is_empty() {
            return Ok(());
        }

        let command_buffers: Vec<_> = buffers.into_iter().map(|tcb| tcb.buffer).collect();
        self.queue.submit(command_buffers);

        Ok(())
    }

    /// Record and submit in one operation
    pub fn record_and_submit<T: CommandTask + 'static>(
        &mut self,
        tasks: Vec<Arc<T>>,
    ) -> Result<(), Box<dyn std::error::Error + Send>> {
        let buffers = self.record_parallel(tasks)?;
        self.submit_buffers(buffers)
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send>)
    }

    /// Record a single task on a worker thread
    fn record_task_on_thread<T: CommandTask>(
        task: Arc<T>,
        device: Arc<Device>,
        queue: Arc<Queue>,
        config: MultiThreadConfig,
        task_index: usize,
    ) -> Result<ThreadCommandBuffer, Box<dyn std::error::Error + Send>> {
        let thread_id = thread::current().id();
        let start_time = if config.enable_profiling {
            Some(Instant::now())
        } else {
            None
        };

        // Create command encoder
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some(&format!(
                "{}_{}_thread_{:?}",
                config.label_prefix, task_index, thread_id
            )),
        });

        // Execute the task
        let command_count = task
            .execute(&mut encoder, &device, &queue)
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send>)?;

        // Finish recording
        let buffer = encoder.finish();

        let record_time_ms = if let Some(start) = start_time {
            start.elapsed().as_millis() as f32
        } else {
            0.0
        };

        Ok(ThreadCommandBuffer {
            buffer,
            thread_id,
            record_time_ms,
            command_count,
        })
    }

    /// Get configuration
    pub fn config(&self) -> &MultiThreadConfig {
        &self.config
    }

    /// Get thread count
    pub fn thread_count(&self) -> usize {
        self.config.thread_count
    }
}

/// Example task for copying buffers
pub struct CopyTask {
    name: String,
    src_buffer: Arc<Buffer>,
    dst_buffer: Arc<Buffer>,
    size: u64,
}

impl CopyTask {
    pub fn new(name: String, src: Arc<Buffer>, dst: Arc<Buffer>, size: u64) -> Self {
        Self {
            name,
            src_buffer: src,
            dst_buffer: dst,
            size,
        }
    }
}

impl CommandTask for CopyTask {
    type Error = RenderError;

    fn execute(
        &self,
        encoder: &mut CommandEncoder,
        _device: &Device,
        _queue: &Queue,
    ) -> Result<usize, Self::Error> {
        encoder.copy_buffer_to_buffer(&self.src_buffer, 0, &self.dst_buffer, 0, self.size);
        Ok(1) // One command recorded
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Example task for clearing textures
pub struct ClearTask {
    name: String,
    texture: Arc<Texture>,
    clear_color: Color,
}

impl ClearTask {
    pub fn new(name: String, texture: Arc<Texture>, clear_color: Color) -> Self {
        Self {
            name,
            texture,
            clear_color,
        }
    }
}

impl CommandTask for ClearTask {
    type Error = RenderError;

    fn execute(
        &self,
        encoder: &mut CommandEncoder,
        _device: &Device,
        _queue: &Queue,
    ) -> Result<usize, Self::Error> {
        let view = self.texture.create_view(&TextureViewDescriptor::default());

        let render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some(&format!("clear_pass_{}", self.name)),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(self.clear_color),
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        drop(render_pass); // End the render pass
        Ok(1) // One render pass recorded
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    // Mock task for testing
    struct MockTask {
        name: String,
        work_duration_ms: u64,
        counter: Arc<AtomicUsize>,
    }

    impl MockTask {
        fn new(name: String, work_duration_ms: u64, counter: Arc<AtomicUsize>) -> Self {
            Self {
                name,
                work_duration_ms,
                counter,
            }
        }
    }

    impl CommandTask for MockTask {
        type Error = RenderError;

        fn execute(
            &self,
            _encoder: &mut CommandEncoder,
            _device: &Device,
            _queue: &Queue,
        ) -> Result<usize, Self::Error> {
            // Simulate some work
            thread::sleep(std::time::Duration::from_millis(self.work_duration_ms));
            self.counter.fetch_add(1, Ordering::SeqCst);
            Ok(1)
        }

        fn name(&self) -> &str {
            &self.name
        }
    }

    // These tests would require a real GPU context, so they're disabled by default
    #[ignore]
    #[test]
    fn test_multi_thread_recorder_basic() {
        // This test requires a real wgpu device
        // Would need to be run with a GPU context available
        assert!(true, "Multi-threading test placeholder");
    }

    #[test]
    fn test_multi_thread_config() {
        let config = MultiThreadConfig::default();
        assert_eq!(config.thread_count, 0); // Auto-detect
        assert_eq!(config.timeout_ms, 5000);
        assert!(!config.enable_profiling);

        let custom_config = MultiThreadConfig {
            thread_count: 8,
            timeout_ms: 10000,
            enable_profiling: true,
            label_prefix: "custom".to_string(),
        };
        assert_eq!(custom_config.thread_count, 8);
        assert!(custom_config.enable_profiling);
    }

    #[test]
    fn test_thread_pool_creation() {
        let pool = ThreadPool::new(4);
        // Pool should be created successfully
        drop(pool); // Test cleanup
    }
}
