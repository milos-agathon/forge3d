//! Async compute prepasses for GPU pipeline parallelization  
//!
//! Provides utilities for running compute shaders asynchronously alongside
//! graphics workloads to improve GPU utilization and performance.

use wgpu::*;
use crate::error::{RenderError, RenderResult};
use std::sync::{Arc, Mutex};
use std::collections::HashMap;

/// Handle for an async compute pass
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ComputePassId(usize);

/// Configuration for async compute execution
#[derive(Debug, Clone)]
pub struct AsyncComputeConfig {
    /// Maximum number of concurrent compute passes
    pub max_concurrent_passes: usize,
    /// Timeout for compute completion (milliseconds)  
    pub timeout_ms: u64,
    /// Whether to enable profiling/timing
    pub enable_profiling: bool,
    /// Label prefix for compute passes
    pub label_prefix: String,
}

impl Default for AsyncComputeConfig {
    fn default() -> Self {
        Self {
            max_concurrent_passes: 4,
            timeout_ms: 1000, // 1 second timeout
            enable_profiling: false,
            label_prefix: "async_compute".to_string(),
        }
    }
}

/// Synchronization point between compute and graphics
#[derive(Debug, Clone, PartialEq)]
pub enum SyncPoint {
    /// Wait for specific compute passes to complete
    WaitForCompute(Vec<ComputePassId>),
    /// Signal completion of graphics work
    SignalGraphics,
    /// Full pipeline flush
    FullFlush,
}

/// Resource barrier for compute/graphics synchronization
#[derive(Debug, Clone)]
pub struct ComputeBarrier {
    /// Buffer being transitioned
    pub buffer: Option<Arc<Buffer>>,
    /// Texture being transitioned
    pub texture: Option<Arc<Texture>>,
    /// Previous usage state
    pub src_usage: ResourceUsage,
    /// New usage state
    pub dst_usage: ResourceUsage,
}

/// Resource usage states for barrier management
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ResourceUsage {
    /// Storage buffer (read/write from compute)
    ComputeStorage,
    /// Uniform buffer (read-only from compute)
    ComputeUniform,
    /// Texture storage (read/write from compute)
    ComputeTexture,
    /// Graphics vertex buffer
    GraphicsVertex,
    /// Graphics uniform buffer  
    GraphicsUniform,
    /// Render target texture
    GraphicsRenderTarget,
    /// Texture sampling in graphics
    GraphicsTexture,
}

/// Compute shader dispatch parameters
#[derive(Debug, Clone)]
pub struct DispatchParams {
    /// Workgroup count in X dimension
    pub workgroups_x: u32,
    /// Workgroup count in Y dimension  
    pub workgroups_y: u32,
    /// Workgroup count in Z dimension
    pub workgroups_z: u32,
}

impl DispatchParams {
    /// Create dispatch parameters for 1D workload
    pub fn linear(workgroups: u32) -> Self {
        Self {
            workgroups_x: workgroups,
            workgroups_y: 1,
            workgroups_z: 1,
        }
    }
    
    /// Create dispatch parameters for 2D workload
    pub fn planar(workgroups_x: u32, workgroups_y: u32) -> Self {
        Self {
            workgroups_x,
            workgroups_y,
            workgroups_z: 1,
        }
    }
    
    /// Create dispatch parameters for 3D workload
    pub fn volumetric(workgroups_x: u32, workgroups_y: u32, workgroups_z: u32) -> Self {
        Self {
            workgroups_x,
            workgroups_y,
            workgroups_z,
        }
    }
    
    /// Calculate total workgroups
    pub fn total_workgroups(&self) -> u32 {
        self.workgroups_x * self.workgroups_y * self.workgroups_z
    }
}

/// Async compute pass descriptor
#[derive(Debug)]
pub struct ComputePassDescriptor {
    /// Human-readable label
    pub label: String,
    /// Compute pipeline to execute
    pub pipeline: Arc<ComputePipeline>,
    /// Bind groups for resources
    pub bind_groups: Vec<Arc<BindGroup>>,
    /// Dispatch parameters  
    pub dispatch: DispatchParams,
    /// Barriers needed before execution
    pub barriers: Vec<ComputeBarrier>,
    /// Priority level (higher = more important)
    pub priority: u32,
}

/// Status of an async compute pass
#[derive(Debug, Clone, PartialEq)]
pub enum ComputePassStatus {
    /// Pass is queued for execution
    Queued,
    /// Pass is currently executing
    Executing, 
    /// Pass completed successfully
    Completed,
    /// Pass failed with error
    Failed(String),
    /// Pass was cancelled
    Cancelled,
}

/// Information about a running compute pass
#[derive(Debug)]
pub struct ComputePassInfo {
    /// Pass descriptor
    pub descriptor: ComputePassDescriptor,
    /// Current status
    pub status: ComputePassStatus,
    /// Start time (for profiling)
    pub start_time: Option<std::time::Instant>,
    /// Completion time (for profiling)
    pub end_time: Option<std::time::Instant>,
    /// Command buffer for this pass
    pub command_buffer: Option<CommandBuffer>,
}

/// Async compute scheduler and executor
pub struct AsyncComputeScheduler {
    /// GPU device reference
    device: Arc<Device>,
    /// GPU queue reference
    queue: Arc<Queue>,
    /// Configuration
    config: AsyncComputeConfig,
    /// Active compute passes
    passes: HashMap<ComputePassId, ComputePassInfo>,
    /// Next pass ID to assign
    next_pass_id: usize,
    /// Resource state tracking
    resource_states: HashMap<String, ResourceUsage>,
    /// Mutex for thread safety
    mutex: Mutex<()>,
}

/// Performance metrics for compute passes
#[derive(Debug, Clone)]
pub struct ComputeMetrics {
    /// Total number of passes submitted
    pub total_passes: usize,
    /// Number of completed passes
    pub completed_passes: usize,
    /// Number of failed passes
    pub failed_passes: usize,
    /// Total execution time in milliseconds
    pub total_execution_time_ms: f32,
    /// Total workgroups dispatched
    pub total_workgroups: u32,
    /// Average execution time per pass
    pub average_execution_time_ms: f32,
}

impl AsyncComputeScheduler {
    /// Create a new async compute scheduler
    pub fn new(device: Arc<Device>, queue: Arc<Queue>, config: AsyncComputeConfig) -> Self {
        Self {
            device,
            queue,
            config,
            passes: HashMap::new(),
            next_pass_id: 0,
            resource_states: HashMap::new(),
            mutex: Mutex::new(()),
        }
    }
    
    /// Submit a compute pass for async execution
    pub fn submit_compute_pass(&mut self, descriptor: ComputePassDescriptor) -> RenderResult<ComputePassId> {
        let _lock = self.mutex.lock().unwrap();
        
        let pass_id = ComputePassId(self.next_pass_id);
        self.next_pass_id += 1;
        
        // Check if we can accept more passes
        let active_passes = self.passes.values()
            .filter(|info| matches!(info.status, ComputePassStatus::Queued | ComputePassStatus::Executing))
            .count();
            
        if active_passes >= self.config.max_concurrent_passes {
            return Err(RenderError::render("Too many concurrent compute passes"));
        }
        
        let pass_info = ComputePassInfo {
            descriptor,
            status: ComputePassStatus::Queued,
            start_time: None,
            end_time: None,
            command_buffer: None,
        };
        
        self.passes.insert(pass_id, pass_info);
        Ok(pass_id)
    }
    
    /// Execute all queued compute passes
    pub fn execute_queued_passes(&mut self) -> RenderResult<Vec<ComputePassId>> {
        let queued_passes = {
            let _lock = self.mutex.lock().unwrap();
            
            // Collect queued passes sorted by priority
            let mut queued_passes: Vec<_> = self.passes.iter()
                .filter(|(_, info)| info.status == ComputePassStatus::Queued)
                .map(|(&id, info)| (id, info.descriptor.priority))
                .collect();
            
            queued_passes.sort_by(|a, b| b.1.cmp(&a.1)); // Sort by priority descending
            queued_passes
        };
        
        let mut executed_passes = Vec::new();
        
        for (pass_id, _) in queued_passes {
            match self.execute_compute_pass_internal(pass_id) {
                Ok(()) => executed_passes.push(pass_id),
                Err(e) => {
                    let _lock = self.mutex.lock().unwrap();
                    if let Some(pass_info) = self.passes.get_mut(&pass_id) {
                        pass_info.status = ComputePassStatus::Failed(e.to_string());
                    }
                }
            }
        }
        
        Ok(executed_passes)
    }
    
    /// Wait for specific compute passes to complete
    pub fn wait_for_passes(&mut self, pass_ids: &[ComputePassId]) -> RenderResult<()> {
        let timeout = std::time::Duration::from_millis(self.config.timeout_ms);
        let start_time = std::time::Instant::now();
        
        loop {
            let all_completed = {
                let _lock = self.mutex.lock().unwrap();
                pass_ids.iter().all(|&pass_id| {
                    self.passes.get(&pass_id)
                        .map(|info| matches!(info.status, ComputePassStatus::Completed | ComputePassStatus::Failed(_)))
                        .unwrap_or(true) // Consider missing passes as completed
                })
            };
            
            if all_completed {
                break;
            }
            
            if start_time.elapsed() > timeout {
                return Err(RenderError::render("Timeout waiting for compute passes to complete"));
            }
            
            // Small sleep to avoid busy waiting
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        
        Ok(())
    }
    
    /// Insert barriers for compute/graphics synchronization
    pub fn insert_barriers(&mut self, barriers: Vec<ComputeBarrier>) -> RenderResult<()> {
        if barriers.is_empty() {
            return Ok(());
        }
        
        // Create a command encoder for barriers
        let encoder = self.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("compute_barriers"),
        });
        
        // GPU barriers are implicit in WebGPU - just update our state tracking
        for barrier in barriers {
            let resource_name = if let Some(ref buffer) = barrier.buffer {
                format!("buffer_{:p}", buffer.as_ref() as *const _)
            } else if let Some(ref texture) = barrier.texture {
                format!("texture_{:p}", texture.as_ref() as *const _)
            } else {
                continue;
            };
            
            self.resource_states.insert(resource_name, barrier.dst_usage);
        }
        
        // Submit empty command buffer to ensure ordering
        let command_buffer = encoder.finish();
        self.queue.submit([command_buffer]);
        
        Ok(())
    }
    
    /// Get status of a compute pass
    pub fn get_pass_status(&self, pass_id: ComputePassId) -> Option<ComputePassStatus> {
        let _lock = self.mutex.lock().unwrap();
        self.passes.get(&pass_id).map(|info| info.status.clone())
    }
    
    /// Get performance metrics for completed passes
    pub fn get_metrics(&self) -> ComputeMetrics {
        let _lock = self.mutex.lock().unwrap();
        
        let mut completed_passes = 0;
        let mut failed_passes = 0;
        let mut total_execution_time_ms = 0.0;
        let mut total_workgroups = 0;
        
        for info in self.passes.values() {
            match &info.status {
                ComputePassStatus::Completed => {
                    completed_passes += 1;
                    total_workgroups += info.descriptor.dispatch.total_workgroups();
                    
                    if let (Some(start), Some(end)) = (info.start_time, info.end_time) {
                        total_execution_time_ms += end.duration_since(start).as_millis() as f32;
                    }
                }
                ComputePassStatus::Failed(_) => failed_passes += 1,
                _ => {}
            }
        }
        
        ComputeMetrics {
            total_passes: self.passes.len(),
            completed_passes,
            failed_passes,
            total_execution_time_ms,
            total_workgroups,
            average_execution_time_ms: if completed_passes > 0 {
                total_execution_time_ms / completed_passes as f32
            } else {
                0.0
            },
        }
    }
    
    /// Cancel a queued compute pass
    pub fn cancel_pass(&mut self, pass_id: ComputePassId) -> RenderResult<()> {
        let _lock = self.mutex.lock().unwrap();
        
        if let Some(pass_info) = self.passes.get_mut(&pass_id) {
            match pass_info.status {
                ComputePassStatus::Queued => {
                    pass_info.status = ComputePassStatus::Cancelled;
                    Ok(())
                }
                _ => Err(RenderError::render("Cannot cancel compute pass that is not queued"))
            }
        } else {
            Err(RenderError::render("Compute pass not found"))
        }
    }
    
    /// Clear completed and failed passes
    pub fn cleanup_completed_passes(&mut self) {
        let _lock = self.mutex.lock().unwrap();
        
        self.passes.retain(|_, info| {
            !matches!(info.status, ComputePassStatus::Completed | ComputePassStatus::Failed(_) | ComputePassStatus::Cancelled)
        });
    }
    
    /// Internal method to execute a single compute pass
    fn execute_compute_pass_internal(&mut self, pass_id: ComputePassId) -> RenderResult<()> {
        // Get pass descriptor info (clone to avoid borrowing issues)
        let (label, label_prefix) = {
            let _lock = self.mutex.lock().unwrap();
            let pass_info = self.passes.get_mut(&pass_id)
                .ok_or_else(|| RenderError::render("Compute pass not found"))?;
                
            // Mark as executing
            pass_info.status = ComputePassStatus::Executing;
            pass_info.start_time = Some(std::time::Instant::now());
            
            (pass_info.descriptor.label.clone(), self.config.label_prefix.clone())
        };
        
        // Create command encoder
        let encoder = self.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some(&format!("{}_{}", label_prefix, label)),
        });
        
        // For now, just create an empty command buffer
        let command_buffer = encoder.finish();
        
        // Mark as completed
        {
            let _lock = self.mutex.lock().unwrap();
            if let Some(pass_info) = self.passes.get_mut(&pass_id) {
                pass_info.command_buffer = Some(command_buffer);
                pass_info.status = ComputePassStatus::Completed;
                pass_info.end_time = Some(std::time::Instant::now());
            }
        }
        
        Ok(())
    }
}

/// Utility functions for common compute patterns
pub mod patterns {
    use super::*;
    
    /// Create a simple buffer copy compute pass
    pub fn create_buffer_copy_pass(
        _device: &Device,
        _src_buffer: Arc<Buffer>,
        _dst_buffer: Arc<Buffer>,
        _size: u64,
    ) -> RenderResult<ComputePassDescriptor> {
        // This would typically require a compute shader for buffer copying
        // For now, return a placeholder
        Err(RenderError::render("Buffer copy compute shader not implemented"))
    }
    
    /// Create a parallel reduction compute pass
    pub fn create_reduction_pass(
        _device: &Device,
        _input_buffer: Arc<Buffer>,
        _output_buffer: Arc<Buffer>,
        _element_count: u32,
    ) -> RenderResult<ComputePassDescriptor> {
        // This would typically require a reduction compute shader
        Err(RenderError::render("Reduction compute shader not implemented"))
    }
    
    /// Create a parallel prefix sum (scan) compute pass
    pub fn create_scan_pass(
        _device: &Device,
        _input_buffer: Arc<Buffer>,
        _output_buffer: Arc<Buffer>,
        _element_count: u32,
    ) -> RenderResult<ComputePassDescriptor> {
        // This would typically require a scan compute shader
        Err(RenderError::render("Scan compute shader not implemented"))
    }
}

/// Helper function to estimate post-processing workgroup count
pub fn estimate_postfx_workgroups(width: u32, height: u32, local_size_x: u32, local_size_y: u32) -> (u32, u32, u32) {
    let workgroups_x = (width + local_size_x - 1) / local_size_x;
    let workgroups_y = (height + local_size_y - 1) / local_size_y;
    (workgroups_x, workgroups_y, 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_c7_dispatch_params_linear() {
        let dispatch = DispatchParams::linear(256);
        assert_eq!(dispatch.workgroups_x, 256);
        assert_eq!(dispatch.workgroups_y, 1);
        assert_eq!(dispatch.workgroups_z, 1);
        assert_eq!(dispatch.total_workgroups(), 256);
    }
}
