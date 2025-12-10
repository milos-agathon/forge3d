//! Types and data structures for async compute passes.

use std::sync::Arc;
use wgpu::{BindGroup, Buffer, CommandBuffer, ComputePipeline, Texture};

/// Handle for an async compute pass
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ComputePassId(pub(crate) usize);

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
            timeout_ms: 1000,
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
