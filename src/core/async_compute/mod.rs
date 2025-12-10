//! Async compute prepasses for GPU pipeline parallelization
//!
//! Provides utilities for running compute shaders asynchronously alongside
//! graphics workloads to improve GPU utilization and performance.

mod scheduler;
mod types;

pub use scheduler::AsyncComputeScheduler;
pub use types::{
    AsyncComputeConfig, ComputeBarrier, ComputeMetrics, ComputePassDescriptor, ComputePassId,
    ComputePassInfo, ComputePassStatus, DispatchParams, ResourceUsage, SyncPoint,
};

use crate::core::error::{RenderError, RenderResult};
use std::sync::Arc;
use wgpu::Buffer;

/// Utility functions for common compute patterns
pub mod patterns {
    use super::*;

    /// Create a simple buffer copy compute pass
    pub fn create_buffer_copy_pass(
        _device: &wgpu::Device,
        _src_buffer: Arc<Buffer>,
        _dst_buffer: Arc<Buffer>,
        _size: u64,
    ) -> RenderResult<ComputePassDescriptor> {
        Err(RenderError::render(
            "Buffer copy compute shader not implemented",
        ))
    }

    /// Create a parallel reduction compute pass
    pub fn create_reduction_pass(
        _device: &wgpu::Device,
        _input_buffer: Arc<Buffer>,
        _output_buffer: Arc<Buffer>,
        _element_count: u32,
    ) -> RenderResult<ComputePassDescriptor> {
        Err(RenderError::render(
            "Reduction compute shader not implemented",
        ))
    }

    /// Create a parallel prefix sum (scan) compute pass
    pub fn create_scan_pass(
        _device: &wgpu::Device,
        _input_buffer: Arc<Buffer>,
        _output_buffer: Arc<Buffer>,
        _element_count: u32,
    ) -> RenderResult<ComputePassDescriptor> {
        Err(RenderError::render("Scan compute shader not implemented"))
    }
}

/// Helper function to estimate post-processing workgroup count
pub fn estimate_postfx_workgroups(
    width: u32,
    height: u32,
    local_size_x: u32,
    local_size_y: u32,
) -> (u32, u32, u32) {
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
