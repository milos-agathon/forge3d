//! TLAS-style instanced geometry for path tracing.
//!
//! Supports per-instance transforms with forward and inverse matrices for ray traversal.

use glam::Mat4;
use wgpu::Device;

/// Maximum VRAM budget for instance data (512 MiB per memory_budget.rst).
const MAX_INSTANCE_VRAM_BYTES: usize = 512 * 1024 * 1024;

/// Per-instance data for TLAS ray traversal.
///
/// Layout is GPU-friendly (128 bytes, 16-byte aligned).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InstanceData {
    /// Object-to-world transform (column-major 4x4 matrix).
    pub transform: [f32; 16],
    /// World-to-object transform for ray transformation.
    pub inv_transform: [f32; 16],
    /// Index into the BLAS array for this instance's geometry.
    pub blas_index: u32,
    /// Material ID for shading.
    pub material_id: u32,
    /// Padding for 16-byte alignment.
    pub _padding: [u32; 2],
}

/// Top-Level Acceleration Structure for instanced geometry.
///
/// Manages a collection of instances, each referencing a BLAS with a unique transform.
pub struct TLAS {
    instances: Vec<InstanceData>,
    max_instances: usize,
    #[allow(dead_code)]
    device: std::sync::Arc<Device>,
}

impl TLAS {
    /// Create a new TLAS with the given maximum instance capacity.
    pub fn new(device: std::sync::Arc<Device>, max_instances: usize) -> Self {
        Self {
            instances: Vec::new(),
            max_instances,
            device,
        }
    }

    /// Add an instance with the given transform, BLAS index, and material.
    ///
    /// Returns the instance index on success, or an error if capacity is exceeded.
    pub fn add_instance(
        &mut self,
        transform: Mat4,
        blas_index: u32,
        material_id: u32,
    ) -> Result<usize, String> {
        if self.instances.len() >= self.max_instances {
            return Err("Maximum instances exceeded".to_string());
        }

        let inv_transform = transform.inverse();

        let instance = InstanceData {
            transform: transform.to_cols_array(),
            inv_transform: inv_transform.to_cols_array(),
            blas_index,
            material_id,
            _padding: [0; 2],
        };

        self.instances.push(instance);
        Ok(self.instances.len() - 1)
    }

    /// Get current memory usage in bytes for all instances.
    pub fn get_memory_usage(&self) -> usize {
        self.instances.len() * std::mem::size_of::<InstanceData>()
    }

    /// Check if current usage is within the 512 MiB VRAM budget.
    pub fn validate_memory_budget(&self) -> bool {
        self.get_memory_usage() <= MAX_INSTANCE_VRAM_BYTES
    }
}
