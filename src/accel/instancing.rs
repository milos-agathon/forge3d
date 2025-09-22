//! A22: Instanced Geometry (PT) - TLAS-style instances with per-instance transforms

use glam::Mat4;
use wgpu::*;

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InstanceData {
    pub transform: [f32; 16], // 4x4 matrix
    pub inv_transform: [f32; 16],
    pub blas_index: u32,
    pub material_id: u32,
    pub _padding: [u32; 2],
}

pub struct TLAS {
    instances: Vec<InstanceData>,
    max_instances: usize,
    #[allow(dead_code)]
    device: std::sync::Arc<Device>,
}

impl TLAS {
    pub fn new(device: std::sync::Arc<Device>, max_instances: usize) -> Self {
        Self {
            instances: Vec::new(),
            max_instances,
            device,
        }
    }

    // A22: 10k instances with one BLAS; ≤512MiB VRAM
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

    pub fn get_memory_usage(&self) -> usize {
        self.instances.len() * std::mem::size_of::<InstanceData>()
    }

    pub fn validate_memory_budget(&self) -> bool {
        const MAX_VRAM: usize = 512 * 1024 * 1024; // 512 MiB
        self.get_memory_usage() <= MAX_VRAM
    }
}
