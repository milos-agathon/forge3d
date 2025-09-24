//! GPU context management - thin re-export layer
//!
//! This module provides a consistent interface for GPU context operations
//! by re-exporting functionality from the gpu module.

use std::fmt;

// Re-export GPU context and accessor
pub use crate::gpu::{ctx, GpuContext};

/// High-level engine/adapter information collected from the active GPU context
#[derive(Debug, Clone)]
pub struct EngineInfo {
    /// Backend identifier (e.g., "vulkan", "metal", "dx12", "gl")
    pub backend: String,
    /// Adapter name reported by the driver
    pub adapter_name: String,
    /// Device name (same as adapter for now)
    pub device_name: String,
    /// Maximum 2D texture dimension
    pub max_texture_dimension_2d: u32,
    /// Maximum buffer size
    pub max_buffer_size: u64,
}

impl fmt::Display for EngineInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "EngineInfo{{ backend: {}, adapter: {}, max_tex2d: {}, max_buf: {} }}",
            self.backend, self.adapter_name, self.max_texture_dimension_2d, self.max_buffer_size
        )
    }
}

/// Retrieve high-level engine/device information from the active GPU context
pub fn engine_info() -> EngineInfo {
    let g = ctx();
    let adapter_info = g.adapter.get_info();
    let device_limits = g.device.limits();

    EngineInfo {
        backend: format!("{:?}", adapter_info.backend).to_lowercase(),
        adapter_name: adapter_info.name.clone(),
        device_name: adapter_info.name.clone(),
        max_texture_dimension_2d: device_limits.max_texture_dimension_2d,
        max_buffer_size: device_limits.max_buffer_size,
    }
}
