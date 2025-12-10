//! GPU context management and engine information.
//!
//! Re-exports the core GPU context and provides high-level engine introspection.

use std::fmt;

pub use super::gpu::{ctx, GpuContext};

/// High-level engine/adapter information collected from the active GPU context.
#[derive(Debug, Clone)]
pub struct EngineInfo {
    /// Backend identifier (e.g., "vulkan", "metal", "dx12", "gl").
    pub backend: String,
    /// Adapter name reported by the driver.
    pub adapter_name: String,
    /// Device name (currently mirrors `adapter_name`; reserved for multi-device scenarios).
    pub device_name: String,
    /// Maximum 2D texture dimension supported.
    pub max_texture_dimension_2d: u32,
    /// Maximum buffer size in bytes.
    pub max_buffer_size: u64,
}

impl fmt::Display for EngineInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "EngineInfo(backend={}, adapter={}, max_tex2d={}, max_buf={})",
            self.backend, self.adapter_name, self.max_texture_dimension_2d, self.max_buffer_size
        )
    }
}

/// Retrieve high-level engine/device information from the active GPU context.
pub fn engine_info() -> EngineInfo {
    let ctx = ctx();
    let adapter_info = ctx.adapter.get_info();
    let limits = ctx.device.limits();

    let backend = format!("{:?}", adapter_info.backend).to_lowercase();
    let adapter_name = adapter_info.name.clone();

    EngineInfo {
        backend,
        adapter_name: adapter_name.clone(),
        device_name: adapter_name,
        max_texture_dimension_2d: limits.max_texture_dimension_2d,
        max_buffer_size: limits.max_buffer_size,
    }
}
