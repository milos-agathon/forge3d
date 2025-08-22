//! Device capabilities and diagnostics
//!
//! Provides structured access to GPU device capabilities, limits, and features.

use crate::gpu::ctx;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};
use std::collections::HashMap;

/// Device capabilities structure
#[derive(Debug, Clone)]
pub struct DeviceCaps {
    /// Backend identifier (vulkan, dx12, metal, gl)
    pub backend: String,
    
    /// Adapter name from driver
    pub adapter_name: String,
    
    /// Device name 
    pub device_name: String,
    
    /// Maximum 2D texture dimension
    pub max_texture_dimension_2d: u32,
    
    /// Maximum buffer size
    pub max_buffer_size: u64,
    
    /// MSAA support (sample count > 1)
    pub msaa_supported: bool,
    
    /// Maximum supported sample count
    pub max_samples: u32,
    
    /// Device type (integrated, discrete, virtual, cpu, other)
    pub device_type: String,
}

impl DeviceCaps {
    /// Create DeviceCaps from current GPU context
    pub fn from_current_device() -> PyResult<Self> {
        let g = ctx();
        let adapter_info = g.adapter.get_info();
        let device_limits = g.device.limits();
        
        // Check MSAA support by testing common sample counts
        let msaa_supported = [2u32, 4, 8].iter().any(|&samples| {
            g.adapter.get_texture_format_features(wgpu::TextureFormat::Rgba8UnormSrgb)
                .flags.sample_count_supported(samples)
        });
        
        let max_samples = if msaa_supported {
            [8u32, 4, 2].into_iter()
                .find(|&samples| {
                    g.adapter.get_texture_format_features(wgpu::TextureFormat::Rgba8UnormSrgb)
                        .flags.sample_count_supported(samples)
                })
                .unwrap_or(1)
        } else {
            1
        };
        
        Ok(DeviceCaps {
            backend: format!("{:?}", adapter_info.backend).to_lowercase(),
            adapter_name: adapter_info.name.clone(),
            device_name: adapter_info.name.clone(), // Same as adapter for now
            max_texture_dimension_2d: device_limits.max_texture_dimension_2d,
            max_buffer_size: device_limits.max_buffer_size,
            msaa_supported,
            max_samples,
            device_type: format!("{:?}", adapter_info.device_type).to_lowercase(),
        })
    }
    
    /// Convert to Python dictionary
    pub fn to_py_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new_bound(py);
        
        dict.set_item("backend", &self.backend)?;
        dict.set_item("adapter_name", &self.adapter_name)?;
        dict.set_item("device_name", &self.device_name)?;
        dict.set_item("max_texture_dimension_2d", self.max_texture_dimension_2d)?;
        dict.set_item("max_buffer_size", self.max_buffer_size)?;
        dict.set_item("msaa_supported", self.msaa_supported)?;
        dict.set_item("max_samples", self.max_samples)?;
        dict.set_item("device_type", &self.device_type)?;
        
        Ok(dict.into())
    }
}