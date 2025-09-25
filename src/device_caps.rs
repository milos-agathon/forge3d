//! Device capabilities and diagnostics
//!
//! Provides structured access to GPU device capabilities, limits, and features.

use crate::gpu::ctx;
use pyo3::prelude::*;
use pyo3::types::PyDict;

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

    /// Descriptor indexing support (bindless textures)
    pub descriptor_indexing: bool,

    /// Maximum textures in descriptor array
    pub max_texture_array_layers: u32,

    /// Maximum samplers in descriptor array
    pub max_sampler_array_size: u32,

    /// Vertex shader texture array support
    pub vertex_shader_array_support: bool,
    /// Support for linear filtering on 32-bit float textures (e.g., R32Float)
    pub float32_filterable: bool,
}

impl DeviceCaps {
    /// Create DeviceCaps from current GPU context
    pub fn from_current_device() -> PyResult<Self> {
        let g = ctx();
        let adapter_info = g.adapter.get_info();
        let device_limits = g.device.limits();

        // Check MSAA support by testing common sample counts
        let msaa_supported = [2u32, 4, 8].iter().any(|&samples| {
            g.adapter
                .get_texture_format_features(wgpu::TextureFormat::Rgba8UnormSrgb)
                .flags
                .sample_count_supported(samples)
        });

        let max_samples = if msaa_supported {
            [8u32, 4, 2]
                .into_iter()
                .find(|&samples| {
                    g.adapter
                        .get_texture_format_features(wgpu::TextureFormat::Rgba8UnormSrgb)
                        .flags
                        .sample_count_supported(samples)
                })
                .unwrap_or(1)
        } else {
            1
        };

        // Detect descriptor indexing capabilities
        let device_features = g.device.features();
        let descriptor_indexing = device_features.contains(wgpu::Features::TEXTURE_BINDING_ARRAY)
            && device_features.contains(
                wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
            );
        // Detect float32 filterable textures support
        let float32_filterable = device_features.contains(wgpu::Features::FLOAT32_FILTERABLE);

        // Backend-specific array limits
        let (max_texture_array_layers, max_sampler_array_size, vertex_shader_array_support) =
            Self::detect_array_limits(&adapter_info.backend, &device_limits);

        Ok(DeviceCaps {
            backend: format!("{:?}", adapter_info.backend).to_lowercase(),
            adapter_name: adapter_info.name.clone(),
            device_name: adapter_info.name.clone(), // Same as adapter for now
            max_texture_dimension_2d: device_limits.max_texture_dimension_2d,
            max_buffer_size: device_limits.max_buffer_size,
            msaa_supported,
            max_samples,
            device_type: format!("{:?}", adapter_info.device_type).to_lowercase(),
            descriptor_indexing,
            max_texture_array_layers,
            max_sampler_array_size,
            vertex_shader_array_support,
            float32_filterable,
        })
    }

    /// Detect backend-specific array limits
    fn detect_array_limits(backend: &wgpu::Backend, limits: &wgpu::Limits) -> (u32, u32, bool) {
        match backend {
            wgpu::Backend::Vulkan => {
                // Vulkan has generous limits for descriptor arrays
                // Based on typical Vulkan 1.2 implementations
                let max_textures = limits.max_texture_array_layers.min(2048);
                let max_samplers = 32; // Conservative limit for samplers
                let vertex_support = true; // Vulkan supports arrays in vertex shaders
                (max_textures, max_samplers, vertex_support)
            }
            wgpu::Backend::Metal => {
                // Metal has more restrictive limits
                // Based on Metal argument buffer limitations
                let max_textures = limits.max_texture_array_layers.min(512);
                let max_samplers = 16; // Metal sampler limits
                let vertex_support = true; // Metal supports texture arrays in vertex shaders
                (max_textures, max_samplers, vertex_support)
            }
            wgpu::Backend::Dx12 => {
                // DX12 has variable support depending on feature level
                let max_textures = limits.max_texture_array_layers.min(1024);
                let max_samplers = 32; // DX12 descriptor heap limits
                let vertex_support = true; // DX12 supports arrays in vertex shaders
                (max_textures, max_samplers, vertex_support)
            }
            wgpu::Backend::Gl => {
                // OpenGL has more limited support, especially older versions
                let max_textures = limits.max_texture_array_layers.min(256);
                let max_samplers = 8; // Conservative GL limit
                let vertex_support = false; // GL often lacks vertex shader array support
                (max_textures, max_samplers, vertex_support)
            }
            _ => {
                // Unknown backend - use conservative limits
                (32, 8, false)
            }
        }
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
        dict.set_item("descriptor_indexing", self.descriptor_indexing)?;
        dict.set_item("max_texture_array_layers", self.max_texture_array_layers)?;
        dict.set_item("max_sampler_array_size", self.max_sampler_array_size)?;
        dict.set_item(
            "vertex_shader_array_support",
            self.vertex_shader_array_support,
        )?;
        dict.set_item("float32_filterable", self.float32_filterable)?;

        Ok(dict.into())
    }
}
