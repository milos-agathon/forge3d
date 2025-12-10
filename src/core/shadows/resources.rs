//! GPU resource creation for shadow mapping
//!
//! Provides helper functions for creating shadow map textures, views,
//! samplers, and uniform buffers.

use super::types::{CsmConfig, CsmUniforms};

/// Create depth views for each cascade
pub fn create_cascade_depth_views(
    shadow_maps: &wgpu::Texture,
    config: &CsmConfig,
) -> Vec<wgpu::TextureView> {
    (0..config.cascade_count)
        .map(|i| {
            shadow_maps.create_view(&wgpu::TextureViewDescriptor {
                label: Some(&format!("CSM Shadow Map Cascade {}", i)),
                format: Some(wgpu::TextureFormat::Depth32Float),
                dimension: Some(wgpu::TextureViewDimension::D2),
                aspect: wgpu::TextureAspect::DepthOnly,
                base_mip_level: 0,
                mip_level_count: Some(1),
                base_array_layer: i,
                array_layer_count: Some(1),
            })
        })
        .collect()
}

/// Create array view for shader sampling
pub fn create_shadow_array_view(
    shadow_maps: &wgpu::Texture,
    config: &CsmConfig,
) -> wgpu::TextureView {
    shadow_maps.create_view(&wgpu::TextureViewDescriptor {
        label: Some("CSM Shadow Map Array"),
        format: Some(wgpu::TextureFormat::Depth32Float),
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        aspect: wgpu::TextureAspect::DepthOnly,
        base_mip_level: 0,
        mip_level_count: Some(1),
        base_array_layer: 0,
        array_layer_count: Some(config.cascade_count),
    })
}

/// Create shadow sampler with comparison for PCF
pub fn create_shadow_sampler(device: &wgpu::Device) -> wgpu::Sampler {
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("CSM Shadow Sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Nearest,
        compare: Some(wgpu::CompareFunction::LessEqual),
        ..Default::default()
    })
}

/// Create CSM uniform buffer
pub fn create_uniform_buffer(device: &wgpu::Device) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("CSM Uniforms"),
        size: std::mem::size_of::<CsmUniforms>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}
