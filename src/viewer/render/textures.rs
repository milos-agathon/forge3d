// src/viewer/render/textures.rs
// Texture creation helpers for the interactive viewer

use wgpu::{Device, Texture, TextureView};

/// Create a storage/texture binding texture with common settings
pub fn create_hdr_texture(
    device: &Device,
    width: u32,
    height: u32,
    label: &str,
    format: wgpu::TextureFormat,
) -> (Texture, TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

/// Create an RGBA16Float HDR texture
pub fn create_rgba16f_texture(
    device: &Device,
    width: u32,
    height: u32,
    label: &str,
) -> (Texture, TextureView) {
    create_hdr_texture(
        device,
        width,
        height,
        label,
        wgpu::TextureFormat::Rgba16Float,
    )
}

/// Create an RGBA8Unorm texture with storage binding
pub fn create_rgba8_storage_texture(
    device: &Device,
    width: u32,
    height: u32,
    label: &str,
) -> (Texture, TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

/// Create a depth texture
pub fn create_depth_texture(
    device: &Device,
    width: u32,
    height: u32,
    label: &str,
) -> (Texture, TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}
