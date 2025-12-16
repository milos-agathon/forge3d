// src/viewer/render/bind_groups.rs
// Bind group creation helpers for the interactive viewer

use wgpu::{BindGroup, BindGroupLayout, Buffer, Device, Sampler, Texture, TextureView};

/// Create a fallback 1x1 albedo texture
pub fn create_fallback_albedo_texture(device: &Device) -> Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("viewer.geom.albedo.empty"),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

/// Create geometry bind group for camera + albedo texture + sampler
pub fn create_geom_bind_group(
    device: &Device,
    layout: &BindGroupLayout,
    cam_buf: &Buffer,
    albedo_view: &TextureView,
    sampler: &Sampler,
) -> BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("viewer.gbuf.geom.bg.runtime"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: cam_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(albedo_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    })
}
