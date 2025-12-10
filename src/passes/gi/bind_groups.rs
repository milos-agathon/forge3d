//! Bind group layout creation helpers for GI passes.

use wgpu::{
    BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType,
    BufferBindingType, Device, ShaderStages, StorageTextureAccess, TextureFormat,
    TextureSampleType, TextureViewDimension,
};

/// Create the composite pass bind group layout (10 bindings).
pub fn create_composite_bind_group_layout(device: &Device) -> BindGroupLayout {
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("p5.gi.composite.bgl"),
        entries: &[
            texture_entry(0), // baseline_lighting
            texture_entry(1), // diffuse_view
            texture_entry(2), // spec_view
            texture_entry(3), // ao_view
            texture_entry(4), // ssgi_view
            texture_entry(5), // ssr_view
            texture_entry(6), // normal_view
            texture_entry(7), // material_view
            storage_texture_entry(8), // output
            uniform_buffer_entry(9), // params
        ],
    })
}

/// Create the debug pass bind group layout (4 bindings).
pub fn create_debug_bind_group_layout(device: &Device) -> BindGroupLayout {
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("p5.gi.debug.bgl"),
        entries: &[
            texture_entry(0), // ao_view
            texture_entry(1), // ssgi_view
            texture_entry(2), // ssr_view
            storage_texture_entry(3), // debug_output
        ],
    })
}

fn texture_entry(binding: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Texture {
            sample_type: TextureSampleType::Float { filterable: false },
            view_dimension: TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn storage_texture_entry(binding: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::StorageTexture {
            access: StorageTextureAccess::WriteOnly,
            format: TextureFormat::Rgba16Float,
            view_dimension: TextureViewDimension::D2,
        },
        count: None,
    }
}

fn uniform_buffer_entry(binding: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
