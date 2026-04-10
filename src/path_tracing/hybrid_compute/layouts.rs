use super::*;

impl HybridPathTracer {
    pub(super) fn create_uniforms_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hybrid-pt-bgl0-uniforms"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        })
    }

    pub(super) fn create_scene_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hybrid-pt-bgl1-scene"),
            entries: &[
                storage_entry(0, true),
                uniform_entry(1),
                storage_entry(2, true),
                storage_entry(3, true),
                storage_entry(4, true),
            ],
        })
    }

    pub(super) fn create_accum_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hybrid-pt-bgl2-accum"),
            entries: &[storage_entry(0, false)],
        })
    }

    pub(super) fn create_output_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hybrid-pt-bgl3-out"),
            entries: &[
                texture_entry(0, wgpu::TextureFormat::Rgba16Float),
                texture_entry(1, wgpu::TextureFormat::Rgba16Float),
                texture_entry(2, wgpu::TextureFormat::Rgba16Float),
                texture_entry(3, wgpu::TextureFormat::R32Float),
                texture_entry(4, wgpu::TextureFormat::Rgba16Float),
                texture_entry(5, wgpu::TextureFormat::Rgba16Float),
                texture_entry(6, wgpu::TextureFormat::Rgba16Float),
                texture_entry(7, wgpu::TextureFormat::Rgba8Unorm),
            ],
        })
    }

    pub(super) fn create_lighting_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hybrid-pt-bgl4-lighting"),
            entries: &[uniform_entry(0)],
        })
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn texture_entry(binding: u32, format: wgpu::TextureFormat) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    }
}
