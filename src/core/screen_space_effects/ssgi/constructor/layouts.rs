use super::*;

pub(super) fn create_layouts(device: &Device) -> ConstructorLayouts {
    ConstructorLayouts {
        trace_bind_group_layout: device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("ssgi_trace_bgl"),
            entries: &[
                texture_entry(0, false, TextureViewDimension::D2),
                texture_entry(1, false, TextureViewDimension::D2),
                texture_entry(2, false, TextureViewDimension::D2),
                storage_texture_entry(3, TextureFormat::Rgba16Float),
                uniform_buffer_entry(4),
                uniform_buffer_entry(5),
            ],
        }),
        shade_bind_group_layout: device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("ssgi_shade_bgl"),
            entries: &[
                texture_entry(0, true, TextureViewDimension::D2),
                sampler_entry(1),
                texture_entry(2, true, TextureViewDimension::Cube),
                sampler_entry(3),
                texture_entry(4, false, TextureViewDimension::D2),
                storage_texture_entry(5, TextureFormat::Rgba16Float),
                uniform_buffer_entry(6),
                uniform_buffer_entry(7),
                texture_entry(8, false, TextureViewDimension::D2),
                texture_entry(9, false, TextureViewDimension::D2),
            ],
        }),
        temporal_bind_group_layout: device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("ssgi_temporal_bind_group_layout"),
            entries: &[
                texture_entry(0, false, TextureViewDimension::D2),
                texture_entry(1, false, TextureViewDimension::D2),
                storage_texture_entry(2, TextureFormat::Rgba16Float),
                uniform_buffer_entry(3),
                texture_entry(4, false, TextureViewDimension::D2),
                texture_entry(5, false, TextureViewDimension::D2),
            ],
        }),
        upsample_bind_group_layout: device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("ssgi_upsample_bind_group_layout"),
            entries: &[
                texture_entry(0, true, TextureViewDimension::D2),
                storage_texture_entry(1, TextureFormat::Rgba16Float),
                sampler_entry(2),
                texture_entry(3, false, TextureViewDimension::D2),
                texture_entry(4, false, TextureViewDimension::D2),
                uniform_buffer_entry(5),
            ],
        }),
        composite_bind_group_layout: device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("ssgi_composite_bind_group_layout"),
            entries: &[
                texture_entry(0, false, TextureViewDimension::D2),
                storage_texture_entry(1, TextureFormat::Rgba8Unorm),
                texture_entry(2, true, TextureViewDimension::D2),
                uniform_buffer_entry(3),
            ],
        }),
    }
}

fn texture_entry(
    binding: u32,
    filterable: bool,
    view_dimension: TextureViewDimension,
) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Texture {
            sample_type: TextureSampleType::Float { filterable },
            view_dimension,
            multisampled: false,
        },
        count: None,
    }
}

fn storage_texture_entry(binding: u32, format: TextureFormat) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::StorageTexture {
            access: StorageTextureAccess::WriteOnly,
            format,
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

fn sampler_entry(binding: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Sampler(SamplerBindingType::Filtering),
        count: None,
    }
}
