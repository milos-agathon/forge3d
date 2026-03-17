use super::*;

pub(super) fn create_layouts(device: &Device) -> ConstructorLayouts {
    ConstructorLayouts {
        trace_bind_group_layout: device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("p5.ssr.trace.bgl"),
            entries: &[
                texture_entry(0, false, TextureViewDimension::D2),
                texture_entry(1, false, TextureViewDimension::D2),
                storage_texture_entry(2, TextureFormat::Rgba16Float),
                uniform_buffer_entry(3),
                uniform_buffer_entry(4),
                storage_buffer_entry(5),
            ],
        }),
        shade_bind_group_layout: device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("p5.ssr.shade.bgl"),
            entries: &[
                texture_entry(0, true, TextureViewDimension::D2),
                sampler_entry(1),
                texture_entry(2, false, TextureViewDimension::D2),
                texture_entry(3, false, TextureViewDimension::D2),
                texture_entry(4, true, TextureViewDimension::D2),
                texture_entry(5, false, TextureViewDimension::D2),
                storage_texture_entry(6, TextureFormat::Rgba16Float),
                uniform_buffer_entry(7),
                uniform_buffer_entry(8),
                storage_buffer_entry(9),
            ],
        }),
        fallback_bind_group_layout: device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("p5.ssr.fallback.bgl"),
            entries: &[
                texture_entry(0, false, TextureViewDimension::D2),
                texture_entry(1, false, TextureViewDimension::D2),
                texture_entry(2, false, TextureViewDimension::D2),
                texture_entry(3, false, TextureViewDimension::D2),
                texture_entry(4, true, TextureViewDimension::Cube),
                sampler_entry(5),
                storage_texture_entry(6, TextureFormat::Rgba16Float),
                uniform_buffer_entry(7),
                uniform_buffer_entry(8),
                storage_buffer_entry(9),
            ],
        }),
        temporal_bind_group_layout: device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("p5.ssr.temporal.bgl"),
            entries: &[
                texture_entry(0, false, TextureViewDimension::D2),
                texture_entry(1, false, TextureViewDimension::D2),
                storage_texture_entry(2, TextureFormat::Rgba16Float),
                uniform_buffer_entry(3),
            ],
        }),
        composite_bind_group_layout: device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("p5.ssr.composite.bgl"),
            entries: &[
                texture_entry(0, true, TextureViewDimension::D2),
                texture_entry(1, false, TextureViewDimension::D2),
                uniform_buffer_entry(2),
                storage_texture_entry(3, TextureFormat::Rgba8Unorm),
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

fn storage_buffer_entry(binding: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Storage { read_only: false },
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
