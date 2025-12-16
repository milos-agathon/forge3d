use crate::core::shadow_mapping::CsmUniforms;
use wgpu::{
    BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType,
    BufferBindingType, Device, SamplerBindingType, ShaderStages, TextureSampleType,
    TextureViewDimension,
};

/// Create default shadow mapping bind group layout
pub fn create_shadow_bind_group_layout(device: &Device) -> BindGroupLayout {
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("shadow_bind_group_layout"),
        entries: &[
            // CSM uniforms
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: Some(
                        std::num::NonZeroU64::new(std::mem::size_of::<CsmUniforms>() as u64)
                            .unwrap(),
                    ),
                },
                count: None,
            },
            // Shadow map texture array
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    multisampled: false,
                    view_dimension: TextureViewDimension::D2Array,
                    sample_type: TextureSampleType::Depth,
                },
                count: None,
            },
            // Shadow sampler
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Sampler(SamplerBindingType::Comparison),
                count: None,
            },
        ],
    })
}
