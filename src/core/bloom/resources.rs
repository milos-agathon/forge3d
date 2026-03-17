use std::borrow::Cow;

use crate::core::error::RenderResult;
use crate::core::postfx::{PostFxResourceDesc, PostFxResourcePool};
use wgpu::*;

pub(super) struct BloomLayouts {
    pub(super) brightpass: BindGroupLayout,
    pub(super) blur: BindGroupLayout,
    pub(super) composite: BindGroupLayout,
}

pub(super) struct BloomPipelines {
    pub(super) brightpass: ComputePipeline,
    pub(super) blur_h: ComputePipeline,
    pub(super) blur_v: ComputePipeline,
    pub(super) composite: ComputePipeline,
}

pub(super) struct BloomUniformBuffers {
    pub(super) brightpass: Buffer,
    pub(super) blur: Buffer,
    pub(super) composite: Buffer,
}

pub(super) fn create_layouts(device: &Device) -> BloomLayouts {
    BloomLayouts {
        brightpass: create_read_write_uniform_layout(device, "bloom_brightpass_layout"),
        blur: create_read_write_uniform_layout(device, "bloom_blur_layout"),
        composite: create_composite_layout(device),
    }
}

pub(super) fn create_pipelines(device: &Device, layouts: &BloomLayouts) -> BloomPipelines {
    let brightpass_shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("bloom_brightpass_shader"),
        source: ShaderSource::Wgsl(Cow::Borrowed(include_str!(
            "../../shaders/bloom_brightpass.wgsl"
        ))),
    });
    let blur_h_shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("bloom_blur_h_shader"),
        source: ShaderSource::Wgsl(Cow::Borrowed(include_str!(
            "../../shaders/bloom_blur_h.wgsl"
        ))),
    });
    let blur_v_shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("bloom_blur_v_shader"),
        source: ShaderSource::Wgsl(Cow::Borrowed(include_str!(
            "../../shaders/bloom_blur_v.wgsl"
        ))),
    });
    let composite_shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("bloom_composite_shader"),
        source: ShaderSource::Wgsl(Cow::Borrowed(include_str!(
            "../../shaders/bloom_composite.wgsl"
        ))),
    });

    BloomPipelines {
        brightpass: create_pipeline(
            device,
            &layouts.brightpass,
            &brightpass_shader,
            "bloom_brightpass_pipeline_layout",
            "bloom_brightpass_pipeline",
        ),
        blur_h: create_pipeline(
            device,
            &layouts.blur,
            &blur_h_shader,
            "bloom_blur_pipeline_layout",
            "bloom_blur_h_pipeline",
        ),
        blur_v: create_pipeline(
            device,
            &layouts.blur,
            &blur_v_shader,
            "bloom_blur_pipeline_layout",
            "bloom_blur_v_pipeline",
        ),
        composite: create_pipeline(
            device,
            &layouts.composite,
            &composite_shader,
            "bloom_composite_pipeline_layout",
            "bloom_composite_pipeline",
        ),
    }
}

pub(super) fn create_uniform_buffers(device: &Device) -> BloomUniformBuffers {
    BloomUniformBuffers {
        brightpass: create_uniform_buffer(
            device,
            "bloom_brightpass_uniforms",
            std::mem::size_of::<super::config::BloomBrightPassUniforms>() as BufferAddress,
        ),
        blur: create_uniform_buffer(
            device,
            "bloom_blur_uniforms",
            std::mem::size_of::<super::config::BloomBlurUniforms>() as BufferAddress,
        ),
        composite: create_uniform_buffer(
            device,
            "bloom_composite_uniforms",
            std::mem::size_of::<super::config::BloomCompositeUniforms>() as BufferAddress,
        ),
    }
}

pub(super) fn allocate_resource_indices(
    device: &Device,
    resource_pool: &mut PostFxResourcePool,
) -> RenderResult<(usize, usize)> {
    let pp_desc = PostFxResourceDesc {
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
        ..PostFxResourceDesc::default()
    };

    let brightpass_idx = resource_pool.allocate_ping_pong_pair(device, &pp_desc)?;
    let blur_temp_idx = resource_pool.allocate_ping_pong_pair(device, &pp_desc)?;
    Ok((brightpass_idx, blur_temp_idx))
}

fn create_read_write_uniform_layout(device: &Device, label: &str) -> BindGroupLayout {
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: false },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::StorageTexture {
                    access: StorageTextureAccess::WriteOnly,
                    format: TextureFormat::Rgba8Unorm,
                    view_dimension: TextureViewDimension::D2,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

fn create_composite_layout(device: &Device) -> BindGroupLayout {
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("bloom_composite_layout"),
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: false },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: false },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::StorageTexture {
                    access: StorageTextureAccess::WriteOnly,
                    format: TextureFormat::Rgba8Unorm,
                    view_dimension: TextureViewDimension::D2,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

fn create_pipeline(
    device: &Device,
    layout: &BindGroupLayout,
    shader: &ShaderModule,
    layout_label: &str,
    pipeline_label: &str,
) -> ComputePipeline {
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some(layout_label),
        bind_group_layouts: &[layout],
        push_constant_ranges: &[],
    });

    device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some(pipeline_label),
        layout: Some(&pipeline_layout),
        module: shader,
        entry_point: "main",
    })
}

fn create_uniform_buffer(device: &Device, label: &str, size: BufferAddress) -> Buffer {
    device.create_buffer(&BufferDescriptor {
        label: Some(label),
        size,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}
