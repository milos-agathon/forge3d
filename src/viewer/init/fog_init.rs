// src/viewer/init/fog_init.rs
// Fog pipeline initialization for the Viewer

use std::sync::Arc;
use wgpu::util::DeviceExt;
use wgpu::{BindGroupLayout, Buffer, ComputePipeline, Device, Sampler, Texture, TextureView};

/// Resources created during fog initialization
pub struct FogResources {
    pub fog_bgl0: BindGroupLayout,
    pub fog_bgl1: BindGroupLayout,
    pub fog_bgl2: BindGroupLayout,
    pub fog_bgl3: BindGroupLayout,
    pub fog_pipeline: ComputePipeline,
    pub fog_params: Buffer,
    pub fog_camera: Buffer,
    pub fog_output: Texture,
    pub fog_output_view: TextureView,
    pub fog_history: Texture,
    pub fog_history_view: TextureView,
    pub fog_depth_sampler: Sampler,
    pub fog_history_sampler: Sampler,
    pub fog_shadow_map: Texture,
    pub fog_shadow_view: TextureView,
    pub fog_shadow_sampler: Sampler,
    pub fog_shadow_matrix: Buffer,
    pub fog_zero_tex: Texture,
    pub fog_zero_view: TextureView,
    pub froxel_tex: Texture,
    pub froxel_view: TextureView,
    pub froxel_sampler: Sampler,
    pub froxel_build_pipeline: ComputePipeline,
    pub froxel_apply_pipeline: ComputePipeline,
    pub fog_output_half: Texture,
    pub fog_output_half_view: TextureView,
    pub fog_history_half: Texture,
    pub fog_history_half_view: TextureView,
    pub fog_upsample_bgl: BindGroupLayout,
    pub fog_upsample_pipeline: ComputePipeline,
    pub fog_upsample_params: Buffer,
}

/// Create fog compute pipeline and resources
pub fn create_fog_resources(device: &Arc<Device>, width: u32, height: u32) -> FogResources {
    // Fog BGL0: params + camera + depth
    let fog_bgl0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("viewer.fog.bgl0"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });

    // Fog BGL1: shadow map + sampler + matrix
    let fog_bgl1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("viewer.fog.bgl1"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: wgpu::TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    // Fog BGL2: output + history
    let fog_bgl2 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("viewer.fog.bgl2"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba16Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });

    // Fog BGL3: froxel 3D texture
    let fog_bgl3 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("viewer.fog.bgl3"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba16Float,
                    view_dimension: wgpu::TextureViewDimension::D3,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D3,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });

    let fog_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("viewer.fog.shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/volumetric.wgsl").into()),
    });

    let fog_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("viewer.fog.pl"),
        bind_group_layouts: &[&fog_bgl0, &fog_bgl1, &fog_bgl2, &fog_bgl3],
        push_constant_ranges: &[],
    });

    let fog_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("viewer.fog.pipeline"),
        layout: Some(&fog_pl),
        module: &fog_shader,
        entry_point: "cs_volumetric",
    });

    let froxel_build_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("viewer.fog.froxel.build"),
        layout: Some(&fog_pl),
        module: &fog_shader,
        entry_point: "cs_build_froxels",
    });

    let froxel_apply_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("viewer.fog.froxel.apply"),
        layout: Some(&fog_pl),
        module: &fog_shader,
        entry_point: "cs_apply_froxels",
    });

    // Fog params buffer
    let fog_params_data: [u8; 80] = [0; 80];
    let fog_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("viewer.fog.params"),
        contents: &fog_params_data,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // Fog camera buffer
    let fog_camera_data: [u8; 400] = [0; 400];
    let fog_camera = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("viewer.fog.camera"),
        contents: &fog_camera_data,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // Fog output textures
    let fog_output = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("viewer.fog.output"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let fog_output_view = fog_output.create_view(&wgpu::TextureViewDescriptor::default());

    let fog_history = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("viewer.fog.history"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let fog_history_view = fog_history.create_view(&wgpu::TextureViewDescriptor::default());

    // Half-res fog textures
    let half_w = width.max(1) / 2;
    let half_h = height.max(1) / 2;
    let fog_output_half = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("viewer.fog.output.half"),
        size: wgpu::Extent3d { width: half_w, height: half_h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let fog_output_half_view = fog_output_half.create_view(&wgpu::TextureViewDescriptor::default());

    let fog_history_half = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("viewer.fog.history.half"),
        size: wgpu::Extent3d { width: half_w, height: half_h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let fog_history_half_view = fog_history_half.create_view(&wgpu::TextureViewDescriptor::default());

    // Samplers
    let fog_depth_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("viewer.fog.depth.sampler"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    let fog_history_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("viewer.fog.history.sampler"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    // Dummy shadow map until fog integrates with the shadow pipeline.
    let fog_shadow_map = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("viewer.fog.shadow.map"),
        size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 4 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let fog_shadow_view = fog_shadow_map.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    });

    let fog_shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("viewer.fog.shadow.sampler"),
        compare: Some(wgpu::CompareFunction::LessEqual),
        ..Default::default()
    });

    let fog_shadow_matrix = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("viewer.fog.shadow.matrix"),
        contents: bytemuck::cast_slice(&glam::Mat4::IDENTITY.to_cols_array()),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // Zero fallback texture
    let fog_zero_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("viewer.fog.zero"),
        size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let fog_zero_view = fog_zero_tex.create_view(&wgpu::TextureViewDescriptor::default());

    // Froxel 3D texture
    let froxel_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("viewer.fog.froxel"),
        size: wgpu::Extent3d { width: 16, height: 8, depth_or_array_layers: 64 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let froxel_view = froxel_tex.create_view(&wgpu::TextureViewDescriptor::default());
    let froxel_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("viewer.fog.froxel.sampler"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    // Fog upsample pipeline
    let fog_upsample_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("viewer.fog.upsample.bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba16Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let fog_upsample_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("viewer.fog.upsample.shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/fog_upsample.wgsl").into()),
    });

    let fog_upsample_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("viewer.fog.upsample.pl"),
        bind_group_layouts: &[&fog_upsample_bgl],
        push_constant_ranges: &[],
    });

    let fog_upsample_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("viewer.fog.upsample.pipeline"),
        layout: Some(&fog_upsample_pl),
        module: &fog_upsample_shader,
        entry_point: "cs_main",
    });

    let fog_upsample_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("viewer.fog.upsample.params"),
        contents: &[0u8; 16],
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    FogResources {
        fog_bgl0,
        fog_bgl1,
        fog_bgl2,
        fog_bgl3,
        fog_pipeline,
        fog_params,
        fog_camera,
        fog_output,
        fog_output_view,
        fog_history,
        fog_history_view,
        fog_depth_sampler,
        fog_history_sampler,
        fog_shadow_map,
        fog_shadow_view,
        fog_shadow_sampler,
        fog_shadow_matrix,
        fog_zero_tex,
        fog_zero_view,
        froxel_tex,
        froxel_view,
        froxel_sampler,
        froxel_build_pipeline,
        froxel_apply_pipeline,
        fog_output_half,
        fog_output_half_view,
        fog_history_half,
        fog_history_half_view,
        fog_upsample_bgl,
        fog_upsample_pipeline,
        fog_upsample_params,
    }
}
