// src/viewer/init/sky_init.rs
// Sky pipeline initialization for the Viewer

use std::sync::Arc;
use wgpu::util::DeviceExt;
use wgpu::{BindGroupLayout, Buffer, ComputePipeline, Device, Texture, TextureView};

/// Resources created during sky initialization
pub struct SkyResources {
    pub sky_bind_group_layout0: BindGroupLayout,
    pub sky_bind_group_layout1: BindGroupLayout,
    pub sky_pipeline: ComputePipeline,
    pub sky_params: Buffer,
    pub sky_camera: Buffer,
    pub sky_output: Texture,
    pub sky_output_view: TextureView,
}

/// Create sky compute pipeline and resources
pub fn create_sky_resources(device: &Arc<Device>, width: u32, height: u32) -> SkyResources {
    // Sky BGL0: params (binding 0) + output texture (binding 1)
    let sky_bgl0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("viewer.sky.bgl0"),
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
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
        ],
    });

    // Sky BGL1: camera uniform (binding 0)
    let sky_bgl1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("viewer.sky.bgl1"),
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
    });

    let sky_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("viewer.sky.shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/sky.wgsl").into()),
    });

    let sky_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("viewer.sky.pl"),
        bind_group_layouts: &[&sky_bgl0, &sky_bgl1],
        push_constant_ranges: &[],
    });

    let sky_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("viewer.sky.pipeline"),
        layout: Some(&sky_pl),
        module: &sky_shader,
        entry_point: "cs_render_sky",
    });

    // Sky params buffer - matches SkyParams struct in sky.wgsl (48 bytes with alignment)
    // Layout: sun_direction(vec3+pad=16) + turbidity(4) + ground_albedo(4) + model(4) + 
    //         sun_intensity(4) + exposure(4) + _pad(8) = 48 bytes
    let sky_params_data: [f32; 12] = [
        0.3, 0.8, -0.5, 0.0,  // sun_direction.xyz + padding
        2.0,                   // turbidity
        0.3,                   // ground_albedo
        0.0,                   // model (as f32, will be read as u32)
        5.0,                   // sun_intensity
        1.0,                   // exposure
        0.0, 0.0, 0.0,        // _pad
    ];
    let sky_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("viewer.sky.params"),
        contents: bytemuck::cast_slice(&sky_params_data),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // Sky camera buffer - matches CameraUniforms struct in sky.wgsl (272 bytes)
    // Layout: view(64) + proj(64) + inv_view(64) + inv_proj(64) + eye_position(12) + _pad0(4)
    let sky_camera_data: [f32; 68] = [0.0; 68]; // 272 bytes
    let sky_camera = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("viewer.sky.camera"),
        contents: bytemuck::cast_slice(&sky_camera_data),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // Sky output texture
    let sky_output = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("viewer.sky.output"),
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
    let sky_output_view = sky_output.create_view(&wgpu::TextureViewDescriptor::default());

    SkyResources {
        sky_bind_group_layout0: sky_bgl0,
        sky_bind_group_layout1: sky_bgl1,
        sky_pipeline,
        sky_params,
        sky_camera,
        sky_output,
        sky_output_view,
    }
}
