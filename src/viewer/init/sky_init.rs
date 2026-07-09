// src/viewer/init/sky_init.rs
// Sky pipeline initialization for the Viewer

use std::sync::Arc;
use wgpu::{BindGroupLayout, ComputePipeline, Device, TextureView};

use crate::core::error::RenderResult;
use crate::core::resource_tracker::{
    tracked_create_buffer_init, tracked_create_texture, TrackedBuffer, TrackedTexture,
};

use super::super::viewer_types::SkyUniforms;

/// Resources created during sky initialization
pub struct SkyResources {
    pub sky_bind_group_layout0: BindGroupLayout,
    pub sky_bind_group_layout1: BindGroupLayout,
    pub sky_pipeline: ComputePipeline,
    pub sky_params: TrackedBuffer,
    pub sky_camera: TrackedBuffer,
    pub sky_output: TrackedTexture,
    pub sky_output_view: TextureView,
}

/// Create sky compute pipeline and resources
pub fn create_sky_resources(
    device: &Arc<Device>,
    width: u32,
    height: u32,
) -> RenderResult<SkyResources> {
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

    let sky_shader = crate::core::shader_registry::create_labeled_shader_module(
        device,
        "viewer.sky.shader",
        include_str!("../../shaders/sky.wgsl"),
    );

    let sky_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("viewer.sky.pl"),
        bind_group_layouts: &[&sky_bgl0, &sky_bgl1],
        push_constant_ranges: &[],
    });

    let sky_pipeline =
        crate::core::shader_registry::with_error_scope(device, "viewer.sky.pipeline", || {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("viewer.sky.pipeline"),
                layout: Some(&sky_pl),
                module: &sky_shader,
                entry_point: "cs_render_sky",
            })
        });

    let sky_params_data = SkyUniforms::new([0.3, 0.8, -0.5], 2.0, 0.3, 1.0, 5.0, 1.0, 0);
    let sky_params = tracked_create_buffer_init(
        device,
        &wgpu::util::BufferInitDescriptor {
            label: Some("viewer.sky.params"),
            contents: bytemuck::bytes_of(&sky_params_data),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        },
    )?;

    // Sky camera buffer - matches CameraUniforms struct in sky.wgsl (272 bytes)
    // Layout: view(64) + proj(64) + inv_view(64) + inv_proj(64) + eye_position(12) + _pad0(4)
    let sky_camera_data: [f32; 68] = [0.0; 68]; // 272 bytes
    let sky_camera = tracked_create_buffer_init(
        device,
        &wgpu::util::BufferInitDescriptor {
            label: Some("viewer.sky.camera"),
            contents: bytemuck::cast_slice(&sky_camera_data),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        },
    )?;

    // Sky output texture
    let sky_output = tracked_create_texture(
        device,
        &wgpu::TextureDescriptor {
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
        },
    )?;
    let sky_output_view = sky_output.create_view(&wgpu::TextureViewDescriptor::default());

    Ok(SkyResources {
        sky_bind_group_layout0: sky_bgl0,
        sky_bind_group_layout1: sky_bgl1,
        sky_pipeline,
        sky_params,
        sky_camera,
        sky_output,
        sky_output_view,
    })
}

#[cfg(test)]
mod tests {
    use super::create_sky_resources;
    use std::sync::Arc;

    #[test]
    fn creates_sky_pipeline_when_adapter_available() {
        let instance = wgpu::Instance::default();
        let Some(adapter) =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
        else {
            eprintln!("No GPU adapter available, skipping viewer sky pipeline test");
            return;
        };
        let Ok((device, _queue)) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None))
        else {
            eprintln!("Could not request GPU device, skipping viewer sky pipeline test");
            return;
        };

        let _resources = create_sky_resources(&Arc::new(device), 16, 16).expect("sky resources");
    }
}
