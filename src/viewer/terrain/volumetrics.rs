// src/viewer/terrain/volumetrics.rs
// P5: Volumetric fog pass for terrain viewer

use std::sync::Arc;
use wgpu;

/// Uniforms for volumetrics shader (must match viewer_volumetrics.wgsl)
/// Note: vec3 in WGSL has 16-byte alignment, so we use [f32; 4] for vec3 fields
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VolumetricsUniforms {
    pub inv_view_proj: [[f32; 4]; 4],  // 64 bytes, offset 0
    pub camera_pos: [f32; 4],           // 16 bytes (vec3 + pad), offset 64
    pub near_far: [f32; 4],             // near, far, density, height_falloff; offset 80
    pub scatter_absorb: [f32; 4],       // scattering, absorption, pad, pad; offset 96
    pub sun_direction: [f32; 4],        // 16 bytes (vec3 + intensity), offset 112
    pub shaft_params: [f32; 4],         // shaft_intensity, light_shafts_enabled, steps, mode; offset 128
    pub screen_dims: [f32; 4],          // width, height, pad, pad; offset 144
}

/// Volumetric fog pass manager
pub struct VolumetricsPass {
    device: Arc<wgpu::Device>,
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    uniform_buffer: wgpu::Buffer,
    color_sampler: wgpu::Sampler,
    depth_sampler: wgpu::Sampler,
}

impl VolumetricsPass {
    pub fn new(device: Arc<wgpu::Device>, target_format: wgpu::TextureFormat) -> Self {
        // Create shader module
        let shader_source = include_str!("../../shaders/viewer_volumetrics.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("viewer_volumetrics.wgsl"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("volumetrics.bind_group_layout"),
            entries: &[
                // Uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Color texture
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Color sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // Depth texture (Depth32Float, non-filterable on some backends)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Depth sampler (non-filtering for depth)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("volumetrics.pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create render pipeline
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("volumetrics.pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        // Create uniform buffer
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("volumetrics.uniforms"),
            size: std::mem::size_of::<VolumetricsUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create color sampler (filtering for color texture)
        let color_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("volumetrics.color_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            compare: None,
            ..Default::default()
        });

        // Create depth sampler (non-filtering for depth texture)
        let depth_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("volumetrics.depth_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            compare: None,
            ..Default::default()
        });

        Self {
            device,
            pipeline,
            bind_group_layout,
            uniform_buffer,
            color_sampler,
            depth_sampler,
        }
    }

    /// Apply volumetric fog to the scene
    #[allow(clippy::too_many_arguments)]
    pub fn apply(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        color_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        output_view: &wgpu::TextureView,
        width: u32,
        height: u32,
        inv_view_proj: [[f32; 4]; 4],
        camera_pos: [f32; 3],
        near: f32,
        far: f32,
        sun_direction: [f32; 3],
        sun_intensity: f32,
        config: &super::pbr_renderer::VolumetricsConfig,
    ) {
        // Update uniforms with proper alignment
        let mode = match config.mode.as_str() {
            "uniform" => 0u32,
            "height" => 1u32,
            _ => 1u32,
        };

        let uniforms = VolumetricsUniforms {
            inv_view_proj,
            camera_pos: [camera_pos[0], camera_pos[1], camera_pos[2], 0.0],
            near_far: [near, far, config.density, config.height_falloff],
            scatter_absorb: [config.scattering, config.absorption, 0.0, 0.0],
            sun_direction: [sun_direction[0], sun_direction[1], sun_direction[2], sun_intensity],
            shaft_params: [
                config.shaft_intensity,
                if config.light_shafts { 1.0 } else { 0.0 },
                config.steps as f32,
                mode as f32,
            ],
            screen_dims: [width as f32, height as f32, 0.0, 0.0],
        };

        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("volumetrics.bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.color_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&self.depth_sampler),
                },
            ],
        });

        // Render pass - clear to black, shader samples from color_input and writes composited result
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("volumetrics.pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: output_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}
