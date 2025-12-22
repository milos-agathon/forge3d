// src/viewer/terrain/motion_blur.rs
// Motion blur via temporal accumulation across shutter interval

use std::sync::Arc;
use wgpu::util::DeviceExt;

/// Motion blur accumulation uniforms for the resolve shader
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MotionBlurUniforms {
    /// Screen dimensions: width, height, 1/width, 1/height
    pub screen_dims: [f32; 4],
    /// Sample count for normalization
    pub sample_count: [f32; 4],
}

/// Motion blur configuration (mirrors pbr_renderer::MotionBlurConfig)
#[derive(Debug, Clone)]
pub struct MotionBlurConfig {
    pub enabled: bool,
    pub samples: u32,
    pub shutter_open: f32,
    pub shutter_close: f32,
    pub cam_phi_delta: f32,
    pub cam_theta_delta: f32,
    pub cam_radius_delta: f32,
}

impl Default for MotionBlurConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            samples: 16,
            shutter_open: 0.0,
            shutter_close: 0.5,
            cam_phi_delta: 0.0,
            cam_theta_delta: 0.0,
            cam_radius_delta: 0.0,
        }
    }
}

/// Motion blur accumulator for temporal frame blending
pub struct MotionBlurAccumulator {
    device: Arc<wgpu::Device>,
    resolve_pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    uniform_buffer: wgpu::Buffer,
    // Accumulation texture (Rgba32Float for HDR accumulation)
    accum_texture: Option<wgpu::Texture>,
    accum_view: Option<wgpu::TextureView>,
    current_size: (u32, u32),
    current_sample: u32,
    total_samples: u32,
}

impl MotionBlurAccumulator {
    pub fn new(device: Arc<wgpu::Device>, surface_format: wgpu::TextureFormat) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("motion_blur.bind_group_layout"),
            entries: &[
                // Accumulated color texture
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // Uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("motion_blur.pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("motion_blur.shader"),
            source: wgpu::ShaderSource::Wgsl(MOTION_BLUR_SHADER.into()),
        });

        let resolve_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("motion_blur.resolve_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_resolve",
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("motion_blur.sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("motion_blur.uniforms"),
            contents: bytemuck::cast_slice(&[MotionBlurUniforms {
                screen_dims: [1.0, 1.0, 1.0, 1.0],
                sample_count: [1.0, 0.0, 0.0, 0.0],
            }]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        Self {
            device,
            resolve_pipeline,
            bind_group_layout,
            sampler,
            uniform_buffer,
            accum_texture: None,
            accum_view: None,
            current_size: (0, 0),
            current_sample: 0,
            total_samples: 1,
        }
    }

    /// Ensure accumulation texture is allocated
    fn ensure_textures(&mut self, width: u32, height: u32) {
        if self.current_size != (width, height) || self.accum_texture.is_none() {
            let accum_tex = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("motion_blur.accum"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            self.accum_view = Some(accum_tex.create_view(&wgpu::TextureViewDescriptor::default()));
            self.accum_texture = Some(accum_tex);
            self.current_size = (width, height);
        }
    }

    /// Get the accumulation texture view for rendering
    pub fn get_accum_view(&mut self, width: u32, height: u32) -> &wgpu::TextureView {
        self.ensure_textures(width, height);
        self.accum_view.as_ref().unwrap()
    }

    /// Begin a new motion blur accumulation sequence
    pub fn begin_accumulation(&mut self, samples: u32, width: u32, height: u32) {
        self.ensure_textures(width, height);
        self.current_sample = 0;
        self.total_samples = samples.max(1);
    }

    /// Check if accumulation is in progress
    pub fn is_accumulating(&self) -> bool {
        self.current_sample < self.total_samples
    }

    /// Get current sample index
    pub fn current_sample(&self) -> u32 {
        self.current_sample
    }

    /// Increment sample counter after a frame is rendered
    pub fn advance_sample(&mut self) {
        self.current_sample += 1;
    }

    /// Calculate interpolation factor for current sample
    pub fn get_interpolation_t(&self, config: &MotionBlurConfig) -> f32 {
        let shutter_range = config.shutter_close - config.shutter_open;
        let sample_t = (self.current_sample as f32 + 0.5) / self.total_samples as f32;
        config.shutter_open + shutter_range * sample_t
    }

    /// Get interpolated camera parameters for current sample
    pub fn get_interpolated_camera(
        &self,
        config: &MotionBlurConfig,
        base_phi: f32,
        base_theta: f32,
        base_radius: f32,
    ) -> (f32, f32, f32) {
        let t = self.get_interpolation_t(config);
        let phi = base_phi + config.cam_phi_delta * t;
        let theta = base_theta + config.cam_theta_delta * t;
        let radius = base_radius + config.cam_radius_delta * t;
        (phi, theta, radius)
    }

    /// Resolve the accumulated buffer to the final output
    pub fn resolve(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        accum_view: &wgpu::TextureView,
        output_view: &wgpu::TextureView,
        width: u32,
        height: u32,
        sample_count: u32,
    ) {
        let uniforms = MotionBlurUniforms {
            screen_dims: [
                width as f32,
                height as f32,
                1.0 / width as f32,
                1.0 / height as f32,
            ],
            sample_count: [sample_count as f32, 0.0, 0.0, 0.0],
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("motion_blur.resolve_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(accum_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
            ],
        });

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("motion_blur.resolve_pass"),
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

        pass.set_pipeline(&self.resolve_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

/// Shader for motion blur resolve (divide accumulated color by sample count)
const MOTION_BLUR_SHADER: &str = r#"
struct Uniforms {
    screen_dims: vec4<f32>,
    sample_count: vec4<f32>,
}

@group(0) @binding(0) var accum_tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<uniform> u: Uniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32((vertex_index << 1u) & 2u);
    let y = f32(vertex_index & 2u);
    out.position = vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(x, 1.0 - y);
    return out;
}

@fragment
fn fs_resolve(in: VertexOutput) -> @location(0) vec4<f32> {
    let accum = textureSample(accum_tex, samp, in.uv);
    let sample_count = u.sample_count.x;
    
    // Divide accumulated color by sample count
    let color = accum.rgb / max(sample_count, 1.0);
    
    return vec4<f32>(color, 1.0);
}
"#;
