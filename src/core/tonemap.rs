//! C8: Full linear→tonemap→sRGB pipeline
//!
//! Provides a dedicated post-processing pass for tone mapping from HDR linear color
//! to sRGB output with exposure control.

use crate::error::RenderResult;
use std::borrow::Cow;
use wgpu::*;

/// Tonemap post-processor for converting HDR linear to sRGB
pub struct TonemapProcessor {
    /// Render pipeline for tonemap pass
    pipeline: RenderPipeline,
    /// Bind group layout for tonemap uniforms
    bind_group_layout: BindGroupLayout,
    /// Sampler for HDR input texture
    sampler: Sampler,
    /// Current exposure value
    exposure: f32,
    /// Uniform buffer for exposure
    uniform_buffer: Buffer,
}

/// Tonemap uniform data
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct TonemapUniforms {
    /// Exposure multiplier
    exposure: f32,
    /// Padding for 16-byte alignment
    _pad: [f32; 3],
}

impl TonemapProcessor {
    /// Create a new tonemap processor
    pub fn new(device: &Device, output_format: TextureFormat) -> RenderResult<Self> {
        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("tonemap_bind_group_layout"),
            entries: &[
                // HDR input texture
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Sampler
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
                // Uniforms (exposure)
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("tonemap_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create shader module
        let shader_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("tonemap_shader"),
            source: ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "../shaders/postprocess_tonemap.wgsl"
            ))),
        });

        // Create render pipeline
        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("tonemap_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: &shader_module,
                entry_point: "vs_main",
                buffers: &[], // Full-screen triangle needs no vertex buffer
            },
            fragment: Some(FragmentState {
                module: &shader_module,
                entry_point: "fs_main",
                targets: &[Some(ColorTargetState {
                    format: output_format,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: FrontFace::Ccw,
                cull_mode: None, // Don't cull for full-screen triangle
                unclipped_depth: false,
                polygon_mode: PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        // Create sampler
        let sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("tonemap_sampler"),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Linear,
            ..Default::default()
        });

        // Create uniform buffer
        let uniform_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("tonemap_uniforms"),
            size: std::mem::size_of::<TonemapUniforms>() as BufferAddress,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self {
            pipeline,
            bind_group_layout,
            sampler,
            exposure: 1.0, // Default exposure
            uniform_buffer,
        })
    }

    /// Set the exposure value
    pub fn set_exposure(&mut self, exposure: f32) {
        self.exposure = exposure;
    }

    /// Get the current exposure value
    pub fn exposure(&self) -> f32 {
        self.exposure
    }

    /// Render tone-mapped output from HDR input
    pub fn render(
        &self,
        encoder: &mut CommandEncoder,
        device: &Device,
        queue: &Queue,
        hdr_input: &TextureView,
        srgb_output: &TextureView,
    ) -> RenderResult<()> {
        // Update uniforms
        let uniforms = TonemapUniforms {
            exposure: self.exposure,
            _pad: [0.0; 3],
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        // Create bind group
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("tonemap_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(hdr_input),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&self.sampler),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
            ],
        });

        // Record render pass
        {
            let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("tonemap_pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: srgb_output,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(Color::BLACK),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_pipeline(&self.pipeline);
            render_pass.set_bind_group(0, &bind_group, &[]);

            // Draw full-screen triangle (3 vertices, no vertex buffer needed)
            render_pass.draw(0..3, 0..1);
        }

        Ok(())
    }

    /// Create a bind group for the tonemap pass
    pub fn create_bind_group(&self, device: &Device, hdr_input: &TextureView) -> BindGroup {
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("tonemap_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(hdr_input),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&self.sampler),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
            ],
        })
    }

    /// Get the pipeline for manual rendering
    pub fn pipeline(&self) -> &RenderPipeline {
        &self.pipeline
    }

    /// Update uniforms buffer manually
    pub fn update_uniforms(&self, queue: &Queue) {
        let uniforms = TonemapUniforms {
            exposure: self.exposure,
            _pad: [0.0; 3],
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
    }

    /// Create compute-based tone mapping effect for post-processing chain
    ///
    /// This method creates a compute-based version of the tone mapping effect
    /// that can be integrated into the post-processing pipeline.
    pub fn create_compute_effect(&self, device: &Device) -> RenderResult<ComputePipeline> {
        // Create compute shader for tone mapping
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("tonemap_compute_shader"),
            source: ShaderSource::Wgsl(
                r#"
                @group(0) @binding(0) var<uniform> uniforms: TonemapUniforms;
                @group(0) @binding(1) var input_texture: texture_2d<f32>;
                @group(0) @binding(2) var output_texture: texture_storage_2d<rgba8unorm, write>;
                
                struct TonemapUniforms {
                    exposure: f32,
                    _pad: vec3<f32>,
                };
                
                @compute @workgroup_size(16, 16)
                fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                    let dimensions = textureDimensions(input_texture);
                    let coord = global_id.xy;
                    
                    if (coord.x >= dimensions.x || coord.y >= dimensions.y) {
                        return;
                    }
                    
                    let hdr_color = textureLoad(input_texture, coord, 0);
                    
                    // Apply exposure
                    let exposed = hdr_color.rgb * uniforms.exposure;
                    
                    // Simple Reinhard tone mapping
                    let tone_mapped = exposed / (exposed + vec3<f32>(1.0));
                    
                    // Gamma correction (sRGB approximation)
                    let gamma_corrected = pow(tone_mapped, vec3<f32>(1.0 / 2.2));
                    
                    textureStore(output_texture, coord, vec4<f32>(gamma_corrected, hdr_color.a));
                }
            "#
                .into(),
            ),
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("tonemap_compute_pipeline_layout"),
            bind_group_layouts: &[&self.bind_group_layout],
            push_constant_ranges: &[],
        });

        let compute_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("tonemap_compute_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });

        Ok(compute_pipeline)
    }
}
