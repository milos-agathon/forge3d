//! C8: Full linear->tonemap->sRGB pipeline
//!
//! Provides a dedicated post-processing pass for tone mapping from HDR linear color
//! to sRGB output with exposure control.

use super::error::RenderResult;
use std::borrow::Cow;
use wgpu::*;

/// Tonemap post-processor for converting HDR linear to sRGB
/// M6: Extended with LUT and white balance support
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
    // M6: Extended settings
    /// White point for extended operators
    white_point: f32,
    /// Gamma correction value
    gamma: f32,
    /// Tonemap operator index
    operator_index: u32,
    /// LUT enabled
    lut_enabled: bool,
    /// LUT blend strength
    lut_strength: f32,
    /// LUT dimension
    lut_size: f32,
    /// White balance enabled
    white_balance_enabled: bool,
    /// Color temperature in Kelvin
    temperature: f32,
    /// Green-magenta tint
    tint: f32,
    /// M6: Default 1x1x1 LUT texture (identity)
    #[allow(dead_code)]
    default_lut_texture: Texture,
    default_lut_view: TextureView,
    /// M6: LUT sampler
    lut_sampler: Sampler,
}

/// Tonemap uniform data
/// M6: Extended with LUT and white balance parameters
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct TonemapUniforms {
    /// Exposure multiplier
    exposure: f32,
    /// White point for extended operators
    white_point: f32,
    /// Gamma correction
    gamma: f32,
    /// Tonemap operator index: 0=Reinhard, 1=ReinhardExtended, 2=ACES, 3=Uncharted2, 4=Exposure
    operator_index: u32,
    // M6: LUT and white balance
    /// LUT enabled (0=disabled, 1=enabled)
    lut_enabled: u32,
    /// LUT blend strength (0-1)
    lut_strength: f32,
    /// LUT dimension (e.g., 32 for 32x32x32)
    lut_size: f32,
    /// White balance enabled (0=disabled, 1=enabled)
    white_balance_enabled: u32,
    /// Color temperature in Kelvin
    temperature: f32,
    /// Green-magenta tint (-1 to 1)
    tint: f32,
    /// Padding for 16-byte alignment
    _pad0: f32,
    _pad1: f32,
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
                // M6: 3D LUT texture
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                // M6: LUT sampler
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
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

        // M6: Create default 1x1x1 identity LUT texture
        let default_lut_texture = device.create_texture(&TextureDescriptor {
            label: Some("tonemap_default_lut"),
            size: Extent3d {
                width: 2,
                height: 2,
                depth_or_array_layers: 2,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D3,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let default_lut_view = default_lut_texture.create_view(&TextureViewDescriptor::default());

        // M6: Create LUT sampler with trilinear filtering
        let lut_sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("tonemap_lut_sampler"),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Linear,
            ..Default::default()
        });

        Ok(Self {
            pipeline,
            bind_group_layout,
            sampler,
            exposure: 1.0,
            uniform_buffer,
            // M6: Extended settings with defaults
            white_point: 4.0,
            gamma: 2.2,
            operator_index: 2, // ACES
            lut_enabled: false,
            lut_strength: 1.0,
            lut_size: 0.0,
            white_balance_enabled: false,
            temperature: 6500.0,
            tint: 0.0,
            default_lut_texture,
            default_lut_view,
            lut_sampler,
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

    // M6: Extended setters for tonemap settings
    /// Set the tonemap operator index
    pub fn set_operator(&mut self, index: u32) {
        self.operator_index = index;
    }

    /// Set white point for extended operators
    pub fn set_white_point(&mut self, white_point: f32) {
        self.white_point = white_point.max(0.1);
    }

    /// Set gamma correction value
    pub fn set_gamma(&mut self, gamma: f32) {
        self.gamma = gamma.max(0.1);
    }

    /// Enable/disable LUT
    pub fn set_lut_enabled(&mut self, enabled: bool) {
        self.lut_enabled = enabled;
    }

    /// Set LUT blend strength
    pub fn set_lut_strength(&mut self, strength: f32) {
        self.lut_strength = strength.clamp(0.0, 1.0);
    }

    /// Set LUT dimension
    pub fn set_lut_size(&mut self, size: f32) {
        self.lut_size = size;
    }

    /// Enable/disable white balance
    pub fn set_white_balance_enabled(&mut self, enabled: bool) {
        self.white_balance_enabled = enabled;
    }

    /// Set color temperature in Kelvin
    pub fn set_temperature(&mut self, temp: f32) {
        self.temperature = temp.clamp(2000.0, 12000.0);
    }

    /// Set green-magenta tint
    pub fn set_tint(&mut self, tint: f32) {
        self.tint = tint.clamp(-1.0, 1.0);
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
        // Update uniforms with all M6 fields
        let uniforms = TonemapUniforms {
            exposure: self.exposure,
            white_point: self.white_point,
            gamma: self.gamma,
            operator_index: self.operator_index,
            lut_enabled: if self.lut_enabled { 1 } else { 0 },
            lut_strength: self.lut_strength,
            lut_size: self.lut_size,
            white_balance_enabled: if self.white_balance_enabled { 1 } else { 0 },
            temperature: self.temperature,
            tint: self.tint,
            _pad0: 0.0,
            _pad1: 0.0,
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        // Create bind group with M6 LUT bindings
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
                // M6: LUT texture (use default if not set)
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::TextureView(&self.default_lut_view),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::Sampler(&self.lut_sampler),
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
                // M6: LUT texture bindings
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::TextureView(&self.default_lut_view),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::Sampler(&self.lut_sampler),
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
            white_point: self.white_point,
            gamma: self.gamma,
            operator_index: self.operator_index,
            lut_enabled: if self.lut_enabled { 1 } else { 0 },
            lut_strength: self.lut_strength,
            lut_size: self.lut_size,
            white_balance_enabled: if self.white_balance_enabled { 1 } else { 0 },
            temperature: self.temperature,
            tint: self.tint,
            _pad0: 0.0,
            _pad1: 0.0,
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
