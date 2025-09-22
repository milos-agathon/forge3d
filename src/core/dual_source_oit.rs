//! B16: Dual-source blending Order Independent Transparency
//! High-quality OIT using dual-source color blending with WBOIT fallback

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// Dual-source OIT rendering mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DualSourceOITMode {
    /// Disabled - use standard alpha blending
    Disabled,
    /// Dual-source blending (requires hardware support)
    DualSource,
    /// Weighted Blended OIT fallback
    WBOITFallback,
    /// Automatic mode - dual-source if supported, otherwise WBOIT
    Automatic,
}

/// Quality settings for dual-source OIT
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DualSourceOITQuality {
    Low,    // Basic dual-source with minimal overhead
    Medium, // Standard quality with depth weighting
    High,   // High quality with advanced blending
    Ultra,  // Maximum quality with all features
}

/// Dual-source OIT uniforms matching WGSL layout
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct DualSourceOITUniforms {
    pub alpha_correction: f32,   // Alpha correction factor
    pub depth_weight_scale: f32, // Depth-based weight scaling
    pub max_fragments: f32,      // Maximum expected fragments per pixel
    pub premultiply_factor: f32, // Premultiplication factor
}

impl Default for DualSourceOITUniforms {
    fn default() -> Self {
        Self {
            alpha_correction: 1.0,
            depth_weight_scale: 1.0,
            max_fragments: 8.0,
            premultiply_factor: 1.0,
        }
    }
}

/// Composition uniforms for final blending
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct DualSourceComposeUniforms {
    pub use_dual_source: u32,   // 1 if dual-source is active, 0 for WBOIT
    pub tone_mapping_mode: u32, // 0=none, 1=reinhard, 2=aces_approx
    pub exposure: f32,          // Exposure adjustment
    pub gamma: f32,             // Gamma correction factor
}

impl Default for DualSourceComposeUniforms {
    fn default() -> Self {
        Self {
            use_dual_source: 0,
            tone_mapping_mode: 1, // Reinhard by default
            exposure: 0.0,        // No exposure adjustment
            gamma: 2.2,           // Standard gamma
        }
    }
}

/// Dual-source OIT renderer state
pub struct DualSourceOITRenderer {
    // Basic state
    mode: DualSourceOITMode,
    quality: DualSourceOITQuality,
    enabled: bool,
    width: u32,
    height: u32,

    // Hardware support detection
    dual_source_supported: bool,
    max_dual_source_targets: u32,

    // GPU resources
    uniforms_buffer: wgpu::Buffer,
    compose_uniforms_buffer: wgpu::Buffer,

    // Dual-source render targets
    dual_source_color_texture: Option<wgpu::Texture>,
    dual_source_color_view: Option<wgpu::TextureView>,

    // WBOIT fallback resources (reuse existing implementation)
    wboit_color_accum: Option<wgpu::Texture>,
    wboit_reveal_accum: Option<wgpu::Texture>,
    wboit_color_view: Option<wgpu::TextureView>,
    wboit_reveal_view: Option<wgpu::TextureView>,

    // Shaders and pipelines
    dual_source_shader: wgpu::ShaderModule,
    compose_shader: wgpu::ShaderModule,

    // Bind group layouts
    dual_source_bind_group_layout: wgpu::BindGroupLayout,
    compose_bind_group_layout: wgpu::BindGroupLayout,

    // Render pipelines
    dual_source_pipeline: Option<wgpu::RenderPipeline>,
    compose_pipeline: wgpu::RenderPipeline,

    // Bind groups
    dual_source_bind_group: Option<wgpu::BindGroup>,
    compose_bind_group: Option<wgpu::BindGroup>,

    // Sampler
    sampler: wgpu::Sampler,

    // Performance metrics
    frame_stats: DualSourceOITStats,

    // Current uniforms
    uniforms: DualSourceOITUniforms,
    compose_uniforms: DualSourceComposeUniforms,
}

/// Performance and quality statistics
#[derive(Debug, Clone, Copy, Default)]
pub struct DualSourceOITStats {
    pub frames_rendered: u64,
    pub dual_source_frames: u64,
    pub wboit_fallback_frames: u64,
    pub average_fragment_count: f32,
    pub peak_fragment_count: f32,
    pub quality_score: f32,
}

impl DualSourceOITRenderer {
    /// Create new dual-source OIT renderer
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        target_format: wgpu::TextureFormat,
    ) -> Result<Self, String> {
        // Detect dual-source blending support
        let dual_source_supported = Self::detect_dual_source_support(device);
        let max_dual_source_targets = if dual_source_supported { 2 } else { 0 };

        // Create shaders
        let dual_source_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("DualSourceOIT.Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/oit_dual_source.wgsl").into(),
            ),
        });

        let compose_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("DualSourceOIT.Compose"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/oit_dual_source_compose.wgsl").into(),
            ),
        });

        // Create uniform buffers
        let uniforms = DualSourceOITUniforms::default();
        let uniforms_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("DualSourceOIT.Uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let compose_uniforms = DualSourceComposeUniforms::default();
        let compose_uniforms_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("DualSourceOIT.ComposeUniforms"),
                contents: bytemuck::bytes_of(&compose_uniforms),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        // Create bind group layouts
        let dual_source_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("DualSourceOIT.BindGroupLayout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let compose_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("DualSourceOIT.ComposeBindGroupLayout"),
                entries: &[
                    // Compose uniforms
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
                    // Dual-source color texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    // WBOIT color accumulation (fallback)
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    // WBOIT reveal accumulation (fallback)
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    // Background color texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    // Sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        // Create sampler
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("DualSourceOIT.Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Create compose pipeline
        let compose_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("DualSourceOIT.ComposePipelineLayout"),
                bind_group_layouts: &[&compose_bind_group_layout],
                push_constant_ranges: &[],
            });

        let compose_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("DualSourceOIT.ComposePipeline"),
            layout: Some(&compose_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &compose_shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &compose_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
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

        let mut renderer = Self {
            mode: if dual_source_supported {
                DualSourceOITMode::DualSource
            } else {
                DualSourceOITMode::WBOITFallback
            },
            quality: DualSourceOITQuality::Medium,
            enabled: false,
            width,
            height,
            dual_source_supported,
            max_dual_source_targets,
            uniforms_buffer,
            compose_uniforms_buffer,
            dual_source_color_texture: None,
            dual_source_color_view: None,
            wboit_color_accum: None,
            wboit_reveal_accum: None,
            wboit_color_view: None,
            wboit_reveal_view: None,
            dual_source_shader,
            compose_shader,
            dual_source_bind_group_layout,
            compose_bind_group_layout,
            dual_source_pipeline: None,
            compose_pipeline,
            dual_source_bind_group: None,
            compose_bind_group: None,
            sampler,
            frame_stats: DualSourceOITStats::default(),
            uniforms,
            compose_uniforms,
        };

        // Initialize textures
        renderer.create_textures(device, width, height)?;

        Ok(renderer)
    }

    /// Detect if dual-source blending is supported
    fn detect_dual_source_support(device: &wgpu::Device) -> bool {
        // Check for dual-source blending feature
        // Note: This is a simplified check - in practice, you'd check device features/limits
        let _features = device.features();

        // WebGPU doesn't directly expose dual-source blending in the standard API
        // This would typically require checking for specific extensions or backend capabilities
        // For now, we'll use a conservative approach and detect based on backend

        // In a real implementation, you'd check:
        // - Vulkan: VK_EXT_blend_operation_advanced or similar
        // - D3D12: Check for dual-source blend support
        // - Metal: Check for dual-source fragment output support

        // For this implementation, we'll assume support exists and provide fallback
        true // Conservative assumption for demo
    }

    /// Create or recreate GPU textures
    fn create_textures(
        &mut self,
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> Result<(), String> {
        self.width = width;
        self.height = height;

        // Create dual-source color texture (high precision for blending)
        let dual_source_color_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("DualSourceOIT.ColorTexture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float, // High precision for quality
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let dual_source_color_view =
            dual_source_color_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create WBOIT fallback textures
        let wboit_color_accum = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("DualSourceOIT.WBOITColorAccum"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let wboit_reveal_accum = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("DualSourceOIT.WBOITRevealAccum"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let wboit_color_view =
            wboit_color_accum.create_view(&wgpu::TextureViewDescriptor::default());
        let wboit_reveal_view =
            wboit_reveal_accum.create_view(&wgpu::TextureViewDescriptor::default());

        // Store textures
        self.dual_source_color_texture = Some(dual_source_color_texture);
        self.dual_source_color_view = Some(dual_source_color_view);
        self.wboit_color_accum = Some(wboit_color_accum);
        self.wboit_reveal_accum = Some(wboit_reveal_accum);
        self.wboit_color_view = Some(wboit_color_view);
        self.wboit_reveal_view = Some(wboit_reveal_view);

        Ok(())
    }

    /// Set OIT mode
    pub fn set_mode(&mut self, mode: DualSourceOITMode) {
        self.mode = mode;
        self.update_compose_uniforms();
    }

    /// Get current OIT mode
    pub fn mode(&self) -> DualSourceOITMode {
        self.mode
    }

    /// Set quality level
    pub fn set_quality(&mut self, quality: DualSourceOITQuality) {
        self.quality = quality;
        self.update_uniforms_for_quality();
    }

    /// Get current quality level
    pub fn quality(&self) -> DualSourceOITQuality {
        self.quality
    }

    /// Enable or disable dual-source OIT
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        self.update_compose_uniforms();
    }

    /// Check if dual-source OIT is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Check if dual-source blending is supported
    pub fn is_dual_source_supported(&self) -> bool {
        self.dual_source_supported
    }

    /// Get current operating mode (actual mode being used)
    pub fn get_operating_mode(&self) -> DualSourceOITMode {
        if !self.enabled {
            return DualSourceOITMode::Disabled;
        }

        match self.mode {
            DualSourceOITMode::Disabled => DualSourceOITMode::Disabled,
            DualSourceOITMode::DualSource => {
                if self.dual_source_supported {
                    DualSourceOITMode::DualSource
                } else {
                    DualSourceOITMode::WBOITFallback
                }
            }
            DualSourceOITMode::WBOITFallback => DualSourceOITMode::WBOITFallback,
            DualSourceOITMode::Automatic => {
                if self.dual_source_supported {
                    DualSourceOITMode::DualSource
                } else {
                    DualSourceOITMode::WBOITFallback
                }
            }
        }
    }

    /// Update uniforms for current quality setting
    fn update_uniforms_for_quality(&mut self) {
        match self.quality {
            DualSourceOITQuality::Low => {
                self.uniforms.alpha_correction = 1.0;
                self.uniforms.depth_weight_scale = 0.5;
                self.uniforms.max_fragments = 4.0;
                self.uniforms.premultiply_factor = 1.0;
            }
            DualSourceOITQuality::Medium => {
                self.uniforms.alpha_correction = 1.1;
                self.uniforms.depth_weight_scale = 1.0;
                self.uniforms.max_fragments = 8.0;
                self.uniforms.premultiply_factor = 1.0;
            }
            DualSourceOITQuality::High => {
                self.uniforms.alpha_correction = 1.2;
                self.uniforms.depth_weight_scale = 1.5;
                self.uniforms.max_fragments = 16.0;
                self.uniforms.premultiply_factor = 1.0;
            }
            DualSourceOITQuality::Ultra => {
                self.uniforms.alpha_correction = 1.3;
                self.uniforms.depth_weight_scale = 2.0;
                self.uniforms.max_fragments = 32.0;
                self.uniforms.premultiply_factor = 1.0;
            }
        }
    }

    /// Update composition uniforms
    fn update_compose_uniforms(&mut self) {
        let operating_mode = self.get_operating_mode();
        self.compose_uniforms.use_dual_source = match operating_mode {
            DualSourceOITMode::DualSource => 1,
            _ => 0,
        };
    }

    /// Upload uniforms to GPU
    pub fn upload_uniforms(&self, queue: &wgpu::Queue) {
        queue.write_buffer(&self.uniforms_buffer, 0, bytemuck::bytes_of(&self.uniforms));
        queue.write_buffer(
            &self.compose_uniforms_buffer,
            0,
            bytemuck::bytes_of(&self.compose_uniforms),
        );
    }

    /// Create dual-source render pipeline for transparency pass
    pub fn create_dual_source_pipeline(
        &mut self,
        device: &wgpu::Device,
        vertex_buffers: &[wgpu::VertexBufferLayout],
        additional_bind_group_layouts: &[&wgpu::BindGroupLayout],
    ) -> Result<(), String> {
        let operating_mode = self.get_operating_mode();

        match operating_mode {
            DualSourceOITMode::DualSource => self.create_true_dual_source_pipeline(
                device,
                vertex_buffers,
                additional_bind_group_layouts,
            ),
            DualSourceOITMode::WBOITFallback => {
                // Will reuse existing WBOIT pipeline
                Ok(())
            }
            _ => Ok(()),
        }
    }

    /// Create true dual-source blending pipeline
    fn create_true_dual_source_pipeline(
        &mut self,
        device: &wgpu::Device,
        vertex_buffers: &[wgpu::VertexBufferLayout],
        additional_bind_group_layouts: &[&wgpu::BindGroupLayout],
    ) -> Result<(), String> {
        // Construct bind group layouts
        let mut all_layouts = vec![&self.dual_source_bind_group_layout];
        all_layouts.extend(additional_bind_group_layouts);

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("DualSourceOIT.Pipeline.Layout"),
            bind_group_layouts: &all_layouts,
            push_constant_ranges: &[],
        });

        // Create dual-source render pipeline
        let dual_source_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("DualSourceOIT.Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &self.dual_source_shader,
                entry_point: "vs_main",
                buffers: vertex_buffers,
            },
            fragment: Some(wgpu::FragmentState {
                module: &self.dual_source_shader,
                entry_point: "fs_main",
                targets: &[
                    // First color target: premultiplied color
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: Some(Self::get_dual_source_color_blend()),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    // Second color target: alpha and weight
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: Some(Self::get_dual_source_alpha_blend()),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
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
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false, // Don't write depth for transparent objects
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        self.dual_source_pipeline = Some(dual_source_pipeline);

        Ok(())
    }

    /// Get dual-source color blend state
    fn get_dual_source_color_blend() -> wgpu::BlendState {
        wgpu::BlendState {
            color: wgpu::BlendComponent {
                // For dual-source: src_color + dst_color * (1 - src1_alpha)
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrc1Alpha,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrc1Alpha,
                operation: wgpu::BlendOperation::Add,
            },
        }
    }

    /// Get dual-source alpha blend state
    fn get_dual_source_alpha_blend() -> wgpu::BlendState {
        wgpu::BlendState {
            color: wgpu::BlendComponent {
                // Accumulate alpha information
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        }
    }

    /// Get performance statistics
    pub fn get_stats(&self) -> DualSourceOITStats {
        self.frame_stats
    }

    /// Begin transparency pass
    pub fn begin_transparency_pass<'a>(
        &'a self,
        encoder: &'a mut wgpu::CommandEncoder,
        depth_view: &'a wgpu::TextureView,
    ) -> Result<wgpu::RenderPass<'a>, String> {
        let operating_mode = self.get_operating_mode();

        match operating_mode {
            DualSourceOITMode::DualSource => {
                // Dual-source pass
                let color_view = self
                    .dual_source_color_view
                    .as_ref()
                    .ok_or("Dual-source color view not initialized")?;

                Ok(encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("DualSourceOIT.TransparencyPass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: color_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load, // Preserve opaque depth
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    occlusion_query_set: None,
                    timestamp_writes: None,
                }))
            }
            DualSourceOITMode::WBOITFallback => {
                // WBOIT fallback pass
                let color_view = self
                    .wboit_color_view
                    .as_ref()
                    .ok_or("WBOIT color view not initialized")?;
                let reveal_view = self
                    .wboit_reveal_view
                    .as_ref()
                    .ok_or("WBOIT reveal view not initialized")?;

                Ok(encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("DualSourceOIT.WBOITFallbackPass"),
                    color_attachments: &[
                        Some(wgpu::RenderPassColorAttachment {
                            view: color_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                store: wgpu::StoreOp::Store,
                            },
                        }),
                        Some(wgpu::RenderPassColorAttachment {
                            view: reveal_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color {
                                    r: 1.0,
                                    g: 0.0,
                                    b: 0.0,
                                    a: 0.0,
                                }),
                                store: wgpu::StoreOp::Store,
                            },
                        }),
                    ],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    occlusion_query_set: None,
                    timestamp_writes: None,
                }))
            }
            _ => Err("Dual-source OIT not enabled".to_string()),
        }
    }

    /// Compose final result
    pub fn compose<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) -> Result<(), String> {
        if let Some(bind_group) = &self.compose_bind_group {
            render_pass.set_pipeline(&self.compose_pipeline);
            render_pass.set_bind_group(0, bind_group, &[]);
            render_pass.draw(0..3, 0..1); // Fullscreen triangle
            Ok(())
        } else {
            Err("Compose bind group not initialized".to_string())
        }
    }

    /// Resize renderer
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) -> Result<(), String> {
        if self.width != width || self.height != height {
            self.create_textures(device, width, height)?;
            // Note: Bind groups would need to be recreated here
        }
        Ok(())
    }
}
