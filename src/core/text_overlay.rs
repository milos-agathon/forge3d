// src/core/text_overlay.rs
// Native text overlay pass for SDF/MSDF glyph quads.
// Renders screen-space quads (pixel coords) with alpha blending on top of the scene color target.

use wgpu::{
    vertex_attr_array, BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout,
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType, BufferAddress, BufferBindingType,
    BufferDescriptor, BufferUsages, ColorTargetState, ColorWrites, Device, FragmentState,
    PipelineLayoutDescriptor, PrimitiveState, PrimitiveTopology, Queue, RenderPipeline,
    RenderPipelineDescriptor, Sampler, SamplerBindingType, SamplerDescriptor, ShaderStages,
    TextureDescriptor, TextureDimension, TextureFormat, TextureSampleType, TextureUsages,
    TextureView, TextureViewDescriptor, VertexBufferLayout, VertexState, VertexStepMode,
};

use crate::core::error::{RenderError, RenderResult};
use crate::core::resource_tracker::{
    tracked_create_buffer, tracked_create_texture, TrackedBuffer, TrackedTexture,
};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TextOverlayUniforms {
    pub resolution: [f32; 2], // (width, height)
    pub alpha: f32,
    pub enabled: f32,
    pub channels: f32,  // 1.0 for SDF, 3.0 for MSDF
    pub smoothing: f32, // smoothing factor (pixels)
}

impl Default for TextOverlayUniforms {
    fn default() -> Self {
        Self {
            resolution: [1.0, 1.0],
            alpha: 1.0,
            enabled: 0.0,
            channels: 3.0,
            smoothing: 1.0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuTextOverlayUniforms {
    base: TextOverlayUniforms,
    union_scale: f32,
    px_range: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TextInstance {
    pub rect_min: [f32; 2], // x0, y0 in pixels
    pub rect_max: [f32; 2], // x1, y1 in pixels
    pub uv_min: [f32; 2],   // u0, v0 in atlas
    pub uv_max: [f32; 2],   // u1, v1 in atlas
    pub color: [f32; 4],    // rgba in linear 0..1
    pub halo_color: [f32; 4],
    pub halo_width: f32, // outward halo width in screen pixels
    pub rotation: f32,   // radians around rect center in screen space
}

impl TextInstance {
    pub fn new(
        rect_min: [f32; 2],
        rect_max: [f32; 2],
        uv_min: [f32; 2],
        uv_max: [f32; 2],
        color: [f32; 4],
    ) -> Self {
        Self {
            rect_min,
            rect_max,
            uv_min,
            uv_max,
            color,
            halo_color: [0.0, 0.0, 0.0, 0.0],
            halo_width: 0.0,
            rotation: 0.0,
        }
    }

    pub fn with_halo(mut self, halo_color: [f32; 4], halo_width: f32) -> Self {
        self.halo_color = halo_color;
        self.halo_width = halo_width.max(0.0);
        self
    }
}

struct UnionTargets {
    _fill: TrackedTexture,
    _halo: TrackedTexture,
    fill_view: TextureView,
    halo_view: TextureView,
    bind_group: BindGroup,
    size: [u32; 2],
}

fn validate_union_target_extents(size: [u32; 2], limit: u32) -> RenderResult<()> {
    if size.into_iter().any(|dimension| dimension > limit) {
        return Err(RenderError::degraded_capability(format!(
            "text union target {}x{} exceeds max_texture_dimension_2d={limit}",
            size[0], size[1]
        )));
    }
    Ok(())
}

fn finalize_union_target(
    resolution: [u32; 2],
    jacobian_scale: f64,
    limit: u32,
) -> RenderResult<([u32; 2], f32)> {
    if !jacobian_scale.is_finite() {
        return Err(RenderError::degraded_capability(
            "text union target scale is not finite",
        ));
    }
    let required_scale = jacobian_scale.max(1.0).ceil();
    let mut shader_scale = required_scale as f32;
    if f64::from(shader_scale) < required_scale {
        shader_scale = f32::from_bits(shader_scale.to_bits() + 1);
    }
    if !shader_scale.is_finite() || f64::from(shader_scale) > f64::from(u32::MAX) {
        return Err(RenderError::degraded_capability(format!(
            "text union target at {required_scale} texels/display-pixel exceeds max_texture_dimension_2d={limit}"
        )));
    }
    let scale = shader_scale as u32;
    let mut size = [0; 2];
    for (target, dimension) in size.iter_mut().zip(resolution) {
        *target = dimension.checked_mul(scale).ok_or_else(|| {
            RenderError::degraded_capability(format!(
                "text union target at {shader_scale} texels/display-pixel exceeds max_texture_dimension_2d={limit}"
            ))
        })?;
    }
    validate_union_target_extents(size, limit)?;
    Ok((size, shader_scale))
}

pub struct TextOverlayRenderer {
    pub uniforms: TextOverlayUniforms,
    pub uniform_buffer: TrackedBuffer,
    pub bind_group_layout: BindGroupLayout,
    pub bind_group: BindGroup,
    pub pipeline: RenderPipeline,
    distance_pipeline: RenderPipeline,
    union_pipeline: RenderPipeline,
    union_layout: BindGroupLayout,
    union_targets: Option<UnionTargets>,
    label_starts: Vec<usize>,
    union_groups: Vec<std::ops::Range<u32>>,
    resolution: [u32; 2],
    union_scale: f32,
    atlas_px_range: f32,

    pub quad_vbuf: TrackedBuffer,
    pub instance_buf: Option<TrackedBuffer>,
    pub instance_count: u32,

    pub atlas_tex: Option<TrackedTexture>,
    pub atlas_view: Option<TextureView>,
    pub atlas_sampler: Sampler,
}

impl TextOverlayRenderer {
    pub fn new(device: &Device, color_format: TextureFormat) -> RenderResult<Self> {
        let uniforms = TextOverlayUniforms::default();
        let uniform_buffer = tracked_create_buffer(
            device,
            &BufferDescriptor {
                label: Some("text_overlay_uniforms"),
                size: std::mem::size_of::<GpuTextOverlayUniforms>() as u64,
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            },
        )?;

        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("text_overlay_bgl"),
            entries: &[
                // uniforms
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX_FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // atlas texture (optional)
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                // atlas sampler
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // Fallback 1x1 white atlas for empty text.
        let dummy_tex = tracked_create_texture(
            device,
            &TextureDescriptor {
                label: Some("text_dummy_atlas"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba8Unorm,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
                view_formats: &[],
            },
        )?;
        let dummy_view = dummy_tex.create_view(&TextureViewDescriptor::default());
        let atlas_sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("text_atlas_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("text_overlay_bg"),
            layout: &bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&dummy_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&atlas_sampler),
                },
            ],
        });

        let shader = crate::core::shader_registry::create_labeled_shader_module(
            device,
            "text_overlay_shader",
            include_str!("../shaders/text_overlay.wgsl"),
        );

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("text_overlay_pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Vertex buffer layouts: 0) unit quad verts, 1) instance data (rect/color)
        let quad_layout = VertexBufferLayout {
            array_stride: (std::mem::size_of::<[f32; 2]>() as BufferAddress),
            step_mode: VertexStepMode::Vertex,
            attributes: &vertex_attr_array![0 => Float32x2],
        };
        let inst_layout = VertexBufferLayout {
            array_stride: (std::mem::size_of::<TextInstance>() as BufferAddress),
            step_mode: VertexStepMode::Instance,
            attributes: &vertex_attr_array![
                1 => Float32x2, // rect_min
                2 => Float32x2, // rect_max
                3 => Float32x2, // uv_min
                4 => Float32x2, // uv_max
                5 => Float32x4, // color
                6 => Float32x4, // halo_color
                7 => Float32,   // halo_width
                8 => Float32    // rotation
            ],
        };

        let vertex_buffers = [quad_layout, inst_layout];
        let pipeline = crate::core::shader_registry::create_render_pipeline_scoped(
            device,
            &RenderPipelineDescriptor {
                label: Some("text_overlay_pipeline"),
                layout: Some(&pipeline_layout),
                vertex: VertexState {
                    module: &shader,
                    entry_point: "vs_main",
                    buffers: &vertex_buffers,
                },
                primitive: PrimitiveState {
                    topology: PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(FragmentState {
                    module: &shader,
                    entry_point: "fs_main",
                    targets: &[Some(ColorTargetState {
                        format: color_format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: ColorWrites::ALL,
                    })],
                }),
                multiview: None,
            },
        );

        let union_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("text_union_bgl"),
            entries: &[0, 1].map(|binding| BindGroupLayoutEntry {
                binding,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    sample_type: TextureSampleType::Float { filterable: true },
                },
                count: None,
            }),
        });
        let union_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("text_union_pl"),
            bind_group_layouts: &[&bind_group_layout, &union_layout],
            push_constant_ranges: &[],
        });
        let maximum = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Max,
        };
        let distance_target = Some(ColorTargetState {
            format: TextureFormat::R16Float,
            blend: Some(wgpu::BlendState {
                color: maximum,
                alpha: maximum,
            }),
            write_mask: ColorWrites::RED,
        });
        let distance_pipeline = crate::core::shader_registry::create_render_pipeline_scoped(
            device,
            &RenderPipelineDescriptor {
                label: Some("text_union_distance_pipeline"),
                layout: Some(&pipeline_layout),
                vertex: VertexState {
                    module: &shader,
                    entry_point: "vs_main",
                    buffers: &vertex_buffers,
                },
                primitive: PrimitiveState {
                    topology: PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(FragmentState {
                    module: &shader,
                    entry_point: "fs_distance",
                    targets: &[distance_target.clone(), distance_target],
                }),
                multiview: None,
            },
        );
        let union_pipeline = crate::core::shader_registry::create_render_pipeline_scoped(
            device,
            &RenderPipelineDescriptor {
                label: Some("text_union_colour_pipeline"),
                layout: Some(&union_pipeline_layout),
                vertex: VertexState {
                    module: &shader,
                    entry_point: "vs_union",
                    buffers: &vertex_buffers,
                },
                primitive: PrimitiveState {
                    topology: PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(FragmentState {
                    module: &shader,
                    entry_point: "fs_union",
                    targets: &[Some(ColorTargetState {
                        format: color_format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: ColorWrites::ALL,
                    })],
                }),
                multiview: None,
            },
        );

        // Unit quad (0,0)-(1,1)
        let quad_data: [[f32; 2]; 6] = [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ];
        let quad_vbuf = tracked_create_buffer(
            device,
            &BufferDescriptor {
                label: Some("text_overlay_quad"),
                size: (quad_data.len() * std::mem::size_of::<[f32; 2]>()) as u64,
                usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                mapped_at_creation: true,
            },
        )?;
        quad_vbuf
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(bytemuck::cast_slice(&quad_data));
        quad_vbuf.unmap();

        Ok(Self {
            uniforms,
            uniform_buffer,
            bind_group_layout,
            bind_group,
            pipeline,
            distance_pipeline,
            union_pipeline,
            union_layout,
            union_targets: None,
            label_starts: Vec::new(),
            union_groups: Vec::new(),
            resolution: [1, 1],
            union_scale: 1.0,
            atlas_px_range: 8.0,
            quad_vbuf,
            instance_buf: None,
            instance_count: 0,
            atlas_tex: None,
            atlas_view: None,
            atlas_sampler,
        })
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.uniforms.enabled = if enabled { 1.0 } else { 0.0 };
    }
    pub fn set_alpha(&mut self, alpha: f32) {
        self.uniforms.alpha = alpha.clamp(0.0, 1.0);
    }
    pub fn set_resolution(&mut self, w: u32, h: u32) {
        self.resolution = [w, h];
        self.uniforms.resolution = [w as f32, h as f32];
    }
    pub fn set_channels(&mut self, channels: u32) {
        self.uniforms.channels = if channels == 4 {
            4.0
        } else if channels >= 3 {
            3.0
        } else {
            1.0
        };
    }
    pub fn set_smoothing(&mut self, px: f32) {
        self.uniforms.smoothing = px.max(0.1);
    }

    pub fn set_distance_range(&mut self, px_range: f32) {
        self.atlas_px_range = px_range;
    }

    pub fn upload_uniforms(&self, queue: &Queue) {
        let uniforms = GpuTextOverlayUniforms {
            base: self.uniforms,
            union_scale: self.union_scale,
            px_range: self.atlas_px_range,
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
    }

    pub fn upload_instances(
        &mut self,
        device: &Device,
        _queue: &Queue,
        instances: &[TextInstance],
    ) -> RenderResult<()> {
        self.union_groups.clear();
        if self.uniforms.channels == 4.0 {
            let atlas_size = self.atlas_tex.as_ref().map(|texture| texture.size());
            let mut k = 1.0_f64;
            let mut start = 0usize;
            for (index, glyph) in instances.iter().enumerate() {
                if index > start
                    && (self.label_starts.contains(&index)
                        || glyph.color != instances[start].color
                        || glyph.halo_color != instances[start].halo_color)
                {
                    self.union_groups.push(
                        u32::try_from(start)
                            .map_err(|_| RenderError::upload("text group exceeds u32"))?
                            ..u32::try_from(index)
                                .map_err(|_| RenderError::upload("text group exceeds u32"))?,
                    );
                    start = index;
                }
                if let Some(size) = atlas_size {
                    let w = f64::from(glyph.rect_max[0]) - f64::from(glyph.rect_min[0]);
                    let h = f64::from(glyph.rect_max[1]) - f64::from(glyph.rect_min[1]);
                    if w > 0.0 && h > 0.0 {
                        let sx = f64::from(size.width)
                            * (f64::from(glyph.uv_max[0]) - f64::from(glyph.uv_min[0]))
                            / w;
                        let sy = f64::from(size.height)
                            * (f64::from(glyph.uv_max[1]) - f64::from(glyph.uv_min[1]))
                            / h;
                        let (sin, cos) = f64::from(glyph.rotation).sin_cos();
                        k = k.max((sx * cos).hypot(sy * sin).max((sx * sin).hypot(sy * cos)));
                    }
                }
            }
            if !instances.is_empty() {
                self.union_groups.push(
                    u32::try_from(start)
                        .map_err(|_| RenderError::upload("text group exceeds u32"))?
                        ..u32::try_from(instances.len())
                            .map_err(|_| RenderError::upload("text group exceeds u32"))?,
                );
            }
            let limit = device.limits().max_texture_dimension_2d;
            let (size, union_scale) = finalize_union_target(self.resolution, k, limit)?;
            self.union_scale = union_scale;
            if !instances.is_empty() {
                self.prepare_union_targets(device, size)?;
            }
        }
        self.instance_count = u32::try_from(instances.len()).map_err(|_| {
            RenderError::Upload("text instance count exceeds the GPU u32 draw range".to_owned())
        })?;
        if self.instance_count == 0 {
            return Ok(());
        }
        let size = (instances.len() * std::mem::size_of::<TextInstance>()) as u64;
        let buf = tracked_create_buffer(
            device,
            &BufferDescriptor {
                label: Some("text_overlay_instances"),
                size,
                usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                mapped_at_creation: true,
            },
        )?;
        buf.slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(bytemuck::cast_slice(instances));
        buf.unmap();
        self.instance_buf = Some(buf);
        Ok(())
    }

    pub fn recreate_bind_group(
        &mut self,
        device: &Device,
        atlas_view: Option<&TextureView>,
    ) -> RenderResult<()> {
        // Use 1x1 fallback atlas when no view is available.
        let (dummy_tex, dummy_view) = if atlas_view.is_none() && self.atlas_view.is_none() {
            let t = tracked_create_texture(
                device,
                &TextureDescriptor {
                    label: Some("text_dummy_atlas_tmp"),
                    size: wgpu::Extent3d {
                        width: 1,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format: TextureFormat::Rgba8UnormSrgb,
                    usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
                    view_formats: &[],
                },
            )?;
            let v = t.create_view(&TextureViewDescriptor::default());
            (Some(t), Some(v))
        } else {
            (None, None)
        };
        let view = atlas_view
            .or(self.atlas_view.as_ref())
            .or(dummy_view.as_ref())
            .unwrap();
        self.bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("text_overlay_bg"),
            layout: &self.bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.atlas_sampler),
                },
            ],
        });
        drop(dummy_tex);
        Ok(())
    }

    pub fn set_atlas(&mut self, atlas_tex: TrackedTexture, atlas_view: TextureView) {
        self.atlas_tex = Some(atlas_tex);
        self.atlas_view = Some(atlas_view);
    }

    pub fn set_label_starts(&mut self, starts: &[usize]) {
        self.label_starts.clear();
        self.label_starts.extend_from_slice(starts);
    }

    fn prepare_union_targets(&mut self, device: &Device, size: [u32; 2]) -> RenderResult<()> {
        if self
            .union_targets
            .as_ref()
            .is_none_or(|target| target.size != size)
        {
            let make = |label| {
                tracked_create_texture(
                    device,
                    &TextureDescriptor {
                        label: Some(label),
                        size: wgpu::Extent3d {
                            width: size[0],
                            height: size[1],
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: TextureDimension::D2,
                        format: TextureFormat::R16Float,
                        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
                        view_formats: &[],
                    },
                )
            };
            let fill = make("text_union_fill")?;
            let halo = make("text_union_halo")?;
            let fill_view = fill.create_view(&TextureViewDescriptor::default());
            let halo_view = halo.create_view(&TextureViewDescriptor::default());
            let bind_group = device.create_bind_group(&BindGroupDescriptor {
                label: Some("text_union_bg"),
                layout: &self.union_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&fill_view),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&halo_view),
                    },
                ],
            });
            self.union_targets = Some(UnionTargets {
                _fill: fill,
                _halo: halo,
                fill_view,
                halo_view,
                bind_group,
                size,
            });
        }
        Ok(())
    }

    pub fn encode_union(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        colour: &TextureView,
    ) -> RenderResult<()> {
        if self.uniforms.channels != 4.0 || self.uniforms.enabled < 0.5 || self.instance_count == 0
        {
            return Ok(());
        }
        let (Some(targets), Some(instances)) = (&self.union_targets, &self.instance_buf) else {
            return Err(RenderError::upload("text union resources have not been prepared"));
        };
        for group in &self.union_groups {
            let attachment = |view| {
                Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: -65504.0,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })
            };
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("text_label_distance_union"),
                    color_attachments: &[
                        attachment(&targets.fill_view),
                        attachment(&targets.halo_view),
                    ],
                    depth_stencil_attachment: None,
                    ..Default::default()
                });
                pass.set_pipeline(&self.distance_pipeline);
                pass.set_bind_group(0, &self.bind_group, &[]);
                pass.set_vertex_buffer(0, self.quad_vbuf.slice(..));
                pass.set_vertex_buffer(1, instances.slice(..));
                pass.draw(0..6, group.clone());
            }
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("text_label_union_colour"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: colour,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            pass.set_pipeline(&self.union_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.set_bind_group(1, &targets.bind_group, &[]);
            pass.set_vertex_buffer(0, self.quad_vbuf.slice(..));
            pass.set_vertex_buffer(1, instances.slice(..));
            pass.draw(0..6, group.start..group.start + 1);
        }
        Ok(())
    }

    pub fn render<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        if self.uniforms.enabled < 0.5 || self.instance_count == 0 || self.uniforms.channels == 4.0
        {
            return;
        }
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_vertex_buffer(0, self.quad_vbuf.slice(..));
        if let Some(inst) = &self.instance_buf {
            pass.set_vertex_buffer(1, inst.slice(..));
            pass.draw(0..6, 0..self.instance_count);
        }
    }
}

#[cfg(test)]
mod union_extent_tests {
    use super::finalize_union_target;

    #[test]
    fn exact_device_limit_is_preserved_above_f32_integer_precision() {
        let limit = 16_777_219;
        assert!(finalize_union_target([16_777_220, 1], 1.0, limit).is_err());
        let (size, scale) = finalize_union_target([limit, 1], 1.0, limit).unwrap();
        assert_eq!(size, [limit, 1]);
        assert_eq!(scale, 1.0);
    }

    #[test]
    fn shader_scale_rounds_up_without_reducing_required_resolution() {
        let required = 16_777_217.0;
        let (size, scale) = finalize_union_target([1, 1], required, u32::MAX).unwrap();
        assert!(f64::from(scale) >= required);
        assert_eq!(size, [16_777_218, 16_777_218]);
        assert!(finalize_union_target([1, 1], required, 16_777_217).is_err());
    }

    #[test]
    fn target_multiplication_rejects_overflow() {
        assert!(finalize_union_target([u32::MAX, 1], 2.0, u32::MAX).is_err());
        let (size, scale) = finalize_union_target([448, 128], 3.01, 4096).unwrap();
        assert_eq!(size, [1792, 512]);
        assert_eq!(scale, 4.0);
    }
}
