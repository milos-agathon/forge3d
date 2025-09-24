// src/core/text_overlay.rs
// Native text overlay pass (rectangle placeholder for MSDF text).
// Renders screen-space quads (pixel coords) with alpha blending on top of the scene color target.

use wgpu::{
    vertex_attr_array, BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout,
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType, Buffer, BufferAddress,
    BufferBindingType, BufferDescriptor, BufferUsages, ColorTargetState, ColorWrites, Device,
    FragmentState, PipelineLayoutDescriptor, PrimitiveState, PrimitiveTopology, Queue,
    RenderPipeline, RenderPipelineDescriptor, Sampler, SamplerBindingType, SamplerDescriptor,
    ShaderModuleDescriptor, ShaderSource, ShaderStages, Texture, TextureDescriptor, TextureDimension,
    TextureFormat, TextureSampleType, TextureUsages, TextureView, TextureViewDescriptor,
    VertexBufferLayout, VertexState, VertexStepMode,
};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TextOverlayUniforms {
    pub resolution: [f32; 2], // (width, height)
    pub alpha: f32,
    pub enabled: f32,
    pub channels: f32, // 1.0 for SDF, 3.0 for MSDF
    pub smoothing: f32, // smoothing factor (pixels)
}

impl Default for TextOverlayUniforms {
    fn default() -> Self {
        Self { resolution: [1.0, 1.0], alpha: 1.0, enabled: 0.0, channels: 3.0, smoothing: 1.0 }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TextInstance {
    pub rect_min: [f32; 2], // x0, y0 in pixels
    pub rect_max: [f32; 2], // x1, y1 in pixels
    pub uv_min: [f32; 2],   // u0, v0 in atlas
    pub uv_max: [f32; 2],   // u1, v1 in atlas
    pub color: [f32; 4],    // rgba in linear 0..1
}

pub struct TextOverlayRenderer {
    pub uniforms: TextOverlayUniforms,
    pub uniform_buffer: Buffer,
    pub bind_group_layout: BindGroupLayout,
    pub bind_group: BindGroup,
    pub pipeline: RenderPipeline,

    pub quad_vbuf: Buffer,
    pub instance_buf: Option<Buffer>,
    pub instance_count: u32,

    pub atlas_tex: Option<Texture>,
    pub atlas_view: Option<TextureView>,
    pub atlas_sampler: Sampler,
}

impl TextOverlayRenderer {
    pub fn new(device: &Device, color_format: TextureFormat) -> Self {
        let uniforms = TextOverlayUniforms::default();
        let uniform_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("text_overlay_uniforms"),
            size: std::mem::size_of::<TextOverlayUniforms>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("text_overlay_bgl"),
            entries: &[
                // uniforms
                BindGroupLayoutEntry { binding: 0, visibility: ShaderStages::VERTEX_FRAGMENT, ty: BindingType::Buffer { ty: BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                // atlas texture (optional)
                BindGroupLayoutEntry { binding: 1, visibility: ShaderStages::FRAGMENT, ty: BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: TextureSampleType::Float { filterable: true } }, count: None },
                // atlas sampler
                BindGroupLayoutEntry { binding: 2, visibility: ShaderStages::FRAGMENT, ty: BindingType::Sampler(SamplerBindingType::Filtering), count: None },
            ],
        });

        // Dummy 1x1 white atlas
        let dummy_tex = device.create_texture(&TextureDescriptor {
            label: Some("text_dummy_atlas"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8UnormSrgb,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let dummy_view = dummy_tex.create_view(&TextureViewDescriptor::default());
        let atlas_sampler = device.create_sampler(&SamplerDescriptor { label: Some("text_atlas_sampler"), mag_filter: wgpu::FilterMode::Linear, min_filter: wgpu::FilterMode::Linear, mipmap_filter: wgpu::FilterMode::Nearest, ..Default::default() });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("text_overlay_bg"),
            layout: &bind_group_layout,
            entries: &[
                BindGroupEntry { binding: 0, resource: uniform_buffer.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&dummy_view) },
                BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&atlas_sampler) },
            ],
        });

        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("text_overlay_shader"),
            source: ShaderSource::Wgsl(include_str!("../shaders/text_overlay.wgsl").into()),
        });

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
                5 => Float32x4  // color
            ],
        };

        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("text_overlay_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: VertexState { module: &shader, entry_point: "vs_main", buffers: &[quad_layout, inst_layout] },
            primitive: PrimitiveState { topology: PrimitiveTopology::TriangleList, ..Default::default() },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(ColorTargetState { format: color_format, blend: Some(wgpu::BlendState::ALPHA_BLENDING), write_mask: ColorWrites::ALL })],
            }),
            multiview: None,
        });

        // Unit quad (0,0)-(1,1)
        let quad_data: [[f32; 2]; 6] = [
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
            [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],
        ];
        let quad_vbuf = device.create_buffer(&BufferDescriptor {
            label: Some("text_overlay_quad"),
            size: (quad_data.len() * std::mem::size_of::<[f32; 2]>()) as u64,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        quad_vbuf.slice(..).get_mapped_range_mut().copy_from_slice(bytemuck::cast_slice(&quad_data));
        quad_vbuf.unmap();

        Self {
            uniforms,
            uniform_buffer,
            bind_group_layout,
            bind_group,
            pipeline,
            quad_vbuf,
            instance_buf: None,
            instance_count: 0,
            atlas_tex: None,
            atlas_view: None,
            atlas_sampler,
        }
    }

    pub fn set_enabled(&mut self, enabled: bool) { self.uniforms.enabled = if enabled { 1.0 } else { 0.0 }; }
    pub fn set_alpha(&mut self, alpha: f32) { self.uniforms.alpha = alpha.clamp(0.0, 1.0); }
    pub fn set_resolution(&mut self, w: u32, h: u32) { self.uniforms.resolution = [w as f32, h as f32]; }
    pub fn set_channels(&mut self, channels: u32) { self.uniforms.channels = if channels >= 3 { 3.0 } else { 1.0 }; }
    pub fn set_smoothing(&mut self, px: f32) { self.uniforms.smoothing = px.max(0.1); }

    pub fn upload_uniforms(&self, queue: &Queue) { queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&self.uniforms)); }

    pub fn upload_instances(&mut self, device: &Device, _queue: &Queue, instances: &[TextInstance]) {
        self.instance_count = instances.len() as u32;
        if self.instance_count == 0 { return; }
        let size = (instances.len() * std::mem::size_of::<TextInstance>()) as u64;
        let buf = device.create_buffer(&BufferDescriptor {
            label: Some("text_overlay_instances"),
            size,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        buf.slice(..).get_mapped_range_mut().copy_from_slice(bytemuck::cast_slice(instances));
        buf.unmap();
        self.instance_buf = Some(buf);
    }

    pub fn recreate_bind_group(&mut self, device: &Device, atlas_view: Option<&TextureView>) {
        // Fallback to dummy if needed
        let (dummy_tex, dummy_view) = if atlas_view.is_none() && self.atlas_view.is_none() {
            let t = device.create_texture(&TextureDescriptor {
                label: Some("text_dummy_atlas_tmp"),
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba8UnormSrgb,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let v = t.create_view(&TextureViewDescriptor::default());
            (Some(t), Some(v))
        } else { (None, None) };
        let view = atlas_view.or(self.atlas_view.as_ref()).or(dummy_view.as_ref()).unwrap();
        self.bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("text_overlay_bg"),
            layout: &self.bind_group_layout,
            entries: &[
                BindGroupEntry { binding: 0, resource: self.uniform_buffer.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(view) },
                BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.atlas_sampler) },
            ],
        });
        drop(dummy_tex);
    }

    pub fn set_atlas(&mut self, atlas_tex: Texture, atlas_view: TextureView) {
        self.atlas_tex = Some(atlas_tex);
        self.atlas_view = Some(atlas_view);
    }

    pub fn render<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        if self.uniforms.enabled < 0.5 || self.instance_count == 0 { return; }
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_vertex_buffer(0, self.quad_vbuf.slice(..));
        if let Some(inst) = &self.instance_buf {
            pass.set_vertex_buffer(1, inst.slice(..));
            pass.draw(0..6, 0..self.instance_count);
        }
    }
}
