// src/core/overlays.rs
// Native overlay compositor for drape overlays and altitude color ramp.
// Draws a fullscreen quad with alpha blending onto the scene color target.

use glam::Mat4;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType, BufferDescriptor, BufferUsages,
    ColorTargetState, ColorWrites, Device, FragmentState, PipelineLayoutDescriptor, PrimitiveState,
    PrimitiveTopology, Queue, RenderPipeline, RenderPipelineDescriptor, Sampler,
    SamplerBindingType, SamplerDescriptor, ShaderModuleDescriptor, ShaderSource, ShaderStages,
    Texture, TextureDescriptor, TextureDimension, TextureFormat, TextureSampleType,
    TextureUsages, TextureView, TextureViewDescriptor, TextureViewDimension, VertexState,
};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct OverlayUniforms {
    pub view_proj: [[f32; 4]; 4], // reserved for future (e.g., 3D text quads)
    pub overlay_params: [f32; 4], // x: overlay_enabled, y: overlay_alpha, z: altitude_enabled, w: altitude_alpha
    pub overlay_uv: [f32; 4],     // x: uv_offset_x, y: uv_offset_y, z: uv_scale_x, w: uv_scale_y
    pub contour_params: [f32; 4], // x: contour_enabled, y: interval, z: thickness_mul, w: unused
    pub contour_color: [f32; 4],  // rgba for contour lines
}

impl Default for OverlayUniforms {
    fn default() -> Self {
        Self {
            view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            // x: overlay_enabled, y: overlay_alpha, z: altitude_enabled, w: altitude_alpha
            overlay_params: [0.0, 1.0, 0.0, 0.35],
            overlay_uv: [0.0, 0.0, 1.0, 1.0],
            contour_params: [0.0, 0.1, 1.0, 0.0],
            contour_color: [0.0, 0.0, 0.0, 0.75],
        }
    }
}

pub struct OverlayRenderer {
    pub uniforms: OverlayUniforms,
    pub uniform_buffer: Buffer,
    pub bind_group_layout: BindGroupLayout,
    pub bind_group: BindGroup,
    pub pipeline: RenderPipeline,

    // resources
    pub overlay_tex: Option<Texture>,
    pub overlay_view: Option<TextureView>,
    pub overlay_sampler: Sampler,
    pub height_sampler: Sampler,

    // formats
    pub overlay_format: TextureFormat,
}

impl OverlayRenderer {
    pub fn new(device: &Device, color_format: TextureFormat) -> Self {
        let uniforms = OverlayUniforms::default();
        let uniform_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("overlay_uniforms"),
            size: std::mem::size_of::<OverlayUniforms>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("overlay_bgl"),
            entries: &[
                // uniforms
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // overlay texture (optional: will bind a 1x1 if none)
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        multisampled: false,
                        view_dimension: TextureViewDimension::D2,
                        sample_type: TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                // overlay sampler
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
                // height view (optional)
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        multisampled: false,
                        view_dimension: TextureViewDimension::D2,
                        sample_type: TextureSampleType::Float { filterable: false },
                    },
                    count: None,
                },
                // height sampler (non-filtering)
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });

        // Create default 1x1 white overlay and height views to bind initially
        let overlay_sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("overlay_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let height_sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("overlay_height_sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Dummy 1x1 RGBA texture for overlay
        let dummy_tex = device.create_texture(&TextureDescriptor {
            label: Some("overlay_dummy"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8UnormSrgb,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let dummy_view = dummy_tex.create_view(&TextureViewDescriptor::default());

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("overlay_bg"),
            layout: &bind_group_layout,
            entries: &[
                BindGroupEntry { binding: 0, resource: uniform_buffer.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&dummy_view) },
                BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&overlay_sampler) },
                BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&dummy_view) },
                BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&height_sampler) },
            ],
        });

        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("overlays_shader"),
            source: ShaderSource::Wgsl(include_str!("../shaders/overlays.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("overlays_pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("overlays_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: VertexState { module: &module, entry_point: "vs_fullscreen", buffers: &[] },
            primitive: PrimitiveState { topology: PrimitiveTopology::TriangleList, ..Default::default() },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(FragmentState {
                module: &module,
                entry_point: "fs_overlay",
                targets: &[Some(ColorTargetState {
                    format: color_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: ColorWrites::ALL,
                })],
            }),
            multiview: None,
        });

        Self {
            uniforms,
            uniform_buffer,
            bind_group_layout,
            bind_group,
            pipeline,
            overlay_tex: None,
            overlay_view: None,
            overlay_sampler,
            height_sampler,
            overlay_format: TextureFormat::Rgba8UnormSrgb,
        }
    }

    pub fn set_enabled(&mut self, enabled: bool) { self.uniforms.overlay_params[0] = if enabled { 1.0 } else { 0.0 }; }
    pub fn set_overlay_alpha(&mut self, alpha: f32) { self.uniforms.overlay_params[1] = alpha.clamp(0.0, 1.0); }
    pub fn set_altitude_enabled(&mut self, enabled: bool) { self.uniforms.overlay_params[2] = if enabled { 1.0 } else { 0.0 }; }
    pub fn set_altitude_alpha(&mut self, alpha: f32) { self.uniforms.overlay_params[3] = alpha.clamp(0.0, 1.0); }
    pub fn set_overlay_uv(&mut self, off_x: f32, off_y: f32, scale_x: f32, scale_y: f32) {
        self.uniforms.overlay_uv = [off_x, off_y, scale_x, scale_y];
    }
    pub fn set_contours_enabled(&mut self, enabled: bool) {
        self.uniforms.contour_params[0] = if enabled { 1.0 } else { 0.0 };
    }
    pub fn set_contour_interval(&mut self, interval: f32) {
        self.uniforms.contour_params[1] = interval.max(1e-6);
    }
    pub fn set_contour_thickness_mul(&mut self, thickness: f32) {
        self.uniforms.contour_params[2] = thickness.max(0.1);
    }
    pub fn set_contour_color(&mut self, r: f32, g: f32, b: f32, a: f32) {
        self.uniforms.contour_color = [r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0), a.clamp(0.0, 1.0)];
    }

    pub fn upload_uniforms(&self, queue: &Queue) {
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&self.uniforms));
    }

    pub fn recreate_bind_group(
        &mut self,
        device: &Device,
        overlay_view: Option<&TextureView>,
        height_view: Option<&TextureView>,
    ) {
        // fallback to dummy 1x1 white created via a small temporary texture
        // Prefer provided view, then stored view, else dummy
        let use_overlay_view = overlay_view.or(self.overlay_view.as_ref());
        let (dummy_tex, dummy_view) = if use_overlay_view.is_none() || height_view.is_none() {
            let t = device.create_texture(&TextureDescriptor {
                label: Some("overlay_dummy_tmp"),
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

        let overlay_view = use_overlay_view.unwrap_or_else(|| dummy_view.as_ref().unwrap());
        let height_view = height_view.unwrap_or_else(|| dummy_view.as_ref().unwrap());

        self.bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("overlay_bg"),
            layout: &self.bind_group_layout,
            entries: &[
                BindGroupEntry { binding: 0, resource: self.uniform_buffer.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(overlay_view) },
                BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.overlay_sampler) },
                BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(height_view) },
                BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&self.height_sampler) },
            ],
        });

        // keep dummy_tex alive until function ends
        drop(dummy_tex);
    }

    /// Store overlay texture/view so GPU resources live as long as the renderer
    pub fn set_overlay_texture(&mut self, tex: Texture, view: TextureView) {
        self.overlay_tex = Some(tex);
        self.overlay_view = Some(view);
    }

    pub fn render<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        let overlay_en = self.uniforms.overlay_params[0] >= 0.5;
        let altitude_en = self.uniforms.overlay_params[2] >= 0.5;
        let contours_en = self.uniforms.contour_params[0] >= 0.5;
        if !overlay_en && !altitude_en && !contours_en { return; }
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}
