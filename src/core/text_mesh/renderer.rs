use super::types::{MeshUniforms, VertexPN};
use glam::Mat4;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindingType, Buffer, BufferDescriptor, BufferUsages, ColorTargetState, ColorWrites, Device,
    FragmentState, IndexFormat, PipelineLayoutDescriptor, PrimitiveState, PrimitiveTopology, Queue,
    RenderPass, RenderPipeline, RenderPipelineDescriptor, ShaderModuleDescriptor, ShaderSource,
    ShaderStages, TextureFormat, VertexAttribute, VertexBufferLayout, VertexFormat, VertexState,
    VertexStepMode,
};

pub struct TextMeshRenderer {
    pipeline: RenderPipeline,
    pub uniforms: MeshUniforms,
    uniforms_buf: Buffer,
    pub bind_group_layout: BindGroupLayout,
    pub bind_group: BindGroup,
    vbuf: Option<Buffer>,
    ibuf: Option<Buffer>,
    index_count: u32,
}

impl TextMeshRenderer {
    pub fn new(
        device: &Device,
        color_format: TextureFormat,
        depth_format: Option<TextureFormat>,
    ) -> Self {
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("mesh_basic_shader"),
            source: ShaderSource::Wgsl(include_str!("../../shaders/mesh_basic.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("text_mesh_bgl"),
            entries: &[
                // uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("text_mesh_pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let vertex_layout = VertexBufferLayout {
            array_stride: std::mem::size_of::<VertexPN>() as u64,
            step_mode: VertexStepMode::Vertex,
            attributes: &[
                VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: VertexFormat::Float32x3,
                },
                VertexAttribute {
                    offset: 12,
                    shader_location: 1,
                    format: VertexFormat::Float32x3,
                },
            ],
        };

        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("text_mesh_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[vertex_layout],
            },
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: depth_format.map(|df| wgpu::DepthStencilState {
                format: df,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
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
        });

        let uniforms = MeshUniforms::default();
        let uniforms_buf = device.create_buffer(&BufferDescriptor {
            label: Some("text_mesh_uniforms"),
            size: std::mem::size_of::<MeshUniforms>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("text_mesh_bg"),
            layout: &bind_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: uniforms_buf.as_entire_binding(),
            }],
        });

        Self {
            pipeline,
            uniforms,
            uniforms_buf,
            bind_group_layout,
            bind_group,
            vbuf: None,
            ibuf: None,
            index_count: 0,
        }
    }

    pub fn set_mesh(
        &mut self,
        device: &Device,
        queue: &Queue,
        vertices: &[VertexPN],
        indices: &[u32],
    ) {
        let vsize = (vertices.len() * std::mem::size_of::<VertexPN>()) as u64;
        let isize = (indices.len() * std::mem::size_of::<u32>()) as u64;
        let vbuf = device.create_buffer(&BufferDescriptor {
            label: Some("text_mesh_vbuf"),
            size: vsize,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let ibuf = device.create_buffer(&BufferDescriptor {
            label: Some("text_mesh_ibuf"),
            size: isize,
            usage: BufferUsages::INDEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&vbuf, 0, bytemuck::cast_slice(vertices));
        queue.write_buffer(&ibuf, 0, bytemuck::cast_slice(indices));
        self.vbuf = Some(vbuf);
        self.ibuf = Some(ibuf);
        self.index_count = indices.len() as u32;
    }

    pub fn set_model(&mut self, model: Mat4) {
        self.uniforms.model = model.to_cols_array_2d();
    }
    pub fn set_view_proj(&mut self, view: Mat4, proj: Mat4) {
        self.uniforms.view = view.to_cols_array_2d();
        self.uniforms.proj = proj.to_cols_array_2d();
    }
    pub fn set_color(&mut self, r: f32, g: f32, b: f32, a: f32) {
        self.uniforms.color = [r, g, b, a];
    }
    pub fn set_light(&mut self, dir: [f32; 3], intensity: f32) {
        self.uniforms.light_dir_ws = [dir[0], dir[1], dir[2], intensity.max(0.0)];
    }
    pub fn set_light_dir(&mut self, dir: [f32; 3]) {
        self.set_light(dir, 1.0);
    }
    pub fn set_material(&mut self, metallic: f32, roughness: f32) {
        self.uniforms.mr = [metallic.clamp(0.0, 1.0), roughness.clamp(0.04, 1.0)];
    }
    pub fn upload_uniforms(&self, queue: &Queue) {
        queue.write_buffer(&self.uniforms_buf, 0, bytemuck::bytes_of(&self.uniforms));
    }

    pub fn render<'a>(&'a self, pass: &mut RenderPass<'a>) {
        if self.index_count == 0 {
            return;
        }
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        if let Some(ref v) = self.vbuf {
            pass.set_vertex_buffer(0, v.slice(..));
        }
        if let Some(ref i) = self.ibuf {
            pass.set_index_buffer(i.slice(..), IndexFormat::Uint32);
        }
        pass.draw_indexed(0..self.index_count, 0, 0..1);
    }

    pub fn draw_instance_with_light<'rp>(
        &'rp self,
        pass: &mut RenderPass<'rp>,
        queue: &Queue,
        model: Mat4,
        color: [f32; 4],
        light_dir: [f32; 3],
        light_intensity: f32,
        metallic: f32,
        roughness: f32,
        vbuf: &'rp Buffer,
        ibuf: &'rp Buffer,
        index_count: u32,
    ) {
        // Stage uniforms without mutating self.uniforms to avoid &mut self borrows
        let mut u = self.uniforms;
        u.model = model.to_cols_array_2d();
        u.color = color;
        u.light_dir_ws = [
            light_dir[0],
            light_dir[1],
            light_dir[2],
            light_intensity.max(0.0),
        ];
        u.mr = [metallic.clamp(0.0, 1.0), roughness.clamp(0.04, 1.0)];
        queue.write_buffer(&self.uniforms_buf, 0, bytemuck::bytes_of(&u));
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_vertex_buffer(0, vbuf.slice(..));
        pass.set_index_buffer(ibuf.slice(..), IndexFormat::Uint32);
        pass.draw_indexed(0..index_count, 0, 0..1);
    }
}
