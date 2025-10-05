// src/render/mesh_instanced.rs
// GPU instanced mesh renderer (feature-gated by enable-gpu-instancing)

#![allow(dead_code)]

use bytemuck::{Pod, Zeroable};
use glam::Mat4;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindingType, Buffer, BufferDescriptor, BufferUsages, ColorTargetState, ColorWrites, Device,
    FragmentState, IndexFormat, PipelineLayoutDescriptor, PrimitiveState, PrimitiveTopology, Queue,
    RenderPass, RenderPipeline, RenderPipelineDescriptor, ShaderModuleDescriptor, ShaderSource,
    ShaderStages, TextureFormat, VertexAttribute, VertexBufferLayout, VertexFormat, VertexState,
    VertexStepMode,
};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SceneUniforms {
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
    color: [f32; 4],
    light_dir_ws: [f32; 4], // xyz + intensity
}

impl Default for SceneUniforms {
    fn default() -> Self {
        Self {
            view: Mat4::IDENTITY.to_cols_array_2d(),
            proj: Mat4::IDENTITY.to_cols_array_2d(),
            color: [1.0, 1.0, 1.0, 1.0],
            light_dir_ws: [0.0, -1.0, 0.0, 1.0],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct VertexPN {
    pub position: [f32; 3],
    pub normal: [f32; 3],
}

pub struct MeshInstancedRenderer {
    pipeline: RenderPipeline,
    uniforms: SceneUniforms,
    uniforms_buf: Buffer,
    bind_group_layout: BindGroupLayout,
    bind_group: BindGroup,
    vbuf: Option<Buffer>,
    ibuf: Option<Buffer>,
    instbuf: Option<Buffer>,
    index_count: u32,
    instance_capacity: usize,
}

impl MeshInstancedRenderer {
    pub fn new(
        device: &Device,
        color_format: TextureFormat,
        depth_format: Option<TextureFormat>,
    ) -> Self {
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("mesh_instanced_shader"),
            source: ShaderSource::Wgsl(include_str!("../shaders/mesh_instanced.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("mesh_instanced_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("mesh_instanced_pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Vertex layout 0: per-vertex position/normal
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
        // Vertex layout 1: per-instance transform as 4x vec4 (column-major)
        let instance_layout = VertexBufferLayout {
            array_stride: 64, // 4 * vec4<f32>
            step_mode: VertexStepMode::Instance,
            attributes: &[
                VertexAttribute {
                    offset: 0,
                    shader_location: 2,
                    format: VertexFormat::Float32x4,
                },
                VertexAttribute {
                    offset: 16,
                    shader_location: 3,
                    format: VertexFormat::Float32x4,
                },
                VertexAttribute {
                    offset: 32,
                    shader_location: 4,
                    format: VertexFormat::Float32x4,
                },
                VertexAttribute {
                    offset: 48,
                    shader_location: 5,
                    format: VertexFormat::Float32x4,
                },
            ],
        };

        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("mesh_instanced_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[vertex_layout, instance_layout],
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

        let uniforms = SceneUniforms::default();
        let uniforms_buf = device.create_buffer(&BufferDescriptor {
            label: Some("mesh_instanced_uniforms"),
            size: std::mem::size_of::<SceneUniforms>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("mesh_instanced_bg"),
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
            instbuf: None,
            index_count: 0,
            instance_capacity: 0,
        }
    }

    pub fn set_view_proj(&mut self, view: Mat4, proj: Mat4) {
        self.uniforms.view = view.to_cols_array_2d();
        self.uniforms.proj = proj.to_cols_array_2d();
    }
    pub fn set_color(&mut self, color: [f32; 4]) {
        self.uniforms.color = color;
    }
    pub fn set_light(&mut self, dir: [f32; 3], intensity: f32) {
        self.uniforms.light_dir_ws = [dir[0], dir[1], dir[2], intensity.max(0.0)];
    }
    pub fn upload_uniforms(&self, queue: &Queue) {
        queue.write_buffer(&self.uniforms_buf, 0, bytemuck::bytes_of(&self.uniforms));
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
            label: Some("mesh_instanced_vbuf"),
            size: vsize,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let ibuf = device.create_buffer(&BufferDescriptor {
            label: Some("mesh_instanced_ibuf"),
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

    pub fn upload_instances_from_mat4(
        &mut self,
        device: &Device,
        queue: &Queue,
        transforms: &[Mat4],
    ) {
        if transforms.is_empty() {
            return;
        }
        let needed = transforms.len();
        if needed > self.instance_capacity {
            let new_cap = (needed * 2).max(128);
            self.instbuf = Some(device.create_buffer(&BufferDescriptor {
                label: Some("mesh_instanced_instance_buf"),
                size: (new_cap * 64) as u64, // 64 bytes per transform
                usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.instance_capacity = new_cap;
        }
        let mut packed: Vec<f32> = Vec::with_capacity(needed * 16);
        for m in transforms {
            // Pack as column-major (Mat4 is column-major to_cols_array)
            let cols = m.to_cols_array();
            packed.extend_from_slice(&cols);
        }
        if let Some(inst) = &self.instbuf {
            queue.write_buffer(inst, 0, bytemuck::cast_slice(&packed));
        }
    }

    pub fn upload_instances_from_rowmajor(
        &mut self,
        device: &Device,
        queue: &Queue,
        row_major_4x4: &[[f32; 16]],
    ) {
        if row_major_4x4.is_empty() {
            return;
        }
        let needed = row_major_4x4.len();
        if needed > self.instance_capacity {
            let new_cap = (needed * 2).max(128);
            self.instbuf = Some(device.create_buffer(&BufferDescriptor {
                label: Some("mesh_instanced_instance_buf"),
                size: (new_cap * 64) as u64,
                usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.instance_capacity = new_cap;
        }
        // Convert row-major to column-major packing
        let mut packed: Vec<f32> = Vec::with_capacity(needed * 16);
        for r in row_major_4x4 {
            // r is row-major; convert to column-major by transposing
            let m = Mat4::from_cols_array(&[
                r[0], r[4], r[8], r[12], r[1], r[5], r[9], r[13], r[2], r[6], r[10], r[14], r[3],
                r[7], r[11], r[15],
            ]);
            packed.extend_from_slice(&m.to_cols_array());
        }
        if let Some(inst) = &self.instbuf {
            queue.write_buffer(inst, 0, bytemuck::cast_slice(&packed));
        }
    }

    pub fn render<'rp>(&'rp self, pass: &mut RenderPass<'rp>, queue: &Queue, instance_count: u32) {
        if self.index_count == 0 {
            return;
        }
        let Some(vbuf) = &self.vbuf else {
            return;
        };
        let Some(ibuf) = &self.ibuf else {
            return;
        };
        let Some(inst) = &self.instbuf else {
            return;
        };
        // Update uniforms
        queue.write_buffer(&self.uniforms_buf, 0, bytemuck::bytes_of(&self.uniforms));

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_vertex_buffer(0, vbuf.slice(..));
        pass.set_vertex_buffer(1, inst.slice(..));
        pass.set_index_buffer(ibuf.slice(..), IndexFormat::Uint32);
        pass.draw_indexed(0..self.index_count, 0, 0..instance_count);
    }

    /// Draw a batch using explicit uniform parameters (no mutation of renderer state).
    pub fn draw_batch_params<'rp>(
        &'rp self,
        pass: &mut RenderPass<'rp>,
        queue: &Queue,
        view: Mat4,
        proj: Mat4,
        color: [f32; 4],
        light_dir: [f32; 3],
        light_intensity: f32,
        vbuf: &'rp Buffer,
        ibuf: &'rp Buffer,
        instbuf: &'rp Buffer,
        index_count: u32,
        instance_count: u32,
    ) {
        if index_count == 0 || instance_count == 0 {
            return;
        }
        // Stage uniforms without mutating self
        let mut u = self.uniforms;
        u.view = view.to_cols_array_2d();
        u.proj = proj.to_cols_array_2d();
        u.color = color;
        u.light_dir_ws = [
            light_dir[0],
            light_dir[1],
            light_dir[2],
            light_intensity.max(0.0),
        ];
        queue.write_buffer(&self.uniforms_buf, 0, bytemuck::bytes_of(&u));

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_vertex_buffer(0, vbuf.slice(..));
        pass.set_vertex_buffer(1, instbuf.slice(..));
        pass.set_index_buffer(ibuf.slice(..), IndexFormat::Uint32);
        pass.draw_indexed(0..index_count, 0, 0..instance_count);
    }
}
