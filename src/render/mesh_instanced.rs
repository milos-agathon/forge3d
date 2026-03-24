// src/render/mesh_instanced.rs
// GPU instanced mesh renderer (feature-gated by enable-gpu-instancing)

#![allow(dead_code)]

use bytemuck::{Pod, Zeroable};
use glam::Mat4;
use std::cell::Cell;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindingType, Buffer, BufferDescriptor, BufferUsages, ColorTargetState, ColorWrites, Device,
    FragmentState, IndexFormat, PipelineLayoutDescriptor, PrimitiveState, PrimitiveTopology, Queue,
    RenderPass, RenderPipeline, RenderPipelineDescriptor, ShaderModuleDescriptor, ShaderSource,
    ShaderStages, TextureFormat, VertexAttribute, VertexBufferLayout, VertexFormat, VertexState,
    VertexStepMode,
};

const DRAW_BATCH_UNIFORM_SLOT_COUNT: usize = 512;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ScatterBatchUniforms {
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
    color: [f32; 4],
    light_dir_ws: [f32; 4], // xyz + intensity
    wind_phase: [f32; 4],
    wind_vec_bounds: [f32; 4],
    wind_bend_fade: [f32; 4],
}

impl Default for ScatterBatchUniforms {
    fn default() -> Self {
        Self {
            view: Mat4::IDENTITY.to_cols_array_2d(),
            proj: Mat4::IDENTITY.to_cols_array_2d(),
            color: [1.0, 1.0, 1.0, 1.0],
            light_dir_ws: [0.0, -1.0, 0.0, 1.0],
            wind_phase: [0.0; 4],
            wind_vec_bounds: [0.0; 4],
            wind_bend_fade: [0.0; 4],
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
    uniforms: ScatterBatchUniforms,
    uniforms_buf: Buffer,
    bind_group_layout: BindGroupLayout,
    bind_group: BindGroup,
    per_draw_uniforms: Vec<Buffer>,
    per_draw_bind_groups: Vec<BindGroup>,
    per_draw_cursor: Cell<usize>,
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
        Self::new_with_sample_count(device, color_format, depth_format, 1)
    }

    pub fn new_with_sample_count(
        device: &Device,
        color_format: TextureFormat,
        depth_format: Option<TextureFormat>,
        sample_count: u32,
    ) -> Self {
        Self::new_with_depth_state(
            device,
            color_format,
            depth_format,
            sample_count,
            wgpu::CompareFunction::Less,
            true,
        )
    }

    pub fn new_with_depth_state(
        device: &Device,
        color_format: TextureFormat,
        depth_format: Option<TextureFormat>,
        sample_count: u32,
        depth_compare: wgpu::CompareFunction,
        depth_write_enabled: bool,
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
                depth_write_enabled,
                depth_compare,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: sample_count.max(1),
                ..Default::default()
            },
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

        let uniforms = ScatterBatchUniforms::default();
        let uniforms_buf = device.create_buffer(&BufferDescriptor {
            label: Some("mesh_instanced_uniforms"),
            size: std::mem::size_of::<ScatterBatchUniforms>() as u64,
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

        let mut per_draw_uniforms = Vec::with_capacity(DRAW_BATCH_UNIFORM_SLOT_COUNT);
        let mut per_draw_bind_groups = Vec::with_capacity(DRAW_BATCH_UNIFORM_SLOT_COUNT);
        for _ in 0..DRAW_BATCH_UNIFORM_SLOT_COUNT {
            let uniforms_buf = device.create_buffer(&BufferDescriptor {
                label: Some("mesh_instanced_draw_uniforms"),
                size: std::mem::size_of::<ScatterBatchUniforms>() as u64,
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let bind_group = device.create_bind_group(&BindGroupDescriptor {
                label: Some("mesh_instanced_draw_bg"),
                layout: &bind_group_layout,
                entries: &[BindGroupEntry {
                    binding: 0,
                    resource: uniforms_buf.as_entire_binding(),
                }],
            });
            per_draw_uniforms.push(uniforms_buf);
            per_draw_bind_groups.push(bind_group);
        }

        Self {
            pipeline,
            uniforms,
            uniforms_buf,
            bind_group_layout,
            bind_group,
            per_draw_uniforms,
            per_draw_bind_groups,
            per_draw_cursor: Cell::new(0),
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

    pub fn reset_draw_batch_uniforms(&self) {
        self.per_draw_cursor.set(0);
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

    fn next_draw_batch_uniform_slot(&self) -> Option<usize> {
        let slot = self.per_draw_cursor.get();
        if slot >= self.per_draw_uniforms.len() {
            return None;
        }
        self.per_draw_cursor.set(slot + 1);
        Some(slot)
    }

    /// Draw a batch using explicit uniform parameters (no mutation of renderer state).
    ///
    /// Silently skips the draw if the per-draw uniform slot pool is exhausted.
    pub fn draw_batch_params<'rp>(
        &'rp self,
        _device: &Device,
        pass: &mut RenderPass<'rp>,
        queue: &Queue,
        view: Mat4,
        proj: Mat4,
        color: [f32; 4],
        light_dir: [f32; 3],
        light_intensity: f32,
        wind_phase: [f32; 4],
        wind_vec_bounds: [f32; 4],
        wind_bend_fade: [f32; 4],
        vbuf: &'rp Buffer,
        ibuf: &'rp Buffer,
        instbuf: &'rp Buffer,
        index_count: u32,
        instance_count: u32,
    ) {
        if index_count == 0 || instance_count == 0 {
            return;
        }
        let Some(slot) = self.next_draw_batch_uniform_slot() else {
            static WARN_ONCE: std::sync::Once = std::sync::Once::new();
            WARN_ONCE.call_once(|| {
                eprintln!(
                    "[mesh_instanced] per-draw uniform slot pool exhausted ({} slots); \
                     excess draws will be skipped this frame",
                    DRAW_BATCH_UNIFORM_SLOT_COUNT
                );
            });
            return;
        };
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
        u.wind_phase = wind_phase;
        u.wind_vec_bounds = wind_vec_bounds;
        u.wind_bend_fade = wind_bend_fade;
        queue.write_buffer(&self.per_draw_uniforms[slot], 0, bytemuck::bytes_of(&u));

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.per_draw_bind_groups[slot], &[]);
        pass.set_vertex_buffer(0, vbuf.slice(..));
        pass.set_vertex_buffer(1, instbuf.slice(..));
        pass.set_index_buffer(ibuf.slice(..), IndexFormat::Uint32);
        pass.draw_indexed(0..index_count, 0, 0..instance_count);
    }
}

#[cfg(all(test, feature = "enable-gpu-instancing"))]
mod tests {
    use super::*;

    fn render_non_black_pixels<F>(device: &Device, queue: &Queue, draw: F) -> Option<usize>
    where
        F: FnOnce(&mut wgpu::CommandEncoder, &wgpu::TextureView, &wgpu::TextureView),
    {
        let width = 96u32;
        let height = 96u32;
        let color = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("mesh_instanced.test.color"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let color_view = color.create_view(&wgpu::TextureViewDescriptor::default());
        let depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("mesh_instanced.test.depth"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());
        let row_bytes = width * 4;
        let padded_bpr = crate::core::gpu::align_copy_bpr(row_bytes);
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mesh_instanced.test.readback"),
            size: (padded_bpr * height) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("mesh_instanced.test.encoder"),
        });
        draw(&mut encoder, &color_view, &depth_view);
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &color,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &readback,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bpr),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);

        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv().ok()?.ok()?;

        let data = slice.get_mapped_range();
        let row_bytes = row_bytes as usize;
        let padded_bpr = padded_bpr as usize;
        let mut non_black = 0usize;
        for y in 0..height as usize {
            let row = &data[(y * padded_bpr)..(y * padded_bpr + row_bytes)];
            non_black += row
                .chunks_exact(4)
                .filter(|px| px[0] != 0 || px[1] != 0 || px[2] != 0)
                .count();
        }
        drop(data);
        readback.unmap();
        Some(non_black)
    }

    #[test]
    fn draw_batch_params_renders_pixels() {
        let Some((device, queue)) = crate::core::gpu::create_device_and_queue_for_test() else {
            return;
        };

        let vertices = [
            VertexPN {
                position: [-0.5, -0.5, 0.0],
                normal: [0.0, 0.0, 1.0],
            },
            VertexPN {
                position: [0.5, -0.5, 0.0],
                normal: [0.0, 0.0, 1.0],
            },
            VertexPN {
                position: [0.0, 0.5, 0.0],
                normal: [0.0, 0.0, 1.0],
            },
        ];
        let indices = [0u32, 1, 2];
        let instance = [[
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ]];
        let view = Mat4::look_at_rh(
            glam::Vec3::new(0.0, 0.0, 2.0),
            glam::Vec3::ZERO,
            glam::Vec3::Y,
        );
        let proj = crate::camera::perspective_wgpu(45.0f32.to_radians(), 1.0, 0.1, 10.0);
        let color = [0.9, 0.3, 0.2, 1.0];
        let light_dir = [0.0, 0.0, -1.0];
        let light_intensity = 1.0;

        let mut shared_renderer = MeshInstancedRenderer::new(
            &device,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            Some(wgpu::TextureFormat::Depth32Float),
        );
        shared_renderer.set_mesh(&device, &queue, &vertices, &indices);
        shared_renderer.upload_instances_from_rowmajor(&device, &queue, &instance);
        shared_renderer.set_view_proj(view, proj);
        shared_renderer.set_color(color);
        shared_renderer.set_light(light_dir, light_intensity);

        let shared_pixels =
            render_non_black_pixels(&device, &queue, |encoder, color_view, depth_view| {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("mesh_instanced.test.shared_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: color_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                shared_renderer.render(&mut pass, &queue, 1);
            })
            .expect("shared render readback should succeed");

        let mut per_draw_renderer = MeshInstancedRenderer::new(
            &device,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            Some(wgpu::TextureFormat::Depth32Float),
        );
        per_draw_renderer.set_mesh(&device, &queue, &vertices, &indices);
        per_draw_renderer.upload_instances_from_rowmajor(&device, &queue, &instance);
        per_draw_renderer.reset_draw_batch_uniforms();

        let per_draw_pixels =
            render_non_black_pixels(&device, &queue, |encoder, color_view, depth_view| {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("mesh_instanced.test.per_draw_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: color_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                per_draw_renderer.draw_batch_params(
                    &device,
                    &mut pass,
                    &queue,
                    view,
                    proj,
                    color,
                    light_dir,
                    light_intensity,
                    [0.0; 4],
                    [0.0; 4],
                    [0.0; 4],
                    per_draw_renderer.vbuf.as_ref().unwrap(),
                    per_draw_renderer.ibuf.as_ref().unwrap(),
                    per_draw_renderer.instbuf.as_ref().unwrap(),
                    indices.len() as u32,
                    1,
                );
            })
            .expect("per-draw render readback should succeed");

        assert!(
            shared_pixels > 0,
            "shared instanced render should draw visible pixels"
        );
        assert!(
            per_draw_pixels > 0,
            "draw_batch_params should draw visible pixels"
        );
    }
}
