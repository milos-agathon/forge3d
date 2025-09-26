// src/core/text_mesh.rs
// 3D Text mesh renderer and mesh builder (extruded outlines)

use bytemuck::{Pod, Zeroable};
use glam::Mat4;
use lyon_path::Path;
use lyon_tessellation::{FillOptions, FillTessellator, BuffersBuilder, VertexBuffers, FillVertex, FillVertexConstructor};
use ttf_parser::{Face, OutlineBuilder};
use wgpu::{BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor, BindingType, Buffer, BufferDescriptor, BufferUsages, ColorTargetState, ColorWrites, Device, FragmentState, PipelineLayoutDescriptor, PrimitiveState, PrimitiveTopology, Queue, RenderPipeline, RenderPipelineDescriptor, ShaderModuleDescriptor, ShaderSource, ShaderStages, TextureFormat, VertexAttribute, VertexBufferLayout, VertexFormat, VertexState, VertexStepMode, RenderPass, IndexFormat};
use lyon_path::math;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct MeshUniforms {
    pub model: [[f32;4];4],
    pub view: [[f32;4];4],
    pub proj: [[f32;4];4],
    pub color: [f32;4],
    pub light_dir_ws: [f32;4],
    pub mr: [f32; 2], // metallic, roughness
    pub _pad_mr: [f32; 2],
}

impl Default for MeshUniforms {
    fn default() -> Self {
        Self {
            model: Mat4::IDENTITY.to_cols_array_2d(),
            view: Mat4::IDENTITY.to_cols_array_2d(),
            proj: Mat4::IDENTITY.to_cols_array_2d(),
            color: [1.0,1.0,1.0,1.0],
            light_dir_ws: [0.0, -1.0, 0.0, 0.0],
            mr: [0.0, 1.0],
            _pad_mr: [0.0, 0.0],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct VertexPN {
    pub position: [f32;3],
    pub normal: [f32;3],
}

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
    pub fn new(device: &Device, color_format: TextureFormat, depth_format: Option<TextureFormat>) -> Self {
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("mesh_basic_shader"),
            source: ShaderSource::Wgsl(include_str!("../shaders/mesh_basic.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("text_mesh_bgl"),
            entries: &[
                // uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
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
                VertexAttribute { offset: 0, shader_location: 0, format: VertexFormat::Float32x3 },
                VertexAttribute { offset: 12, shader_location: 1, format: VertexFormat::Float32x3 },
            ],
        };

        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("text_mesh_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: VertexState { module: &shader, entry_point: "vs_main", buffers: &[vertex_layout] },
            primitive: PrimitiveState { topology: PrimitiveTopology::TriangleList, ..Default::default() },
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
                targets: &[Some(ColorTargetState { format: color_format, blend: Some(wgpu::BlendState::ALPHA_BLENDING), write_mask: ColorWrites::ALL })],
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
            entries: &[BindGroupEntry { binding: 0, resource: uniforms_buf.as_entire_binding() }],
        });

        Self { pipeline, uniforms, uniforms_buf, bind_group_layout, bind_group, vbuf: None, ibuf: None, index_count: 0 }
    }

    pub fn set_mesh(&mut self, device: &Device, queue: &Queue, vertices: &[VertexPN], indices: &[u32]) {
        let vsize = (vertices.len() * std::mem::size_of::<VertexPN>()) as u64;
        let isize = (indices.len() * std::mem::size_of::<u32>()) as u64;
        let vbuf = device.create_buffer(&BufferDescriptor { label: Some("text_mesh_vbuf"), size: vsize, usage: BufferUsages::VERTEX | BufferUsages::COPY_DST, mapped_at_creation: false });
        let ibuf = device.create_buffer(&BufferDescriptor { label: Some("text_mesh_ibuf"), size: isize, usage: BufferUsages::INDEX | BufferUsages::COPY_DST, mapped_at_creation: false });
        queue.write_buffer(&vbuf, 0, bytemuck::cast_slice(vertices));
        queue.write_buffer(&ibuf, 0, bytemuck::cast_slice(indices));
        self.vbuf = Some(vbuf);
        self.ibuf = Some(ibuf);
        self.index_count = indices.len() as u32;
    }

    pub fn set_model(&mut self, model: Mat4) { self.uniforms.model = model.to_cols_array_2d(); }
    pub fn set_view_proj(&mut self, view: Mat4, proj: Mat4) {
        self.uniforms.view = view.to_cols_array_2d();
        self.uniforms.proj = proj.to_cols_array_2d();
    }
    pub fn set_color(&mut self, r:f32,g:f32,b:f32,a:f32) { self.uniforms.color = [r,g,b,a]; }
    pub fn set_light(&mut self, dir: [f32;3], intensity: f32) {
        self.uniforms.light_dir_ws = [dir[0], dir[1], dir[2], intensity.max(0.0)];
    }
    pub fn set_light_dir(&mut self, dir: [f32;3]) { self.set_light(dir, 1.0); }
    pub fn set_material(&mut self, metallic: f32, roughness: f32) {
        self.uniforms.mr = [metallic.clamp(0.0, 1.0), roughness.clamp(0.04, 1.0)];
    }
    pub fn upload_uniforms(&self, queue: &Queue) { queue.write_buffer(&self.uniforms_buf, 0, bytemuck::bytes_of(&self.uniforms)); }

    pub fn render<'a>(&'a self, pass: &mut RenderPass<'a>) {
        if self.index_count == 0 { return; }
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        if let Some(ref v) = self.vbuf { pass.set_vertex_buffer(0, v.slice(..)); }
        if let Some(ref i) = self.ibuf { pass.set_index_buffer(i.slice(..), IndexFormat::Uint32); }
        pass.draw_indexed(0..self.index_count, 0, 0..1);
    }

    pub fn draw_instance_with_light<'rp>(&'rp self, pass: &mut RenderPass<'rp>, queue: &Queue, model: Mat4, color: [f32;4], light_dir: [f32;3], light_intensity: f32, metallic: f32, roughness: f32, vbuf: &'rp Buffer, ibuf: &'rp Buffer, index_count: u32) {
        // Stage uniforms without mutating self.uniforms to avoid &mut self borrows
        let mut u = self.uniforms;
        u.model = model.to_cols_array_2d();
        u.color = color;
        u.light_dir_ws = [light_dir[0], light_dir[1], light_dir[2], light_intensity.max(0.0)];
        u.mr = [metallic.clamp(0.0, 1.0), roughness.clamp(0.04, 1.0)];
        queue.write_buffer(&self.uniforms_buf, 0, bytemuck::bytes_of(&u));
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_vertex_buffer(0, vbuf.slice(..));
        pass.set_index_buffer(ibuf.slice(..), IndexFormat::Uint32);
        pass.draw_indexed(0..index_count, 0, 0..1);
    }
}

// --------------------
// Mesh builder
// --------------------
struct PathSink<'a> {
    builder: &'a mut lyon_path::path::Builder,
    scale: f32,
    offset: glam::Vec2,
}
impl OutlineBuilder for PathSink<'_> {
    fn move_to(&mut self, x: f32, y: f32) {
        self.builder.begin(math::point(self.offset.x + x * self.scale, self.offset.y - y * self.scale));
    }
    fn line_to(&mut self, x: f32, y: f32) {
        self.builder.line_to(math::point(self.offset.x + x * self.scale, self.offset.y - y * self.scale));
    }
    fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
        self.builder.quadratic_bezier_to(
            math::point(self.offset.x + x1 * self.scale, self.offset.y - y1 * self.scale),
            math::point(self.offset.x + x * self.scale, self.offset.y - y * self.scale),
        );
    }
    fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
        self.builder.cubic_bezier_to(
            math::point(self.offset.x + x1 * self.scale, self.offset.y - y1 * self.scale),
            math::point(self.offset.x + x2 * self.scale, self.offset.y - y2 * self.scale),
            math::point(self.offset.x + x * self.scale, self.offset.y - y * self.scale),
        );
    }
    fn close(&mut self) {
        self.builder.end(true);
    }
}

#[derive(Clone, Copy)]
struct FrontVertexCtor;
impl FillVertexConstructor<VertexPN> for FrontVertexCtor {
    fn new_vertex(&mut self, v: FillVertex) -> VertexPN {
        let p = v.position();
        VertexPN { position: [p.x, p.y, 0.0], normal: [0.0, 0.0, 1.0] }
    }
}

pub fn build_text_mesh(text: &str, font_bytes: &[u8], size_px: f32, depth: f32, bevel_width: f32, bevel_segments: u32) -> anyhow::Result<(Vec<VertexPN>, Vec<u32>)> {
    let face = Face::parse(font_bytes, 0).map_err(|_| anyhow::anyhow!("Invalid font data"))?;
    let units = face.units_per_em() as f32;
    let scale = (size_px / units).max(1e-6);

    // Build combined path for all glyphs
    let mut path_builder = Path::builder();
    let mut x_cursor = 0.0f32;

    for ch in text.chars() {
        if ch == '\n' { // simple newline support
            // Move down by ascent (approx) and reset x
            x_cursor = 0.0;
            // skip vertical offset in this simple pass
            continue;
        }
        if let Some(gid) = face.glyph_index(ch) {
            // Outline glyph into builder with current offset
            let mut sink = PathSink { builder: &mut path_builder, scale, offset: glam::Vec2::new(x_cursor, 0.0) };
            // outline_glyph returns Option<Rect>; we ignore the bounds and just build when present
            let _ = face.outline_glyph(gid, &mut sink);
            // advance x_cursor
            let adv = face.glyph_hor_advance(gid).unwrap_or(0) as f32 * scale;
            x_cursor += adv;
        }
    }

    let path = path_builder.build();

    // Tessellate front face
    let mut buffers: VertexBuffers<VertexPN, u32> = VertexBuffers::new();
    let mut tess = FillTessellator::new();
    tess.tessellate_path(
        path.as_slice(),
        &FillOptions::tolerance(0.1),
        &mut BuffersBuilder::new(&mut buffers, FrontVertexCtor)
    ).map_err(|e| anyhow::anyhow!("Tessellation failed: {:?}", e))?;

    let mut vertices = buffers.vertices.clone();
    let mut indices = buffers.indices.clone();

    // Back face (reverse winding)
    let back_offset = vertices.len() as u32;
    for v in &buffers.vertices {
        vertices.push(VertexPN { position: [v.position[0], v.position[1], depth], normal: [0.0, 0.0, -1.0] });
    }
    let mut back_tris: Vec<u32> = Vec::with_capacity(indices.len());
    for tri in indices.chunks_exact(3) {
        back_tris.push(back_offset + tri[0]);
        back_tris.push(back_offset + tri[2]);
        back_tris.push(back_offset + tri[1]);
    }
    indices.extend(back_tris);

    // Side walls with geometric bevels: iterate flattened path edges
    let tol = 0.5f32; // px tolerance for flattening
    let mut _first_in_subpath = None;
    let mut _prev = None;
    for e in path.iter() {
        match e {
            lyon_path::Event::Begin { at } => {
                _first_in_subpath = Some(at);
                _prev = Some(at);
            }
            lyon_path::Event::Line { from, to } => {
                add_side_bevel_strips(&mut vertices, &mut indices, from, to, depth, bevel_width, bevel_segments.max(1));
                _prev = Some(to);
            }
            lyon_path::Event::Quadratic { from, ctrl, to } => {
                let seg = lyon_geom::QuadraticBezierSegment { from, ctrl, to };
                seg.for_each_flattened(tol, &mut |ls: &lyon_geom::LineSegment<f32>| {
                    add_side_bevel_strips(&mut vertices, &mut indices, ls.from, ls.to, depth, bevel_width, bevel_segments.max(1));
                });
                _prev = Some(to);
            }
            lyon_path::Event::Cubic { from, ctrl1, ctrl2, to } => {
                let seg = lyon_geom::CubicBezierSegment { from, ctrl1, ctrl2, to };
                seg.for_each_flattened(tol, &mut |ls: &lyon_geom::LineSegment<f32>| {
                    add_side_bevel_strips(&mut vertices, &mut indices, ls.from, ls.to, depth, bevel_width, bevel_segments.max(1));
                });
                _prev = Some(to);
            }
            lyon_path::Event::End { last, first, close } => {
                if close {
                    add_side_bevel_strips(&mut vertices, &mut indices, last, first, depth, bevel_width, bevel_segments.max(1));
                }
                _first_in_subpath = None;
                _prev = None;
            }
        }
    }

    Ok((vertices, indices))
}

fn add_side_bevel_strips(vertices: &mut Vec<VertexPN>, indices: &mut Vec<u32>, from: math::Point, to: math::Point, depth: f32, bevel_width: f32, bevel_segments: u32) {
    let p0 = glam::Vec2::new(from.x, from.y);
    let p1 = glam::Vec2::new(to.x, to.y);
    let edge = p1 - p0;
    let len = edge.length();
    if len < 1e-6 { return; }
    let n = glam::Vec2::new(edge.y / len, -edge.x / len); // outward approx

    let bw = bevel_width.max(0.0);
    let segs = bevel_segments as usize;
    // Generate front bevel rings from z=0 to z=bw
    let mut rings: Vec<(f32, glam::Vec2)> = Vec::new();
    for k in 0..=segs {
        let t = k as f32 / (segs as f32);
        let z = bw * t;
        let off = n * (bw * t);
        rings.push((z, off));
    }
    // Add middle section (constant offset) if depth allows
    if depth > 2.0 * bw {
        rings.push((depth - bw, n * bw));
    }
    // Back bevel rings from z=depth-bw to z=depth
    for k in (0..=segs).rev() {
        let t = k as f32 / (segs as f32);
        let z = depth - bw * t;
        let off = n * (bw * t);
        rings.push((z, off));
    }

    // Build quad strips between successive rings
    for r in 0..(rings.len().saturating_sub(1)) {
        let (z0, off0) = rings[r];
        let (z1, off1) = rings[r+1];
        let v_start = vertices.len() as u32;
        let n3 = [n.x, n.y, 0.0];
        // p0, p1 at z0
        vertices.push(VertexPN { position: [p0.x + off0.x, p0.y + off0.y, z0], normal: n3 });
        vertices.push(VertexPN { position: [p1.x + off0.x, p1.y + off0.y, z0], normal: n3 });
        // p1, p0 at z1
        vertices.push(VertexPN { position: [p1.x + off1.x, p1.y + off1.y, z1], normal: n3 });
        vertices.push(VertexPN { position: [p0.x + off1.x, p0.y + off1.y, z1], normal: n3 });
        indices.extend_from_slice(&[v_start, v_start+1, v_start+2, v_start, v_start+2, v_start+3]);
    }
}
