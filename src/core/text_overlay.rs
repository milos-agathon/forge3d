// src/core/text_overlay.rs
// Native text overlay pass for SDF/MSDF glyph quads.
// Renders screen-space quads (pixel coords) with alpha blending on top of the scene color target.

use wgpu::{
    vertex_attr_array, BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout,
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType, BufferAddress, BufferBindingType,
    BufferDescriptor, BufferUsages, ColorTargetState, ColorWrites, Device, FragmentState,
    PipelineLayoutDescriptor, PrimitiveState, PrimitiveTopology, Queue, RenderPipeline,
    RenderPipelineDescriptor, ShaderStages, TextureFormat, VertexBufferLayout, VertexState,
    VertexStepMode,
};

use crate::core::error::RenderResult;
use crate::core::resource_tracker::{tracked_create_buffer, TrackedBuffer};

const _: () = {
    assert!(std::mem::size_of::<TextOverlayUniforms>() == 32);
    assert!(std::mem::align_of::<TextOverlayUniforms>() == 8);
};

#[repr(C, align(8))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TextOverlayUniforms {
    pub resolution: [f32; 2], // (width, height)
    pub alpha: f32,
    pub enabled: f32,
    pub channels: f32,  // 1.0 for SDF, 3.0 for MSDF
    pub smoothing: f32, // smoothing factor (pixels)
    pub atlas_size: [u32; 2],
}

impl Default for TextOverlayUniforms {
    fn default() -> Self {
        Self {
            resolution: [1.0, 1.0],
            alpha: 1.0,
            enabled: 0.0,
            channels: 3.0,
            smoothing: 1.0,
            atlas_size: [1, 1],
        }
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

pub struct TextOverlayRenderer {
    pub uniforms: TextOverlayUniforms,
    pub uniform_buffer: TrackedBuffer,
    pub bind_group_layout: BindGroupLayout,
    pub bind_group: BindGroup,
    pub pipeline: RenderPipeline,

    pub quad_vbuf: TrackedBuffer,
    pub instance_buf: Option<TrackedBuffer>,
    pub instance_count: u32,

    pub atlas_buf: TrackedBuffer,
}

impl TextOverlayRenderer {
    pub fn new(device: &Device, color_format: TextureFormat) -> RenderResult<Self> {
        let uniforms = TextOverlayUniforms::default();
        let uniform_buffer = tracked_create_buffer(
            device,
            &BufferDescriptor {
                label: Some("text_overlay_uniforms"),
                size: std::mem::size_of::<TextOverlayUniforms>() as u64,
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
                // Packed RGBA8 atlas pixels. A buffer avoids the Metal Scene
                // texture-upload path; the shader performs normalized bilinear sampling.
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let atlas_buf = tracked_create_buffer(
            device,
            &BufferDescriptor {
                label: Some("text_dummy_atlas"),
                size: std::mem::size_of::<u32>() as u64,
                usage: BufferUsages::STORAGE,
                mapped_at_creation: true,
            },
        )?;
        atlas_buf
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(&0u32.to_ne_bytes());
        atlas_buf.unmap();

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
                    resource: atlas_buf.as_entire_binding(),
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

        let pipeline = crate::core::shader_registry::create_render_pipeline_scoped(
            device,
            &RenderPipelineDescriptor {
                label: Some("text_overlay_pipeline"),
                layout: Some(&pipeline_layout),
                vertex: VertexState {
                    module: &shader,
                    entry_point: "vs_main",
                    buffers: &[quad_layout, inst_layout],
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
            quad_vbuf,
            instance_buf: None,
            instance_count: 0,
            atlas_buf,
        })
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.uniforms.enabled = if enabled { 1.0 } else { 0.0 };
    }
    pub fn set_alpha(&mut self, alpha: f32) {
        self.uniforms.alpha = alpha.clamp(0.0, 1.0);
    }
    pub fn set_resolution(&mut self, w: u32, h: u32) {
        self.uniforms.resolution = [w as f32, h as f32];
    }
    pub fn set_channels(&mut self, channels: u32) {
        self.uniforms.channels = if channels >= 3 { 3.0 } else { 1.0 };
    }
    pub fn set_smoothing(&mut self, px: f32) {
        self.uniforms.smoothing = px.max(0.1);
    }
    pub fn set_atlas_size(&mut self, width: u32, height: u32) {
        self.uniforms.atlas_size = [width, height];
    }

    pub fn upload_uniforms(&self, queue: &Queue) {
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&self.uniforms));
    }

    pub fn upload_instances(
        &mut self,
        device: &Device,
        _queue: &Queue,
        instances: &[TextInstance],
    ) -> RenderResult<()> {
        self.instance_count = instances.len() as u32;
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

    pub fn recreate_bind_group(&mut self, device: &Device, atlas_buf: Option<&TrackedBuffer>) {
        let atlas_buf = atlas_buf.unwrap_or(&self.atlas_buf);
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
                    resource: atlas_buf.as_entire_binding(),
                },
            ],
        });
    }

    pub fn set_atlas(&mut self, atlas_buf: TrackedBuffer, width: u32, height: u32) {
        self.atlas_buf = atlas_buf;
        self.set_atlas_size(width, height);
    }

    pub fn render<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        if self.uniforms.enabled < 0.5 || self.instance_count == 0 {
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
mod tests {
    use super::TextOverlayUniforms;
    use naga::{ScalarKind, TypeInner, VectorSize};

    #[test]
    fn shader_storage_atlas_contract_parses_and_matches_host_uniform_layout() {
        assert_eq!(std::mem::size_of::<TextOverlayUniforms>(), 32);
        assert_eq!(std::mem::align_of::<TextOverlayUniforms>(), 8);
        assert_eq!(std::mem::offset_of!(TextOverlayUniforms, resolution), 0);
        assert_eq!(std::mem::offset_of!(TextOverlayUniforms, alpha), 8);
        assert_eq!(std::mem::offset_of!(TextOverlayUniforms, enabled), 12);
        assert_eq!(std::mem::offset_of!(TextOverlayUniforms, channels), 16);
        assert_eq!(std::mem::offset_of!(TextOverlayUniforms, smoothing), 20);
        assert_eq!(std::mem::offset_of!(TextOverlayUniforms, atlas_size), 24);
        let source = include_str!("../shaders/text_overlay.wgsl");
        let module = naga::front::wgsl::parse_str(source).expect("valid text overlay WGSL");
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("valid text overlay module");

        let uniforms = module
            .types
            .iter()
            .find_map(|(_, ty)| (ty.name.as_deref() == Some("TextOverlayUniforms")).then_some(ty))
            .expect("text overlay uniforms");
        let TypeInner::Struct { members, span } = &uniforms.inner else {
            panic!("TextOverlayUniforms must remain a WGSL struct");
        };
        assert_eq!(*span, std::mem::size_of::<TextOverlayUniforms>() as u32);
        let expected = [
            ("resolution", 0, ScalarKind::Float, true),
            ("alpha", 8, ScalarKind::Float, false),
            ("enabled", 12, ScalarKind::Float, false),
            ("channels", 16, ScalarKind::Float, false),
            ("smoothing", 20, ScalarKind::Float, false),
            ("atlas_size", 24, ScalarKind::Uint, true),
        ];
        assert_eq!(members.len(), expected.len());
        for (member, (name, offset, kind, vector)) in members.iter().zip(expected) {
            assert_eq!(member.name.as_deref(), Some(name));
            assert_eq!(member.offset, offset);
            match (&module.types[member.ty].inner, vector) {
                (TypeInner::Scalar(scalar), false) => {
                    assert_eq!(scalar.kind, kind);
                    assert_eq!(scalar.width, 4);
                }
                (TypeInner::Vector { size, scalar }, true) => {
                    assert_eq!(*size, VectorSize::Bi);
                    assert_eq!(scalar.kind, kind);
                    assert_eq!(scalar.width, 4);
                }
                (actual, _) => panic!("unexpected WGSL type for {name}: {actual:?}"),
            }
        }
    }
}
