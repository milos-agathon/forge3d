//! H8,H7,H9: Anti-aliased line rendering with instanced segment expansion
//! GPU-based line segment expansion for smooth anti-aliased lines

use crate::error::RenderError;
use crate::vector::api::PolylineDef;
use crate::vector::layer::Layer;
use bytemuck::{Pod, Zeroable};
use glam::Vec2;
use wgpu::util::DeviceExt;

/// H9: Line cap styles
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineCap {
    /// Square cap ending exactly at vertex
    Butt = 0,
    /// Rounded cap extending beyond vertex
    Round = 1,
    /// Square cap extending beyond vertex
    Square = 2,
}

/// H9: Line join styles
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineJoin {
    /// Sharp miter join with limit
    Miter = 0,
    /// Beveled join (flat cut)
    Bevel = 1,
    /// Rounded join
    Round = 2,
}

/// Anti-aliased line renderer with GPU expansion
pub struct LineRenderer {
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: Option<wgpu::Buffer>,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    vertex_capacity: usize,
}

/// Line rendering uniforms with H9 caps/joins support
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct LineUniform {
    transform: [[f32; 4]; 4], // View-projection matrix
    stroke_color: [f32; 4],   // RGBA stroke color
    stroke_width: f32,        // Line width in world units
    viewport_size: [f32; 2],  // Viewport dimensions for AA
    miter_limit: f32,         // H9: Miter limit for joins
    cap_style: u32,           // H9: LineCap as u32
    join_style: u32,          // H9: LineJoin as u32
    _pad: [f32; 2],           // Alignment padding
}

/// Line segment instance data for GPU expansion
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct LineInstance {
    pub start_pos: [f32; 2], // Segment start position
    pub end_pos: [f32; 2],   // Segment end position
    pub width: f32,          // Line width in world units
    pub color: [f32; 4],     // RGBA color
    pub miter_limit: f32,    // Miter limit for joins
    pub _pad: [f32; 2],      // Alignment padding
}

impl LineRenderer {
    pub fn new(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
    ) -> Result<Self, RenderError> {
        // Load shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("line_aa.wgsl"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "../shaders/line_aa.wgsl"
            ))),
        });

        // Create uniform buffer
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vf.Vector.Line.Uniform"),
            size: std::mem::size_of::<LineUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vf.Vector.Line.BindGroupLayout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vf.Vector.Line.BindGroup"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("vf.Vector.Line.PipelineLayout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create render pipeline with instanced rendering
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("vf.Vector.Line.Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[
                    // Per-instance line segment data
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<LineInstance>() as u64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &[
                            // start_pos
                            wgpu::VertexAttribute {
                                offset: 0,
                                shader_location: 0,
                                format: wgpu::VertexFormat::Float32x2,
                            },
                            // end_pos
                            wgpu::VertexAttribute {
                                offset: 8,
                                shader_location: 1,
                                format: wgpu::VertexFormat::Float32x2,
                            },
                            // width
                            wgpu::VertexAttribute {
                                offset: 16,
                                shader_location: 2,
                                format: wgpu::VertexFormat::Float32,
                            },
                            // color
                            wgpu::VertexAttribute {
                                offset: 20,
                                shader_location: 3,
                                format: wgpu::VertexFormat::Float32x4,
                            },
                            // miter_limit (+ padding)
                            wgpu::VertexAttribute {
                                offset: 36,
                                shader_location: 4,
                                format: wgpu::VertexFormat::Float32,
                            },
                        ],
                    },
                ],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
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

        Ok(Self {
            render_pipeline,
            vertex_buffer: None,
            uniform_buffer,
            bind_group,
            vertex_capacity: 0,
        })
    }

    /// Convert polylines to line instances for GPU expansion
    pub fn pack_polylines(
        &self,
        polylines: &[PolylineDef],
    ) -> Result<Vec<LineInstance>, RenderError> {
        let mut instances = Vec::new();

        for polyline in polylines {
            // Validate path has at least 2 points
            if polyline.path.len() < 2 {
                return Err(RenderError::Upload(format!(
                    "Polyline must have at least 2 points, got {}",
                    polyline.path.len()
                )));
            }

            // Create line instances for each segment
            for i in 0..polyline.path.len() - 1 {
                let start = polyline.path[i];
                let end = polyline.path[i + 1];

                // Skip degenerate segments (duplicate consecutive points)
                let segment_length = (end - start).length();
                if segment_length < 1e-6 {
                    continue;
                }

                instances.push(LineInstance {
                    start_pos: [start.x, start.y],
                    end_pos: [end.x, end.y],
                    width: polyline.style.stroke_width,
                    color: polyline.style.stroke_color,
                    miter_limit: 4.0, // Standard miter limit
                    _pad: [0.0; 2],
                });
            }
        }

        Ok(instances)
    }

    /// Upload line instances to GPU buffer
    pub fn upload_lines(
        &mut self,
        device: &wgpu::Device,
        instances: &[LineInstance],
    ) -> Result<(), RenderError> {
        if instances.is_empty() {
            return Ok(());
        }

        // Reallocate buffer if needed
        if instances.len() > self.vertex_capacity {
            let new_capacity = (instances.len() * 2).max(1024);
            self.vertex_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vf.Vector.Line.InstanceBuffer"),
                size: (new_capacity * std::mem::size_of::<LineInstance>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.vertex_capacity = new_capacity;
        }

        // Upload instance data
        if let Some(vertex_buffer) = &self.vertex_buffer {
            let instance_data = bytemuck::cast_slice(instances);

            // Use staging buffer for upload
            let staging_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("vf.Vector.Line.StagingBuffer"),
                contents: instance_data,
                usage: wgpu::BufferUsages::COPY_SRC,
            });

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("vf.Vector.Line.Upload"),
            });

            encoder.copy_buffer_to_buffer(
                &staging_buffer,
                0,
                vertex_buffer,
                0,
                instance_data.len() as u64,
            );

            // Note: In production, this command buffer should be submitted
            // by the calling renderer, not here
        }

        Ok(())
    }

    /// Render anti-aliased lines with H9 caps/joins support
    pub fn render<'pass>(
        &'pass self,
        render_pass: &mut wgpu::RenderPass<'pass>,
        queue: &wgpu::Queue,
        transform: &[[f32; 4]; 4],
        viewport_size: [f32; 2],
        instance_count: u32,
        cap_style: LineCap,
        join_style: LineJoin,
        miter_limit: f32,
    ) -> Result<(), RenderError> {
        if let Some(vertex_buffer) = &self.vertex_buffer {
            // Update uniforms
            let uniform = LineUniform {
                transform: *transform,
                stroke_color: [1.0, 1.0, 1.0, 1.0], // Default white, overridden per-instance
                stroke_width: 1.0,
                viewport_size,
                miter_limit,
                cap_style: cap_style as u32,
                join_style: join_style as u32,
                _pad: [0.0; 2],
            };

            queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniform]));

            // Set pipeline and resources
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.bind_group, &[]);
            render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));

            // Draw instanced - each instance generates a quad (4 vertices as triangle strip)
            render_pass.draw(0..4, 0..instance_count);
        }

        Ok(())
    }

    /// Get layer for line rendering
    pub fn layer() -> Layer {
        Layer::Vector
    }
}

/// Calculate line join points for smooth connections
pub fn calculate_line_joins(path: &[Vec2], width: f32) -> Vec<Vec2> {
    if path.len() < 2 {
        return Vec::new();
    }

    let mut joins = Vec::with_capacity(path.len());
    let half_width = width * 0.5;

    for i in 0..path.len() {
        if i == 0 || i == path.len() - 1 {
            // Endpoints don't need join calculation
            joins.push(path[i]);
            continue;
        }

        let prev = path[i - 1];
        let curr = path[i];
        let next = path[i + 1];

        // Calculate join normal
        let seg1 = (curr - prev).normalize_or_zero();
        let seg2 = (next - curr).normalize_or_zero();
        let join_normal = (seg1 + seg2).normalize_or_zero();

        // Calculate miter offset
        let miter_dot = seg1.dot(join_normal);
        let miter_length = if miter_dot.abs() > 0.01 {
            half_width / miter_dot
        } else {
            half_width
        };

        // Apply miter limit
        let limited_length = miter_length.min(half_width * 4.0);
        joins.push(curr + join_normal.perp() * limited_length);
    }

    joins
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::api::VectorStyle;

    #[test]
    fn test_pack_simple_polyline() {
        let device = crate::gpu::create_device_for_test();
        let renderer = LineRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb).unwrap();

        let polyline = PolylineDef {
            path: vec![
                Vec2::new(0.0, 0.0),
                Vec2::new(1.0, 1.0),
                Vec2::new(2.0, 0.0),
            ],
            style: VectorStyle {
                stroke_width: 2.0,
                stroke_color: [1.0, 0.0, 0.0, 1.0],
                ..Default::default()
            },
        };

        let instances = renderer.pack_polylines(&[polyline]).unwrap();

        assert_eq!(instances.len(), 2); // 3 points = 2 segments
        assert_eq!(instances[0].width, 2.0);
        assert_eq!(instances[0].color, [1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_skip_degenerate_segments() {
        let device = crate::gpu::create_device_for_test();
        let renderer = LineRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb).unwrap();

        let polyline = PolylineDef {
            path: vec![
                Vec2::new(0.0, 0.0),
                Vec2::new(0.0, 0.0), // Duplicate point
                Vec2::new(1.0, 1.0),
            ],
            style: VectorStyle::default(),
        };

        let instances = renderer.pack_polylines(&[polyline]).unwrap();

        // Should skip the degenerate segment
        assert_eq!(instances.len(), 1);
    }

    #[test]
    fn test_line_joins() {
        let path = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(1.0, 1.0),
        ];

        let joins = calculate_line_joins(&path, 1.0);
        assert_eq!(joins.len(), 3);

        // First and last points should remain unchanged
        assert_eq!(joins[0], path[0]);
        assert_eq!(joins[2], path[2]);

        // Middle point should be offset for smooth join
        assert_ne!(joins[1], path[1]);
    }

    #[test]
    fn test_reject_short_polyline() {
        let device = crate::gpu::create_device_for_test();
        let renderer = LineRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb).unwrap();

        let short_line = PolylineDef {
            path: vec![Vec2::new(0.0, 0.0)], // Only 1 point
            style: VectorStyle::default(),
        };

        let result = renderer.pack_polylines(&[short_line]);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("at least 2 points"));
    }
}
