//! H5: Polygon fill pipeline with winding order and hole support
//! GPU tessellation with proper sRGB target rendering

use crate::error::RenderError;
use crate::vector::api::PolygonDef;
use crate::vector::data::{validate_polygon_vertices, PackedPolygon, PolygonVertex};
use crate::vector::layer::Layer;
use bytemuck::{Pod, Zeroable};

/// Polygon renderer with GPU tessellation
pub struct PolygonRenderer {
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: Option<wgpu::Buffer>,
    index_buffer: Option<wgpu::Buffer>,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    vertex_capacity: usize,
    index_capacity: usize,
}

/// Polygon uniform data (16-byte aligned)
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct PolygonUniform {
    transform: [[f32; 4]; 4], // View-projection matrix
    fill_color: [f32; 4],     // RGBA fill color
    stroke_color: [f32; 4],   // RGBA stroke color
    stroke_width: f32,        // Stroke width in pixels
    _pad: [f32; 3],           // Alignment padding
}

impl PolygonRenderer {
    pub fn new(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
    ) -> Result<Self, RenderError> {
        // Load and compile shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("polygon_fill.wgsl"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "../shaders/polygon_fill.wgsl"
            ))),
        });

        // Create uniform buffer
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vf.Vector.Polygon.Uniform"),
            size: std::mem::size_of::<PolygonUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vf.Vector.Polygon.BindGroupLayout"),
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
            label: Some("vf.Vector.Polygon.BindGroup"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("vf.Vector.Polygon.PipelineLayout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create render pipeline
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("vf.Vector.Polygon.Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<PolygonVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        // Position
                        wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: wgpu::VertexFormat::Float32x2,
                        },
                        // UV
                        wgpu::VertexAttribute {
                            offset: 8,
                            shader_location: 1,
                            format: wgpu::VertexFormat::Float32x2,
                        },
                    ],
                }],
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
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, // Support both orientations for holes
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
            index_buffer: None,
            uniform_buffer,
            bind_group,
            vertex_capacity: 0,
            index_capacity: 0,
        })
    }

    /// Tessellate polygon with proper winding order and hole support
    pub fn tessellate_polygon(&self, polygon: &PolygonDef) -> Result<PackedPolygon, RenderError> {
        use lyon_path::Path;
        use lyon_tessellation::{FillOptions, FillTessellator, FillVertex, VertexBuffers};

        let mut tessellator = FillTessellator::new();
        let mut buffers = VertexBuffers::new();

        // Build Lyon path with exterior and holes
        let mut path_builder = Path::builder();

        // Add exterior ring (should be CCW for filled areas)
        if polygon.exterior.len() < 3 {
            return Err(RenderError::Upload(
                "Polygon exterior must have at least 3 vertices".to_string(),
            ));
        }

        let first_point = lyon_path::math::Point::new(polygon.exterior[0].x, polygon.exterior[0].y);
        path_builder.begin(first_point);

        for vertex in polygon.exterior.iter().skip(1) {
            path_builder.line_to(lyon_path::math::Point::new(vertex.x, vertex.y));
        }
        path_builder.close();

        // Add holes (should be CW)
        for hole in &polygon.holes {
            if hole.len() < 3 {
                return Err(RenderError::Upload(
                    "Polygon hole must have at least 3 vertices".to_string(),
                ));
            }

            let first_hole_point = lyon_path::math::Point::new(hole[0].x, hole[0].y);
            path_builder.begin(first_hole_point);

            for vertex in hole.iter().skip(1) {
                path_builder.line_to(lyon_path::math::Point::new(vertex.x, vertex.y));
            }
            path_builder.close();
        }

        let path = path_builder.build();

        // Tessellate with appropriate fill rule
        let result = tessellator.tessellate_path(
            &path,
            &FillOptions::default(),
            &mut lyon_tessellation::BuffersBuilder::new(&mut buffers, |vertex: FillVertex| {
                PolygonVertex {
                    position: [vertex.position().x, vertex.position().y],
                    uv: [0.0, 0.0], // UV will be computed in shader based on bbox
                }
            }),
        );

        if let Err(e) = result {
            return Err(RenderError::Upload(format!(
                "Polygon tessellation failed: {:?}",
                e
            )));
        }

        // Validate tessellation result
        let validation_result = validate_polygon_vertices(&buffers.vertices, &buffers.indices);
        if !validation_result.is_valid {
            return Err(RenderError::Upload(
                validation_result
                    .error_message
                    .unwrap_or_else(|| "Polygon tessellation validation failed".to_string()),
            ));
        }

        Ok(PackedPolygon {
            vertices: buffers.vertices,
            indices: buffers.indices,
            hole_offsets: vec![], // Lyon handles holes internally
        })
    }

    /// Upload polygon data to GPU buffers
    pub fn upload_polygons(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        polygons: &[PackedPolygon],
    ) -> Result<(), RenderError> {
        if polygons.is_empty() {
            return Ok(());
        }

        // Calculate total buffer sizes needed
        let total_vertices: usize = polygons.iter().map(|p| p.vertices.len()).sum();
        let total_indices: usize = polygons.iter().map(|p| p.indices.len()).sum();

        // Reallocate vertex buffer if needed
        if total_vertices > self.vertex_capacity {
            let new_capacity = (total_vertices * 2).max(1024);
            self.vertex_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vf.Vector.Polygon.VertexBuffer"),
                size: (new_capacity * std::mem::size_of::<PolygonVertex>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.vertex_capacity = new_capacity;
        }

        // Reallocate index buffer if needed
        if total_indices > self.index_capacity {
            let new_capacity = (total_indices * 2).max(3072); // At least 1024 triangles
            self.index_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vf.Vector.Polygon.IndexBuffer"),
                size: (new_capacity * std::mem::size_of::<u32>()) as u64,
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.index_capacity = new_capacity;
        }

        // Build combined vertex/index data
        let mut combined_vertices = Vec::with_capacity(total_vertices);
        let mut combined_indices = Vec::with_capacity(total_indices);

        for polygon in polygons {
            let vertex_offset = combined_vertices.len() as u32;
            combined_vertices.extend_from_slice(&polygon.vertices);

            // Adjust indices by current vertex offset
            for &index in &polygon.indices {
                combined_indices.push(vertex_offset + index);
            }
        }

        // Upload vertices to GPU
        if let Some(vertex_buffer) = &self.vertex_buffer {
            if !combined_vertices.is_empty() {
                let vertex_data = bytemuck::cast_slice(&combined_vertices);
                queue.write_buffer(vertex_buffer, 0, vertex_data);
            }
        }

        if let Some(index_buffer) = &self.index_buffer {
            if !combined_indices.is_empty() {
                let index_data = bytemuck::cast_slice(&combined_indices);
                queue.write_buffer(index_buffer, 0, index_data);
            }
        }

        Ok(())
    }

    /// Render polygons with the given transform and style
    pub fn render<'pass>(
        &'pass self,
        render_pass: &mut wgpu::RenderPass<'pass>,
        queue: &wgpu::Queue,
        transform: &[[f32; 4]; 4],
        fill_color: [f32; 4],
        stroke_color: [f32; 4],
        stroke_width: f32,
        index_count: u32,
    ) -> Result<(), RenderError> {
        if let (Some(vertex_buffer), Some(index_buffer)) = (&self.vertex_buffer, &self.index_buffer)
        {
            // Update uniforms
            let uniform = PolygonUniform {
                transform: *transform,
                fill_color,
                stroke_color,
                stroke_width,
                _pad: [0.0; 3],
            };

            queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniform]));

            // Set pipeline and resources
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.bind_group, &[]);
            render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
            render_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);

            // Draw indexed
            render_pass.draw_indexed(0..index_count, 0, 0..1);
        }

        Ok(())
    }

    /// Get layer for polygon rendering (filled polygons render in background)
    pub fn layer() -> Layer {
        Layer::Background
    }
}

/// H6: Polygon outline generation for line pipeline integration
pub struct PolygonOutlines {
    pub exterior_lines: Vec<crate::vector::api::PolylineDef>,
    pub hole_lines: Vec<crate::vector::api::PolylineDef>,
}

impl PolygonOutlines {
    /// Generate outlines from polygon definition
    pub fn from_polygon(
        polygon: &PolygonDef,
        outline_width: f32,
        outline_color: [f32; 4],
    ) -> Result<Self, RenderError> {
        use crate::vector::api::{PolylineDef, VectorStyle};

        let mut exterior_lines = Vec::new();
        let mut hole_lines = Vec::new();

        // Create outline style
        let outline_style = VectorStyle {
            fill_color: [0.0, 0.0, 0.0, 0.0], // Not used for lines
            stroke_color: outline_color,
            stroke_width: outline_width,
            point_size: 4.0, // Not used for lines
        };

        // Generate exterior outline
        if polygon.exterior.len() >= 3 {
            let mut exterior_path = polygon.exterior.clone();
            // Close the path by adding first point at end
            exterior_path.push(polygon.exterior[0]);

            exterior_lines.push(PolylineDef {
                path: exterior_path,
                style: outline_style.clone(),
            });
        }

        // Generate hole outlines
        for hole in &polygon.holes {
            if hole.len() >= 3 {
                let mut hole_path = hole.clone();
                // Close the hole path
                hole_path.push(hole[0]);

                hole_lines.push(PolylineDef {
                    path: hole_path,
                    style: outline_style.clone(),
                });
            }
        }

        Ok(Self {
            exterior_lines,
            hole_lines,
        })
    }

    /// Get all outline polylines as a single vector
    pub fn all_lines(&self) -> Vec<&crate::vector::api::PolylineDef> {
        let mut lines = Vec::new();
        lines.extend(self.exterior_lines.iter());
        lines.extend(self.hole_lines.iter());
        lines
    }

    /// Get total number of outline segments
    pub fn segment_count(&self) -> usize {
        self.exterior_lines
            .iter()
            .map(|line| line.path.len().saturating_sub(1))
            .sum::<usize>()
            + self
                .hole_lines
                .iter()
                .map(|line| line.path.len().saturating_sub(1))
                .sum::<usize>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::api::VectorStyle;
    use glam::Vec2;

    #[test]
    fn test_tessellate_simple_triangle() {
        let device = crate::gpu::create_device_for_test();
        let renderer = PolygonRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb).unwrap();

        let triangle = PolygonDef {
            exterior: vec![
                Vec2::new(0.0, 0.0),
                Vec2::new(1.0, 0.0),
                Vec2::new(0.5, 1.0),
            ],
            holes: vec![],
            style: VectorStyle::default(),
        };

        let packed = renderer.tessellate_polygon(&triangle).unwrap();

        assert!(packed.vertices.len() >= 3);
        assert!(packed.indices.len() >= 3);
        assert_eq!(packed.indices.len() % 3, 0); // Must be triangles
    }

    #[test]
    fn test_tessellate_polygon_with_hole() {
        let device = crate::gpu::create_device_for_test();
        let renderer = PolygonRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb).unwrap();

        let polygon_with_hole = PolygonDef {
            exterior: vec![
                Vec2::new(-2.0, -2.0),
                Vec2::new(2.0, -2.0),
                Vec2::new(2.0, 2.0),
                Vec2::new(-2.0, 2.0),
            ],
            holes: vec![vec![
                Vec2::new(-0.5, -0.5),
                Vec2::new(0.5, -0.5),
                Vec2::new(0.5, 0.5),
                Vec2::new(-0.5, 0.5),
            ]],
            style: VectorStyle::default(),
        };

        let packed = renderer.tessellate_polygon(&polygon_with_hole).unwrap();

        // Should have more vertices than a simple quad due to hole tessellation
        assert!(packed.vertices.len() > 4);
        assert!(packed.indices.len() > 6); // More than 2 triangles
        assert_eq!(packed.indices.len() % 3, 0);
    }

    #[test]
    fn test_reject_degenerate_polygon() {
        let device = crate::gpu::create_device_for_test();
        let renderer = PolygonRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb).unwrap();

        let degenerate = PolygonDef {
            exterior: vec![Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0)], // Only 2 vertices
            holes: vec![],
            style: VectorStyle::default(),
        };

        let result = renderer.tessellate_polygon(&degenerate);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("at least 3 vertices"));
    }
}
