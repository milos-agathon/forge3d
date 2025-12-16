//! Anti-aliased line rendering with GPU-based instanced segment expansion.
//!
//! Supports configurable line caps (butt, round, square) and joins (miter, bevel, round)
//! with smooth anti-aliasing via shader-based quad expansion.

use crate::core::error::RenderError;
use crate::vector::api::PolylineDef;
use crate::vector::layer::Layer;
use wgpu::util::DeviceExt;

// Re-export types from line_types module
pub use super::line_types::{LineCap, LineInstance, LineJoin, LineUniform};

// Import pipeline creation helpers
use super::line_pipeline::{
    create_oit_pipeline, create_pick_bind_group, create_pick_bind_group_layout,
    create_pick_pipeline, create_pick_uniform_buffer, create_pipeline_layout,
    create_render_pipeline, create_uniform_bind_group, create_uniform_bind_group_layout,
};

// Re-export helper function
pub use super::line_helpers::calculate_line_joins;

/// Anti-aliased line renderer with GPU-based quad expansion.
///
/// Uses instanced rendering where each line segment expands to a screen-aligned quad
/// in the vertex shader, with anti-aliasing computed in the fragment shader.
pub struct LineRenderer {
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: Option<wgpu::Buffer>,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    vertex_capacity: usize,
    // Picking resources for object selection
    pick_pipeline: wgpu::RenderPipeline,
    pick_uniform_buffer: wgpu::Buffer,
    pick_bind_group: wgpu::BindGroup,
    // Weighted OIT pipeline (MRT) for transparency
    oit_pipeline: wgpu::RenderPipeline,
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

        // Create bind group layout and bind group
        let bind_group_layout = create_uniform_bind_group_layout(device);
        let bind_group = create_uniform_bind_group(device, &bind_group_layout, &uniform_buffer);

        // Create pipeline layout and render pipeline
        let pipeline_layout = create_pipeline_layout(device, &bind_group_layout);
        let render_pipeline =
            create_render_pipeline(device, &shader, &pipeline_layout, target_format);

        // H5: Picking resources
        let pick_bind_group_layout = create_pick_bind_group_layout(device);
        let pick_uniform_buffer = create_pick_uniform_buffer(device);
        let pick_bind_group = create_pick_bind_group(
            device,
            &pick_bind_group_layout,
            &uniform_buffer,
            &pick_uniform_buffer,
        );
        let pick_pipeline_layout = create_pipeline_layout(device, &pick_bind_group_layout);
        let pick_pipeline = create_pick_pipeline(device, &shader, &pick_pipeline_layout);

        // H4: OIT pipeline
        let oit_pipeline_layout = create_pipeline_layout(device, &bind_group_layout);
        let oit_pipeline = create_oit_pipeline(device, &shader, &oit_pipeline_layout);

        Ok(Self {
            render_pipeline,
            vertex_buffer: None,
            uniform_buffer,
            bind_group,
            vertex_capacity: 0,
            pick_pipeline,
            pick_uniform_buffer,
            pick_bind_group,
            oit_pipeline,
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

    /// H5: Render picking IDs to an R32Uint attachment
    pub fn render_pick<'pass>(
        &'pass self,
        render_pass: &mut wgpu::RenderPass<'pass>,
        queue: &wgpu::Queue,
        transform: &[[f32; 4]; 4],
        viewport_size: [f32; 2],
        instance_count: u32,
        base_pick_id: u32,
    ) -> Result<(), RenderError> {
        if let Some(vertex_buffer) = &self.vertex_buffer {
            // Update uniforms
            let uniform = LineUniform {
                transform: *transform,
                stroke_color: [1.0, 1.0, 1.0, 1.0],
                stroke_width: 1.0,
                _pad0: 0.0,
                viewport_size,
                miter_limit: 4.0,
                cap_style: LineCap::Butt as u32,
                join_style: LineJoin::Miter as u32,
                _pad1: [0.0; 5],
            };
            queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniform]));

            // Write pick uniform
            let pick_data: [u32; 4] = [base_pick_id, 0, 0, 0];
            queue.write_buffer(&self.pick_uniform_buffer, 0, bytemuck::bytes_of(&pick_data));

            render_pass.set_pipeline(&self.pick_pipeline);
            render_pass.set_bind_group(0, &self.pick_bind_group, &[]);
            render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
            render_pass.draw(0..4, 0..instance_count);
        }
        Ok(())
    }

    /// H4: Render using weighted OIT MRT. Render pass must be created with color attachments
    /// matching Rgba16Float and R16Float targets.
    pub fn render_oit<'pass>(
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
            let uniform = LineUniform {
                transform: *transform,
                stroke_color: [1.0, 1.0, 1.0, 1.0],
                stroke_width: 1.0,
                _pad0: 0.0,
                viewport_size,
                miter_limit,
                cap_style: cap_style as u32,
                join_style: join_style as u32,
                _pad1: [0.0; 5],
            };
            queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniform]));

            render_pass.set_pipeline(&self.oit_pipeline);
            render_pass.set_bind_group(0, &self.bind_group, &[]);
            render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
            render_pass.draw(0..4, 0..instance_count);
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
                _pad0: 0.0,
                viewport_size,
                miter_limit,
                cap_style: cap_style as u32,
                join_style: join_style as u32,
                _pad1: [0.0; 5],
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
