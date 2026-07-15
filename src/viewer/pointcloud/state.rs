use crate::core::error::RenderResult;
use crate::core::resource_tracker::{tracked_create_buffer, TrackedBuffer};
use crate::viewer::pointcloud::load::load_laz_points;
use crate::viewer::pointcloud::shader::POINTCLOUD_SHADER;
use crate::viewer::pointcloud::{ColorMode, PointCloudUniforms, PointInstance3D, PointSource3D};
use glam::DVec3;

/// Point cloud state for the viewer.
pub struct PointCloudState {
    pub source_points: Vec<PointSource3D>,
    pub points: Vec<PointInstance3D>,
    pub instance_buffer: Option<TrackedBuffer>,
    pub uniform_buffer: TrackedBuffer,
    pub bind_group: wgpu::BindGroup,
    pub pipeline: wgpu::RenderPipeline,
    pub point_count: usize,
    pub point_size: f32,
    pub visible: bool,
    pub color_mode: ColorMode,
    pub bounds_min: DVec3,
    pub bounds_max: DVec3,
    pub center: DVec3,
    pub extent_render: f32,
    pub has_rgb: bool,
    pub has_intensity: bool,
    pub cam_phi: f32,
    pub cam_theta: f32,
    pub cam_radius: f32,
}

impl PointCloudState {
    pub fn new(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
        _depth_format: wgpu::TextureFormat,
    ) -> RenderResult<Self> {
        let shader = crate::core::shader_registry::create_labeled_shader_module(
            device,
            "pointcloud.wgsl",
            POINTCLOUD_SHADER,
        );

        let uniform_buffer = tracked_create_buffer(
            device,
            &wgpu::BufferDescriptor {
                label: Some("PointCloud.Uniforms"),
                size: std::mem::size_of::<PointCloudUniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            },
        )?;

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PointCloud.BindGroupLayout"),
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

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PointCloud.BindGroup"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PointCloud.PipelineLayout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = crate::core::shader_registry::create_render_pipeline_scoped(
            device,
            &wgpu::RenderPipelineDescriptor {
                label: Some("PointCloud.Pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: "vs_main",
                    buffers: &[wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<PointInstance3D>() as u64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &[
                            wgpu::VertexAttribute {
                                offset: 0,
                                shader_location: 0,
                                format: wgpu::VertexFormat::Float32x3,
                            },
                            wgpu::VertexAttribute {
                                offset: 12,
                                shader_location: 1,
                                format: wgpu::VertexFormat::Float32,
                            },
                            wgpu::VertexAttribute {
                                offset: 16,
                                shader_location: 2,
                                format: wgpu::VertexFormat::Float32x3,
                            },
                            wgpu::VertexAttribute {
                                offset: 28,
                                shader_location: 3,
                                format: wgpu::VertexFormat::Float32,
                            },
                            wgpu::VertexAttribute {
                                offset: 32,
                                shader_location: 4,
                                format: wgpu::VertexFormat::Float32,
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
            },
        );

        Ok(Self {
            source_points: Vec::new(),
            points: Vec::new(),
            instance_buffer: None,
            uniform_buffer,
            bind_group,
            pipeline,
            point_count: 0,
            point_size: 2.0,
            visible: true,
            color_mode: ColorMode::Elevation,
            bounds_min: DVec3::ZERO,
            bounds_max: DVec3::ZERO,
            center: DVec3::ZERO,
            extent_render: 100.0,
            has_rgb: false,
            has_intensity: false,
            cam_phi: 0.7,
            cam_theta: 0.5,
            cam_radius: 1.0,
        })
    }

    pub fn handle_mouse_drag(&mut self, dx: f32, dy: f32) {
        let sensitivity = 0.005;
        self.cam_phi += dx * sensitivity;
        self.cam_theta = (self.cam_theta - dy * sensitivity).clamp(0.1, 1.5);
    }

    pub fn handle_scroll(&mut self, delta: f32) {
        let zoom_speed = 0.1;
        self.cam_radius *= 1.0 - delta * zoom_speed;
        self.cam_radius = self.cam_radius.clamp(0.1, 100.0);
    }

    pub fn handle_keys(&mut self, forward: f32, right: f32, up: f32) {
        let rotate_speed = 0.02;
        let zoom_speed = 0.02;

        self.cam_phi += right * rotate_speed;
        self.cam_theta = (self.cam_theta + forward * rotate_speed).clamp(0.1, 1.5);
        self.cam_radius *= 1.0 - up * zoom_speed;
        self.cam_radius = self.cam_radius.clamp(0.1, 100.0);
    }

    pub fn load_from_file(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        anchor: &crate::camera::Anchor,
        rebase_to_center: bool,
        path: &str,
        max_points: u64,
        color_mode: ColorMode,
    ) -> Result<(), String> {
        let load_result = load_laz_points(path, max_points as usize)?;
        let points = load_result.points;

        println!(
            "[pointcloud] Data flags - has_rgb: {}, has_intensity: {}",
            load_result.has_rgb, load_result.has_intensity
        );

        if points.is_empty() {
            return Err("No points loaded".to_string());
        }

        let mut min = DVec3::splat(f64::INFINITY);
        let mut max = DVec3::splat(f64::NEG_INFINITY);
        for p in &points {
            if !p.position.is_finite() {
                return Err("point cloud contains a non-finite world position".to_string());
            }
            min = min.min(p.position);
            max = max.max(p.position);
        }

        let center = (min + max) * 0.5;
        let mut candidate_anchor = *anchor;
        if rebase_to_center {
            candidate_anchor.rebase_if_needed(center);
        }
        for point in [min, max] {
            crate::viewer::camera_controller::validate_world_point(
                crate::viewer::camera_controller::CoordRole::Content,
                point,
                &candidate_anchor,
            )
            .map_err(|err| err.to_string())?;
        }
        let extent_render = crate::camera::Anchor::new()
            .to_render_direction(max - min)
            .max_element()
            .max(100.0);

        eprintln!(
            "[pointcloud] Original center: ({:.1}, {:.1}, {:.1})",
            center.x, center.y, center.z
        );
        eprintln!(
            "[pointcloud] Extent: ({:.1}, {:.1}, {:.1})",
            max.x - min.x,
            max.y - min.y,
            max.z - min.z
        );

        let render_points = points
            .iter()
            .map(|point| PointInstance3D {
                position: candidate_anchor.to_render_vec3(point.position).to_array(),
                elevation_norm: point.elevation_norm,
                rgb: point.rgb,
                intensity: point.intensity,
                size: point.size,
                _pad: [0.0; 3],
            })
            .collect::<Vec<_>>();
        let buffer = tracked_create_buffer(
            device,
            &wgpu::BufferDescriptor {
                label: Some("PointCloud.InstanceBuffer"),
                size: (render_points.len() * std::mem::size_of::<PointInstance3D>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            },
        )
        .map_err(|err| err.to_string())?;
        queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&render_points));

        // Commit only after validation, packing, allocation, and upload all succeed.
        self.has_rgb = load_result.has_rgb;
        self.has_intensity = load_result.has_intensity;
        self.bounds_min = min;
        self.bounds_max = max;
        self.center = center;
        self.extent_render = extent_render;
        self.point_count = points.len();
        self.color_mode = color_mode;
        self.source_points = points;
        self.points = render_points;
        self.instance_buffer = Some(buffer);

        Ok(())
    }

    fn pack_for_anchor(&mut self, anchor: &crate::camera::Anchor) {
        self.points = self
            .source_points
            .iter()
            .map(|point| PointInstance3D {
                position: anchor.to_render_vec3(point.position).to_array(),
                elevation_norm: point.elevation_norm,
                rgb: point.rgb,
                intensity: point.intensity,
                size: point.size,
                _pad: [0.0; 3],
            })
            .collect();
    }

    /// Repack into the existing COPY_DST instance buffer. Routine rebases do
    /// not allocate or replace GPU resources.
    pub fn repack_for_anchor(
        &mut self,
        queue: &wgpu::Queue,
        anchor: &crate::camera::Anchor,
    ) -> Result<(), String> {
        if self.source_points.is_empty() {
            return Ok(());
        }
        self.pack_for_anchor(anchor);
        let buffer = self
            .instance_buffer
            .as_ref()
            .ok_or_else(|| "point-cloud instance buffer is missing".to_string())?;
        queue.write_buffer(buffer, 0, bytemuck::cast_slice(&self.points));
        Ok(())
    }

    pub fn camera_pose_world(&self) -> (DVec3, DVec3, f32, f32) {
        let extent = self.extent_render.max(100.0);
        let radius = extent * 2.0 * self.cam_radius;
        let offset = DVec3::new(
            f64::from(radius * self.cam_theta.cos() * self.cam_phi.cos()),
            f64::from(radius * self.cam_theta.sin()),
            f64::from(radius * self.cam_theta.cos() * self.cam_phi.sin()),
        );
        (
            self.center + offset,
            self.center,
            (extent * 0.01).max(0.01),
            extent * 10.0,
        )
    }

    pub fn render<'pass>(
        &'pass self,
        render_pass: &mut wgpu::RenderPass<'pass>,
        queue: &wgpu::Queue,
        view_proj: [[f32; 4]; 4],
        viewport_size: [f32; 2],
    ) {
        if !self.visible || self.point_count == 0 {
            return;
        }

        let Some(instance_buffer) = &self.instance_buffer else {
            return;
        };

        let uniforms = PointCloudUniforms {
            view_proj,
            viewport_size,
            point_size: self.point_size,
            color_mode: self.color_mode as u32,
            has_rgb: if self.has_rgb { 1 } else { 0 },
            has_intensity: if self.has_intensity { 1 } else { 0 },
            _pad: [0, 0],
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.set_vertex_buffer(0, instance_buffer.slice(..));
        render_pass.draw(0..4, 0..self.point_count as u32);
    }

    pub fn set_point_size(&mut self, size: f32) {
        self.point_size = size.max(0.5).min(50.0);
    }

    pub fn set_visible(&mut self, visible: bool) {
        self.visible = visible;
    }

    pub fn clear(&mut self) {
        self.source_points.clear();
        self.points.clear();
        self.instance_buffer = None;
        self.point_count = 0;
    }
}
