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
    /// Number of instances surviving the most recent frame frustum cull.
    pub visible_point_count: usize,
    visible_source_indices: Vec<usize>,
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
            visible_point_count: 0,
            visible_source_indices: Vec::new(),
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

    fn validate_camera_state(
        &self,
        anchor: &crate::camera::Anchor,
    ) -> Result<(), crate::viewer::camera_controller::CameraFrameError> {
        let (eye, target, _, _) = self.camera_pose_world();
        crate::viewer::camera_controller::validate_camera_pose(anchor, self.center, eye, target)
            .map(|_| ())
    }

    pub fn handle_mouse_drag(
        &mut self,
        anchor: &crate::camera::Anchor,
        dx: f32,
        dy: f32,
    ) -> Result<(), crate::viewer::camera_controller::CameraFrameError> {
        let old = (self.cam_phi, self.cam_theta);
        let sensitivity = 0.005;
        self.cam_phi += dx * sensitivity;
        self.cam_theta = (self.cam_theta - dy * sensitivity).clamp(0.1, 1.5);
        if let Err(error) = self.validate_camera_state(anchor) {
            (self.cam_phi, self.cam_theta) = old;
            return Err(error);
        }
        Ok(())
    }

    pub fn handle_scroll(
        &mut self,
        anchor: &crate::camera::Anchor,
        delta: f32,
    ) -> Result<(), crate::viewer::camera_controller::CameraFrameError> {
        let old = self.cam_radius;
        let zoom_speed = 0.1;
        self.cam_radius *= 1.0 - delta * zoom_speed;
        self.cam_radius = self.cam_radius.clamp(0.1, 100.0);
        if let Err(error) = self.validate_camera_state(anchor) {
            self.cam_radius = old;
            return Err(error);
        }
        Ok(())
    }

    pub fn handle_keys(
        &mut self,
        anchor: &crate::camera::Anchor,
        forward: f32,
        right: f32,
        up: f32,
    ) -> Result<(), crate::viewer::camera_controller::CameraFrameError> {
        let old = (self.cam_phi, self.cam_theta, self.cam_radius);
        let rotate_speed = 0.02;
        let zoom_speed = 0.02;

        self.cam_phi += right * rotate_speed;
        self.cam_theta = (self.cam_theta + forward * rotate_speed).clamp(0.1, 1.5);
        self.cam_radius *= 1.0 - up * zoom_speed;
        self.cam_radius = self.cam_radius.clamp(0.1, 100.0);
        if let Err(error) = self.validate_camera_state(anchor) {
            (self.cam_phi, self.cam_theta, self.cam_radius) = old;
            return Err(error);
        }
        Ok(())
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
        let extent_render = crate::camera::Anchor::direction_to_render(max - min)
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
        self.visible_point_count = points.len();
        self.visible_source_indices = (0..points.len()).collect();
        self.color_mode = color_mode;
        self.source_points = points;
        self.points = render_points;
        self.instance_buffer = Some(buffer);

        Ok(())
    }

    fn packed_point(point: &PointSource3D, anchor: &crate::camera::Anchor) -> PointInstance3D {
        PointInstance3D {
            position: anchor.to_render_vec3(point.position).to_array(),
            elevation_norm: point.elevation_norm,
            rgb: point.rgb,
            intensity: point.intensity,
            size: point.size,
            _pad: [0.0; 3],
        }
    }

    fn pack_for_anchor(&mut self, anchor: &crate::camera::Anchor) {
        debug_assert!(self.points.capacity() >= self.source_points.len());
        debug_assert!(self.visible_source_indices.capacity() >= self.source_points.len());
        self.points.clear();
        self.visible_source_indices.clear();
        for (index, point) in self.source_points.iter().enumerate() {
            self.points.push(Self::packed_point(point, anchor));
            self.visible_source_indices.push(index);
        }
        self.visible_point_count = self.points.len();
    }

    /// Repack into the existing COPY_DST instance buffer. Routine rebases do
    /// not allocate or replace GPU resources.
    pub fn repack_for_anchor(&mut self, queue: &wgpu::Queue, anchor: &crate::camera::Anchor) {
        if self.source_points.is_empty() {
            return;
        }
        assert!(
            self.instance_buffer.is_some(),
            "point-cloud invariant: non-empty source owns an instance buffer"
        );
        self.pack_for_anchor(anchor);
        let buffer = self.instance_buffer.as_ref().expect("invariant checked");
        queue.write_buffer(buffer, 0, bytemuck::cast_slice(&self.points));
    }

    /// Compact the anchor-packed cache in-place to points inside the current
    /// wgpu clip volume. The backing allocation and GPU buffer stay unchanged.
    pub fn prepare_visible(
        &mut self,
        queue: &wgpu::Queue,
        anchor: &crate::camera::Anchor,
        view_proj: glam::Mat4,
    ) {
        debug_assert!(self.points.capacity() >= self.source_points.len());
        debug_assert!(self.visible_source_indices.capacity() >= self.source_points.len());
        self.points.clear();
        self.visible_source_indices.clear();
        for (index, point) in self.source_points.iter().enumerate() {
            let packed = Self::packed_point(point, anchor);
            let clip = view_proj * glam::Vec3::from(packed.position).extend(1.0);
            if clip.w > 0.0
                && clip.x.abs() <= clip.w
                && clip.y.abs() <= clip.w
                && clip.z >= 0.0
                && clip.z <= clip.w
            {
                self.points.push(packed);
                self.visible_source_indices.push(index);
            }
        }
        self.visible_point_count = self.points.len();
        if let Some(buffer) = self.instance_buffer.as_ref() {
            if !self.points.is_empty() {
                queue.write_buffer(buffer, 0, bytemuck::cast_slice(&self.points));
            }
        }
    }

    /// CPU point pick over the same cull result and anchor-packed positions
    /// consumed by the renderer. The returned world coordinate is the exact
    /// source f64 value, not a widened render-space value.
    pub fn pick_ray(&self, ray: &crate::picking::Ray) -> Option<(usize, f32, DVec3)> {
        let origin = glam::Vec3::from(ray.origin);
        let direction = glam::Vec3::from(ray.direction).normalize_or_zero();
        if direction == glam::Vec3::ZERO {
            return None;
        }
        let radius = (self.extent_render * 0.002).max(0.01);
        self.points
            .iter()
            .zip(self.visible_source_indices.iter().copied())
            .take(self.visible_point_count)
            .filter_map(|(point, source_index)| {
                let delta = glam::Vec3::from(point.position) - origin;
                let t = delta.dot(direction);
                if t < 0.0 || (delta - direction * t).length() > radius {
                    return None;
                }
                Some((source_index, t, self.source_points[source_index].position))
            })
            .min_by(|left, right| left.1.total_cmp(&right.1))
    }

    pub fn instance_buffer_bytes(&self) -> u64 {
        self.instance_buffer
            .as_ref()
            .map_or(0, |buffer| buffer.size())
    }

    pub fn instance_buffer_id(&self) -> u64 {
        self.instance_buffer
            .as_ref()
            .map_or(0, TrackedBuffer::ledger_id)
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
        render_pass.draw(0..4, 0..self.visible_point_count as u32);
    }

    pub fn set_point_size(&mut self, size: f32) {
        self.point_size = size.max(0.5).min(50.0);
    }

    pub fn set_visible(&mut self, visible: bool) {
        self.visible = visible;
    }

    /// Validate the complete prospective point-cloud state before publishing
    /// any field. This keeps invalid camera/parameter commands transactional.
    #[allow(clippy::too_many_arguments)]
    pub fn try_set_params(
        &mut self,
        anchor: &crate::camera::Anchor,
        owns_frame: bool,
        point_size: Option<f32>,
        visible: Option<bool>,
        color_mode: Option<&str>,
        phi: Option<f32>,
        theta: Option<f32>,
        radius: Option<f32>,
    ) -> Result<(), String> {
        let next_point_size = point_size.unwrap_or(self.point_size);
        let next_phi = phi.unwrap_or(self.cam_phi);
        let next_theta = theta.unwrap_or(self.cam_theta).clamp(0.1, 1.5);
        let next_radius = radius.unwrap_or(self.cam_radius).clamp(0.1, 100.0);
        if ![next_point_size, next_phi, next_theta, next_radius]
            .into_iter()
            .all(f32::is_finite)
            || next_point_size <= 0.0
        {
            return Err("invalid point-cloud parameter".to_string());
        }

        let extent = self.extent_render.max(100.0);
        let orbit_radius = extent * 2.0 * next_radius;
        let prospective_eye = self.center
            + DVec3::new(
                f64::from(orbit_radius * next_theta.cos() * next_phi.cos()),
                f64::from(orbit_radius * next_theta.sin()),
                f64::from(orbit_radius * next_theta.cos() * next_phi.sin()),
            );
        let rebase_focus = if owns_frame && visible.unwrap_or(self.visible) {
            self.center
        } else {
            anchor.origin()
        };
        let validation_anchor =
            crate::viewer::camera_controller::prospective_anchor(anchor, rebase_focus);
        crate::viewer::camera_controller::validate_camera_pose(
            &validation_anchor,
            self.center,
            prospective_eye,
            self.center,
        )
        .map_err(|error| error.to_string())?;

        self.point_size = next_point_size.clamp(0.5, 50.0);
        self.cam_phi = next_phi;
        self.cam_theta = next_theta;
        self.cam_radius = next_radius;
        self.visible = visible.unwrap_or(self.visible);
        if let Some(mode) = color_mode {
            self.color_mode = ColorMode::from_str(mode);
        }
        Ok(())
    }

    pub fn clear(&mut self) {
        self.source_points.clear();
        self.points.clear();
        self.instance_buffer = None;
        self.point_count = 0;
        self.visible_point_count = 0;
        self.visible_source_indices.clear();
    }
}
