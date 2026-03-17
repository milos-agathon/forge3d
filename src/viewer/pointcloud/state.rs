use crate::viewer::pointcloud::load::load_laz_points;
use crate::viewer::pointcloud::shader::POINTCLOUD_SHADER;
use crate::viewer::pointcloud::{ColorMode, PointCloudUniforms, PointInstance3D};

/// Point cloud state for the viewer.
pub struct PointCloudState {
    pub points: Vec<PointInstance3D>,
    pub instance_buffer: Option<wgpu::Buffer>,
    pub uniform_buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
    pub pipeline: wgpu::RenderPipeline,
    pub point_count: usize,
    pub point_size: f32,
    pub visible: bool,
    pub color_mode: ColorMode,
    pub bounds_min: [f32; 3],
    pub bounds_max: [f32; 3],
    pub center: [f32; 3],
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
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pointcloud.wgsl"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(POINTCLOUD_SHADER)),
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PointCloud.Uniforms"),
            size: std::mem::size_of::<PointCloudUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

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

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
        });

        Self {
            points: Vec::new(),
            instance_buffer: None,
            uniform_buffer,
            bind_group,
            pipeline,
            point_count: 0,
            point_size: 2.0,
            visible: true,
            color_mode: ColorMode::Elevation,
            bounds_min: [0.0; 3],
            bounds_max: [0.0; 3],
            center: [0.0; 3],
            has_rgb: false,
            has_intensity: false,
            cam_phi: 0.7,
            cam_theta: 0.5,
            cam_radius: 1.0,
        }
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
        path: &str,
        max_points: u64,
        color_mode: ColorMode,
    ) -> Result<(), String> {
        let load_result = load_laz_points(path, max_points as usize)?;
        let mut points = load_result.points;

        self.has_rgb = load_result.has_rgb;
        self.has_intensity = load_result.has_intensity;

        println!(
            "[pointcloud] Data flags - has_rgb: {}, has_intensity: {}",
            self.has_rgb, self.has_intensity
        );

        if points.is_empty() {
            return Err("No points loaded".to_string());
        }

        let mut min = [f32::MAX; 3];
        let mut max = [f32::MIN; 3];
        for p in &points {
            for i in 0..3 {
                min[i] = min[i].min(p.position[i]);
                max[i] = max[i].max(p.position[i]);
            }
        }

        let center = [
            (min[0] + max[0]) / 2.0,
            (min[1] + max[1]) / 2.0,
            (min[2] + max[2]) / 2.0,
        ];

        for p in &mut points {
            p.position[0] -= center[0];
            p.position[1] -= center[1];
            p.position[2] -= center[2];
        }

        self.bounds_min = [min[0] - center[0], min[1] - center[1], min[2] - center[2]];
        self.bounds_max = [max[0] - center[0], max[1] - center[1], max[2] - center[2]];
        self.center = [0.0, 0.0, 0.0];

        eprintln!(
            "[pointcloud] Original center: ({:.1}, {:.1}, {:.1})",
            center[0], center[1], center[2]
        );
        eprintln!(
            "[pointcloud] Extent: ({:.1}, {:.1}, {:.1})",
            self.bounds_max[0] - self.bounds_min[0],
            self.bounds_max[1] - self.bounds_min[1],
            self.bounds_max[2] - self.bounds_min[2]
        );

        self.points = points;
        self.point_count = self.points.len();
        self.color_mode = color_mode;
        self.upload_points(device, queue);

        Ok(())
    }

    fn upload_points(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if self.points.is_empty() {
            return;
        }

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PointCloud.InstanceBuffer"),
            size: (self.points.len() * std::mem::size_of::<PointInstance3D>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&self.points));
        self.instance_buffer = Some(buffer);
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
        self.points.clear();
        self.instance_buffer = None;
        self.point_count = 0;
    }
}
