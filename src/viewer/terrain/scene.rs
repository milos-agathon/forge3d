// src/viewer/terrain/scene.rs
// Terrain scene management for the interactive viewer

use super::render::TerrainUniforms;
use super::shader::TERRAIN_SHADER;
use anyhow::Result;
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// Stored terrain data for interactive viewer rendering
pub struct ViewerTerrainData {
    pub heightmap: Vec<f32>,
    pub dimensions: (u32, u32),
    pub domain: (f32, f32),
    pub heightmap_texture: wgpu::Texture,
    pub heightmap_view: wgpu::TextureView,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub index_count: u32,
    pub uniform_buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
    // Camera
    pub cam_radius: f32,
    pub cam_phi_deg: f32,
    pub cam_theta_deg: f32,
    pub cam_fov_deg: f32,
    // Sun/lighting
    pub sun_azimuth_deg: f32,
    pub sun_elevation_deg: f32,
    pub sun_intensity: f32,
    pub ambient: f32,
    // Terrain rendering
    pub z_scale: f32,
    pub shadow_intensity: f32,
    pub background_color: [f32; 3],
    pub water_level: f32,
    pub water_color: [f32; 3],
}

/// Simple terrain scene for interactive viewer
pub struct ViewerTerrainScene {
    pub(super) device: Arc<wgpu::Device>,
    pub(super) queue: Arc<wgpu::Queue>,
    pub(super) pipeline: wgpu::RenderPipeline,
    pub(super) bind_group_layout: wgpu::BindGroupLayout,
    pub(super) depth_texture: Option<wgpu::Texture>,
    pub(super) depth_view: Option<wgpu::TextureView>,
    pub(super) depth_size: (u32, u32),
    pub terrain: Option<ViewerTerrainData>,
}

impl ViewerTerrainScene {
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        target_format: wgpu::TextureFormat,
    ) -> Result<Self> {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("terrain_viewer.bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("terrain_viewer.shader"),
            source: wgpu::ShaderSource::Wgsl(TERRAIN_SHADER.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("terrain_viewer.pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("terrain_viewer.pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 16,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2,
                            offset: 0,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2,
                            offset: 8,
                            shader_location: 1,
                        },
                    ],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
            depth_texture: None,
            depth_view: None,
            depth_size: (0, 0),
            terrain: None,
        })
    }

    pub fn load_terrain(&mut self, path: &str) -> Result<()> {
        use std::fs::File;

        let file = File::open(path)?;
        let mut decoder = tiff::decoder::Decoder::new(file)?;
        let (width, height) = decoder.dimensions()?;
        let image = decoder.read_image()?;

        let heightmap: Vec<f32> = match image {
            tiff::decoder::DecodingResult::F32(data) => data,
            tiff::decoder::DecodingResult::F64(data) => data.iter().map(|&v| v as f32).collect(),
            tiff::decoder::DecodingResult::I16(data) => data.iter().map(|&v| v as f32).collect(),
            tiff::decoder::DecodingResult::U16(data) => data.iter().map(|&v| v as f32).collect(),
            tiff::decoder::DecodingResult::I32(data) => data.iter().map(|&v| v as f32).collect(),
            tiff::decoder::DecodingResult::U32(data) => data.iter().map(|&v| v as f32).collect(),
            tiff::decoder::DecodingResult::U8(data) => data.iter().map(|&v| v as f32).collect(),
            tiff::decoder::DecodingResult::I8(data) => data.iter().map(|&v| v as f32).collect(),
            tiff::decoder::DecodingResult::U64(data) => data.iter().map(|&v| v as f32).collect(),
            tiff::decoder::DecodingResult::I64(data) => data.iter().map(|&v| v as f32).collect(),
        };

        let (min_h, max_h) = heightmap
            .iter()
            .filter(|h| h.is_finite())
            .fold((f32::MAX, f32::MIN), |(min, max), &h| {
                (min.min(h), max.max(h))
            });

        let heightmap_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("terrain_viewer.heightmap"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &heightmap_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&heightmap),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        let heightmap_view = heightmap_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let grid_res = 256u32.min(width.min(height));
        let (vertices, indices) = create_grid_mesh(grid_res);

        let vertex_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("terrain_viewer.vertex_buffer"),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });

        let index_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("terrain_viewer.index_buffer"),
                contents: bytemuck::cast_slice(&indices),
                usage: wgpu::BufferUsages::INDEX,
            });

        let terrain_span = width.max(height) as f32;
        let cam_radius = terrain_span * 1.5;

        let uniforms = TerrainUniforms {
            view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
            sun_dir: [0.5, 0.8, 0.3, 0.0],
            terrain_params: [min_h, max_h - min_h, terrain_span, 1.0],
            lighting: [1.0, 0.3, 0.5, -999999.0],
            background: [0.5, 0.7, 0.9, 0.0],
            water_color: [0.2, 0.4, 0.6, 0.0],
        };

        let uniform_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("terrain_viewer.uniform_buffer"),
                contents: bytemuck::cast_slice(&[uniforms]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("terrain_viewer.sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain_viewer.bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&heightmap_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        self.terrain = Some(ViewerTerrainData {
            heightmap,
            dimensions: (width, height),
            domain: (min_h, max_h),
            heightmap_texture,
            heightmap_view,
            vertex_buffer,
            index_buffer,
            index_count: indices.len() as u32,
            uniform_buffer,
            bind_group,
            cam_radius,
            cam_phi_deg: 135.0,
            cam_theta_deg: 45.0,
            cam_fov_deg: 55.0,
            sun_azimuth_deg: 135.0,
            sun_elevation_deg: 35.0,
            sun_intensity: 1.0,
            ambient: 0.3,
            z_scale: 0.3,
            shadow_intensity: 0.5,
            background_color: [0.5, 0.7, 0.9],
            water_level: -999999.0,
            water_color: [0.2, 0.4, 0.6],
        });

        println!(
            "[terrain] Loaded {}x{} DEM, domain: {:.1}..{:.1}",
            width, height, min_h, max_h
        );
        Ok(())
    }

    pub fn has_terrain(&self) -> bool {
        self.terrain.is_some()
    }

    pub fn set_camera(&mut self, phi: f32, theta: f32, radius: f32, fov: f32) {
        if let Some(ref mut t) = self.terrain {
            t.cam_phi_deg = phi;
            t.cam_theta_deg = theta;
            t.cam_radius = radius;
            t.cam_fov_deg = fov;
        }
    }

    pub fn set_sun(&mut self, azimuth: f32, elevation: f32, intensity: f32) {
        if let Some(ref mut t) = self.terrain {
            t.sun_azimuth_deg = azimuth;
            t.sun_elevation_deg = elevation;
            t.sun_intensity = intensity;
        }
    }

    pub fn get_params(&self) -> Option<String> {
        self.terrain.as_ref().map(|t| format!(
            "phi={:.1} theta={:.1} radius={:.0} fov={:.1} | sun_az={:.1} sun_el={:.1} intensity={:.2} ambient={:.2} | zscale={:.2} shadow={:.2}",
            t.cam_phi_deg, t.cam_theta_deg, t.cam_radius, t.cam_fov_deg,
            t.sun_azimuth_deg, t.sun_elevation_deg, t.sun_intensity, t.ambient,
            t.z_scale, t.shadow_intensity
        ))
    }

    pub fn handle_mouse_drag(&mut self, dx: f32, dy: f32) {
        if let Some(ref mut t) = self.terrain {
            t.cam_phi_deg += dx * 0.3;
            t.cam_theta_deg = (t.cam_theta_deg - dy * 0.3).clamp(5.0, 85.0);
        }
    }

    pub fn handle_scroll(&mut self, delta: f32) {
        if let Some(ref mut t) = self.terrain {
            let factor = (-delta * 0.05).exp();
            t.cam_radius = (t.cam_radius * factor).clamp(100.0, 50000.0);
        }
    }

    pub fn handle_keys(&mut self, forward: f32, right: f32, up: f32) {
        if let Some(ref mut t) = self.terrain {
            t.cam_phi_deg += right * 2.0;
            t.cam_theta_deg = (t.cam_theta_deg - forward * 2.0).clamp(5.0, 85.0);
            t.cam_radius = (t.cam_radius * (1.0 - up * 0.02)).clamp(100.0, 50000.0);
        }
    }

    // ensure_depth and render moved to terrain/render.rs
}

fn create_grid_mesh(resolution: u32) -> (Vec<f32>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let inv = 1.0 / (resolution - 1) as f32;

    for y in 0..resolution {
        for x in 0..resolution {
            let u = x as f32 * inv;
            let v = y as f32 * inv;
            vertices.extend_from_slice(&[u, v, u, v]);
        }
    }

    for y in 0..(resolution - 1) {
        for x in 0..(resolution - 1) {
            let i = y * resolution + x;
            indices.extend_from_slice(&[
                i,
                i + resolution,
                i + 1,
                i + 1,
                i + resolution,
                i + resolution + 1,
            ]);
        }
    }

    (vertices, indices)
}
