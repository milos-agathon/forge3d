// src/viewer/viewer_terrain.rs
// Minimal terrain viewer for interactive viewing (no PyO3 dependencies)

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

/// Terrain uniforms for the simple terrain shader
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct TerrainUniforms {
    view_proj: [[f32; 4]; 4],
    sun_dir: [f32; 4],
    terrain_params: [f32; 4],   // domain_min, domain_range, terrain_width, z_scale
    lighting: [f32; 4],         // sun_intensity, ambient, shadow_intensity, water_level
    background: [f32; 4],       // r, g, b, _
    water_color: [f32; 4],      // r, g, b, _
}

/// Simple terrain scene for interactive viewer
pub struct ViewerTerrainScene {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    depth_texture: Option<wgpu::Texture>,
    depth_view: Option<wgpu::TextureView>,
    depth_size: (u32, u32),
    pub terrain: Option<ViewerTerrainData>,
}

impl ViewerTerrainScene {
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        target_format: wgpu::TextureFormat,
    ) -> Result<Self> {
        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("terrain_viewer.bind_group_layout"),
            entries: &[
                // Uniforms
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
                // Heightmap texture
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
                // Heightmap sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });

        // Create shader module
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("terrain_viewer.shader"),
            source: wgpu::ShaderSource::Wgsl(TERRAIN_SHADER.into()),
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("terrain_viewer.pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create render pipeline
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("terrain_viewer.pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 16, // 4 floats: x, y, u, v
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

    /// Load terrain from a GeoTIFF file
    pub fn load_terrain(&mut self, path: &str) -> Result<()> {
        use std::fs::File;

        let file = File::open(path)?;
        let mut decoder = tiff::decoder::Decoder::new(file)?;
        let (width, height) = decoder.dimensions()?;
        let image = decoder.read_image()?;

        // Convert to f32
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

        // Calculate domain
        let mut min_h = f32::MAX;
        let mut max_h = f32::MIN;
        for &h in &heightmap {
            if h.is_finite() {
                min_h = min_h.min(h);
                max_h = max_h.max(h);
            }
        }

        // Create heightmap texture
        let heightmap_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("terrain_viewer.heightmap"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
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
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        );

        let heightmap_view = heightmap_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create terrain mesh
        let grid_res = 256u32.min(width.min(height));
        let (vertices, indices) = create_grid_mesh(grid_res);

        let vertex_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("terrain_viewer.vertex_buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("terrain_viewer.index_buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        // Create uniform buffer
        let terrain_span = width.max(height) as f32;
        let cam_radius = terrain_span * 1.5;  // More zoomed out default

        let uniforms = TerrainUniforms {
            view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
            sun_dir: [0.5, 0.8, 0.3, 0.0],
            terrain_params: [min_h, max_h - min_h, terrain_span, 1.0],
            lighting: [1.0, 0.3, 0.5, -999999.0],
            background: [0.5, 0.7, 0.9, 0.0],
            water_color: [0.2, 0.4, 0.6, 0.0],
        };

        let uniform_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("terrain_viewer.uniform_buffer"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create sampler
        let sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("terrain_viewer.sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain_viewer.bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&heightmap_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&sampler) },
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
            water_level: -999999.0, // disabled by default
            water_color: [0.2, 0.4, 0.6],
        });

        println!("[terrain] Loaded {}x{} DEM, domain: {:.1}..{:.1}", width, height, min_h, max_h);
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

    /// Set z-scale (vertical exaggeration)
    pub fn set_zscale(&mut self, z_scale: f32) {
        if let Some(ref mut t) = self.terrain {
            t.z_scale = z_scale.max(0.01);
        }
    }

    /// Set ambient light level
    pub fn set_ambient(&mut self, ambient: f32) {
        if let Some(ref mut t) = self.terrain {
            t.ambient = ambient.clamp(0.0, 1.0);
        }
    }

    /// Set shadow intensity
    pub fn set_shadow_intensity(&mut self, intensity: f32) {
        if let Some(ref mut t) = self.terrain {
            t.shadow_intensity = intensity.clamp(0.0, 1.0);
        }
    }

    /// Set background color (RGB 0-1)
    pub fn set_background(&mut self, r: f32, g: f32, b: f32) {
        if let Some(ref mut t) = self.terrain {
            t.background_color = [r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0)];
        }
    }

    /// Set water level and color
    pub fn set_water(&mut self, level: f32, r: f32, g: f32, b: f32) {
        if let Some(ref mut t) = self.terrain {
            t.water_level = level;
            t.water_color = [r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0)];
        }
    }

    /// Get current parameters as JSON-like string for display
    pub fn get_params(&self) -> Option<String> {
        self.terrain.as_ref().map(|t| {
            format!(
                "phi={:.1} theta={:.1} radius={:.0} fov={:.1} | sun_az={:.1} sun_el={:.1} intensity={:.2} ambient={:.2} | zscale={:.2} shadow={:.2}",
                t.cam_phi_deg, t.cam_theta_deg, t.cam_radius, t.cam_fov_deg,
                t.sun_azimuth_deg, t.sun_elevation_deg, t.sun_intensity, t.ambient,
                t.z_scale, t.shadow_intensity
            )
        })
    }

    /// Handle mouse drag to orbit camera (delta in pixels)
    pub fn handle_mouse_drag(&mut self, dx: f32, dy: f32) {
        if let Some(ref mut t) = self.terrain {
            // Horizontal drag rotates phi (azimuth)
            t.cam_phi_deg += dx * 0.3;
            // Vertical drag rotates theta (elevation), clamped to avoid flipping
            t.cam_theta_deg = (t.cam_theta_deg - dy * 0.3).clamp(5.0, 85.0);
        }
    }

    /// Handle scroll wheel to zoom (positive = zoom in, negative = zoom out)
    pub fn handle_scroll(&mut self, delta: f32) {
        if let Some(ref mut t) = self.terrain {
            // Exponential zoom for smooth feel on both mouse wheel and trackpad
            // delta > 0 = zoom in (smaller radius), delta < 0 = zoom out (larger radius)
            let factor = (-delta * 0.05).exp();
            t.cam_radius = (t.cam_radius * factor).clamp(100.0, 50000.0);
        }
    }

    /// Handle keyboard input for camera movement
    pub fn handle_keys(&mut self, forward: f32, right: f32, up: f32) {
        if let Some(ref mut t) = self.terrain {
            // Arrow keys / WASD adjust phi and theta
            t.cam_phi_deg += right * 2.0;
            t.cam_theta_deg = (t.cam_theta_deg - forward * 2.0).clamp(5.0, 85.0);
            // Q/E adjust radius
            t.cam_radius = (t.cam_radius * (1.0 - up * 0.02)).clamp(100.0, 50000.0);
        }
    }

    fn ensure_depth(&mut self, width: u32, height: u32) {
        if self.depth_size != (width, height) {
            let tex = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("terrain_viewer.depth"),
                size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            self.depth_view = Some(tex.create_view(&wgpu::TextureViewDescriptor::default()));
            self.depth_texture = Some(tex);
            self.depth_size = (width, height);
        }
    }

    /// Render terrain to the given view
    pub fn render(&mut self, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView, width: u32, height: u32) -> bool {
        if self.terrain.is_none() {
            return false;
        }

        self.ensure_depth(width, height);
        
        let terrain = self.terrain.as_ref().unwrap();
        let depth_view = self.depth_view.as_ref().unwrap();

        // Update uniforms
        let phi = terrain.cam_phi_deg.to_radians();
        let theta = terrain.cam_theta_deg.to_radians();
        let r = terrain.cam_radius;

        let (tw, th) = terrain.dimensions;
        let center = glam::Vec3::new(tw as f32 * 0.5, (terrain.domain.0 + terrain.domain.1) * 0.5, th as f32 * 0.5);

        let eye = glam::Vec3::new(
            center.x + r * theta.sin() * phi.cos(),
            center.y + r * theta.cos(),
            center.z + r * theta.sin() * phi.sin(),
        );

        let view_mat = glam::Mat4::look_at_rh(eye, center, glam::Vec3::Y);
        let proj = glam::Mat4::perspective_rh(terrain.cam_fov_deg.to_radians(), width as f32 / height as f32, 1.0, r * 10.0);
        let view_proj = proj * view_mat;

        let sun_az = terrain.sun_azimuth_deg.to_radians();
        let sun_el = terrain.sun_elevation_deg.to_radians();
        let sun_dir = glam::Vec3::new(sun_el.cos() * sun_az.sin(), sun_el.sin(), sun_el.cos() * sun_az.cos()).normalize();

        let uniforms = TerrainUniforms {
            view_proj: view_proj.to_cols_array_2d(),
            sun_dir: [sun_dir.x, sun_dir.y, sun_dir.z, 0.0],
            terrain_params: [terrain.domain.0, terrain.domain.1 - terrain.domain.0, tw as f32, terrain.z_scale],
            lighting: [terrain.sun_intensity, terrain.ambient, terrain.shadow_intensity, terrain.water_level],
            background: [terrain.background_color[0], terrain.background_color[1], terrain.background_color[2], 0.0],
            water_color: [terrain.water_color[0], terrain.water_color[1], terrain.water_color[2], 0.0],
        };

        self.queue.write_buffer(&terrain.uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));

        let bg = &terrain.background_color;
        // Render
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("terrain_viewer.render_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: bg[0] as f64, g: bg[1] as f64, b: bg[2] as f64, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &terrain.bind_group, &[]);
            pass.set_vertex_buffer(0, terrain.vertex_buffer.slice(..));
            pass.set_index_buffer(terrain.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..terrain.index_count, 0, 0..1);
        }

        true
    }
}

fn create_grid_mesh(resolution: u32) -> (Vec<f32>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let inv = 1.0 / (resolution - 1) as f32;

    for y in 0..resolution {
        for x in 0..resolution {
            let u = x as f32 * inv;
            let v = y as f32 * inv;
            vertices.extend_from_slice(&[u, v, u, v]); // pos.xy = uv for simplicity
        }
    }

    for y in 0..(resolution - 1) {
        for x in 0..(resolution - 1) {
            let i = y * resolution + x;
            indices.extend_from_slice(&[i, i + resolution, i + 1, i + 1, i + resolution, i + resolution + 1]);
        }
    }

    (vertices, indices)
}

const TERRAIN_SHADER: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    sun_dir: vec4<f32>,
    terrain_params: vec4<f32>,  // min_h, h_range, terrain_width, z_scale
    lighting: vec4<f32>,        // sun_intensity, ambient, shadow_intensity, water_level
    background: vec4<f32>,      // r, g, b, _
    water_color: vec4<f32>,     // r, g, b, _
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var heightmap: texture_2d<f32>;
@group(0) @binding(2) var height_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) raw_height: f32,
};

@vertex
fn vs_main(@location(0) pos: vec2<f32>, @location(1) uv: vec2<f32>) -> VertexOutput {
    let dims = vec2<f32>(textureDimensions(heightmap));
    let texel = vec2<i32>(i32(uv.x * dims.x), i32(uv.y * dims.y));
    let h = textureLoad(heightmap, texel, 0).r;
    
    let z_scale = u.terrain_params.w;
    let center_h = u.terrain_params.x + u.terrain_params.y * 0.5;
    let scaled_h = center_h + (h - center_h) * z_scale;
    
    let world_x = uv.x * u.terrain_params.z;
    let world_z = uv.y * u.terrain_params.z;
    let world_y = scaled_h;
    
    var out: VertexOutput;
    out.world_pos = vec3<f32>(world_x, world_y, world_z);
    out.position = u.view_proj * vec4<f32>(out.world_pos, 1.0);
    out.uv = uv;
    out.raw_height = h;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let sun_intensity = u.lighting.x;
    let ambient = u.lighting.y;
    let shadow_strength = u.lighting.z;
    let water_level = u.lighting.w;
    
    // Check if below water level
    let is_water = in.raw_height < water_level;
    
    // Simple height-based coloring with sun shading
    let h_norm = clamp((in.raw_height - u.terrain_params.x) / max(u.terrain_params.y, 1.0), 0.0, 1.0);
    
    // Terrain colormap (green valleys, brown slopes, white peaks)
    var color: vec3<f32>;
    if is_water {
        color = u.water_color.rgb;
    } else if h_norm < 0.3 {
        color = mix(vec3<f32>(0.2, 0.5, 0.2), vec3<f32>(0.4, 0.6, 0.3), h_norm / 0.3);
    } else if h_norm < 0.7 {
        color = mix(vec3<f32>(0.4, 0.6, 0.3), vec3<f32>(0.5, 0.4, 0.3), (h_norm - 0.3) / 0.4);
    } else {
        color = mix(vec3<f32>(0.5, 0.4, 0.3), vec3<f32>(0.95, 0.95, 0.95), (h_norm - 0.7) / 0.3);
    }
    
    // Approximate normal from height gradient (finite differences via dFdx/dFdy)
    let dx = dpdx(in.world_pos);
    let dy = dpdy(in.world_pos);
    let normal = normalize(cross(dy, dx));
    
    // Diffuse lighting with shadow
    let sun_dir = normalize(u.sun_dir.xyz);
    let ndotl = max(dot(normal, sun_dir), 0.0);
    
    // Shadow darkening for faces away from sun
    let shadow = mix(1.0, 1.0 - shadow_strength, 1.0 - ndotl);
    
    // Final lighting
    let diffuse = ndotl * sun_intensity;
    let lit = ambient + (1.0 - ambient) * diffuse * shadow;
    
    // Water gets specular highlight
    var final_color = color * lit;
    if is_water {
        let view_dir = normalize(-in.world_pos);
        let reflect_dir = reflect(-sun_dir, vec3<f32>(0.0, 1.0, 0.0));
        let spec = pow(max(dot(view_dir, reflect_dir), 0.0), 32.0);
        final_color = final_color + vec3<f32>(spec * sun_intensity * 0.5);
    }
    
    return vec4<f32>(final_color, 1.0);
}
"#;
