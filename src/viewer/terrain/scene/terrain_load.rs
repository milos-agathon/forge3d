use super::*;
use crate::viewer::event_loop::update_terrain_volumetrics_report;
use crate::viewer::ipc::TerrainVolumetricsReport;

const MAX_VIEWER_TERRAIN_GRID_RESOLUTION: u32 = 2048;

impl ViewerTerrainScene {
    pub fn load_terrain(&mut self, path: &str) -> Result<()> {
        use std::fs::File;

        let file = File::open(path)?;
        let mut decoder = tiff::decoder::Decoder::new(file)?;
        let (width, height) = decoder.dimensions()?;
        let image = decoder.read_image()?;

        let mut heightmap: Vec<f32> = match image {
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

        // Filter out nodata values (common nodata: -9999, -32768, etc.)
        let (min_h, max_h) = heightmap
            .iter()
            .filter(|h| h.is_finite() && **h > -1000.0 && **h < 10000.0)
            .fold((f32::MAX, f32::MIN), |(min, max), &h| {
                (min.min(h), max.max(h))
            });

        // Debug: print height range to diagnose flat terrain issue
        let h_range = max_h - min_h;
        println!(
            "[terrain] Height range: {:.1} to {:.1} (range: {:.1})",
            min_h, max_h, h_range
        );

        // Replace NoData values with min_h to prevent edge artifacts
        // NoData values are typically: NaN, Inf, < -1000, > 10000
        for h in heightmap.iter_mut() {
            if !h.is_finite() || *h < -1000.0 || *h > 10000.0 {
                *h = min_h;
            }
        }

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
        let grid_res = terrain_grid_resolution(width, height);
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

        let terrain_width = width as f32;
        let terrain_span = terrain_width.max(height as f32);
        let cam_radius = terrain_span * 1.5;

        let uniforms = TerrainUniforms {
            view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
            sun_dir: [0.5, 0.8, 0.3, 0.0],
            terrain_params: [min_h, max_h - min_h, terrain_width, 1.0],
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

        self.terrain_revision_counter = self.terrain_revision_counter.wrapping_add(1);
        let mut terrain = ViewerTerrainData {
            heightmap,
            dimensions: (width, height),
            domain: (min_h, max_h),
            revision: self.terrain_revision_counter,
            _heightmap_texture: heightmap_texture,
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
            cam_target: [0.0, 0.0, 0.0],
            sun_azimuth_deg: 135.0,
            sun_elevation_deg: 35.0,
            sun_intensity: 1.0,
            ambient: 0.3,
            z_scale: 1.0,
            shadow_intensity: 0.5,
            background_color: [0.5, 0.7, 0.9],
            water_level: -999999.0,
            water_color: [0.2, 0.4, 0.6],
        };
        terrain.cam_target = terrain.default_camera_target();
        self.terrain = Some(terrain);
        update_terrain_volumetrics_report(TerrainVolumetricsReport::default());

        println!(
            "[terrain] Loaded {}x{} DEM, domain: {:.1}..{:.1}, grid={}x{}",
            width, height, min_h, max_h, grid_res, grid_res
        );
        Ok(())
    }

    pub fn has_terrain(&self) -> bool {
        self.terrain.is_some()
    }

    pub fn set_camera(
        &mut self,
        phi: f32,
        theta: f32,
        radius: f32,
        fov: f32,
        target: Option<[f32; 3]>,
    ) {
        if let Some(ref mut t) = self.terrain {
            t.set_camera_state(phi, theta, radius, fov, target);
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
            "phi={:.1} theta={:.1} radius={:.0} fov={:.1} target=({:.1}, {:.1}, {:.1}) | sun_az={:.1} sun_el={:.1} intensity={:.2} ambient={:.2} | zscale={:.2} shadow={:.2}",
            t.cam_phi_deg, t.cam_theta_deg, t.cam_radius, t.cam_fov_deg,
            t.cam_target[0], t.cam_target[1], t.cam_target[2],
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
}

fn terrain_grid_resolution(width: u32, height: u32) -> u32 {
    width
        .min(height)
        .clamp(2, MAX_VIEWER_TERRAIN_GRID_RESOLUTION)
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

#[cfg(test)]
mod tests {
    use super::terrain_grid_resolution;

    #[test]
    fn viewer_terrain_grid_resolution_keeps_small_heightmaps_native() {
        assert_eq!(terrain_grid_resolution(512, 384), 384);
    }

    #[test]
    fn viewer_terrain_grid_resolution_caps_large_heightmaps() {
        assert_eq!(terrain_grid_resolution(11589, 10518), 2048);
        assert_eq!(terrain_grid_resolution(5794, 5259), 2048);
    }
}
