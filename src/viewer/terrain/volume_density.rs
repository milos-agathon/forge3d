use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use half::f16;

use crate::viewer::ipc::{TerrainVolumetricsReport, TerrainVolumetricsVolumeReport};
use crate::viewer::terrain::pbr_renderer::DensityVolumeConfig;

pub(crate) const MAX_DENSITY_VOLUMES: usize = 4;
pub(crate) const MAX_VOLUME_RESOLUTION_AXIS: u32 = 96;
pub(crate) const DENSITY_VOLUME_MEMORY_BUDGET_BYTES: u64 = 16 * 1024 * 1024;
const DENSITY_VOLUME_TEXEL_BYTES: u64 = 2;

#[derive(Debug, Clone, Copy)]
pub struct TerrainVolumeContext<'a> {
    pub heightmap: &'a [f32],
    pub height_dims: (u32, u32),
    pub terrain_width: f32,
    pub domain: (f32, f32),
    pub z_scale: f32,
    pub terrain_revision: u64,
}

#[derive(Debug, Clone)]
pub struct DensityVolumeGpuMetadata {
    pub min_corner: [f32; 3],
    pub inv_size: [f32; 3],
    pub atlas_offset: [f32; 3],
    pub atlas_scale: [f32; 3],
}

#[derive(Debug, Clone)]
pub struct DensityVolumeAtlasData {
    pub dimensions: [u32; 3],
    pub voxels: Vec<f32>,
    pub metadata: Vec<DensityVolumeGpuMetadata>,
    pub fingerprint: u64,
    pub report: TerrainVolumetricsReport,
}

pub struct DensityVolumeAtlasGpu {
    pub _texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
    pub metadata: Vec<DensityVolumeGpuMetadata>,
    pub fingerprint: u64,
    pub report: TerrainVolumetricsReport,
}

impl DensityVolumeAtlasGpu {
    pub fn upload(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        data: DensityVolumeAtlasData,
        raymarch_steps: u32,
        half_res: bool,
    ) -> Self {
        let dimensions = data.dimensions;
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("terrain_viewer.density_volume_atlas"),
            size: wgpu::Extent3d {
                width: dimensions[0],
                height: dimensions[1],
                depth_or_array_layers: dimensions[2],
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::R16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let texels = data
            .voxels
            .iter()
            .map(|value| f16::from_f32(*value).to_bits())
            .collect::<Vec<_>>();

        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&texels),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(dimensions[0] * DENSITY_VOLUME_TEXEL_BYTES as u32),
                rows_per_image: Some(dimensions[1]),
            },
            wgpu::Extent3d {
                width: dimensions[0],
                height: dimensions[1],
                depth_or_array_layers: dimensions[2],
            },
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("terrain_viewer.density_volume_atlas.view"),
            dimension: Some(wgpu::TextureViewDimension::D3),
            ..Default::default()
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("terrain_viewer.density_volume_atlas.sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let mut report = data.report;
        report.raymarch_steps = raymarch_steps;
        report.half_res = half_res;

        Self {
            _texture: texture,
            view,
            sampler,
            metadata: data.metadata,
            fingerprint: data.fingerprint,
            report,
        }
    }
}

pub fn build_density_volume_atlas_data(
    context: TerrainVolumeContext<'_>,
    configs: &[DensityVolumeConfig],
) -> Option<DensityVolumeAtlasData> {
    let active_configs = configs
        .iter()
        .take(MAX_DENSITY_VOLUMES)
        .map(sanitize_config)
        .collect::<Vec<_>>();

    if active_configs.is_empty() {
        return None;
    }

    let atlas_width = active_configs
        .iter()
        .map(|config| config.resolution[0])
        .max()
        .unwrap_or(1);
    let atlas_height = active_configs
        .iter()
        .map(|config| config.resolution[1])
        .max()
        .unwrap_or(1);
    let atlas_depth = active_configs
        .iter()
        .map(|config| config.resolution[2])
        .sum::<u32>()
        .max(1);

    let total_voxels = atlas_width as u64 * atlas_height as u64 * atlas_depth as u64;
    let texture_bytes = total_voxels * DENSITY_VOLUME_TEXEL_BYTES;
    if texture_bytes > DENSITY_VOLUME_MEMORY_BUDGET_BYTES {
        return None;
    }

    let mut atlas = vec![0.0; total_voxels as usize];
    let mut metadata = Vec::with_capacity(active_configs.len());
    let mut volume_reports = Vec::with_capacity(active_configs.len());
    let fingerprint = fingerprint_configs(context, &active_configs);
    let mut z_cursor = 0u32;

    for config in active_configs {
        let density = generate_density_volume(context, &config);
        let resolution = config.resolution;

        for z in 0..resolution[2] {
            for y in 0..resolution[1] {
                for x in 0..resolution[0] {
                    let src_index = ((z * resolution[1] + y) * resolution[0] + x) as usize;
                    let dst_index =
                        (((z_cursor + z) * atlas_height + y) * atlas_width + x) as usize;
                    atlas[dst_index] = density[src_index];
                }
            }
        }

        let min_corner = [
            config.center[0] - config.size[0] * 0.5,
            config.center[1] - config.size[1] * 0.5,
            config.center[2] - config.size[2] * 0.5,
        ];
        metadata.push(DensityVolumeGpuMetadata {
            min_corner,
            inv_size: [
                1.0 / config.size[0].max(1e-3),
                1.0 / config.size[1].max(1e-3),
                1.0 / config.size[2].max(1e-3),
            ],
            atlas_offset: [0.0, 0.0, z_cursor as f32 / atlas_depth as f32],
            atlas_scale: [
                resolution[0] as f32 / atlas_width as f32,
                resolution[1] as f32 / atlas_height as f32,
                resolution[2] as f32 / atlas_depth as f32,
            ],
        });
        volume_reports.push(TerrainVolumetricsVolumeReport {
            preset: config.preset.clone(),
            center: config.center,
            size: config.size,
            resolution,
            atlas_offset: [0, 0, z_cursor],
            voxel_count: resolution[0] as u64 * resolution[1] as u64 * resolution[2] as u64,
        });
        z_cursor += resolution[2];
    }

    Some(DensityVolumeAtlasData {
        dimensions: [atlas_width, atlas_height, atlas_depth],
        voxels: atlas,
        metadata,
        fingerprint,
        report: TerrainVolumetricsReport {
            active_volume_count: volume_reports.len() as u32,
            atlas_dimensions: [atlas_width, atlas_height, atlas_depth],
            total_voxels,
            texture_bytes,
            memory_budget_bytes: DENSITY_VOLUME_MEMORY_BUDGET_BYTES,
            raymarch_steps: 0,
            half_res: false,
            volumes: volume_reports,
        },
    })
}

fn sanitize_config(config: &DensityVolumeConfig) -> DensityVolumeConfig {
    DensityVolumeConfig {
        preset: match config.preset.as_str() {
            "plume" => "plume".to_string(),
            "localized_haze" => "localized_haze".to_string(),
            _ => "valley_fog".to_string(),
        },
        center: config.center,
        size: [
            config.size[0].max(1.0),
            config.size[1].max(1.0),
            config.size[2].max(1.0),
        ],
        resolution: [
            config.resolution[0].clamp(8, MAX_VOLUME_RESOLUTION_AXIS),
            config.resolution[1].clamp(8, MAX_VOLUME_RESOLUTION_AXIS),
            config.resolution[2].clamp(8, MAX_VOLUME_RESOLUTION_AXIS),
        ],
        density_scale: config.density_scale.clamp(0.0, 4.0),
        edge_softness: config.edge_softness.clamp(0.02, 0.95),
        noise_strength: config.noise_strength.clamp(0.0, 1.0),
        floor_offset: config.floor_offset,
        ceiling: config.ceiling.clamp(0.0, 1.0),
        plume_spread: config.plume_spread.clamp(0.05, 2.0),
        wind: config.wind,
        seed: config.seed,
    }
}

fn fingerprint_configs(context: TerrainVolumeContext<'_>, configs: &[DensityVolumeConfig]) -> u64 {
    let mut hasher = DefaultHasher::new();
    context.terrain_revision.hash(&mut hasher);
    context.height_dims.hash(&mut hasher);
    context.terrain_width.to_bits().hash(&mut hasher);
    context.domain.0.to_bits().hash(&mut hasher);
    context.domain.1.to_bits().hash(&mut hasher);
    context.z_scale.to_bits().hash(&mut hasher);
    for config in configs {
        config.preset.hash(&mut hasher);
        for value in config.center {
            value.to_bits().hash(&mut hasher);
        }
        for value in config.size {
            value.to_bits().hash(&mut hasher);
        }
        config.resolution.hash(&mut hasher);
        config.density_scale.to_bits().hash(&mut hasher);
        config.edge_softness.to_bits().hash(&mut hasher);
        config.noise_strength.to_bits().hash(&mut hasher);
        config.floor_offset.to_bits().hash(&mut hasher);
        config.ceiling.to_bits().hash(&mut hasher);
        config.plume_spread.to_bits().hash(&mut hasher);
        for value in config.wind {
            value.to_bits().hash(&mut hasher);
        }
        config.seed.hash(&mut hasher);
    }
    hasher.finish()
}

fn generate_density_volume(
    context: TerrainVolumeContext<'_>,
    config: &DensityVolumeConfig,
) -> Vec<f32> {
    let resolution = config.resolution;
    let voxel_count = resolution[0] as usize * resolution[1] as usize * resolution[2] as usize;
    let mut voxels = vec![0.0; voxel_count];
    let min_corner = [
        config.center[0] - config.size[0] * 0.5,
        config.center[1] - config.size[1] * 0.5,
        config.center[2] - config.size[2] * 0.5,
    ];

    for z in 0..resolution[2] {
        for y in 0..resolution[1] {
            for x in 0..resolution[0] {
                let local = [
                    x as f32 / (resolution[0] - 1) as f32,
                    y as f32 / (resolution[1] - 1) as f32,
                    z as f32 / (resolution[2] - 1) as f32,
                ];
                let world = [
                    min_corner[0] + config.size[0] * local[0],
                    min_corner[1] + config.size[1] * local[1],
                    min_corner[2] + config.size[2] * local[2],
                ];
                let noise = fbm3(local, config.seed, 3);
                let noise_mix = lerp(1.0, 0.55 + 0.45 * noise, config.noise_strength);
                let density = match config.preset.as_str() {
                    "plume" => plume_density(local, config, noise_mix),
                    "localized_haze" => localized_haze_density(local, config, noise_mix),
                    _ => valley_fog_density(context, world, local, config, noise_mix),
                };
                let density = density * config.density_scale * 0.028;
                let index = ((z * resolution[1] + y) * resolution[0] + x) as usize;
                voxels[index] = density.clamp(0.0, 1.0);
            }
        }
    }

    voxels
}

fn valley_fog_density(
    context: TerrainVolumeContext<'_>,
    world: [f32; 3],
    local: [f32; 3],
    config: &DensityVolumeConfig,
    noise_mix: f32,
) -> f32 {
    let terrain_height = sample_terrain_height(context, world[0], world[2]) + config.floor_offset;
    let top_height = config.size[1] * config.ceiling.max(0.08);
    let height_above_ground = (world[1] - terrain_height).max(0.0);
    let ground_lock = 1.0 - smoothstep(0.0, top_height.max(1.0), height_above_ground);
    let top_fade = 1.0 - smoothstep(config.ceiling, 1.0, local[1]);
    soft_box_falloff(local, config.edge_softness) * ground_lock * top_fade * noise_mix
}

fn plume_density(local: [f32; 3], config: &DensityVolumeConfig, noise_mix: f32) -> f32 {
    let mut wind = glam::Vec3::from_array(config.wind);
    if wind.length_squared() < 1e-6 {
        wind = glam::Vec3::Y;
    }
    let wind = wind.normalize();
    let progress = local[1];
    let centerline = glam::Vec2::new(0.5 + wind.x * progress * 0.2, 0.5 + wind.z * progress * 0.2);
    let pos = glam::Vec2::new(local[0], local[2]);
    let radius = 0.12 + progress * config.plume_spread * 0.35;
    let radial = 1.0 - smoothstep(radius, radius * 1.85, pos.distance(centerline));
    let base = smoothstep(0.02, 0.16, progress);
    let top = 1.0 - smoothstep(0.82, 1.0, progress);
    let axial = base * top;
    soft_box_falloff(local, config.edge_softness * 0.7) * radial * axial * noise_mix
}

fn localized_haze_density(local: [f32; 3], config: &DensityVolumeConfig, noise_mix: f32) -> f32 {
    let centered = glam::Vec3::new(local[0] - 0.5, local[1] - config.ceiling, local[2] - 0.5);
    let ellipsoid = glam::Vec3::new(centered.x / 0.5, centered.y / 0.32, centered.z / 0.5);
    let body = 1.0 - smoothstep(0.55, 1.1, ellipsoid.length());
    let layer = 1.0 - smoothstep(0.25, 0.9, (local[1] - config.ceiling).abs());
    soft_box_falloff(local, config.edge_softness) * body.max(layer * 0.7) * noise_mix
}

fn sample_terrain_height(context: TerrainVolumeContext<'_>, world_x: f32, world_z: f32) -> f32 {
    if context.height_dims.0 < 2 || context.height_dims.1 < 2 || context.heightmap.is_empty() {
        return 0.0;
    }

    let u = (world_x / context.terrain_width).clamp(0.0, 1.0);
    let v = (world_z / context.terrain_width).clamp(0.0, 1.0);
    let x = u * (context.height_dims.0 - 1) as f32;
    let y = v * (context.height_dims.1 - 1) as f32;
    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = (x0 + 1).min(context.height_dims.0 - 1);
    let y1 = (y0 + 1).min(context.height_dims.1 - 1);
    let fx = x.fract();
    let fy = y.fract();
    let raw = bilinear_sample(
        context.heightmap,
        context.height_dims,
        x0,
        y0,
        x1,
        y1,
        fx,
        fy,
    );
    (raw - context.domain.0) * context.z_scale
}

fn bilinear_sample(
    heightmap: &[f32],
    dims: (u32, u32),
    x0: u32,
    y0: u32,
    x1: u32,
    y1: u32,
    fx: f32,
    fy: f32,
) -> f32 {
    let idx = |x: u32, y: u32| -> usize { (y * dims.0 + x) as usize };
    let h00 = heightmap.get(idx(x0, y0)).copied().unwrap_or(0.0);
    let h10 = heightmap.get(idx(x1, y0)).copied().unwrap_or(h00);
    let h01 = heightmap.get(idx(x0, y1)).copied().unwrap_or(h00);
    let h11 = heightmap.get(idx(x1, y1)).copied().unwrap_or(h10);
    let h0 = lerp(h00, h10, fx);
    let h1 = lerp(h01, h11, fx);
    lerp(h0, h1, fy)
}

fn soft_box_falloff(local: [f32; 3], edge_softness: f32) -> f32 {
    edge_axis(local[0], edge_softness)
        * edge_axis(local[1], edge_softness)
        * edge_axis(local[2], edge_softness)
}

fn edge_axis(value: f32, edge_softness: f32) -> f32 {
    let width = (edge_softness * 0.5).clamp(0.01, 0.49);
    let low = smoothstep(0.0, width, value);
    let high = 1.0 - smoothstep(1.0 - width, 1.0, value);
    low * high
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0).max(1e-6)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn fbm3(local: [f32; 3], seed: u32, octaves: u32) -> f32 {
    let mut amplitude = 0.5;
    let mut frequency = 1.0;
    let mut sum = 0.0;
    let mut norm = 0.0;
    for octave in 0..octaves.max(1) {
        let sample = value_noise_3d(
            [
                local[0] * frequency * 3.1,
                local[1] * frequency * 2.3,
                local[2] * frequency * 3.7,
            ],
            seed.wrapping_add(octave * 811),
        );
        sum += sample * amplitude;
        norm += amplitude;
        amplitude *= 0.5;
        frequency *= 2.0;
    }
    if norm > 0.0 {
        sum / norm
    } else {
        0.5
    }
}

fn value_noise_3d(point: [f32; 3], seed: u32) -> f32 {
    let cell = glam::IVec3::new(
        point[0].floor() as i32,
        point[1].floor() as i32,
        point[2].floor() as i32,
    );
    let frac = glam::Vec3::new(point[0].fract(), point[1].fract(), point[2].fract());
    let w = frac * frac * (glam::Vec3::splat(3.0) - 2.0 * frac);

    let n000 = lattice_hash(cell.x, cell.y, cell.z, seed);
    let n100 = lattice_hash(cell.x + 1, cell.y, cell.z, seed);
    let n010 = lattice_hash(cell.x, cell.y + 1, cell.z, seed);
    let n110 = lattice_hash(cell.x + 1, cell.y + 1, cell.z, seed);
    let n001 = lattice_hash(cell.x, cell.y, cell.z + 1, seed);
    let n101 = lattice_hash(cell.x + 1, cell.y, cell.z + 1, seed);
    let n011 = lattice_hash(cell.x, cell.y + 1, cell.z + 1, seed);
    let n111 = lattice_hash(cell.x + 1, cell.y + 1, cell.z + 1, seed);

    let nx00 = lerp(n000, n100, w.x);
    let nx10 = lerp(n010, n110, w.x);
    let nx01 = lerp(n001, n101, w.x);
    let nx11 = lerp(n011, n111, w.x);
    let nxy0 = lerp(nx00, nx10, w.y);
    let nxy1 = lerp(nx01, nx11, w.y);
    lerp(nxy0, nxy1, w.z)
}

fn lattice_hash(x: i32, y: i32, z: i32, seed: u32) -> f32 {
    let mut value = seed
        ^ (x as u32).wrapping_mul(0x9E37_79B9)
        ^ (y as u32).wrapping_mul(0x85EB_CA6B)
        ^ (z as u32).wrapping_mul(0xC2B2_AE35);
    value ^= value >> 16;
    value = value.wrapping_mul(0x7FEB_352D);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846C_A68B);
    value ^= value >> 16;
    (value as f32) / (u32::MAX as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_context() -> TerrainVolumeContext<'static> {
        static HEIGHTS: [f32; 64] = [0.0; 64];
        TerrainVolumeContext {
            heightmap: &HEIGHTS,
            height_dims: (8, 8),
            terrain_width: 8.0,
            domain: (0.0, 1.0),
            z_scale: 1.0,
            terrain_revision: 7,
        }
    }

    fn config(preset: &str) -> DensityVolumeConfig {
        DensityVolumeConfig {
            preset: preset.to_string(),
            center: [4.0, 2.0, 4.0],
            size: [6.0, 4.0, 6.0],
            resolution: [16, 12, 16],
            density_scale: 1.0,
            edge_softness: 0.25,
            noise_strength: 0.3,
            floor_offset: 0.0,
            ceiling: 0.4,
            plume_spread: 0.35,
            wind: [0.2, 1.0, 0.1],
            seed: 13,
        }
    }

    #[test]
    fn build_density_volume_data_is_deterministic() {
        let a = build_density_volume_atlas_data(flat_context(), &[config("valley_fog")]).unwrap();
        let b = build_density_volume_atlas_data(flat_context(), &[config("valley_fog")]).unwrap();
        assert_eq!(a.dimensions, b.dimensions);
        assert_eq!(a.fingerprint, b.fingerprint);
        assert_eq!(a.voxels, b.voxels);
    }

    #[test]
    fn density_volume_report_tracks_budget_and_truncation() {
        let configs = vec![
            config("valley_fog"),
            config("plume"),
            config("localized_haze"),
            config("valley_fog"),
            config("plume"),
        ];
        let atlas = build_density_volume_atlas_data(flat_context(), &configs).unwrap();
        assert_eq!(atlas.report.active_volume_count, MAX_DENSITY_VOLUMES as u32);
        assert!(atlas.report.texture_bytes <= DENSITY_VOLUME_MEMORY_BUDGET_BYTES);
        assert_eq!(atlas.report.volumes.len(), MAX_DENSITY_VOLUMES);
    }

    #[test]
    fn valley_fog_prefers_ground_layer() {
        let atlas =
            build_density_volume_atlas_data(flat_context(), &[config("valley_fog")]).unwrap();
        let dims = atlas.dimensions;
        let low = atlas.voxels[((1 * dims[1] + 1) * dims[0] + 1) as usize];
        let high = atlas.voxels
            [(((dims[2] - 2) * dims[1] + (dims[1] - 2)) * dims[0] + (dims[0] - 2)) as usize];
        assert!(low > high);
    }
}
