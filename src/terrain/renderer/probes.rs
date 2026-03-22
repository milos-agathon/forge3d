use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use wgpu::util::DeviceExt;

use super::*;
use crate::terrain::probes::{
    pack_probes_for_upload, GpuProbeData, HeightfieldAnalyticalBaker, ProbeBaker, ProbeGridDesc,
    ProbeGridUniformsGpu, ProbePlacement,
};
use crate::terrain::render_params::ProbeSettingsNative;

fn hash_probe_bake_inputs(
    settings: &ProbeSettingsNative,
    terrain_span: f32,
    z_scale: f32,
    heightfield: &[f32],
    height_dims: (u32, u32),
) -> u64 {
    let mut hasher = DefaultHasher::new();
    terrain_span.to_bits().hash(&mut hasher);
    z_scale.to_bits().hash(&mut hasher);
    settings.grid_dims.hash(&mut hasher);
    settings
        .origin
        .map(|(x, y)| (x.to_bits(), y.to_bits()))
        .hash(&mut hasher);
    settings
        .spacing
        .map(|(x, y)| (x.to_bits(), y.to_bits()))
        .hash(&mut hasher);
    settings.height_offset.to_bits().hash(&mut hasher);
    settings.ray_count.hash(&mut hasher);
    settings
        .sky_color
        .map(f32::to_bits)
        .into_iter()
        .for_each(|bits| bits.hash(&mut hasher));
    settings.sky_intensity.to_bits().hash(&mut hasher);
    height_dims.hash(&mut hasher);
    for value in heightfield {
        value.to_bits().hash(&mut hasher);
    }
    hasher.finish()
}

fn sample_height_for_placement(
    heightfield: &[f32],
    height_dims: (u32, u32),
    terrain_span: f32,
    world_x: f32,
    world_y: f32,
) -> f32 {
    let (width, height) = height_dims;
    if width == 0 || height == 0 || heightfield.is_empty() {
        return 0.0;
    }
    if width == 1 || height == 1 {
        return heightfield[0]
            .is_finite()
            .then_some(heightfield[0])
            .unwrap_or(0.0);
    }

    let u = ((world_x / terrain_span) + 0.5).clamp(0.0, 1.0);
    let v = ((world_y / terrain_span) + 0.5).clamp(0.0, 1.0);
    let fx = u * (width - 1) as f32;
    let fy = v * (height - 1) as f32;
    let x0 = fx.floor() as u32;
    let y0 = fy.floor() as u32;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);
    let tx = fx - x0 as f32;
    let ty = fy - y0 as f32;

    let sample = |x: u32, y: u32| {
        let value = heightfield[(y * width + x) as usize];
        value.is_finite().then_some(value)
    };
    let samples = [
        ((1.0 - tx) * (1.0 - ty), sample(x0, y0)),
        (tx * (1.0 - ty), sample(x1, y0)),
        ((1.0 - tx) * ty, sample(x0, y1)),
        (tx * ty, sample(x1, y1)),
    ];

    let mut sum = 0.0;
    let mut weight = 0.0;
    for (wgt, value) in samples {
        if let Some(value) = value {
            sum += value * wgt;
            weight += wgt;
        }
    }
    if weight > 0.0 {
        sum / weight
    } else {
        0.0
    }
}

pub(super) fn resolve_placement(
    settings: &ProbeSettingsNative,
    terrain_span: f32,
    heightfield: &[f32],
    height_dims: (u32, u32),
    z_scale: f32,
) -> ProbePlacement {
    let cols = settings.grid_dims.0.max(1);
    let rows = settings.grid_dims.1.max(1);
    let half_span = terrain_span * 0.5;

    let auto_spacing_x = if cols > 1 {
        terrain_span / (cols - 1) as f32
    } else {
        terrain_span
    };
    let auto_spacing_y = if rows > 1 {
        terrain_span / (rows - 1) as f32
    } else {
        terrain_span
    };
    let auto_origin_x = if cols > 1 { -half_span } else { 0.0 };
    let auto_origin_y = if rows > 1 { -half_span } else { 0.0 };

    let origin = settings.origin.unwrap_or((auto_origin_x, auto_origin_y));
    let spacing = settings.spacing.unwrap_or((auto_spacing_x, auto_spacing_y));

    let grid = ProbeGridDesc {
        origin: [origin.0, origin.1],
        spacing: [spacing.0, spacing.1],
        dims: [cols, rows],
        height_offset: settings.height_offset,
        influence_radius: 0.0,
    };

    let positions_ws = (0..rows)
        .flat_map(|row| {
            (0..cols).map(move |col| {
                let wx = grid.origin[0] + grid.spacing[0] * col as f32;
                let wy = grid.origin[1] + grid.spacing[1] * row as f32;
                let wz =
                    sample_height_for_placement(heightfield, height_dims, terrain_span, wx, wy)
                        * z_scale
                        + settings.height_offset;
                [wx, wy, wz]
            })
        })
        .collect();

    ProbePlacement::new(grid, positions_ws)
}

impl TerrainScene {
    pub(super) fn upload_probe_data(
        &mut self,
        grid_uniforms: &ProbeGridUniformsGpu,
        probe_data: &[GpuProbeData],
        active_probe_count: usize,
    ) {
        let required_bytes = (probe_data.len() * std::mem::size_of::<GpuProbeData>()) as u64;
        if self.probe_ssbo_alloc_bytes != required_bytes {
            let tracker = crate::core::memory_tracker::global_tracker();
            if self.probe_ssbo_alloc_bytes > 0 {
                tracker.free_buffer_allocation(self.probe_ssbo_alloc_bytes, false);
            }
            self.probe_ssbo = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("terrain.probes.ssbo"),
                    contents: bytemuck::cast_slice(probe_data),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                });
            tracker.track_buffer_allocation(required_bytes, false);
            self.probe_ssbo_alloc_bytes = required_bytes;
        } else {
            self.queue
                .write_buffer(&self.probe_ssbo, 0, bytemuck::cast_slice(probe_data));
        }

        self.queue.write_buffer(
            &self.probe_grid_uniform_buffer,
            0,
            bytemuck::bytes_of(grid_uniforms),
        );

        self.probe_grid_uniform_bytes = if active_probe_count > 0 {
            std::mem::size_of::<ProbeGridUniformsGpu>() as u64
        } else {
            0
        };
        self.probe_ssbo_bytes = (active_probe_count * std::mem::size_of::<GpuProbeData>()) as u64;
    }
}

pub(super) fn prepare_probes(
    scene: &mut TerrainScene,
    settings: &ProbeSettingsNative,
    terrain_span: f32,
    heightfield: &[f32],
    height_dims: (u32, u32),
    z_scale: f32,
) {
    if !settings.enabled {
        scene.upload_probe_data(
            &ProbeGridUniformsGpu::disabled(),
            &[GpuProbeData::zeroed()],
            0,
        );
        return;
    }

    let bake_key =
        hash_probe_bake_inputs(settings, terrain_span, z_scale, heightfield, height_dims);
    if scene.probe_cache_key != Some(bake_key)
        || scene.probe_cached_grid.is_none()
        || scene.probe_cached_data.is_empty()
    {
        let placement =
            resolve_placement(settings, terrain_span, heightfield, height_dims, z_scale);
        let scaled_heightfield = heightfield
            .iter()
            .map(|value| {
                if value.is_finite() {
                    *value * z_scale
                } else {
                    *value
                }
            })
            .collect();
        let baker = HeightfieldAnalyticalBaker {
            heightfield: scaled_heightfield,
            height_dims,
            terrain_span: [terrain_span, terrain_span],
            sky_color: settings.sky_color,
            sky_intensity: settings.sky_intensity,
            ray_count: settings.ray_count.max(1),
            max_trace_distance: terrain_span,
        };
        let irradiance = baker
            .bake(&placement)
            .expect("HeightfieldAnalyticalBaker should be infallible");
        scene.probe_cache_key = Some(bake_key);
        scene.probe_cached_grid = Some(placement.grid.clone());
        scene.probe_cached_data = pack_probes_for_upload(&irradiance);
    }

    let grid = scene
        .probe_cached_grid
        .clone()
        .expect("probe cache grid should be populated");
    let probe_count = scene.probe_cached_data.len();
    let blend_distance = settings
        .fallback_blend_distance
        .unwrap_or(grid.spacing[0].min(grid.spacing[1]) * 2.0);
    let uniforms = ProbeGridUniformsGpu {
        grid_origin: [grid.origin[0], grid.origin[1], grid.height_offset, 1.0],
        grid_params: [
            grid.spacing[0],
            grid.spacing[1],
            grid.dims[0] as f32,
            grid.dims[1] as f32,
        ],
        blend_params: [blend_distance, probe_count as f32, 0.0, 0.0],
    };
    let gpu_data = scene.probe_cached_data.clone();
    scene.upload_probe_data(&uniforms, &gpu_data, probe_count);
}
