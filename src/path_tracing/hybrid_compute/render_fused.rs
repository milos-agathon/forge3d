// src/path_tracing/hybrid_compute/render_fused.rs
// SPLAT-FUSED accumulation driver: renders Gaussian splats, LiDAR/COPC points
// and a terrain heightfield through the hybrid tracer's `main_terrain` ReSTIR
// kernel (the fused specialization assembled by
// shader_sources::fused_kernel). It is the same integrator and the same
// reuse passes as the terrain reference — this file only adds the fused scene
// bindings and the out-of-core paging service between frames.
//
// Per frame: `main_terrain` (camera samples, visibility-aware ReSTIR
// candidates), pt_restir_temporal, pt_restir_spatial, publish, then the page
// table feedback is serviced. Under the exact paging policy a frame that
// touched a non-resident page restarts the accumulation once the page is in;
// under the progressive policy the kernel shades from the reservoir's
// last-known visibility and the accumulation continues.
// RELEVANT FILES: src/splat/fusion.rs, src/shaders/fusion/unified_occlusion.wgsl,
//                 src/path_tracing/hybrid_compute/render_terrain.rs

use std::sync::Arc;

use super::render_terrain::TerrainStatistics;
use super::terrain_heightfield::{
    build_minmax_mips, AlbedoSampling, EarthCurvatureUniforms, MinMaxPrecision,
    TerrainAlbedoMap, TerrainPtScene,
};
use super::*;
use crate::core::memory_tracker::global_tracker;
use crate::core::resource_tracker::{AllocationOwner, TrackedBuffer};
use crate::path_tracing::lighting::{GpuAreaLight, GpuDirectionalLight};
use crate::path_tracing::restir::{
    create_reservoir_buffer, create_restir_gbuffer, create_restir_gbuffer_pos, Reservoir,
};
use crate::splat::bvh::Aabb;
use crate::splat::fusion::{FusedGpu, FusedScene, PagingPolicy, PagingStats};
use crate::splat::num::f32_from_u32;
use crate::viewer::viewer_types::SkyUniforms;

/// Terrain heightfield input of a fused render.
#[derive(Clone, Debug, PartialEq)]
pub struct FusedTerrainDesc {
    /// Row-major heights, `width * height` samples.
    pub heights: Vec<f32>,
    pub width: u32,
    pub height: u32,
    /// World spacing between samples along x and z.
    pub spacing: (f32, f32),
    pub exaggeration: f32,
    pub albedo: [f32; 3],
    /// Storage precision of the min-max pyramid (`F16Conservative` halves
    /// terrain memory without changing any output bit).
    pub minmax_precision: MinMaxPrecision,
    /// Optional colour map: RGBA8 sRGB, `width * height * 4` bytes,
    /// row-major, row 0 = first heights row. Alpha 255 = mapped; any other
    /// alpha uses `albedo`.
    pub albedo_map: Option<Vec<u8>>,
    pub albedo_sampling: AlbedoSampling,
}

/// Full description of a fused render.
#[derive(Clone)]
pub struct FusedRenderDesc<'a> {
    pub scene: &'a FusedScene,
    /// Shared, so the views of a sequence reference one heightfield (and
    /// colour map) instead of each holding a copy.
    pub terrain: Option<Arc<FusedTerrainDesc>>,
    pub cam_origin: [f32; 3],
    pub cam_look_at: [f32; 3],
    pub cam_up: [f32; 3],
    pub fov_y_deg: f32,
    pub exposure: f32,
    pub sun_azimuth_deg: f32,
    pub sun_elevation_deg: f32,
    pub sun_intensity: f32,
    pub sun_color: [f32; 3],
    /// Hosek-Wilkie sky: turbidity and ground albedo of the baked environment.
    pub sky_turbidity: f32,
    pub sky_ground_albedo: f32,
    /// Scale of the (irradiance-normalised) sky environment.
    pub sky_intensity: f32,
    pub width: u32,
    pub height: u32,
    pub seed: u32,
    /// Camera samples per accumulation frame.
    pub spp: u32,
    /// Accumulation frames.
    pub frames: u32,
    /// Seamless tile size (pixels); `None` uses `default_tile`. Every tile
    /// renders exactly the pixels a monolithic render would.
    pub tile: Option<(u32, u32)>,
    /// Read back the AOVs (albedo, normal, position, transmittance, hit
    /// kind, ...). `false` frees their per-pixel GPU memory; the beauty is
    /// unchanged.
    pub aovs: bool,
}

/// Output of a fused render.
#[derive(Clone, Debug)]
pub struct FusedRenderOutput {
    pub width: u32,
    pub height: u32,
    /// Tonemapped beauty, RGBA8.
    pub rgba: Vec<u8>,
    /// Mean linear radiance, RGB per pixel.
    pub radiance: Vec<f32>,
    pub albedo: Vec<f32>,
    /// World normal of the centre-ray hit (ReSTIR G-buffer, full f32).
    pub normal: Vec<f32>,
    /// Hit distance of the centre ray (NaN on a miss).
    pub depth: Vec<f32>,
    /// Sun radiance reaching the centre-ray surface, RGB per pixel.
    pub direct: Vec<f32>,
    /// Unified occlusion toward the sun at the centre-ray hit, four values
    /// per pixel: (T_total, T_splat, T_lidar, T_terrain).
    pub transmittance: Vec<f32>,
    /// 0 = miss, 3 = terrain, 4 = splat, 5 = LiDAR point.
    pub hit_kind: Vec<u8>,
    /// N.L at the centre-ray hit (negative when facing away from the sun).
    pub sun_cosine: Vec<f32>,
    /// Self-shadow bias applied to secondary rays leaving the centre-ray hit
    /// (zero on terrain).
    pub self_bias: Vec<f32>,
    /// World position of the centre-ray hit (ReSTIR G-buffer), XYZ per pixel.
    pub position: Vec<f32>,
    /// Last-known sun visibility carried by each pixel's merged reservoir
    /// (-1 where the reservoir is empty).
    pub reservoir_visibility: Vec<f32>,
    pub reservoir_valid_count: u64,
    pub frames: u32,
    /// Accumulation restarts forced by page misses (exact policy).
    pub restarts: u32,
    /// Frames accumulated while some requested page was not yet resident
    /// (progressive policy).
    pub stale_frames: u32,
    /// Maximum per-pixel estimated variance of the mean frame luminance.
    pub variance: f32,
    /// Seamless tiles the view was rendered in.
    pub tiles: u32,
    pub logical_primitives: u64,
    pub page_count: u32,
    pub tlas_node_count: u32,
    pub paging: PagingStats,
    /// Tracked GPU bytes of the terrain scene (heights, min-max pyramid,
    /// environment, albedo map).
    pub terrain_bytes: u64,
    /// Whether the terrain min-max pyramid was stored as half floats.
    pub terrain_minmax_f16: bool,
    /// Peaks of every allocation this render made (ledger owner capture).
    pub peak_host_visible_bytes: u64,
    pub peak_device_local_bytes: u64,
    pub peak_total_bytes: u64,
    /// Process-wide tracker state after the render.
    pub tracker_host_visible_bytes: u64,
    pub tracker_peak_host_visible_bytes: u64,
    pub tracker_limit_bytes: u64,
}

const ENV_WIDTH: u32 = 256;
const ENV_HEIGHT: u32 = 128;
const MAX_RESTARTS: u32 = 256;
/// Terrain tiles per axis in the top-level structure (upper bound).
const TERRAIN_TILE_GRID: u32 = 8;

fn validate(desc: &FusedRenderDesc) -> Result<(), RenderError> {
    let fail = |message: String| Err(RenderError::Render(message));
    if desc.width == 0 || desc.height == 0 || desc.frames == 0 {
        return fail("fused render requires non-zero width, height and frames".into());
    }
    if desc.spp == 0 || desc.spp > 64 {
        return fail(format!("spp must be in 1..=64, got {}", desc.spp));
    }
    if let Some((tw, th)) = desc.tile {
        if tw == 0 || th == 0 || tw > desc.width || th > desc.height {
            return fail(format!(
                "tile {tw}x{th} must be at least 1x1 and fit the {}x{} image",
                desc.width, desc.height
            ));
        }
    }
    if desc.scene.page_count() == 0 && desc.terrain.is_none() {
        return fail(
            "fused render has nothing to trace: no splat pages, no point pages and no terrain"
                .into(),
        );
    }
    let finite3 = |v: [f32; 3]| v.iter().all(|c| c.is_finite());
    if !(finite3(desc.cam_origin) && finite3(desc.cam_look_at) && finite3(desc.cam_up)) {
        return fail("camera origin/look_at/up must be finite".into());
    }
    let forward = glam::Vec3::from(desc.cam_look_at) - glam::Vec3::from(desc.cam_origin);
    if forward.length() < 1e-6
        || forward.normalize().cross(glam::Vec3::from(desc.cam_up)).length() < 1e-6
    {
        return fail("camera look_at must differ from origin and not be parallel to up".into());
    }
    if !(desc.fov_y_deg.is_finite() && desc.fov_y_deg > 0.0 && desc.fov_y_deg < 180.0) {
        return fail(format!(
            "fov_y must be in (0, 180) degrees, got {}",
            desc.fov_y_deg
        ));
    }
    if !(desc.exposure.is_finite() && desc.exposure > 0.0) {
        return fail("exposure must be finite and > 0".into());
    }
    if !(desc.sun_azimuth_deg.is_finite() && desc.sun_elevation_deg.is_finite()) {
        return fail("sun azimuth/elevation must be finite".into());
    }
    if !(desc.sun_intensity.is_finite() && desc.sun_intensity >= 0.0)
        || !finite3(desc.sun_color)
        || desc.sun_color.iter().any(|c| *c < 0.0)
    {
        return fail("sun intensity and colour must be finite and non-negative".into());
    }
    if !(1.0..=10.0).contains(&desc.sky_turbidity)
        || !(0.0..=1.0).contains(&desc.sky_ground_albedo)
        || !(desc.sky_intensity.is_finite() && desc.sky_intensity >= 0.0)
    {
        return fail(
            "sky turbidity must be in [1, 10], ground albedo in [0, 1], intensity finite >= 0"
                .into(),
        );
    }
    if let Some(terrain) = &desc.terrain {
        if !(terrain.exaggeration.is_finite() && terrain.exaggeration > 0.0)
            || !(terrain.spacing.0.is_finite() && terrain.spacing.0 > 0.0)
            || !(terrain.spacing.1.is_finite() && terrain.spacing.1 > 0.0)
        {
            return fail("terrain spacing and exaggeration must be finite and > 0".into());
        }
        if !finite3(terrain.albedo) || terrain.albedo.iter().any(|c| *c < 0.0) {
            return fail("terrain albedo must be finite and non-negative".into());
        }
    }
    Ok(())
}

/// Hosek-Wilkie sky radiance for one direction, evaluated from the same
/// `SkyParams` coefficient block sky.wgsl consumes (CPU port of
/// `hosek_wilkie_eval_channel`).
fn hosek_radiance(sky: &SkyUniforms, dir: glam::Vec3) -> [f32; 3] {
    let sun = glam::Vec3::new(
        sky.sun_direction_turbidity[0],
        sky.sun_direction_turbidity[1],
        sky.sun_direction_turbidity[2],
    );
    let cos_theta = dir.y.max(0.0);
    let cos_gamma = dir.dot(sun).clamp(-1.0, 1.0);
    let gamma = cos_gamma.acos();
    let ray_m = cos_gamma * cos_gamma;
    let zenith = cos_theta.sqrt();
    let mut out = [0.0f32; 3];
    for (channel, value) in out.iter_mut().enumerate() {
        let [a, b, c, d] = sky.hosek_coeffs_a_d[channel];
        let [e, f, g, h] = sky.hosek_coeffs_e_h[channel];
        let i = sky.hosek_coeff_i[channel];
        let mie_denom = (1.0 + i * i - 2.0 * i * cos_gamma).max(1e-4);
        let mie_m = (1.0 + ray_m) / mie_denom.powf(1.5);
        *value = (sky.hosek_radiance[channel]
            * (1.0 + a * (b / (cos_theta + 0.01)).exp())
            * (c + d * (e * gamma).exp() + f * ray_m + g * mie_m + h * zenith))
            .max(0.0);
    }
    out
}

/// Bake the Hosek-Wilkie sky into the equirect environment the kernel samples
/// (`terrain_env_radiance`: u = atan2(z, x) / 2pi + 0.5, v = acos(y) / pi).
/// The map is normalised so its cosine-weighted mean over the upper
/// hemisphere is 1; `sky_intensity` then scales it like the constant-white
/// environment of the terrain reference. Below the horizon the ground
/// reflects the horizon radiance by `ground_albedo`.
fn bake_sky_environment(sun_dir: [f32; 3], turbidity: f32, ground_albedo: f32) -> Vec<f32> {
    let sky = SkyUniforms::new(sun_dir, turbidity, ground_albedo, 0.0, 1.0, 1.0, 1);
    let mut pixels = Vec::with_capacity((ENV_WIDTH * ENV_HEIGHT * 3) as usize);
    let mut weighted = [0.0f64; 3];
    let mut weight = 0.0f64;
    for py in 0..ENV_HEIGHT {
        let theta = (f32_from_u32(py) + 0.5) / f32_from_u32(ENV_HEIGHT) * std::f32::consts::PI;
        let (sin_t, cos_t) = theta.sin_cos();
        for px in 0..ENV_WIDTH {
            let phi = ((f32_from_u32(px) + 0.5) / f32_from_u32(ENV_WIDTH) - 0.5)
                * std::f32::consts::TAU;
            let dir = glam::Vec3::new(sin_t * phi.cos(), cos_t, sin_t * phi.sin());
            let mut radiance = hosek_radiance(&sky, glam::Vec3::new(dir.x, dir.y.max(0.0), dir.z).normalize());
            if dir.y < 0.0 {
                radiance = radiance.map(|c| c * ground_albedo);
            } else {
                // Cosine-weighted solid angle of this texel.
                let w = f64::from(cos_t * sin_t);
                for channel in 0..3 {
                    weighted[channel] += w * f64::from(radiance[channel]);
                }
                weight += w;
            }
            pixels.extend_from_slice(&radiance);
        }
    }
    // Normalise on luminance so the sky keeps its colour.
    let mean_luminance =
        (0.2126 * weighted[0] + 0.7152 * weighted[1] + 0.0722 * weighted[2]) / weight.max(1e-12);
    let scale = if mean_luminance > 0.0 {
        1.0 / mean_luminance
    } else {
        0.0
    };
    for value in &mut pixels {
        *value = (f64::from(*value) * scale).min(65_504.0) as f32;
    }
    pixels
}

/// Terrain tiles for the top-level structure: the boxes of a coarse level of
/// the heightfield's min-max pyramid (at most `TERRAIN_TILE_GRID` per axis),
/// in world space.
fn terrain_tiles(terrain: &FusedTerrainDesc) -> Result<Vec<Aabb>, RenderError> {
    let mips = build_minmax_mips(&terrain.heights, terrain.width, terrain.height)?;
    let level = (0..mips.dims.len())
        .find(|&l| mips.dims[l].0 <= TERRAIN_TILE_GRID && mips.dims[l].1 <= TERRAIN_TILE_GRID)
        .unwrap_or(mips.dims.len() - 1);
    let (lw, lh) = mips.dims[level];
    let (sx, sz) = terrain.spacing;
    let ox = -0.5 * (f32_from_u32(terrain.width) - 1.0) * sx;
    let oz = -0.5 * (f32_from_u32(terrain.height) - 1.0) * sz;
    let mut tiles = Vec::new();
    for ty in 0..lh {
        for tx in 0..lw {
            let [lo, hi] = mips.levels[level][(ty * lw + tx) as usize];
            if !(lo.is_finite() && hi.is_finite()) {
                continue; // padding sentinel: no cells under this node
            }
            let cx0 = (tx << level).min(mips.cell_w);
            let cx1 = ((tx + 1) << level).min(mips.cell_w);
            let cz0 = (ty << level).min(mips.cell_h);
            let cz1 = ((ty + 1) << level).min(mips.cell_h);
            if cx0 == cx1 || cz0 == cz1 {
                continue;
            }
            // A hair of vertical slack keeps perfectly flat tiles from
            // degenerating into zero-thickness boxes.
            let pad = 1e-3 * (1.0 + lo.abs().max(hi.abs())) * terrain.exaggeration;
            tiles.push(Aabb::new(
                [
                    ox + f32_from_u32(cx0) * sx,
                    lo * terrain.exaggeration - pad,
                    oz + f32_from_u32(cz0) * sz,
                ],
                [
                    ox + f32_from_u32(cx1) * sx,
                    hi * terrain.exaggeration + pad,
                    oz + f32_from_u32(cz1) * sz,
                ],
            ));
        }
    }
    Ok(tiles)
}

fn read_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    size: u64,
) -> Result<Vec<u8>, RenderError> {
    let staging = tracked_create_buffer(
        device,
        &wgpu::BufferDescriptor {
            label: Some("hybrid-pt-fused-buf-readback"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        },
    )?;
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("hybrid-pt-fused-buf-enc"),
    });
    enc.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
    queue.submit([enc.finish()]);
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|_| RenderError::Readback("fused readback channel closed".into()))?
        .map_err(|e| RenderError::Readback(format!("fused readback map failed: {e:?}")))?;
    let out = slice.get_mapped_range().to_vec();
    staging.unmap();
    Ok(out)
}

fn read_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    width: u32,
    height: u32,
    bytes_per_pixel: u32,
) -> Result<Vec<u8>, RenderError> {
    let unpadded = width * bytes_per_pixel;
    let padded = align_copy_bpr(unpadded);
    let staging = tracked_create_buffer(
        device,
        &wgpu::BufferDescriptor {
            label: Some("hybrid-pt-fused-tex-readback"),
            size: u64::from(padded) * u64::from(height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        },
    )?;
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("hybrid-pt-fused-tex-enc"),
    });
    enc.copy_texture_to_buffer(
        wgpu::ImageCopyTexture {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::ImageCopyBuffer {
            buffer: &staging,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([enc.finish()]);
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|_| RenderError::Readback("fused readback channel closed".into()))?
        .map_err(|e| RenderError::Readback(format!("fused readback map failed: {e:?}")))?;
    let out = {
        let data = slice.get_mapped_range();
        let mut rows = Vec::with_capacity((unpadded * height) as usize);
        for y in 0..height as usize {
            let start = y * padded as usize;
            rows.extend_from_slice(&data[start..start + unpadded as usize]);
        }
        rows
    };
    staging.unmap();
    Ok(out)
}

/// Decode an Rgba16Float readback into `channels` f32 values per pixel.
fn decode_f16(bytes: &[u8], channels: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(bytes.len() / 8 * channels);
    for px in bytes.chunks_exact(8) {
        for c in 0..channels {
            out.push(f16::from_bits(u16::from_le_bytes([px[c * 2], px[c * 2 + 1]])).to_f32());
        }
    }
    out
}

impl HybridPathTracer {
    /// Build the hybrid tracer from the fused (SPLAT-FUSED) kernel: the same
    /// pipelines as `new`, compiled from the specialized source and bound to
    /// the fused scene layout.
    pub fn new_fused() -> Result<Self, RenderError> {
        let device = &try_ctx()?.device;
        Self::new_with_source_and_scene(
            crate::shader_sources::fused_kernel()?,
            Self::create_fused_scene_layout(device),
            true,
        )
    }

    /// Render the fused scene. `self` must come from `new_fused`; a tracer
    /// built from the default kernel has no fused bindings and is rejected.
    pub fn render_fused(&self, desc: &FusedRenderDesc) -> Result<FusedRenderOutput, RenderError> {
        let mut outputs = self.render_fused_sequence(std::slice::from_ref(desc))?;
        Ok(outputs.remove(0))
    }

    /// Render several views of one fused scene (a fly-through) through a
    /// single residency pool: pages streamed in for one view stay resident
    /// for the next and are evicted least-recently-used once the pool is
    /// full. Every view must reference the same scene and terrain. Paging
    /// statistics are cumulative; the memory peaks cover the whole sequence.
    pub fn render_fused_sequence(
        &self,
        descs: &[FusedRenderDesc],
    ) -> Result<Vec<FusedRenderOutput>, RenderError> {
        if !self.fused {
            return Err(RenderError::Render(
                "render_fused requires a tracer built with HybridPathTracer::new_fused(); this \
                 one was compiled from the default hybrid kernel"
                    .into(),
            ));
        }
        let Some(first) = descs.first() else {
            return Err(RenderError::Render(
                "render_fused_sequence requires at least one view".into(),
            ));
        };
        for desc in descs {
            validate(desc)?;
            // The terrain scene is uploaded once from the first view, so
            // every field (heights, colour map, pyramid precision) must match.
            let same_terrain = match (&desc.terrain, &first.terrain) {
                (None, None) => true,
                (Some(a), Some(b)) => Arc::ptr_eq(a, b) || **a == **b,
                _ => false,
            };
            if !std::ptr::eq(desc.scene, first.scene) || !same_terrain {
                return Err(RenderError::Render(
                    "every view of a fused sequence must share one scene and one terrain; \
                     render views of different scenes separately"
                        .into(),
                ));
            }
        }
        let device = &try_ctx()?.device;
        let queue = &try_ctx()?.queue;
        let owner = AllocationOwner::new();
        let _owner_guard = owner.activate();
        let capture = crate::core::resource_tracker::begin_owner_capture(owner.id());

        // --- Fused scene: top-level BVH, residency pool, page table. The
        // terrain tiles come from the heightfield alone, so one structure
        // serves every view.
        let tiles = match &first.terrain {
            Some(terrain) => terrain_tiles(terrain)?,
            None => Vec::new(),
        };
        let mut fused = FusedGpu::new(
            device,
            queue,
            first.scene,
            &tiles,
            Some(owner.clone()),
        )?;
        // --- Terrain scene: heights, min-max pyramid and colour map are
        // uploaded once for the whole sequence; each view only swaps in its
        // own sky environment. A render without terrain still needs the
        // environment the kernel samples, so it binds a flat placeholder DEM
        // with the terrain flag cleared: nothing is traced against it and no
        // tile enters the top-level structure.
        let placeholder = FusedTerrainDesc {
            heights: vec![0.0; 4],
            width: 2,
            height: 2,
            spacing: (1.0, 1.0),
            exaggeration: 1.0,
            albedo: [0.0; 3],
            minmax_precision: MinMaxPrecision::F32,
            albedo_map: None,
            albedo_sampling: AlbedoSampling::Nearest,
        };
        let terrain_desc = first.terrain.as_deref().unwrap_or(&placeholder);
        let mut terrain_scene = TerrainPtScene::new_with_options(
            device,
            queue,
            &terrain_desc.heights,
            terrain_desc.width,
            terrain_desc.height,
            terrain_desc.spacing,
            terrain_desc.exaggeration,
            terrain_desc.albedo,
            None,
            first.sky_intensity,
            match &terrain_desc.albedo_map {
                Some(map) => TerrainAlbedoMap::Rgba8Srgb(map),
                None => TerrainAlbedoMap::None,
            },
            terrain_desc.albedo_sampling,
            // The baked sky already carries the turbidity; the kernel's own
            // aerosol term stays at its clean-air identity.
            1.0,
            terrain_desc.minmax_precision,
        )?;

        // Monotone service tick: the residency pool's notion of "frame".
        let mut tick = 0u64;
        let mut outputs = Vec::with_capacity(descs.len());
        for desc in descs {
            outputs.push(self.render_fused_view(
                desc,
                &mut fused,
                &mut terrain_scene,
                &mut tick,
            )?);
        }
        drop(terrain_scene);
        drop(fused);

        let report = capture.finish();
        let metrics = global_tracker().get_metrics();
        for out in &mut outputs {
            out.peak_host_visible_bytes = report.peak_host_visible_bytes;
            out.peak_device_local_bytes = report.peak_device_local_bytes;
            out.peak_total_bytes = report.peak_total_bytes;
            out.tracker_host_visible_bytes = metrics.host_visible_bytes;
            out.tracker_peak_host_visible_bytes = metrics.peak_host_visible_bytes;
            out.tracker_limit_bytes = metrics.limit_bytes;
        }
        if report.peak_total_bytes > metrics.limit_bytes
            || metrics.peak_host_visible_bytes > metrics.limit_bytes
        {
            return Err(RenderError::Budget(format!(
                "fused render exceeded the {} byte budget: render peak {} (host-visible {}), \
                 process host-visible peak {}",
                metrics.limit_bytes,
                report.peak_total_bytes,
                report.peak_host_visible_bytes,
                metrics.peak_host_visible_bytes
            )));
        }
        Ok(outputs)
    }

    fn render_fused_view(
        &self,
        desc: &FusedRenderDesc,
        fused: &mut FusedGpu,
        terrain_scene: &mut TerrainPtScene,
        tick: &mut u64,
    ) -> Result<FusedRenderOutput, RenderError> {
        let device = &try_ctx()?.device;
        let queue = &try_ctx()?.queue;
        let (width, height) = (desc.width, desc.height);
        fused.begin_view(queue);

        // --- Sun + Hosek-Wilkie sky environment ---
        let az = desc.sun_azimuth_deg.to_radians();
        let el = desc.sun_elevation_deg.to_radians();
        // Direction from the surface TOWARD the sun (kernel convention).
        let light_dir = [az.cos() * el.cos(), el.sin(), az.sin() * el.cos()];
        let env = bake_sky_environment(light_dir, desc.sky_turbidity, desc.sky_ground_albedo);

        // --- Terrain: the sequence's shared heightfield scene with this
        // view's sky environment swapped in.
        terrain_scene.set_environment(
            device,
            queue,
            Some((env.as_slice(), ENV_WIDTH, ENV_HEIGHT)),
            desc.sky_intensity,
        )?;
        let terrain_bytes = terrain_scene.byte_size();
        let terrain_minmax_f16 = desc.terrain.is_some() && terrain_scene.minmax_is_f16();
        drop(env);
        let mut terrain_uniforms = terrain_scene.uniforms(desc.spp, 1);
        if desc.terrain.is_none() {
            terrain_uniforms.mips[1] &= !1;
        }
        // A mapped terrain keeps its albedo map (bit 1) but not the base
        // kernel's spectral residual reservoir (bit 2): that candidate path
        // is not routed through `shadow_transmittance`, so the fused
        // integrator always uses its visibility-aware sun candidate.
        terrain_uniforms.mips[1] &= !4;

        // --- Camera + lighting uniforms. Every fused render is seamless
        // (camera_flags = 1): rays are generated on the full sensor from the
        // global pixel, spatial ReSTIR reuse is self-only, and a tiled render
        // equals the monolithic one bit for bit. Per-tile fields (extent,
        // offset, sensor rectangle) are filled by `render_fused_tile`.
        let origin = glam::Vec3::from(desc.cam_origin);
        let forward = (glam::Vec3::from(desc.cam_look_at) - origin).normalize();
        let right = forward.cross(glam::Vec3::from(desc.cam_up)).normalize();
        let up = right.cross(forward).normalize();
        let base = Uniforms {
            width,
            height,
            frame_index: 0,
            aov_flags: 0xFF,
            cam_origin: desc.cam_origin,
            cam_fov_y: desc.fov_y_deg.to_radians(),
            cam_right: right.into(),
            cam_aspect: f32_from_u32(width) / f32_from_u32(height),
            cam_up: up.into(),
            cam_exposure: desc.exposure,
            cam_forward: forward.into(),
            seed_hi: desc.seed,
            seed_lo: desc
                .seed
                .wrapping_mul(0x9E37_79B9)
                .wrapping_add(0x85EB_CA6B),
            camera_model: 0,
            full_width: width,
            full_height: height,
            pixel_offset_x: 0,
            pixel_offset_y: 0,
            ortho_half_height: 1.0,
            camera_flags: 1,
            sensor_rect: [0.0, 0.0, 1.0, 1.0],
        };
        let uniform = |label: &'static str, bytes: &[u8], extra: wgpu::BufferUsages| {
            tracked_create_buffer_init(
                device,
                &wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: bytes,
                    usage: wgpu::BufferUsages::UNIFORM | extra,
                },
            )
        };
        let base_ubo = uniform(
            "hybrid-pt-fused-base-ubo",
            bytemuck::bytes_of(&base),
            wgpu::BufferUsages::COPY_DST,
        )?;
        // Mesh and SDF lanes are empty; terrain is reached through the fused
        // top-level structure, so the base traversal runs in mesh-only mode.
        let hybrid_uniforms = HybridUniforms {
            sdf_primitive_count: 0,
            sdf_node_count: 0,
            mesh_vertex_count: 0,
            mesh_index_count: 0,
            mesh_bvh_node_count: 0,
            traversal_mode: TraversalMode::MeshOnly as u32,
            _pad: [0; 2],
        };
        let hybrid_ubo = uniform(
            "hybrid-pt-fused-hybrid-ubo",
            bytemuck::bytes_of(&hybrid_uniforms),
            wgpu::BufferUsages::empty(),
        )?;
        let light_color = desc.sun_color.map(|c| c * desc.sun_intensity);
        let lighting = LightingUniforms {
            light_dir,
            lighting_type: 1,
            light_color,
            shadows_enabled: 1,
            ambient_color: [0.0; 3],
            shadow_intensity: 1.0,
            hdri_intensity: desc.sky_intensity,
            hdri_rotation: 0.0,
            specular_power: 32.0,
            _pad: [0; 5],
        };
        let lighting_ubo = uniform(
            "hybrid-pt-fused-lighting-ubo",
            bytemuck::bytes_of(&lighting),
            wgpu::BufferUsages::empty(),
        )?;
        let terrain_ubo = uniform(
            "hybrid-pt-fused-terrain-ubo",
            bytemuck::bytes_of(&terrain_uniforms),
            wgpu::BufferUsages::empty(),
        )?;
        let earth_curvature = <EarthCurvatureUniforms as bytemuck::Zeroable>::zeroed();
        let earth_curvature_ubo = uniform(
            "hybrid-pt-fused-earth-curvature-ubo",
            bytemuck::bytes_of(&earth_curvature),
            wgpu::BufferUsages::empty(),
        )?;

        // --- Scene buffers: the base kernel's sphere and mesh slots are empty ---
        let hybrid_scene = HybridScene::new();
        let scene_buf = tracked_create_buffer(
            device,
            &wgpu::BufferDescriptor {
                label: Some("hybrid-pt-fused-scene"),
                size: std::mem::size_of::<Sphere>() as u64,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            },
        )?;
        let sun_light = GpuDirectionalLight::new(
            [-light_dir[0], -light_dir[1], -light_dir[2]],
            desc.sun_intensity.max(1e-6),
            desc.sun_color,
            1.0,
        );
        let dir_lights_buf = tracked_create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("hybrid-pt-fused-dir-lights"),
                contents: bytemuck::bytes_of(&sun_light),
                usage: wgpu::BufferUsages::STORAGE,
            },
        )?;
        let area_light = <GpuAreaLight as bytemuck::Zeroable>::zeroed();
        let area_lights_buf = tracked_create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("hybrid-pt-fused-area-lights"),
                contents: bytemuck::bytes_of(&area_light),
                usage: wgpu::BufferUsages::STORAGE,
            },
        )?;

        // --- View-level bind groups ---
        let bg0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hybrid-pt-fused-bg0"),
            layout: &self.layouts.uniforms,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: base_ubo.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: lighting_ubo.as_entire_binding(),
                },
            ],
        });
        let mut bg1_entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: scene_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: hybrid_ubo.as_entire_binding(),
            },
        ];
        bg1_entries.extend(
            hybrid_scene
                .get_mesh_bind_entries()?
                .into_iter()
                .enumerate()
                .map(|(i, mut entry)| {
                    entry.binding = (i + 2) as u32;
                    entry
                }),
        );
        bg1_entries.extend(fused.bind_entries());
        let bg1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hybrid-pt-fused-bg1"),
            layout: &self.layouts.scene,
            entries: &bg1_entries,
        });
        drop(bg1_entries);
        let view = |texture: &wgpu::Texture| {
            texture.create_view(&wgpu::TextureViewDescriptor::default())
        };
        let height_view = view(&terrain_scene.pyramid.height_texture);
        let minmax_view = view(&terrain_scene.pyramid.minmax_texture);
        let env_view = view(&terrain_scene.env_texture);
        let albedo_view = view(&terrain_scene.albedo_texture);
        let resources = FusedViewResources {
            base,
            base_ubo,
            bg0,
            bg1,
            height_view,
            minmax_view,
            env_view,
            albedo_view,
            terrain_ubo,
            earth_curvature_ubo,
            dir_lights_buf,
            area_lights_buf,
        };

        // --- Seamless tiles: per-pixel GPU memory is bounded by one tile ---
        let tile = desc.tile.unwrap_or_else(|| default_tile(width, height));
        let rects = fused_tiles(width, height, tile);
        let px_usize = (u64::from(width) * u64::from(height)) as usize;
        let aov_len = |channels: usize| if desc.aovs { px_usize * channels } else { 0 };
        let mut rgba = vec![0u8; px_usize * 4];
        let mut radiance = vec![0.0f32; px_usize * 3];
        let mut albedo = vec![0.0f32; aov_len(3)];
        let mut normal = vec![0.0f32; aov_len(3)];
        let mut depth = vec![0.0f32; aov_len(1)];
        let mut direct = vec![0.0f32; aov_len(3)];
        let mut transmittance = vec![0.0f32; aov_len(4)];
        let mut hit_kind = vec![0u8; aov_len(1)];
        let mut sun_cosine = vec![0.0f32; aov_len(1)];
        let mut self_bias = vec![0.0f32; aov_len(1)];
        let mut position = vec![0.0f32; aov_len(3)];
        let mut reservoir_visibility = vec![0.0f32; aov_len(1)];
        let (mut frames, mut restarts, mut stale_frames) = (0u32, 0u32, 0u32);
        let mut variance = 0.0f32;
        let mut reservoir_valid_count = 0u64;
        // CENSOR F-04: live per-pass timing for the certificate. The first
        // tile's first G-buffer pass and first accumulation frame are each
        // bracketed on the encoder that executes them (one scope per label);
        // without timestamp queries the fallback below records the same
        // labels in the same order with 0.0.
        let mut timing = crate::core::gpu_timing::OneShotTiming::for_current_device();
        for (index, &rect) in rects.iter().enumerate() {
            let tile_out = self.render_fused_tile(
                &resources,
                desc,
                fused,
                tick,
                rect,
                if index == 0 { Some(&mut timing) } else { None },
            )?;
            let (x, y, tw, th) = rect;
            let blit = |dst: &mut [f32], src: &[f32], channels: usize| {
                blit_rows(dst, src, channels, width, (x, y, tw, th));
            };
            blit_rows(&mut rgba, &tile_out.rgba, 4, width, rect);
            blit(&mut radiance, &tile_out.radiance, 3);
            if desc.aovs {
                blit(&mut albedo, &tile_out.albedo, 3);
                blit(&mut normal, &tile_out.normal, 3);
                blit(&mut depth, &tile_out.depth, 1);
                blit(&mut direct, &tile_out.direct, 3);
                blit(&mut transmittance, &tile_out.transmittance, 4);
                blit_rows(&mut hit_kind, &tile_out.hit_kind, 1, width, rect);
                blit(&mut sun_cosine, &tile_out.sun_cosine, 1);
                blit(&mut self_bias, &tile_out.self_bias, 1);
                blit(&mut position, &tile_out.position, 3);
                blit(&mut reservoir_visibility, &tile_out.reservoir_visibility, 1);
            }
            frames = tile_out.frames;
            restarts += tile_out.restarts;
            stale_frames += tile_out.stale_frames;
            let s = tile_out.moment;
            if frames >= 2 {
                let n = f64::from(frames);
                variance = variance.max((f64::from(s.y) / (n * (n - 1.0))) as f32);
            }
            reservoir_valid_count += tile_out.reservoir_valid_count;
        }
        if !timing.record_into_certificate() {
            crate::core::certificate::record_pass("hybrid_pt.fused_gbuffer", 0.0, 1);
            crate::core::certificate::record_pass("hybrid_pt.fused", 0.0, frames);
            crate::core::certificate::record_pass("hybrid_pt.restir_temporal", 0.0, frames);
            crate::core::certificate::record_pass("hybrid_pt.restir_spatial", 0.0, frames);
            crate::core::certificate::record_pass("hybrid_pt.fused_publish", 0.0, frames);
        }

        let lit_scene = desc.sun_elevation_deg > 0.0
            && desc.sun_intensity > 0.0
            && desc.sun_color.iter().any(|c| *c > 0.0)
            && hit_kind.iter().any(|kind| *kind != 0);
        if desc.aovs && lit_scene && frames >= 2 && reservoir_valid_count == 0 {
            return Err(RenderError::Render(
                "fused render ReSTIR reuse chain produced no valid reservoirs for a sun-lit \
                 scene; temporal reuse is broken"
                    .into(),
            ));
        }

        Ok(FusedRenderOutput {
            width,
            height,
            rgba,
            radiance,
            albedo,
            normal,
            depth,
            direct,
            transmittance,
            hit_kind,
            sun_cosine,
            self_bias,
            position,
            reservoir_visibility,
            reservoir_valid_count,
            frames,
            restarts,
            stale_frames,
            variance,
            tiles: u32::try_from(rects.len()).unwrap_or(u32::MAX),
            logical_primitives: desc.scene.logical_primitive_count(),
            page_count: desc.scene.page_count(),
            tlas_node_count: fused.tlas_node_count(),
            paging: fused.stats(),
            terrain_bytes,
            terrain_minmax_f16,
            peak_host_visible_bytes: 0,
            peak_device_local_bytes: 0,
            peak_total_bytes: 0,
            tracker_host_visible_bytes: 0,
            tracker_peak_host_visible_bytes: 0,
            tracker_limit_bytes: 0,
        })
    }

    /// Render one seamless tile `(x, y, tw, th)` of a view: allocate the
    /// tile's accumulation, reservoir, G-buffer and output targets, stream
    /// the pages its rays touch, accumulate, and read the tile back. All
    /// per-pixel GPU memory is released when the tile returns.
    fn render_fused_tile(
        &self,
        view: &FusedViewResources,
        desc: &FusedRenderDesc,
        fused: &mut FusedGpu,
        tick: &mut u64,
        rect: (u32, u32, u32, u32),
        mut timing: Option<&mut crate::core::gpu_timing::OneShotTiming>,
    ) -> Result<FusedTileOutput, RenderError> {
        let device = &try_ctx()?.device;
        let queue = &try_ctx()?.queue;
        let policy = desc.scene.params().policy;
        let (x, y, width, height) = rect;
        let (full_w, full_h) = (desc.width, desc.height);
        let mut base = view.base;
        base.width = width;
        base.height = height;
        base.full_width = full_w;
        base.full_height = full_h;
        base.pixel_offset_x = x;
        base.pixel_offset_y = y;
        base.cam_aspect = f32_from_u32(full_w) / f32_from_u32(full_h);
        base.sensor_rect = [
            f32_from_u32(x) / f32_from_u32(full_w),
            f32_from_u32(y) / f32_from_u32(full_h),
            f32_from_u32(x + width) / f32_from_u32(full_w),
            f32_from_u32(y + height) / f32_from_u32(full_h),
        ];
        if width == 0 || height == 0 || x + width > full_w || y + height > full_h {
            return Err(RenderError::Render(format!(
                "fused tile {width}x{height} at ({x},{y}) exceeds the {full_w}x{full_h} image"
            )));
        }
        let aov_flags = if desc.aovs { 0xFF } else { 0 };
        base.aov_flags = aov_flags;
        queue.write_buffer(&view.base_ubo, 0, bytemuck::bytes_of(&base));

        // --- Accumulation, statistics, canonical ReSTIR reservoirs, G-buffer ---
        let px_count = u64::from(width) * u64::from(height);
        let px_usize = px_count as usize;
        let stats_size = std::mem::size_of::<TerrainStatistics>() as u64;
        let clearable = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let accum_buf = tracked_create_buffer(
            device,
            &wgpu::BufferDescriptor {
                label: Some("hybrid-pt-fused-accum"),
                size: px_count * 16,
                usage: clearable,
                mapped_at_creation: false,
            },
        )?;
        let welford_buf = tracked_create_buffer(
            device,
            &wgpu::BufferDescriptor {
                label: Some("hybrid-pt-fused-welford"),
                size: px_count * stats_size,
                usage: clearable,
                mapped_at_creation: false,
            },
        )?;
        let reservoir_curr = create_reservoir_buffer(device, px_usize)?;
        let reservoir_out = create_reservoir_buffer(device, px_usize)?;
        let reservoir_prev = create_reservoir_buffer(device, px_usize)?;
        let gbuffer_nr = create_restir_gbuffer(device, px_usize)?;
        let gbuffer_pos = create_restir_gbuffer_pos(device, px_usize)?;

        let out_tex = tracked_create_texture(
            device,
            &wgpu::TextureDescriptor {
                label: Some("hybrid-pt-fused-out"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            },
        )?;
        let out_view = out_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let aovs_all = [
            AovKind::Albedo,
            AovKind::Normal,
            AovKind::Depth,
            AovKind::Direct,
            AovKind::Indirect,
            AovKind::Emission,
            AovKind::Visibility,
        ];
        // Without AOVs the output layout still needs bound targets; 1x1
        // placeholders are never written (aov_flags = 0 on every frame).
        let (aov_w, aov_h) = if desc.aovs { (width, height) } else { (1, 1) };
        let aov_frames = AovFrames::new(device, aov_w, aov_h, &aovs_all)?;
        let aov_views: Vec<wgpu::TextureView> = aovs_all
            .iter()
            .map(|kind| {
                aov_frames
                    .get_texture(*kind)
                    .unwrap()
                    .create_view(&wgpu::TextureViewDescriptor::default())
            })
            .collect();

        // --- Memory-budget gate before any dispatch ---
        let metrics = global_tracker().get_metrics();
        if metrics.host_visible_bytes > metrics.limit_bytes {
            return Err(RenderError::Budget(format!(
                "fused render exceeds the memory budget before rendering: host-visible {} > \
                 limit {}",
                metrics.host_visible_bytes, metrics.limit_bytes
            )));
        }

        // --- Tile bind groups ---
        let bg2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hybrid-pt-fused-bg2"),
            layout: &self.layouts.accum,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: accum_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view.height_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&view.minmax_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: view.terrain_ubo.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: welford_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: reservoir_curr.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(&view.env_view),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: reservoir_prev.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: view.earth_curvature_ubo.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 16,
                    resource: wgpu::BindingResource::TextureView(&view.albedo_view),
                },
            ],
        });
        let mut bg3_entries = vec![wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(&out_view),
        }];
        for (i, aov_view) in aov_views.iter().enumerate() {
            bg3_entries.push(wgpu::BindGroupEntry {
                binding: (i as u32) + 1,
                resource: wgpu::BindingResource::TextureView(aov_view),
            });
        }
        let bg3 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hybrid-pt-fused-bg3"),
            layout: &self.layouts.output,
            entries: &bg3_entries,
        });
        let bg_gbuffer = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hybrid-pt-fused-bg-gbuffer"),
            layout: &self.layouts.terrain_gbuffer,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view.height_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&view.minmax_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: view.terrain_ubo.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: gbuffer_nr.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: gbuffer_pos.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: view.earth_curvature_ubo.as_entire_binding(),
                },
            ],
        });
        let bg_empty = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hybrid-pt-fused-bg-empty"),
            layout: &self.layouts.empty,
            entries: &[],
        });
        let bg_temporal = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hybrid-pt-fused-bg-restir-temporal"),
            layout: &self.layouts.restir_temporal,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: reservoir_prev.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: reservoir_curr.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: reservoir_out.as_entire_binding(),
                },
            ],
        });
        let bg_spatial_scene = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hybrid-pt-fused-bg-restir-spatial-scene"),
            layout: &self.layouts.restir_spatial_scene,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: view.area_lights_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: view.dir_lights_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: gbuffer_nr.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: gbuffer_pos.as_entire_binding(),
                },
            ],
        });
        let bg_spatial_reuse = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hybrid-pt-fused-bg-restir-spatial-reuse"),
            layout: &self.layouts.restir_spatial_reuse,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: reservoir_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: reservoir_prev.as_entire_binding(),
                },
            ],
        });

        let wg_x = width.div_ceil(8);
        let wg_y = height.div_ceil(8);
        let px_workgroups = ((px_count as u32) + 255) / 256;
        let px_wg_x = px_workgroups.min(65_535);
        let px_wg_y = px_workgroups.div_ceil(65_535);
        // --- ReSTIR G-buffer pass. The centre rays traverse the fused scene,
        // so pages they miss are streamed in and the pass repeats until its
        // rays saw every page (exact) or once (progressive). ---
        let mut gbuffer_passes = 0u32;
        loop {
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("hybrid-pt-fused-gbuffer-enc"),
            });
            let gbuffer_scope = match timing.as_deref_mut() {
                Some(t) if gbuffer_passes == 0 => t.begin(&mut enc, "hybrid_pt.fused_gbuffer"),
                _ => None,
            };
            {
                let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("hybrid-pt-fused-gbuffer-cpass"),
                    ..Default::default()
                });
                crate::core::shader_registry::record_shader_use("hybrid-pt-kernel");
                cpass.set_pipeline(&self.pipeline_terrain_gbuffer);
                cpass.set_bind_group(0, &view.bg0, &[]);
                cpass.set_bind_group(1, &view.bg1, &[]);
                cpass.set_bind_group(2, &bg_gbuffer, &[]);
                cpass.dispatch_workgroups(wg_x, wg_y, 1);
            }
            if let Some(t) = timing.as_deref_mut() {
                if gbuffer_passes == 0 {
                    t.end(&mut enc, gbuffer_scope, 1);
                }
            }
            queue.submit([enc.finish()]);
            *tick += 1;
            gbuffer_passes += 1;
            let report = fused.service(device, queue, *tick)?;
            if report.miss_events == 0 || policy == PagingPolicy::Progressive {
                break;
            }
            if gbuffer_passes > MAX_RESTARTS {
                return Err(RenderError::Render(format!(
                    "fused G-buffer pass still misses pages after {MAX_RESTARTS} paging rounds"
                )));
            }
        }

        // --- Accumulation ---
        let mut frames = 0u32;
        let mut restarts = 0u32;
        let mut stale_frames = 0u32;
        let mut timed = timing.is_none();
        while frames < desc.frames {
            base.frame_index = frames;
            base.aov_flags = if frames == 0 { aov_flags } else { 0 };
            queue.write_buffer(&view.base_ubo, 0, bytemuck::bytes_of(&base));
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("hybrid-pt-fused-frame"),
            });
            // Only the first submitted frame of the first tile is timed.
            let time_this_frame = !timed;
            let scope_fused = match timing.as_deref_mut() {
                Some(t) if time_this_frame => t.begin(&mut enc, "hybrid_pt.fused"),
                _ => None,
            };
            {
                let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("hybrid-pt-fused-cpass"),
                    ..Default::default()
                });
                crate::core::shader_registry::record_shader_use("hybrid-pt-kernel");
                cpass.set_pipeline(&self.pipeline_terrain);
                cpass.set_bind_group(0, &view.bg0, &[]);
                cpass.set_bind_group(1, &view.bg1, &[]);
                cpass.set_bind_group(2, &bg2, &[]);
                cpass.set_bind_group(3, &bg3, &[]);
                cpass.dispatch_workgroups(wg_x, wg_y, 1);
            }
            if let (Some(t), true) = (timing.as_deref_mut(), time_this_frame) {
                t.end(&mut enc, scope_fused, 1);
            }
            let scope_temporal = match timing.as_deref_mut() {
                Some(t) if time_this_frame => t.begin(&mut enc, "hybrid_pt.restir_temporal"),
                _ => None,
            };
            {
                let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("hybrid-pt-fused-restir-temporal-cpass"),
                    ..Default::default()
                });
                crate::core::shader_registry::record_shader_use("hybrid-pt-restir-temporal");
                cpass.set_pipeline(&self.pipeline_restir_temporal);
                cpass.set_bind_group(0, &view.bg0, &[]);
                cpass.set_bind_group(1, &bg_empty, &[]);
                cpass.set_bind_group(2, &bg_temporal, &[]);
                cpass.dispatch_workgroups(px_wg_x, px_wg_y, 1);
            }
            if let (Some(t), true) = (timing.as_deref_mut(), time_this_frame) {
                t.end(&mut enc, scope_temporal, 1);
            }
            let scope_spatial = match timing.as_deref_mut() {
                Some(t) if time_this_frame => t.begin(&mut enc, "hybrid_pt.restir_spatial"),
                _ => None,
            };
            {
                let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("hybrid-pt-fused-restir-spatial-cpass"),
                    ..Default::default()
                });
                crate::core::shader_registry::record_shader_use("hybrid-pt-restir-spatial");
                cpass.set_pipeline(&self.pipeline_restir_spatial);
                cpass.set_bind_group(0, &view.bg0, &[]);
                cpass.set_bind_group(1, &bg_spatial_scene, &[]);
                cpass.set_bind_group(2, &bg_spatial_reuse, &[]);
                cpass.dispatch_workgroups(px_wg_x, px_wg_y, 1);
            }
            if let (Some(t), true) = (timing.as_deref_mut(), time_this_frame) {
                t.end(&mut enc, scope_spatial, 1);
            }
            let scope_publish = match timing.as_deref_mut() {
                Some(t) if time_this_frame => t.begin(&mut enc, "hybrid_pt.fused_publish"),
                _ => None,
            };
            {
                let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("hybrid-pt-fused-publish-cpass"),
                    ..Default::default()
                });
                crate::core::shader_registry::record_shader_use("hybrid-pt-kernel");
                cpass.set_pipeline(&self.pipeline_terrain_publish);
                cpass.set_bind_group(0, &view.bg0, &[]);
                cpass.set_bind_group(1, &view.bg1, &[]);
                cpass.set_bind_group(2, &bg2, &[]);
                cpass.set_bind_group(3, &bg3, &[]);
                cpass.dispatch_workgroups(wg_x, wg_y, 1);
            }
            if let (Some(t), true) = (timing.as_deref_mut(), time_this_frame) {
                t.end(&mut enc, scope_publish, 1);
                t.resolve(&mut enc);
                timed = true;
            }
            queue.submit([enc.finish()]);
            frames += 1;
            *tick += 1;

            let report = fused.service(device, queue, *tick)?;
            if report.miss_events == 0 {
                continue;
            }
            match policy {
                PagingPolicy::Exact => {
                    // The frame saw incomplete visibility. The pages are in
                    // now: drop the accumulation and the reservoir history
                    // and start over, so every accumulated frame is exact.
                    restarts += 1;
                    if restarts > MAX_RESTARTS {
                        return Err(RenderError::Render(format!(
                            "fused render still misses pages after {MAX_RESTARTS} restarts"
                        )));
                    }
                    let mut enc =
                        device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("hybrid-pt-fused-restart"),
                        });
                    for buffer in [
                        &*accum_buf,
                        &*welford_buf,
                        &*reservoir_curr,
                        &*reservoir_out,
                        &*reservoir_prev,
                    ] {
                        enc.clear_buffer(buffer, 0, None);
                    }
                    queue.submit([enc.finish()]);
                    frames = 0;
                }
                PagingPolicy::Progressive => stale_frames += 1,
            }
        }
        device.poll(wgpu::Maintain::Wait);
        // Release what the readbacks do not need before any staging buffer
        // is allocated, so the tile peak is the render footprint, not the
        // render footprint plus readback staging.
        drop((bg2, bg3, bg_gbuffer, bg_temporal, bg_spatial_scene, bg_spatial_reuse));
        drop(reservoir_curr);
        drop(reservoir_out);

        // --- Readbacks ---
        // ReSTIR reservoir validity + the visibility the chain carries.
        let mut reservoir_valid_count = 0u64;
        let mut reservoir_visibility = Vec::new();
        if desc.aovs {
            let res_stride = std::mem::size_of::<Reservoir>() as u64;
            let res_bytes = read_buffer(device, queue, &reservoir_prev, px_count * res_stride)?;
            let reservoirs: &[Reservoir] = bytemuck::cast_slice(&res_bytes);
            reservoir_visibility.reserve(px_usize);
            for r in reservoirs {
                if !(r.w_sum.is_finite() && r.weight.is_finite() && r.target_pdf.is_finite()) {
                    return Err(RenderError::Render(
                        "fused render reservoir bookkeeping produced non-finite values".into(),
                    ));
                }
                if r.m > 0 && r.weight > 0.0 && r.target_pdf > 0.0 {
                    reservoir_valid_count += 1;
                    reservoir_visibility.push(r.sample.intensity);
                } else {
                    reservoir_visibility.push(-1.0);
                }
            }
        }
        drop(reservoir_prev);
        let stats_bytes = read_buffer(device, queue, &welford_buf, px_count * stats_size)?;
        let stats: &[TerrainStatistics] = bytemuck::cast_slice(&stats_bytes);
        let mut moment = FrameMoment { y: 0.0 };
        for s in stats {
            if s.overflow != 0 {
                return Err(RenderError::render(
                    "fused render traversal counters overflowed u32",
                ));
            }
            if !s.x.is_finite() || s.x < 0.0 || !s.y.is_finite() || s.y < 0.0 {
                return Err(RenderError::render(
                    "fused render produced invalid frame-radiance moments",
                ));
            }
            moment.y = moment.y.max(s.y);
        }
        drop(stats_bytes);
        drop(welford_buf);
        let accum_bytes = read_buffer(device, queue, &accum_buf, px_count * 16)?;
        let accum: &[f32] = bytemuck::cast_slice(&accum_bytes);
        let mut radiance = Vec::with_capacity(px_usize * 3);
        for px in accum.chunks_exact(4) {
            let count = px[3].max(1.0);
            for value in &px[..3] {
                let mean = value / count;
                if !mean.is_finite() {
                    return Err(RenderError::render(
                        "fused render produced non-finite accumulated radiance",
                    ));
                }
                radiance.push(mean);
            }
        }
        drop(accum_bytes);
        drop(accum_buf);
        let beauty = read_texture(device, queue, &out_tex, width, height, 8)?;
        let mut rgba = vec![0u8; px_usize * 4];
        for (i, px) in beauty.chunks_exact(8).enumerate() {
            for c in 0..3 {
                let v = f16::from_bits(u16::from_le_bytes([px[c * 2], px[c * 2 + 1]])).to_f32();
                rgba[i * 4 + c] = (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            }
            rgba[i * 4 + 3] = 255;
        }
        drop(beauty);
        drop(out_tex);
        let mut out = FusedTileOutput {
            rgba,
            radiance,
            frames,
            restarts,
            stale_frames,
            moment,
            reservoir_valid_count,
            reservoir_visibility,
            ..FusedTileOutput::default()
        };
        if !desc.aovs {
            return Ok(out);
        }
        let aov = |kind: AovKind, bytes_per_pixel: u32| {
            read_texture(
                device,
                queue,
                aov_frames.get_texture(kind).unwrap(),
                width,
                height,
                bytes_per_pixel,
            )
        };
        out.albedo = decode_f16(&aov(AovKind::Albedo, 8)?, 3);
        out.depth = bytemuck::cast_slice::<u8, f32>(&aov(AovKind::Depth, 4)?).to_vec();
        out.direct = decode_f16(&aov(AovKind::Direct, 8)?, 3);
        out.transmittance = decode_f16(&aov(AovKind::Indirect, 8)?, 4);
        let emission = decode_f16(&aov(AovKind::Emission, 8)?, 4);
        out.hit_kind.reserve(px_usize);
        for px in emission.chunks_exact(4) {
            out.hit_kind.push(px[0].round().clamp(0.0, 255.0) as u8);
            out.sun_cosine.push(px[1]);
            out.self_bias.push(px[3]);
        }
        let xyz = |bytes: &[u8]| -> Vec<f32> {
            bytemuck::cast_slice::<u8, f32>(bytes)
                .chunks_exact(4)
                .flat_map(|p| [p[0], p[1], p[2]])
                .collect()
        };
        out.position = xyz(&read_buffer(device, queue, &gbuffer_pos, px_count * 16)?);
        out.normal = xyz(&read_buffer(device, queue, &gbuffer_nr, px_count * 16)?);
        Ok(out)
    }
}

/// View-level GPU state shared by every tile of one fused view (owned, so
/// the struct needs no lifetime parameter).
struct FusedViewResources {
    /// Camera/seed template; tiles fill extent, offset and sensor rect.
    base: Uniforms,
    base_ubo: TrackedBuffer,
    bg0: wgpu::BindGroup,
    bg1: wgpu::BindGroup,
    height_view: wgpu::TextureView,
    minmax_view: wgpu::TextureView,
    env_view: wgpu::TextureView,
    albedo_view: wgpu::TextureView,
    terrain_ubo: TrackedBuffer,
    earth_curvature_ubo: TrackedBuffer,
    dir_lights_buf: TrackedBuffer,
    area_lights_buf: TrackedBuffer,
}

/// One tile's read-back outputs (AOV vectors empty when AOVs are off).
#[derive(Default)]
struct FusedTileOutput {
    rgba: Vec<u8>,
    radiance: Vec<f32>,
    albedo: Vec<f32>,
    normal: Vec<f32>,
    depth: Vec<f32>,
    direct: Vec<f32>,
    transmittance: Vec<f32>,
    hit_kind: Vec<u8>,
    sun_cosine: Vec<f32>,
    self_bias: Vec<f32>,
    position: Vec<f32>,
    reservoir_visibility: Vec<f32>,
    frames: u32,
    restarts: u32,
    stale_frames: u32,
    /// Largest per-pixel second moment of the frame radiance.
    moment: FrameMoment,
    reservoir_valid_count: u64,
}

/// The second-moment term of the per-pixel Welford statistics.
#[derive(Clone, Copy, Default)]
struct FrameMoment {
    y: f32,
}

/// Copy a tile's row-major `channels`-interleaved pixels into the full
/// image (`full_width` pixels per row) at `rect = (x, y, tw, th)`.
fn blit_rows<T: Copy>(
    dst: &mut [T],
    src: &[T],
    channels: usize,
    full_width: u32,
    rect: (u32, u32, u32, u32),
) {
    let (x, y, tw, th) = rect;
    let row = tw as usize * channels;
    for r in 0..th as usize {
        let d = ((y as usize + r) * full_width as usize + x as usize) * channels;
        dst[d..d + row].copy_from_slice(&src[r * row..(r + 1) * row]);
    }
}

/// Largest tile (pixels) a fused render dispatches at once. Larger frames
/// are split into seamless tiles of at most 1024 x 1024.
pub(crate) const MAX_TILE_PIXELS: u64 = 1 << 20;

/// The tile a fused render uses when none is requested: the whole frame up
/// to one megapixel, otherwise 1024 x 1024 (clamped to the frame).
pub(crate) fn default_tile(w: u32, h: u32) -> (u32, u32) {
    if u64::from(w) * u64::from(h) <= MAX_TILE_PIXELS {
        (w, h)
    } else {
        (w.min(1024), h.min(1024))
    }
}

/// Row-major tiles `(x, y, tw, th)` covering a `w` x `h` frame; the last
/// column and row may be smaller than `tile`.
pub(crate) fn fused_tiles(w: u32, h: u32, tile: (u32, u32)) -> Vec<(u32, u32, u32, u32)> {
    let (tile_w, tile_h) = (tile.0.max(1), tile.1.max(1));
    let mut rects = Vec::new();
    let mut y = 0;
    while y < h {
        let th = tile_h.min(h - y);
        let mut x = 0;
        while x < w {
            let tw = tile_w.min(w - x);
            rects.push((x, y, tw, th));
            x += tw;
        }
        y += th;
    }
    rects
}
#[cfg(test)]
mod tiling_tests {
    use super::*;

    #[test]
    fn fused_tiles_cover_every_pixel_exactly_once() {
        for (w, h, tile) in [(300u32, 200u32, (128u32, 96u32)), (1920, 1080, (1024, 1024)), (7, 5, (7, 5))] {
            let mut count = vec![0u8; (w * h) as usize];
            for (x, y, tw, th) in fused_tiles(w, h, tile) {
                assert!(tw >= 1 && th >= 1 && tw <= tile.0 && th <= tile.1);
                assert!(x + tw <= w && y + th <= h, "tile ({x},{y},{tw},{th}) out of {w}x{h}");
                for py in y..y + th {
                    for px in x..x + tw {
                        count[(py * w + px) as usize] += 1;
                    }
                }
            }
            assert!(count.iter().all(|c| *c == 1), "{w}x{h} tile {tile:?}");
        }
    }

    #[test]
    fn default_tile_is_single_up_to_one_megapixel() {
        assert_eq!(default_tile(1024, 1024), (1024, 1024));
        assert_eq!(default_tile(640, 480), (640, 480));
        assert_eq!(default_tile(1920, 1080), (1024, 1024));
        assert_eq!(default_tile(3840, 2160), (1024, 1024));
    }
}
