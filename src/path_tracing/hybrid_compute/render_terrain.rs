// src/path_tracing/hybrid_compute/render_terrain.rs
// PROMETHEUS accumulation driver: renders a converged path-traced reference
// of a real DEM under sun + IBL through the `main_terrain` kernel entry,
// optionally mixed with mesh geometry through the shared HybridScene seam.
// Each frame dispatches the terrain kernel (spp jittered samples + canonical
// ReSTIR candidate generation), then the pt_restir_temporal and
// pt_restir_spatial reuse passes over the canonical 80-byte Reservoir layout.
// Convergence gates on the maximum per-pixel estimated variance of the mean
// frame luminance (Welford M2/(N*(N-1)) over ALL accumulated frames, read
// back every WELFORD_WINDOW frames — hard error on cap miss, no silent fake
// convergence). Every GPU allocation is registered with the global
// memory tracker through a drop-guard so error paths cannot leak metrics.
// RELEVANT FILES: src/shaders/hybrid_terrain_traversal.wgsl,
//                 src/path_tracing/hybrid_compute/terrain_heightfield.rs

use super::terrain_heightfield::TerrainPtScene;
use super::*;
use crate::core::atmosphere::AETHER_RADIOMETRIC_SCALE_MAX;
use crate::core::memory_tracker::global_tracker;
use crate::core::shader_contract_runtime::RuntimeContractObservation;
use crate::path_tracing::lighting::{GpuAreaLight, GpuDirectionalLight};
use crate::path_tracing::restir::{
    create_reservoir_buffer, create_restir_gbuffer, create_restir_gbuffer_pos, Reservoir,
};

/// GPU layout of the per-pixel statistics record `main_terrain` writes into
/// the `terrain_welford` storage buffer (binding 4): running Welford moments
/// of the per-frame mean luminance (`x` = mean, `y` = M2 over all frames),
/// a counter-overflow flag, and the last beauty frame's measured traversal
/// counts split into primary/shadow lanes
/// (node_visits, minmax_loads, height_loads, rays).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub(super) struct TerrainStatistics {
    pub x: f32,
    pub y: f32,
    pub overflow: u32,
    /// Set by the kernel when the compiled SDF march failed (non-finite
    /// value, stagnation, or exhausted step budget); mirrors
    /// hybrid_sdf_error in hybrid_traversal.wgsl.
    pub trace_error: u32,
    pub primary: [u32; 4],
    pub shadow: [u32; 4],
}

/// Estimated variance of the mean estimator: M2 / (N * (N - 1)) where N is
/// the accumulated frame count. Errors on degenerate counts or invalid
/// moments — callers must never synthesize a zero variance.
fn estimator_variance(m2: f32, frames: u32) -> Result<f32, RenderError> {
    if frames < 2 || !m2.is_finite() || m2 < 0.0 {
        return Err(RenderError::render(
            "terrain PT produced invalid frame-radiance moments",
        ));
    }
    Ok((m2 as f64 / (frames as f64 * (frames - 1) as f64)) as f32)
}

/// Parse the statistics readback; any counter overflow or invalid moment is
/// a hard error before stats are reported.
fn parse_terrain_statistics(bytes: &[u8]) -> Result<&[TerrainStatistics], RenderError> {
    let stats: &[TerrainStatistics] = bytemuck::cast_slice(bytes);
    for s in stats {
        if s.overflow != 0 {
            return Err(RenderError::render(
                "terrain PT traversal counters overflowed u32",
            ));
        }
        if s.trace_error != 0 {
            return Err(RenderError::render(
                "terrain PT SDF traversal did not converge or produced invalid geometry",
            ));
        }
        if !s.x.is_finite() || s.x < 0.0 || !s.y.is_finite() || s.y < 0.0 {
            return Err(RenderError::render(
                "terrain PT produced invalid frame-radiance moments",
            ));
        }
    }
    Ok(stats)
}

fn finite_min_max(values: impl Iterator<Item = f32>) -> (f32, f32) {
    values.fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), value| {
        if !value.is_finite() || lo.is_nan() {
            (f32::NAN, f32::NAN)
        } else {
            (lo.min(value), hi.max(value))
        }
    })
}

#[allow(clippy::too_many_arguments)]
fn observe_runtime_contract(
    desc: &TerrainReferenceDesc,
    base: &Uniforms,
    hybrid: &HybridUniforms,
    lighting: &LightingUniforms,
    terrain: &super::terrain_heightfield::TerrainPtUniforms,
    earth_curvature: &super::terrain_heightfield::EarthCurvatureUniforms,
    frames: u32,
    reservoirs: &[Reservoir],
    accum: &[f32],
    welford: &[f32],
    beauty: &[u8],
    canonical: bool,
) -> RuntimeContractObservation {
    let mut observed = RuntimeContractObservation::new(
        &format!("hybrid-terrain-{}x{}-{frames}f", desc.width, desc.height),
        "hybrid_pt.terrain",
        "hybrid_terrain_traversal",
        "src/shaders/hybrid_terrain_traversal.wgsl",
        "main_terrain",
        if canonical {
            "shaders/contracts/hybrid_terrain_traversal.toml"
        } else {
            "runtime-safety:hybrid-terrain"
        },
    );
    if canonical {
        let mut check = |name: &str, values: &[f32], lo: f32, hi: f32| {
            let (actual_lo, actual_hi) = finite_min_max(values.iter().copied());
            observed.check_range("uniform", name, None, actual_lo, actual_hi, lo, hi);
        };
        check("uniforms.width", &[base.width as f32], 8.0, 8.0);
        check("uniforms.height", &[base.height as f32], 8.0, 8.0);
        check("uniforms.cam_origin.x", &[base.cam_origin[0]], 0.0, 0.0);
        check("uniforms.cam_origin.y", &[base.cam_origin[1]], 3.0, 3.0);
        check("uniforms.cam_origin.z", &[base.cam_origin[2]], 8.0, 8.0);
        check("uniforms.cam_right", &base.cam_right, 0.0, 1.0);
        check("uniforms.cam_up", &base.cam_up, -0.36, 0.94);
        check("uniforms.cam_forward", &base.cam_forward, -0.94, 0.0);
        check("uniforms.cam_aspect", &[base.cam_aspect], 1.0, 1.0);
        check("uniforms.cam_fov_y", &[base.cam_fov_y], 0.78, 0.79);
        check("uniforms.cam_exposure", &[base.cam_exposure], 1.0, 1.0);
        check(
            "uniforms.frame_index",
            &[0.0, frames.saturating_sub(1) as f32],
            0.0,
            3.0,
        );
        check(
            "uniforms.dispatch_gid",
            &[
                0.0,
                (desc.width.max(desc.height).div_ceil(8) * 8 - 1) as f32,
            ],
            0.0,
            7.0,
        );
        check(
            "hybrid_uniforms.traversal_mode",
            &[hybrid.traversal_mode as f32],
            3.0,
            3.0,
        );
        check(
            "hybrid_uniforms.mesh_vertex_count",
            &[hybrid.mesh_vertex_count as f32],
            0.0,
            0.0,
        );
        check(
            "hybrid_uniforms.mesh_index_count",
            &[hybrid.mesh_index_count as f32],
            0.0,
            0.0,
        );
        check("lighting.light_dir", &lighting.light_dir, -0.51, 0.72);
        check("lighting.light_color", &lighting.light_color, 2.29, 2.5);
        check(
            "lighting.shadows_enabled",
            &[lighting.shadows_enabled as f32],
            1.0,
            1.0,
        );
        check("terrain.origin_spacing", &terrain.origin_spacing, -1.5, 1.0);
        check("terrain.h_params", &terrain.h_params, 0.0, 1.0);
        check("terrain.albedo_pad", &terrain.albedo_pad, 0.0, 0.6);
        check(
            "terrain.dims",
            &terrain.dims.map(|value| value as f32),
            3.0,
            4.0,
        );
        check(
            "terrain.mips",
            &terrain.mips.map(|value| value as f32),
            0.0,
            3.0,
        );
        check(
            "terrain.extra",
            &terrain.extra.map(|value| value as f32),
            0.0,
            32.0,
        );
        check(
            "earth_curvature.inv_two_r_prime",
            &[earth_curvature.inv_two_r_prime],
            0.0,
            1e-6,
        );
        check(
            "earth_curvature.ray_origin_geodetic",
            &earth_curvature.ray_origin_geodetic,
            -180.0,
            180.0,
        );
        check(
            "earth_curvature.enabled",
            &[earth_curvature.enabled as f32],
            0.0,
            1.0,
        );
        check("terrain_height_tex.samples", &desc.heights, 0.0, 0.0);
    } else {
        let mut exact = |name: &str, values: &[f32], expected: &[f32]| {
            for (axis, (&value, &target)) in values.iter().zip(expected).enumerate() {
                let name = if values.len() == 1 {
                    name.to_string()
                } else {
                    format!("{name}.{axis}")
                };
                observed.check_range("uniform", &name, None, value, value, target, target);
            }
        };
        exact("uniforms.width", &[base.width as f32], &[desc.width as f32]);
        exact(
            "uniforms.height",
            &[base.height as f32],
            &[desc.height as f32],
        );
        exact("uniforms.cam_origin", &base.cam_origin, &desc.cam_origin);
        let forward =
            (glam::Vec3::from(desc.cam_look_at) - glam::Vec3::from(desc.cam_origin)).normalize();
        let right = forward.cross(glam::Vec3::from(desc.cam_up)).normalize();
        let up = right.cross(forward).normalize();
        exact(
            "uniforms.cam_forward",
            &base.cam_forward,
            &forward.to_array(),
        );
        exact("uniforms.cam_right", &base.cam_right, &right.to_array());
        exact("uniforms.cam_up", &base.cam_up, &up.to_array());
        exact(
            "uniforms.cam_aspect",
            &[base.cam_aspect],
            &[desc.width as f32 / desc.height as f32],
        );
        exact(
            "uniforms.cam_fov_y",
            &[base.cam_fov_y],
            &[desc.fov_y_deg.to_radians()],
        );
        exact(
            "uniforms.cam_exposure",
            &[base.cam_exposure],
            &[desc.exposure],
        );
        exact(
            "uniforms.frame_index",
            &[base.frame_index as f32],
            &[frames.saturating_sub(1) as f32],
        );
        let (vertices, indices) = desc
            .mesh
            .as_ref()
            .map_or((0, 0), |(v, i)| (v.len() / 3, i.len()));
        exact(
            "hybrid_uniforms.mesh_vertex_count",
            &[hybrid.mesh_vertex_count as f32],
            &[vertices as f32],
        );
        exact(
            "hybrid_uniforms.mesh_index_count",
            &[hybrid.mesh_index_count as f32],
            &[indices as f32],
        );
        exact(
            "hybrid_uniforms.traversal_mode",
            &[hybrid.traversal_mode as f32],
            &[if desc.mesh.is_some() || desc.sdf_scene.is_some() {
                TraversalMode::Hybrid
            } else {
                TraversalMode::TerrainOnly
            } as u32 as f32],
        );
        // The specialized kernel reads the SDF scene from compiled constants,
        // so the uniform count fields stay 0; the validated native counts are
        // diagnostics only, not observed GPU buffer contents.
        let (sdf_prims, sdf_nodes) = desc
            .sdf_scene
            .as_ref()
            .map_or((0usize, 0usize), |s| (s.primitive_count(), s.node_count()));
        exact(
            "sdf_scene.primitive_count",
            &[sdf_prims as f32],
            &[sdf_prims as f32],
        );
        exact(
            "sdf_scene.node_count",
            &[sdf_nodes as f32],
            &[sdf_nodes as f32],
        );
        let az = desc.sun_azimuth_deg.to_radians();
        let el = desc.sun_elevation_deg.to_radians();
        exact(
            "lighting.light_dir",
            &lighting.light_dir,
            &[az.cos() * el.cos(), el.sin(), az.sin() * el.cos()],
        );
        exact(
            "lighting.light_color",
            &lighting.light_color,
            &desc.sun_color.map(|c| c * desc.sun_intensity),
        );
        exact(
            "lighting.shadows_enabled",
            &[lighting.shadows_enabled as f32],
            &[1.0],
        );
        exact(
            "terrain.origin_spacing",
            &terrain.origin_spacing,
            &[
                -0.5 * (desc.dem_width as f32 - 1.0) * desc.spacing.0,
                -0.5 * (desc.dem_height as f32 - 1.0) * desc.spacing.1,
                desc.spacing.0,
                desc.spacing.1,
            ],
        );
        let (h_min, h_max) = finite_min_max(desc.heights.iter().copied());
        exact(
            "terrain.h_params",
            &terrain.h_params,
            &[h_min, h_max, desc.exaggeration, desc.env_intensity],
        );
        exact(
            "terrain.albedo_pad",
            &terrain.albedo_pad,
            &[desc.albedo[0], desc.albedo[1], desc.albedo[2], 0.0],
        );
        let cw = desc.dem_width.saturating_sub(1);
        let ch = desc.dem_height.saturating_sub(1);
        exact(
            "terrain.dims",
            &terrain.dims.map(|v| v as f32),
            &[
                desc.dem_width as f32,
                desc.dem_height as f32,
                cw as f32,
                ch as f32,
            ],
        );
        let mip_count = cw.max(ch).max(1).next_power_of_two().ilog2() + 1;
        let (env_w, env_h) = desc.env_map.as_ref().map_or((0, 0), |(_, w, h)| (*w, *h));
        exact(
            "terrain.mips",
            &terrain.mips.map(|v| v as f32),
            &[mip_count as f32, 1.0, env_w as f32, env_h as f32],
        );
        exact(
            "terrain.extra",
            &terrain.extra.map(|v| v as f32),
            &[desc.spp as f32, WELFORD_WINDOW as f32, 0.0, 0.0],
        );
    }
    {
        let mut check = |name: &str, values: &[f32], lo: f32, hi: f32| {
            let (actual_lo, actual_hi) = finite_min_max(values.iter().copied());
            observed.check_range("buffer", name, None, actual_lo, actual_hi, lo, hi);
        };
        check("accum_hdr.samples", accum, 0.0, 131_026.0);
        let accum_frames = accum.iter().skip(3).step_by(4).copied().collect::<Vec<_>>();
        check(
            "accum_hdr.frames",
            &accum_frames,
            frames as f32,
            frames as f32,
        );
        let welford_mean = welford.iter().step_by(2).copied().collect::<Vec<_>>();
        let welford_m2 = welford
            .iter()
            .skip(1)
            .step_by(2)
            .copied()
            .collect::<Vec<_>>();
        check("terrain_welford.mean", &welford_mean, 0.0, 65_504.0);
        check("terrain_welford.m2", &welford_m2, 0.0, 8_581_615_000.0);
    }

    let pixels = (desc.width as u64) * (desc.height as u64);
    for (name, actual, required) in [
        ("accum_hdr", accum.len() as u64, pixels * 4),
        ("terrain_welford", welford.len() as u64, pixels * 2),
        ("out_tex.bytes", beauty.len() as u64, pixels * 8),
        (
            "terrain_height_tex",
            desc.heights.len() as u64,
            desc.dem_width as u64 * desc.dem_height as u64,
        ),
    ] {
        observed.check_length("buffer", name, None, actual, required);
    }
    observed.check_range(
        "runtime",
        "frames",
        None,
        frames as f32,
        frames as f32,
        desc.min_frames.max(2) as f32,
        desc.max_frames as f32,
    );
    // Same convergence metric as the host gate: maximum estimated variance
    // of the mean frame luminance over ALL `frames` frames. Invalid moments
    // must propagate as NaN — f32::max would silently erase a stored NaN.
    let variance = welford
        .iter()
        .skip(1)
        .step_by(2)
        .copied()
        .try_fold(0.0f32, |vmax, m2| {
            estimator_variance(m2, frames).map(|v| vmax.max(v))
        })
        .unwrap_or(f32::NAN);
    let convergence_valid =
        variance.is_finite() && variance >= 0.0 && variance < desc.variance_threshold;
    observed.check_range(
        "runtime",
        "convergence",
        None,
        u32::from(convergence_valid) as f32,
        u32::from(convergence_valid) as f32,
        1.0,
        1.0,
    );
    let finite_samples = reservoirs.iter().all(|r| {
        r.sample
            .position
            .iter()
            .chain(&r.sample.direction)
            .chain(&r.sample.params)
            .chain(std::iter::once(&r.sample.intensity))
            .all(|v| v.is_finite())
    });
    observed.check_range(
        "buffer",
        "terrain_reservoirs_prev.sample_finite",
        None,
        u32::from(finite_samples) as f32,
        u32::from(finite_samples) as f32,
        1.0,
        1.0,
    );
    let valid = reservoirs
        .iter()
        .filter(|r| r.m > 0 && r.weight > 0.0 && r.target_pdf > 0.0)
        .collect::<Vec<_>>();
    observed.check_length(
        "buffer",
        "terrain_reservoirs_prev.valid",
        None,
        valid.len() as u64,
        u64::from(should_require_valid_sun_reservoirs(
            desc.sun_elevation_deg,
            desc.sun_intensity,
            desc.sun_color,
        )),
    );
    for (name, values, lo, hi) in [
        (
            "light_type",
            valid
                .iter()
                .map(|r| r.sample.light_type as f32)
                .collect::<Vec<_>>(),
            1.0,
            1.0,
        ),
        (
            "light_index",
            valid
                .iter()
                .map(|r| r.sample.light_index as f32)
                .collect::<Vec<_>>(),
            0.0,
            0.0,
        ),
        (
            "target_pdf",
            valid.iter().map(|r| r.target_pdf).collect::<Vec<_>>(),
            1.0,
            1.0,
        ),
    ] {
        if !values.is_empty() {
            let (actual_lo, actual_hi) = finite_min_max(values.into_iter());
            observed.check_range(
                "buffer",
                &format!("terrain_reservoirs_prev.{name}"),
                None,
                actual_lo,
                actual_hi,
                lo,
                hi,
            );
        }
    }
    observed.check_length(
        "buffer",
        "terrain_reservoirs_prev",
        None,
        reservoirs.len() as u64,
        (desc.width as u64) * (desc.height as u64),
    );
    let (m_min, m_max) = reservoirs.iter().fold((u32::MAX, 0u32), |(lo, hi), value| {
        (lo.min(value.m), hi.max(value.m))
    });
    observed.check_range(
        "buffer",
        "terrain_reservoirs_prev.m",
        None,
        m_min as f32,
        m_max as f32,
        0.0,
        512.0,
    );
    let (reservoir_min, reservoir_max) = finite_min_max(
        reservoirs
            .iter()
            .flat_map(|value| [value.w_sum, value.weight, value.target_pdf]),
    );
    observed.check_range(
        "buffer",
        "terrain_reservoirs_prev.weights",
        None,
        reservoir_min,
        reservoir_max,
        0.0,
        65_536.0,
    );
    for axis in 0..3 {
        if valid.is_empty() && !canonical {
            break;
        }
        let directions = valid.iter().map(|value| value.sample.direction[axis]);
        let (lo, hi) = finite_min_max(directions);
        let (expected_lo, expected_hi) = if canonical {
            match axis {
                0 => (0.49, 0.51),
                1 => (0.70, 0.72),
                _ => (-0.51, -0.49),
            }
        } else {
            let az = desc.sun_azimuth_deg.to_radians();
            let el = desc.sun_elevation_deg.to_radians();
            let direction =
                glam::Vec3::new(az.cos() * el.cos(), el.sin(), az.sin() * el.cos()).normalize();
            (direction[axis] - 0.01, direction[axis] + 0.01)
        };
        observed.check_range(
            "buffer",
            &format!("terrain_reservoirs_prev.direction.{axis}"),
            None,
            lo,
            hi,
            expected_lo,
            expected_hi,
        );
    }
    let beauty_values = beauty
        .chunks_exact(2)
        .map(|bytes| f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32());
    let (beauty_min, beauty_max) = finite_min_max(beauty_values);
    observed.check_range(
        "texture",
        "out_tex.samples",
        None,
        beauty_min,
        beauty_max,
        0.0,
        1.0,
    );
    observed
}

fn is_canonical_proof_fixture(desc: &TerrainReferenceDesc) -> bool {
    desc.width == 8
        && desc.height == 8
        && desc.dem_width == 4
        && desc.dem_height == 4
        && desc.heights.len() == 16
        && desc.heights.iter().all(|h| *h == 0.0)
        && desc.cam_origin == [0.0, 3.0, 8.0]
        && desc.cam_look_at == [0.0; 3]
        && desc.cam_up == [0.0, 1.0, 0.0]
        && desc.fov_y_deg == 45.0
        && desc.exposure == 1.0
        && desc.spacing == (1.0, 1.0)
        && desc.exaggeration == 1.0
        && desc.albedo == [0.6; 3]
        && desc.sun_azimuth_deg == 315.0
        && desc.sun_elevation_deg == 45.0
        && desc.sun_intensity == 2.5
        && desc.sun_color == [1.0, 0.97, 0.92]
        && desc.env_map.is_none()
        && desc.env_intensity == 0.35
        && desc.mesh.is_none()
        && desc.sdf_scene.is_none()
        && desc.spp == 1
        && desc.max_frames == 4
        && desc.min_frames == 2
}

#[allow(clippy::too_many_arguments)]
fn record_runtime_contract(
    desc: &TerrainReferenceDesc,
    base: &Uniforms,
    hybrid: &HybridUniforms,
    lighting: &LightingUniforms,
    terrain: &super::terrain_heightfield::TerrainPtUniforms,
    earth_curvature: &super::terrain_heightfield::EarthCurvatureUniforms,
    frames: u32,
    reservoirs: &[Reservoir],
    accum: &[f32],
    welford: &[f32],
    beauty: &[u8],
) -> Result<(), RenderError> {
    for canonical in [false, true] {
        if canonical
            && (!crate::core::shader_contract_runtime::capture_active()
                || !is_canonical_proof_fixture(desc))
        {
            continue;
        }
        let observed = observe_runtime_contract(
            desc,
            base,
            hybrid,
            lighting,
            terrain,
            earth_curvature,
            frames,
            reservoirs,
            accum,
            welford,
            beauty,
            canonical,
        );
        let errors = observed
            .checked_bindings
            .iter()
            .filter_map(|binding| binding.alarm.as_deref())
            .collect::<Vec<_>>()
            .join("; ");
        crate::core::shader_contract_runtime::record_observation(observed)
            .map_err(RenderError::render)?;
        if !errors.is_empty() {
            return Err(RenderError::render(errors));
        }
    }
    Ok(())
}

/// Convergence readback cadence in frames: the estimator variance is
/// evaluated on the GPU statistics buffer every N frames (and at the frame
/// cap). It is only a polling interval — the Welford moments themselves
/// accumulate over ALL frames and are never windowed or reset.
pub const WELFORD_WINDOW: u32 = 32;

/// Full scene description for the terrain reference render.
pub struct TerrainReferenceDesc {
    pub heights: Vec<f32>,
    pub dem_width: u32,
    pub dem_height: u32,
    pub spacing: (f32, f32),
    pub exaggeration: f32,
    pub albedo: [f32; 3],
    pub cam_origin: [f32; 3],
    pub cam_look_at: [f32; 3],
    pub cam_up: [f32; 3],
    pub fov_y_deg: f32,
    pub exposure: f32,
    pub sun_azimuth_deg: f32,
    pub sun_elevation_deg: f32,
    pub sun_intensity: f32,
    pub sun_color: [f32; 3],
    pub observer_geodetic_deg: [f64; 2],
    pub earth_model: crate::geo::refraction::EarthModel,
    pub refraction_model: crate::geo::refraction::RefractionModel,
    /// Optional equirect environment map (RGB f32 rows) + dims; None uses the
    /// constant-white fallback scaled by `env_intensity`.
    pub env_map: Option<(Vec<f32>, u32, u32)>,
    pub env_intensity: f32,
    /// Optional AETHER transport applied as a standalone post over the
    /// authoritative PROMETHEUS accumulation and exact depth AOV. `None`
    /// preserves the original traversal and output byte-for-byte.
    pub atmosphere: Option<crate::core::atmosphere::AtmosphereLutHandle>,
    /// Optional mesh mixed into the scene: flat [x,y,z] vertices + triangle
    /// indices, traversed alongside the heightfield (TraversalMode::Hybrid).
    pub mesh: Option<(Vec<f32>, Vec<u32>)>,
    /// Optional native SDF scene compiled into the hybrid kernel's constant
    /// declarations by `sdf_scene::specialize` before pipeline creation. Not
    /// a storage upload: validation, bounds, and code generation run on the
    /// CPU and the shader executes straight-line CSG expressions.
    pub sdf_scene: Option<crate::sdf::SdfScene>,
    pub width: u32,
    pub height: u32,
    pub seed: u32,
    /// Camera samples per accumulation frame. Traversal cost depends on
    /// visited hierarchy nodes; returned counters measure the last beauty
    /// frame, while throughput is checked separately from 1 to 8 spp.
    pub spp: u32,
    /// Hard frame cap; convergence earlier is allowed.
    pub max_frames: u32,
    /// Frames rendered before the first convergence check.
    pub min_frames: u32,
    /// Converged when the maximum per-pixel estimated variance of the mean
    /// frame luminance (Welford M2/(N*(N-1)) over all frames) < this.
    pub variance_threshold: f32,
}

/// Converged reference output.
pub struct TerrainReferenceOutput {
    pub rgba: Vec<u8>,
    pub albedo: Vec<f32>,
    pub normal: Vec<f32>,
    pub depth: Vec<f32>,
    /// Mean linear radiance per pixel (accumulated RGB / frames), len = 3*W*H.
    pub radiance: Vec<f32>,
    /// Per-pixel estimated variance of the mean frame luminance
    /// (M2/(N*(N-1)) over all frames), len = W*H.
    pub luminance_variance: Vec<f32>,
    pub frames: u32,
    /// Maximum of `luminance_variance` — the scalar convergence metric.
    pub variance: f32,
    /// Measured traversal counters from the last beauty frame, summed over
    /// pixels; lanes = (node_visits, minmax_loads, height_loads, rays).
    pub traversal_primary: [u64; 4],
    pub traversal_shadow: [u64; 4],
    /// Min-max pyramid depth used by the heightfield descent.
    pub traversal_mip_count: u32,
    pub converged: bool,
    pub peak_host_visible_bytes: u64,
    pub minmax_pyramid_bytes: u64,
    /// Sum of every GPU resource this render registered with the memory
    /// tracker (pyramid, env, accum, Welford, reservoirs, G-buffer, UBOs,
    /// output + AOV textures, mesh buffers).
    pub gpu_resource_bytes: u64,
}

/// Accumulates the byte sizes of every GPU resource this render creates so the
/// `gpu_resource_bytes` diagnostic can report the working set. The actual
/// global-memory-tracker and allocation-ledger lifecycle is owned by the
/// `tracked_create_*` wrappers each resource is allocated through (which also
/// free on drop), so this helper only bookkeeps sizes.
struct TrackedGpu {
    buffers: Vec<u64>,
    textures: Vec<(u32, u32, wgpu::TextureFormat, u64)>,
}

impl TrackedGpu {
    fn new() -> Self {
        Self {
            buffers: Vec::new(),
            textures: Vec::new(),
        }
    }

    fn buffer(&mut self, buf: &wgpu::Buffer) {
        self.buffers.push(buf.size());
    }

    fn texture(&mut self, width: u32, height: u32, format: wgpu::TextureFormat) {
        let bpp: u64 = match format {
            wgpu::TextureFormat::Rgba32Float => 16,
            wgpu::TextureFormat::Rgba16Float => 8,
            wgpu::TextureFormat::Rg32Float => 8,
            wgpu::TextureFormat::R32Float | wgpu::TextureFormat::Rgba8Unorm => 4,
            _ => 4,
        };
        self.textures.push((
            width,
            height,
            format,
            (width as u64) * (height as u64) * bpp,
        ));
    }

    fn bytes(&self) -> u64 {
        self.buffers.iter().sum::<u64>() + self.textures.iter().map(|t| t.3).sum::<u64>()
    }
}

fn read_texture_pixels(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    width: u32,
    height: u32,
    bytes_per_pixel: u32,
) -> Result<Vec<u8>, RenderError> {
    let unpadded = width * bytes_per_pixel;
    let padded = align_copy_bpr(unpadded);
    let size = (padded as u64) * (height as u64);
    let staging = tracked_create_buffer(
        device,
        &wgpu::BufferDescriptor {
            label: Some("hybrid-pt-terrain-readback"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        },
    )?;
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("hybrid-pt-terrain-readback-enc"),
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
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::Maintain::Wait);
    let out = {
        let data = slice.get_mapped_range();
        let mut rows = Vec::with_capacity((unpadded as usize) * (height as usize));
        for y in 0..height as usize {
            let start = y * padded as usize;
            rows.extend_from_slice(&data[start..start + unpadded as usize]);
        }
        rows
    };
    staging.unmap();
    drop(staging);
    Ok(out)
}

fn read_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    size: u64,
) -> Result<Vec<u8>, RenderError> {
    if !buffer.usage().contains(wgpu::BufferUsages::COPY_SRC) {
        return Err(RenderError::render(
            "terrain PT readback source buffer requires COPY_SRC usage",
        ));
    }
    let staging = tracked_create_buffer(
        device,
        &wgpu::BufferDescriptor {
            label: Some("hybrid-pt-terrain-buf-readback"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        },
    )?;
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("hybrid-pt-terrain-buf-enc"),
    });
    enc.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
    queue.submit([enc.finish()]);
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::Maintain::Wait);
    let out = {
        let data = slice.get_mapped_range();
        data.to_vec()
    };
    staging.unmap();
    drop(staging);
    Ok(out)
}

fn f16_bytes_to_rgb(data: &[u8], width: u32, height: u32) -> Vec<f32> {
    let mut out = Vec::with_capacity((width as usize) * (height as usize) * 3);
    for px in data.chunks_exact(8) {
        for c in 0..3 {
            let bits = u16::from_le_bytes([px[c * 2], px[c * 2 + 1]]);
            out.push(f16::from_bits(bits).to_f32());
        }
    }
    out
}

fn finite3(v: [f32; 3]) -> bool {
    v.iter().all(|x| x.is_finite())
}

fn factor_sun_lighting(sun_intensity: f32, sun_color: [f32; 3]) -> ([f32; 3], f32, [f32; 3]) {
    (
        [
            sun_intensity * sun_color[0],
            sun_intensity * sun_color[1],
            sun_intensity * sun_color[2],
        ],
        sun_intensity.max(1e-6),
        sun_color,
    )
}

fn should_require_valid_sun_reservoirs(
    sun_elevation_deg: f32,
    sun_intensity: f32,
    sun_color: [f32; 3],
) -> bool {
    sun_elevation_deg > 0.0 && sun_intensity > 0.0 && sun_color.iter().any(|c| *c > 0.0)
}

/// Trust-boundary validation of every public input before any GPU work.
fn validate_desc(desc: &TerrainReferenceDesc) -> Result<(), RenderError> {
    let err = |msg: String| Err(RenderError::Render(msg));
    if desc.width == 0 || desc.height == 0 || desc.max_frames == 0 {
        return err("terrain reference requires non-zero width/height/max_frames".into());
    }
    if desc.min_frames > desc.max_frames {
        return err(format!(
            "min_frames ({}) must be <= max_frames ({})",
            desc.min_frames, desc.max_frames
        ));
    }
    if desc.spp == 0 || desc.spp > 64 {
        return err(format!("spp must be in 1..=64, got {}", desc.spp));
    }
    if !(desc.exaggeration.is_finite() && desc.exaggeration > 0.0) {
        return err("terrain exaggeration must be finite and > 0".into());
    }
    if !(finite3(desc.cam_origin) && finite3(desc.cam_look_at) && finite3(desc.cam_up)) {
        return err("camera origin/look_at/up must be finite".into());
    }
    let origin = glam::Vec3::from(desc.cam_origin);
    let forward = glam::Vec3::from(desc.cam_look_at) - origin;
    if forward.length() < 1e-6 {
        return err("camera look_at must differ from origin".into());
    }
    if forward
        .normalize()
        .cross(glam::Vec3::from(desc.cam_up))
        .length()
        < 1e-6
    {
        return err("camera up vector must not be parallel to the view direction".into());
    }
    if !(desc.fov_y_deg.is_finite() && desc.fov_y_deg > 0.0 && desc.fov_y_deg < 180.0) {
        return err(format!(
            "fov_y must be finite and in (0, 180) degrees, got {}",
            desc.fov_y_deg
        ));
    }
    if !(desc.exposure.is_finite() && desc.exposure > 0.0) {
        return err("exposure must be finite and > 0".into());
    }
    if !(desc.sun_azimuth_deg.is_finite() && desc.sun_elevation_deg.is_finite()) {
        return err("sun azimuth/elevation must be finite".into());
    }
    if !(desc.sun_intensity.is_finite() && desc.sun_intensity >= 0.0) {
        return err("sun intensity must be finite and >= 0".into());
    }
    if !finite3(desc.sun_color) || desc.sun_color.iter().any(|c| *c < 0.0) {
        return err("sun color must have three finite non-negative components".into());
    }
    if !(desc.env_intensity.is_finite() && desc.env_intensity >= 0.0) {
        return err("env intensity must be finite and >= 0".into());
    }
    if !(desc.variance_threshold.is_finite() && desc.variance_threshold > 0.0) {
        return err("variance threshold must be finite and > 0".into());
    }
    if !(desc.spacing.0.is_finite()
        && desc.spacing.0 > 0.0
        && desc.spacing.1.is_finite()
        && desc.spacing.1 > 0.0)
    {
        return err(format!(
            "terrain spacing must be finite and > 0, got {:?}",
            desc.spacing
        ));
    }
    if let Some((verts, idx)) = &desc.mesh {
        if verts.is_empty() || verts.len() % 3 != 0 {
            return err("mesh vertices must be a non-empty flat [x,y,z] list".into());
        }
        if idx.is_empty() || idx.len() % 3 != 0 {
            return err("mesh indices must be a non-empty multiple of 3".into());
        }
        if verts.iter().any(|v| !v.is_finite()) {
            return err("mesh vertices contain non-finite values".into());
        }
        let vcount = (verts.len() / 3) as u32;
        if idx.iter().any(|i| *i >= vcount) {
            return err("mesh indices reference out-of-bounds vertices".into());
        }
    }
    Ok(())
}

impl HybridPathTracer {
    /// Render the converged terrain reference. Errors (rather than returning
    /// a fake image) when inputs are degenerate, the frame cap is hit
    /// without convergence, or the memory budget is exceeded.
    /// Thin public entry: validates the description, and when a native
    /// `sdf_scene` is present compiles it into the shared hybrid kernel's
    /// constant declarations and builds the pipelines from that specialized
    /// source (one new module/pipeline set for this render — the `self`
    /// pipelines keep the default empty-SDF source). With no SDF the
    /// existing pipelines render unchanged.
    pub fn render_terrain_reference(
        &self,
        desc: &TerrainReferenceDesc,
    ) -> Result<TerrainReferenceOutput, RenderError> {
        validate_desc(desc)?;
        if let Some(scene) = &desc.sdf_scene {
            let source = super::sdf_scene::specialize(scene)?;
            return Self::new_with_source(source)?.render_terrain_reference_inner(desc);
        }
        self.render_terrain_reference_inner(desc)
    }

    fn render_terrain_reference_inner(
        &self,
        desc: &TerrainReferenceDesc,
    ) -> Result<TerrainReferenceOutput, RenderError> {
        let device = &try_ctx()?.device;
        let queue = &try_ctx()?.queue;
        let (width, height) = (desc.width, desc.height);
        validate_desc(desc)?;
        let exposure = desc.exposure.clamp(0.0, AETHER_RADIOMETRIC_SCALE_MAX);
        let sun_intensity = desc.sun_intensity.clamp(0.0, AETHER_RADIOMETRIC_SCALE_MAX);
        let sun_color = desc
            .sun_color
            .map(|value| value.clamp(0.0, AETHER_RADIOMETRIC_SCALE_MAX));
        let env_intensity = desc.env_intensity.clamp(0.0, AETHER_RADIOMETRIC_SCALE_MAX);
        let mut tracked = TrackedGpu::new();

        // --- Terrain scene: min-max pyramid + env map (validates the DEM,
        // registers itself with the memory tracker, frees on drop) ---
        let terrain_scene = TerrainPtScene::new(
            device,
            queue,
            &desc.heights,
            desc.dem_width,
            desc.dem_height,
            desc.spacing,
            desc.exaggeration,
            desc.albedo,
            desc.env_map
                .as_ref()
                .map(|(data, w, h)| (data.as_slice(), *w, *h)),
            env_intensity,
        )?;

        // --- Optional mesh mixed through the shared HybridScene seam ---
        let hybrid_scene = match &desc.mesh {
            Some((verts, idx)) => {
                let vertices: Vec<crate::sdf::hybrid::Vertex> = verts
                    .chunks_exact(3)
                    .map(|c| crate::sdf::hybrid::Vertex {
                        position: [c[0], c[1], c[2]],
                        _pad: 0.0,
                    })
                    .collect();
                let tris: Vec<crate::accel::types::Triangle> = idx
                    .chunks_exact(3)
                    .map(|t| {
                        let p = |i: u32| {
                            let b = (i as usize) * 3;
                            [verts[b], verts[b + 1], verts[b + 2]]
                        };
                        crate::accel::types::Triangle::new(p(t[0]), p(t[1]), p(t[2]))
                    })
                    .collect();
                let bvh = crate::accel::build_bvh(
                    &tris,
                    &crate::accel::types::BuildOptions::default(),
                    crate::accel::GpuContext::NotAvailable,
                )
                .map_err(|e| RenderError::Render(format!("mesh BVH build failed: {e}")))?;
                let mut scene = HybridScene::mesh_only(vertices, idx.clone(), bvh);
                scene.prepare_gpu_resources()?;
                scene
            }
            None => HybridScene::new(),
        };
        if let Some(mesh_buffers) = &hybrid_scene.mesh_buffers {
            tracked.buffer(&mesh_buffers.vertices_buffer);
            tracked.buffer(&mesh_buffers.indices_buffer);
            tracked.buffer(&mesh_buffers.bvh_buffer);
        }

        // --- Camera + lighting uniforms ---
        let origin = glam::Vec3::from(desc.cam_origin);
        let forward = (glam::Vec3::from(desc.cam_look_at) - origin).normalize();
        let right = forward.cross(glam::Vec3::from(desc.cam_up)).normalize();
        let up = right.cross(forward).normalize();
        let az = desc.sun_azimuth_deg.to_radians();
        let el = desc.sun_elevation_deg.to_radians();
        // Direction from surface TOWARD the sun (kernel convention).
        let light_dir = [az.cos() * el.cos(), el.sin(), az.sin() * el.cos()];
        let (light_color, restir_sun_intensity, restir_sun_color) =
            factor_sun_lighting(sun_intensity, sun_color);

        let mut base = Uniforms {
            width,
            height,
            frame_index: 0,
            aov_flags: 0xFF, // frame 0 writes all AOVs from the center ray
            cam_origin: desc.cam_origin,
            cam_fov_y: desc.fov_y_deg.to_radians(),
            cam_right: right.into(),
            cam_aspect: width as f32 / height as f32,
            cam_up: up.into(),
            cam_exposure: exposure,
            cam_forward: forward.into(),
            seed_hi: desc.seed,
            // decorrelate seed_hi ^ seed_lo in the kernel's xor mixing so
            // distinct seeds produce distinct streams (seed ^ C cancels).
            seed_lo: desc
                .seed
                .wrapping_mul(0x9E37_79B9)
                .wrapping_add(0x85EB_CA6B),
            _pad_end: [0; 3],
        };
        let base_ubo = tracked_create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("hybrid-pt-terrain-base-ubo"),
                contents: bytemuck::bytes_of(&base),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        )?;
        tracked.buffer(&base_ubo);
        // The SDF scene lives entirely in the specialized shader's constants;
        // the count fields keep the static ABI at 0 while the shader branches
        // on HYBRID_SDF_ENABLED rather than these uniforms. The observer below
        // records the validated counts for runtime evidence.
        let hybrid_uniforms = HybridUniforms {
            sdf_primitive_count: 0,
            sdf_node_count: 0,
            mesh_vertex_count: hybrid_scene.vertices.len() as u32,
            mesh_index_count: hybrid_scene.indices.len() as u32,
            mesh_bvh_node_count: hybrid_scene
                .mesh_buffers
                .as_ref()
                .map(|m| m.bvh_node_count)
                .unwrap_or(0),
            traversal_mode: if desc.mesh.is_some() || desc.sdf_scene.is_some() {
                TraversalMode::Hybrid as u32
            } else {
                TraversalMode::TerrainOnly as u32
            },
            _pad: [0; 2],
        };
        let hybrid_ubo = tracked_create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("hybrid-pt-terrain-hybrid-ubo"),
                contents: bytemuck::bytes_of(&hybrid_uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            },
        )?;
        tracked.buffer(&hybrid_ubo);
        let lighting = LightingUniforms {
            light_dir,
            lighting_type: 1,
            light_color,
            shadows_enabled: 1,
            ambient_color: [0.0, 0.0, 0.0],
            shadow_intensity: 1.0,
            hdri_intensity: env_intensity,
            hdri_rotation: 0.0,
            specular_power: 32.0,
            _pad: [0; 5],
        };
        let lighting_ubo = tracked_create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("hybrid-pt-terrain-lighting-ubo"),
                contents: bytemuck::bytes_of(&lighting),
                usage: wgpu::BufferUsages::UNIFORM,
            },
        )?;
        tracked.buffer(&lighting_ubo);
        let terrain_uniforms = terrain_scene.uniforms(desc.spp, WELFORD_WINDOW);
        let terrain_ubo = tracked_create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("hybrid-pt-terrain-ubo"),
                contents: bytemuck::bytes_of(&terrain_uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            },
        )?;
        tracked.buffer(&terrain_ubo);
        let earth_curvature = super::terrain_heightfield::EarthCurvatureUniforms::new(
            desc.earth_model,
            desc.refraction_model,
            desc.observer_geodetic_deg,
            f64::from(desc.sun_azimuth_deg),
        )?;
        let earth_curvature_ubo = tracked_create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("hybrid-pt-earth-curvature-ubo"),
                contents: bytemuck::bytes_of(&earth_curvature),
                usage: wgpu::BufferUsages::UNIFORM,
            },
        )?;
        tracked.buffer(&earth_curvature_ubo);

        // --- Scene buffers (the spheres slot needs 1 dummy element) ---
        let scene_buf = tracked_create_buffer(
            device,
            &wgpu::BufferDescriptor {
                label: Some("hybrid-pt-terrain-scene"),
                size: std::mem::size_of::<Sphere>() as u64,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            },
        )?;
        tracked.buffer(&scene_buf);

        // --- Directional-light table for the ReSTIR spatial pass ---
        let sun_light = GpuDirectionalLight::new(
            [-light_dir[0], -light_dir[1], -light_dir[2]],
            restir_sun_intensity,
            restir_sun_color,
            1.0,
        );
        let dir_lights_buf = tracked_create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("hybrid-pt-terrain-dir-lights"),
                contents: bytemuck::bytes_of(&sun_light),
                usage: wgpu::BufferUsages::STORAGE,
            },
        )?;
        tracked.buffer(&dir_lights_buf);
        let area_light = <GpuAreaLight as bytemuck::Zeroable>::zeroed();
        let area_lights_buf = tracked_create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("hybrid-pt-terrain-area-lights"),
                contents: bytemuck::bytes_of(&area_light),
                usage: wgpu::BufferUsages::STORAGE,
            },
        )?;
        tracked.buffer(&area_lights_buf);

        // --- Accumulation (same shape as render.rs "hybrid-pt-accum"),
        // per-pixel terrain statistics (Welford moments + traversal
        // counters), canonical ReSTIR reservoirs and G-buffer ---
        let px_count = (width as u64) * (height as u64);
        let px_usize = px_count as usize;
        let stats_size = std::mem::size_of::<TerrainStatistics>() as u64;
        let accum_buf = tracked_create_buffer(
            device,
            &wgpu::BufferDescriptor {
                label: Some("hybrid-pt-accum"),
                size: px_count * 16,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            },
        )?;
        tracked.buffer(&accum_buf);
        let welford_buf = tracked_create_buffer(
            device,
            &wgpu::BufferDescriptor {
                label: Some("hybrid-pt-terrain-welford"),
                size: px_count * stats_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            },
        )?;
        tracked.buffer(&welford_buf);
        // Canonical Reservoir buffers (80-byte stride, matching
        // src/path_tracing/restir/types.rs and the pt_restir_* passes):
        // curr = fresh candidates, out = temporal merge, prev = final merged
        // history (spatial writes back into prev; the kernel shades from it).
        let reservoir_curr = create_reservoir_buffer(device, px_usize)?;
        let reservoir_out = create_reservoir_buffer(device, px_usize)?;
        let prev_init = vec![Reservoir::default(); px_usize];
        let reservoir_prev = tracked_create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("hybrid-pt-terrain-reservoir-prev"),
                contents: bytemuck::cast_slice(&prev_init),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            },
        )?;
        tracked.buffer(&reservoir_curr);
        tracked.buffer(&reservoir_out);
        tracked.buffer(&reservoir_prev);
        let gbuffer_nr = create_restir_gbuffer(device, px_usize)?;
        let gbuffer_pos = create_restir_gbuffer_pos(device, px_usize)?;
        tracked.buffer(&gbuffer_nr);
        tracked.buffer(&gbuffer_pos);

        // --- Output + AOV targets (matching _pt_render_gpu_mesh's plumbing) ---
        let out_tex = tracked_create_texture(
            device,
            &wgpu::TextureDescriptor {
                label: Some("hybrid-pt-terrain-out"),
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
        tracked.texture(width, height, wgpu::TextureFormat::Rgba16Float);
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
        let aov_frames = AovFrames::new(device, width, height, &aovs_all)?;
        for kind in aovs_all {
            tracked.texture(width, height, kind.texture_format());
        }
        let aov_views: Vec<wgpu::TextureView> = aovs_all
            .iter()
            .map(|k| {
                aov_frames
                    .get_texture(*k)
                    .unwrap()
                    .create_view(&wgpu::TextureViewDescriptor::default())
            })
            .collect();

        // --- Memory-budget gate (hard): everything above is registered with
        // the tracker; refuse to render if the working set exceeds the
        // 512 MiB budget. ---
        let base_gpu_resource_bytes = tracked.bytes() + terrain_scene.byte_size();
        let metrics = global_tracker().get_metrics();
        if metrics.total_bytes > metrics.limit_bytes
            || metrics.host_visible_bytes > metrics.limit_bytes
        {
            return Err(RenderError::Render(format!(
                "terrain PT exceeds the memory budget before rendering: tracked total {} \
                 (host-visible {}) > limit {}",
                metrics.total_bytes, metrics.host_visible_bytes, metrics.limit_bytes
            )));
        }

        // --- Bind groups ---
        let bg0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hybrid-pt-terrain-bg0"),
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
        let bg1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hybrid-pt-terrain-bg1"),
            layout: &self.layouts.scene,
            entries: &bg1_entries,
        });
        let height_view = terrain_scene
            .pyramid
            .height_texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let minmax_view = terrain_scene
            .pyramid
            .minmax_texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let env_view = terrain_scene
            .env_texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let bg2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hybrid-pt-terrain-bg2"),
            layout: &self.layouts.accum,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: accum_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&height_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&minmax_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: terrain_ubo.as_entire_binding(),
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
                    resource: wgpu::BindingResource::TextureView(&env_view),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: reservoir_prev.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: earth_curvature_ubo.as_entire_binding(),
                },
            ],
        });
        let mut bg3_entries = vec![wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(&out_view),
        }];
        for (i, view) in aov_views.iter().enumerate() {
            bg3_entries.push(wgpu::BindGroupEntry {
                binding: (i as u32) + 1,
                resource: wgpu::BindingResource::TextureView(view),
            });
        }
        let bg3 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hybrid-pt-terrain-bg3"),
            layout: &self.layouts.output,
            entries: &bg3_entries,
        });
        let bg_gbuffer = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hybrid-pt-terrain-bg-gbuffer"),
            layout: &self.layouts.terrain_gbuffer,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&height_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&minmax_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: terrain_ubo.as_entire_binding(),
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
                    resource: earth_curvature_ubo.as_entire_binding(),
                },
            ],
        });
        let bg_empty = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hybrid-pt-terrain-bg-empty"),
            layout: &self.layouts.empty,
            entries: &[],
        });
        // Temporal: prev + curr -> out.
        let bg_temporal = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hybrid-pt-terrain-bg-restir-temporal"),
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
            label: Some("hybrid-pt-terrain-bg-restir-spatial-scene"),
            layout: &self.layouts.restir_spatial_scene,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: area_lights_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: dir_lights_buf.as_entire_binding(),
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
        // Spatial: out (temporal result) -> prev, which becomes the merged
        // history the kernel shades from next frame.
        let bg_spatial_reuse = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hybrid-pt-terrain-bg-restir-spatial-reuse"),
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

        // --- One-shot ReSTIR G-buffer pass (static scene + camera) ---
        let wg_x = width.div_ceil(8);
        let wg_y = height.div_ceil(8);
        let px_workgroups = ((px_count as u32) + 255) / 256;
        // CENSOR F-04: live per-pass timing for the certificate. The gbuffer
        // pass and frame 0's terrain/temporal/spatial dispatches are each
        // bracketed on the encoder that executes them (one scope per label —
        // timing every accumulation frame would multiply the pass list);
        // when timestamp queries are unavailable the fallback below records
        // the same labels in the same order with 0.0.
        let mut timing = crate::core::gpu_timing::OneShotTiming::for_current_device();
        {
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("hybrid-pt-terrain-gbuffer-enc"),
            });
            let gbuffer_scope = timing.begin(&mut enc, "hybrid_pt.terrain_gbuffer");
            {
                let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("hybrid-pt-terrain-gbuffer-cpass"),
                    ..Default::default()
                });
                crate::core::shader_registry::record_shader_use("hybrid-pt-kernel");
                cpass.set_pipeline(&self.pipeline_terrain_gbuffer);
                cpass.set_bind_group(0, &bg0, &[]);
                cpass.set_bind_group(1, &bg1, &[]);
                cpass.set_bind_group(2, &bg_gbuffer, &[]);
                cpass.dispatch_workgroups(wg_x, wg_y, 1);
            }
            timing.end(&mut enc, gbuffer_scope, 1);
            queue.submit([enc.finish()]);
        }
        if desc.sdf_scene.is_some() {
            // The gbuffer pipeline's layout cannot reach terrain_welford, so
            // the kernel signals a compiled-SDF march failure on the center
            // ray through terrain_gbuffer_pos[pix].w = -1. Reject it here
            // before the accumulation loop consumes the record.
            let pos_bytes = read_buffer(device, queue, &gbuffer_pos, px_count * 16)?;
            let pos: &[f32] = bytemuck::cast_slice(&pos_bytes);
            for px in pos.chunks_exact(4) {
                if !px[3].is_finite() || px[3] < 0.0 {
                    return Err(RenderError::render(
                        "terrain PT SDF traversal did not converge or produced invalid geometry",
                    ));
                }
            }
        }

        // --- Accumulate until converged (estimated variance of the mean
        // frame luminance over all frames, polled every WELFORD_WINDOW
        // frames and at the cap) or capped ---
        let mut frames = 0u32;
        let mut variance = f32::INFINITY;
        let mut converged = false;
        while frames < desc.max_frames {
            base.frame_index = frames;
            base.aov_flags = if frames == 0 { 0xFF } else { 0 };
            queue.write_buffer(&base_ubo, 0, bytemuck::bytes_of(&base));
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("hybrid-pt-terrain-frame"),
            });
            // Only frame 0 carries timing scopes (one certificate pass per
            // label); each scope brackets exactly one dispatch of its
            // pipeline on the encoder that executes it.
            let time_this_frame = frames == 0;
            let terrain_scope = if time_this_frame {
                timing.begin(&mut enc, "hybrid_pt.terrain")
            } else {
                None
            };
            {
                let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("hybrid-pt-terrain-cpass"),
                    ..Default::default()
                });
                crate::core::shader_registry::record_shader_use("hybrid-pt-kernel");
                cpass.set_pipeline(&self.pipeline_terrain);
                cpass.set_bind_group(0, &bg0, &[]);
                cpass.set_bind_group(1, &bg1, &[]);
                cpass.set_bind_group(2, &bg2, &[]);
                cpass.set_bind_group(3, &bg3, &[]);
                cpass.dispatch_workgroups(wg_x, wg_y, 1);
            }
            if time_this_frame {
                timing.end(&mut enc, terrain_scope, 1);
            }
            let temporal_scope = if time_this_frame {
                timing.begin(&mut enc, "hybrid_pt.restir_temporal")
            } else {
                None
            };
            {
                let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("hybrid-pt-restir-temporal-cpass"),
                    ..Default::default()
                });
                crate::core::shader_registry::record_shader_use("hybrid-pt-restir-temporal");
                cpass.set_pipeline(&self.pipeline_restir_temporal);
                cpass.set_bind_group(0, &bg0, &[]);
                cpass.set_bind_group(1, &bg_empty, &[]);
                cpass.set_bind_group(2, &bg_temporal, &[]);
                cpass.dispatch_workgroups(px_workgroups, 1, 1);
            }
            if time_this_frame {
                timing.end(&mut enc, temporal_scope, 1);
            }
            let spatial_scope = if time_this_frame {
                timing.begin(&mut enc, "hybrid_pt.restir_spatial")
            } else {
                None
            };
            {
                let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("hybrid-pt-restir-spatial-cpass"),
                    ..Default::default()
                });
                crate::core::shader_registry::record_shader_use("hybrid-pt-restir-spatial");
                cpass.set_pipeline(&self.pipeline_restir_spatial);
                cpass.set_bind_group(0, &bg0, &[]);
                cpass.set_bind_group(1, &bg_spatial_scene, &[]);
                cpass.set_bind_group(2, &bg_spatial_reuse, &[]);
                cpass.dispatch_workgroups(px_workgroups, 1, 1);
            }
            if time_this_frame {
                timing.end(&mut enc, spatial_scope, 1);
            }
            let publish_scope = if time_this_frame {
                timing.begin(&mut enc, "hybrid_pt.terrain_publish")
            } else {
                None
            };
            {
                let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("hybrid-pt-terrain-publish-cpass"),
                    ..Default::default()
                });
                crate::core::shader_registry::record_shader_use("hybrid-pt-kernel");
                cpass.set_pipeline(&self.pipeline_terrain_publish);
                cpass.set_bind_group(0, &bg0, &[]);
                cpass.set_bind_group(1, &bg1, &[]);
                cpass.set_bind_group(2, &bg2, &[]);
                cpass.set_bind_group(3, &bg3, &[]);
                cpass.dispatch_workgroups(wg_x, wg_y, 1);
            }
            if time_this_frame {
                timing.end(&mut enc, publish_scope, 1);
                // Resolve on frame 0's encoder: the gbuffer scope's stamps
                // were written on the already-submitted gbuffer encoder, so
                // the resolved range is complete once this submit lands.
                timing.resolve(&mut enc);
            }
            queue.submit([enc.finish()]);
            frames += 1;

            let window_full = frames.is_multiple_of(WELFORD_WINDOW);
            if (window_full || frames == desc.max_frames) && frames >= 2 {
                device.poll(wgpu::Maintain::Wait);
                let stats_bytes = read_buffer(device, queue, &welford_buf, px_count * stats_size)?;
                let stats = parse_terrain_statistics(&stats_bytes)?;
                let mut vmax = 0.0f32;
                for s in stats {
                    // Estimated variance of the mean frame luminance over all
                    // `frames` frames.
                    vmax = vmax.max(estimator_variance(s.y, frames)?);
                }
                variance = vmax;
                if frames >= desc.min_frames && variance < desc.variance_threshold {
                    converged = true;
                    break;
                }
            }
        }
        device.poll(wgpu::Maintain::Wait);
        if !converged {
            return Err(RenderError::Render(format!(
                "terrain PT did not converge: estimated variance of the mean frame \
                 luminance {variance:.3e} after {frames} frames (threshold {:.1e}); \
                 raise max_frames or simplify the scene — refusing to return a \
                 fake reference",
                desc.variance_threshold
            )));
        }

        // Allocate and upload AETHER's post-only resources after PROMETHEUS
        // has finished its frame-0 AOV writes and convergence loop. This keeps
        // the reference traversal's resource/queue schedule unchanged.
        let aether_post = desc
            .atmosphere
            .as_ref()
            .map(|lut_handle| {
                super::aether_post::AetherPostPass::new(
                    device,
                    queue,
                    lut_handle,
                    width,
                    height,
                    desc.cam_origin,
                    right.into(),
                    up.into(),
                    forward.into(),
                    desc.fov_y_deg.to_radians(),
                    exposure,
                    light_dir,
                    sun_intensity,
                    &accum_buf,
                    &out_view,
                )
            })
            .transpose()?;
        let gpu_resource_bytes =
            base_gpu_resource_bytes + aether_post.as_ref().map_or(0, |post| post.gpu_bytes());
        let post_metrics = global_tracker().get_metrics();
        if post_metrics.total_bytes > post_metrics.limit_bytes
            || post_metrics.host_visible_bytes > post_metrics.limit_bytes
        {
            return Err(RenderError::Render(format!(
                "terrain PT exceeds the memory budget with AETHER post resources: tracked total {} \
                 (host-visible {}) > limit {}",
                post_metrics.total_bytes,
                post_metrics.host_visible_bytes,
                post_metrics.limit_bytes
            )));
        }

        // AETHER consumes the existing linear accumulation and exact frame-0
        // depth AOV after convergence. The original PROMETHEUS traversal,
        // bind groups, reservoir reuse, and accumulation layout stay untouched.
        let mut aether_post_timing = None;
        if let Some(post) = aether_post.as_ref() {
            let mut post_timing = crate::core::gpu_timing::OneShotTiming::for_current_device();
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("hybrid-pt-aether-post-encoder"),
            });
            let scope = post_timing.begin(&mut encoder, "hybrid_pt.aether_aerial");
            post.encode(
                &mut encoder,
                queue,
                aov_frames.get_texture(AovKind::Depth).unwrap(),
                aov_frames.get_texture(AovKind::Visibility).unwrap(),
                frames,
            );
            post_timing.end(&mut encoder, scope, 1);
            post_timing.resolve(&mut encoder);
            queue.submit([encoder.finish()]);
            device.poll(wgpu::Maintain::Wait);
            // Record only after the already-executed base scopes below so the
            // certificate preserves queue execution order.
            aether_post_timing = Some(post_timing);
        }

        // --- ReSTIR reservoir validity: the merged history must be finite
        // and (for a lit scene) actually populated by the reuse chain ---
        let res_stride = std::mem::size_of::<Reservoir>() as u64;
        let res_bytes = read_buffer(device, queue, &reservoir_prev, px_count * res_stride)?;
        let reservoirs: &[Reservoir] = bytemuck::cast_slice(&res_bytes);
        let mut any_valid = false;
        for r in reservoirs {
            if !(r.w_sum.is_finite() && r.weight.is_finite() && r.target_pdf.is_finite()) {
                return Err(RenderError::Render(
                    "terrain PT reservoir bookkeeping produced non-finite values".into(),
                ));
            }
            if r.m > 0 && r.weight > 0.0 && r.target_pdf > 0.0 {
                any_valid = true;
            }
        }
        if should_require_valid_sun_reservoirs(desc.sun_elevation_deg, sun_intensity, sun_color)
            && !any_valid
        {
            return Err(RenderError::Render(
                "terrain PT ReSTIR reuse chain produced no valid reservoirs for a sun-lit \
                 scene — temporal/spatial reuse is broken"
                    .into(),
            ));
        }

        // --- Readbacks ---
        let beauty = read_texture_pixels(device, queue, &out_tex, width, height, 8)?;
        let accum_bytes = read_buffer(device, queue, &accum_buf, px_count * 16)?;
        let accum: &[f32] = bytemuck::cast_slice(&accum_bytes);
        let welford_bytes = read_buffer(device, queue, &welford_buf, px_count * stats_size)?;
        let stats = parse_terrain_statistics(&welford_bytes)?;
        // Interleaved mean/M2 view keeps the runtime-observer contract
        // (`terrain_welford.mean` / `.m2` ranges, length 2*pixels) unchanged.
        let welford: Vec<f32> = stats.iter().flat_map(|s| [s.x, s.y]).collect();
        let mut luminance_variance = Vec::with_capacity(px_usize);
        let mut traversal_primary = [0u64; 4];
        let mut traversal_shadow = [0u64; 4];
        for s in stats {
            luminance_variance.push(estimator_variance(s.y, frames)?);
            for lane in 0..4 {
                traversal_primary[lane] += s.primary[lane] as u64;
                traversal_shadow[lane] += s.shadow[lane] as u64;
            }
        }
        // Mean linear radiance per pixel; the same data the observer range-
        // checks as accum_hdr.samples.
        let mut radiance = Vec::with_capacity(px_usize * 3);
        for px in accum.chunks_exact(4) {
            for c in 0..3 {
                let v = px[c] / frames as f32;
                if !v.is_finite() {
                    return Err(RenderError::render(
                        "terrain PT produced non-finite accumulated radiance",
                    ));
                }
                radiance.push(v);
            }
        }
        record_runtime_contract(
            desc,
            &base,
            &hybrid_uniforms,
            &lighting,
            &terrain_uniforms,
            &earth_curvature,
            frames,
            reservoirs,
            accum,
            &welford,
            &beauty,
        )?;
        let mut rgba = vec![0u8; (width as usize) * (height as usize) * 4];
        for (i, px) in beauty.chunks_exact(8).enumerate() {
            for c in 0..3 {
                let bits = u16::from_le_bytes([px[c * 2], px[c * 2 + 1]]);
                let v = f16::from_bits(bits).to_f32();
                rgba[i * 4 + c] = (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            }
            rgba[i * 4 + 3] = 255;
        }
        let albedo_bytes = read_texture_pixels(
            device,
            queue,
            aov_frames.get_texture(AovKind::Albedo).unwrap(),
            width,
            height,
            8,
        )?;
        let normal_bytes = read_texture_pixels(
            device,
            queue,
            aov_frames.get_texture(AovKind::Normal).unwrap(),
            width,
            height,
            8,
        )?;
        let depth_bytes = read_texture_pixels(
            device,
            queue,
            aov_frames.get_texture(AovKind::Depth).unwrap(),
            width,
            height,
            4,
        )?;
        let albedo = f16_bytes_to_rgb(&albedo_bytes, width, height);
        let normal = f16_bytes_to_rgb(&normal_bytes, width, height);
        let depth: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&depth_bytes).to_vec();

        // --- Budget guardrail (hard): peak host-visible includes the staging
        // readbacks; the tracked working set was gated before rendering. ---
        let metrics = global_tracker().get_metrics();
        let peak = metrics.peak_host_visible_bytes;
        if peak > metrics.limit_bytes {
            return Err(RenderError::Render(format!(
                "terrain PT exceeded the host-visible budget: peak {peak} > limit {}",
                metrics.limit_bytes
            )));
        }

        // CENSOR F-04: record the timed scopes (gbuffer + frame 0's terrain/
        // temporal/spatial dispatches, draw_calls = 1 dispatch each); when
        // timestamps were unavailable, record the same labels in the same
        // order with 0.0 (draw_calls = accumulated frames, as before).
        if !timing.record_into_certificate() {
            crate::core::certificate::record_pass("hybrid_pt.terrain_gbuffer", 0.0, 1);
            crate::core::certificate::record_pass("hybrid_pt.terrain", 0.0, frames);
            crate::core::certificate::record_pass("hybrid_pt.restir_temporal", 0.0, frames);
            crate::core::certificate::record_pass("hybrid_pt.restir_spatial", 0.0, frames);
            crate::core::certificate::record_pass("hybrid_pt.terrain_publish", 0.0, frames);
        }
        if let Some(post_timing) = aether_post_timing {
            if !post_timing.record_into_certificate() {
                crate::core::certificate::record_pass("hybrid_pt.aether_aerial", 0.0, 1);
            }
        }

        let variance = luminance_variance.iter().copied().fold(0.0f32, f32::max);

        Ok(TerrainReferenceOutput {
            rgba,
            albedo,
            normal,
            depth,
            radiance,
            luminance_variance,
            frames,
            variance,
            traversal_primary,
            traversal_shadow,
            traversal_mip_count: terrain_scene.pyramid.mip_count,
            converged,
            peak_host_visible_bytes: peak,
            minmax_pyramid_bytes: terrain_scene.pyramid.byte_size,
            gpu_resource_bytes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn contract_fixture(
        dawn: bool,
    ) -> (
        TerrainReferenceDesc,
        Uniforms,
        HybridUniforms,
        LightingUniforms,
        super::super::terrain_heightfield::TerrainPtUniforms,
        super::super::terrain_heightfield::EarthCurvatureUniforms,
    ) {
        let desc = TerrainReferenceDesc {
            heights: vec![if dawn { 0.5 } else { 0.0 }; 16],
            dem_width: 4,
            dem_height: 4,
            spacing: (1.0, 1.0),
            exaggeration: 1.0,
            albedo: [0.6; 3],
            cam_origin: if dawn {
                [0.0, 35.0, 90.0]
            } else {
                [0.0, 3.0, 8.0]
            },
            cam_look_at: [0.0; 3],
            cam_up: [0.0, 1.0, 0.0],
            fov_y_deg: 45.0,
            exposure: 1.0,
            sun_azimuth_deg: if dawn {
                1.28047261849929_f64 as f32
            } else {
                315.0
            },
            sun_elevation_deg: if dawn {
                0.8056487084063235_f64 as f32
            } else {
                45.0
            },
            sun_intensity: 2.5,
            sun_color: [1.0, 0.97, 0.92],
            observer_geodetic_deg: [0.0, 0.0],
            earth_model: crate::geo::refraction::EarthModel::Ellipsoid { latitude_deg: 0.0 },
            refraction_model: crate::geo::refraction::RefractionModel::Bennett {
                pressure_mbar: 1013.25,
                temperature_c: 15.0,
            },
            env_map: None,
            env_intensity: 0.35,
            atmosphere: None,
            mesh: None,
            sdf_scene: None,
            width: if dawn { 256 } else { 8 },
            height: 8,
            seed: 7,
            spp: 1,
            max_frames: if dawn { 256 } else { 4 },
            min_frames: 2,
            variance_threshold: 1e-3,
        };
        let forward =
            (glam::Vec3::from(desc.cam_look_at) - glam::Vec3::from(desc.cam_origin)).normalize();
        let right = forward.cross(glam::Vec3::from(desc.cam_up)).normalize();
        let base = Uniforms {
            width: desc.width,
            height: desc.height,
            frame_index: desc.max_frames - 1,
            aov_flags: 0,
            cam_origin: desc.cam_origin,
            cam_fov_y: desc.fov_y_deg.to_radians(),
            cam_right: right.to_array(),
            cam_aspect: desc.width as f32 / desc.height as f32,
            cam_up: right.cross(forward).normalize().to_array(),
            cam_exposure: desc.exposure,
            cam_forward: forward.to_array(),
            seed_hi: 7,
            seed_lo: 7u32.wrapping_mul(0x9E37_79B9).wrapping_add(0x85EB_CA6B),
            _pad_end: [0; 3],
        };
        let hybrid = HybridUniforms {
            traversal_mode: 3,
            ..Zeroable::zeroed()
        };
        let az = desc.sun_azimuth_deg.to_radians();
        let el = desc.sun_elevation_deg.to_radians();
        let lighting = LightingUniforms {
            light_dir: [az.cos() * el.cos(), el.sin(), az.sin() * el.cos()],
            light_color: desc.sun_color.map(|c| c * desc.sun_intensity),
            shadows_enabled: 1,
            ..Zeroable::zeroed()
        };
        let terrain = super::super::terrain_heightfield::TerrainPtUniforms {
            origin_spacing: [-1.5, -1.5, 1.0, 1.0],
            h_params: [desc.heights[0], desc.heights[0], 1.0, 0.35],
            albedo_pad: [0.6, 0.6, 0.6, 0.0],
            dims: [4, 4, 3, 3],
            mips: [3, 1, 0, 0],
            extra: [1, WELFORD_WINDOW, 0, 0],
        };
        let earth_curvature = super::super::terrain_heightfield::EarthCurvatureUniforms::new(
            desc.earth_model,
            desc.refraction_model,
            desc.observer_geodetic_deg,
            f64::from(desc.sun_azimuth_deg),
        )
        .expect("fixture earth-curvature uniforms");
        (desc, base, hybrid, lighting, terrain, earth_curvature)
    }

    fn sample_reservoir(lighting: &LightingUniforms) -> Reservoir {
        let mut r = Reservoir {
            m: 512,
            w_sum: 512.0,
            weight: 1.0,
            target_pdf: 1.0,
            ..Reservoir::default()
        };
        r.sample.light_type = 1;
        r.sample.direction = glam::Vec3::from(lighting.light_dir).normalize().to_array();
        r
    }

    #[test]
    fn canonical_proof_and_scene_runtime_are_distinct_scopes() {
        for dawn in [false, true] {
            let (desc, base, hybrid, lighting, terrain, earth_curvature) = contract_fixture(dawn);
            let pixels = (desc.width * desc.height) as usize;
            let reservoirs = vec![sample_reservoir(&lighting); pixels];
            let mut accum = vec![0.0; pixels * 4];
            for alpha in accum.iter_mut().skip(3).step_by(4) {
                *alpha = desc.max_frames as f32;
            }
            let welford = vec![0.0; pixels * 2];
            let beauty = vec![0; pixels * 8];
            assert_eq!(is_canonical_proof_fixture(&desc), !dawn);
            let runtime = observe_runtime_contract(
                &desc,
                &base,
                &hybrid,
                &lighting,
                &terrain,
                &earth_curvature,
                desc.max_frames,
                &reservoirs,
                &accum,
                &welford,
                &beauty,
                false,
            );
            assert_eq!(runtime.status, "passed", "{:?}", runtime.checked_bindings);
            assert_eq!(runtime.contract, "runtime-safety:hybrid-terrain");
            let proof = observe_runtime_contract(
                &desc,
                &base,
                &hybrid,
                &lighting,
                &terrain,
                &earth_curvature,
                desc.max_frames,
                &reservoirs,
                &accum,
                &welford,
                &beauty,
                true,
            );
            assert_eq!(proof.status, if dawn { "failed" } else { "passed" });
            assert_eq!(
                proof.contract,
                "shaders/contracts/hybrid_terrain_traversal.toml"
            );
        }
    }

    #[test]
    fn runtime_safety_rejects_invalid_readbacks_without_capture() {
        let (desc, base, hybrid, lighting, terrain, earth_curvature) = contract_fixture(true);
        let pixels = (desc.width * desc.height) as usize;
        let reservoirs = vec![sample_reservoir(&lighting); pixels];
        let mut accum = vec![0.0; pixels * 4];
        for alpha in accum.iter_mut().skip(3).step_by(4) {
            *alpha = desc.max_frames as f32;
        }
        let welford = vec![0.0; pixels * 2];
        let beauty = vec![0; pixels * 8];
        let validate = |base: &Uniforms,
                        reservoirs: &[Reservoir],
                        accum: &[f32],
                        welford: &[f32],
                        beauty: &[u8]| {
            record_runtime_contract(
                &desc,
                base,
                &hybrid,
                &lighting,
                &terrain,
                &earth_curvature,
                desc.max_frames,
                reservoirs,
                accum,
                welford,
                beauty,
            )
        };
        assert!(validate(&base, &reservoirs, &accum, &welford, &beauty).is_ok());
        let mut wrong_base = base;
        wrong_base.width = 8;
        assert!(validate(&wrong_base, &reservoirs, &accum, &welford, &beauty).is_err());
        for invalid in [f32::NAN, f32::INFINITY, -1.0, 65_537.0] {
            let mut bad = reservoirs.clone();
            bad[1].w_sum = invalid;
            assert!(validate(&base, &bad, &accum, &welford, &beauty).is_err());
        }
        let mut bad = reservoirs.clone();
        bad[1].m = 513;
        assert!(validate(&base, &bad, &accum, &welford, &beauty).is_err());
        bad[1] = sample_reservoir(&lighting);
        bad[1].sample.direction = [0.5, std::f32::consts::FRAC_1_SQRT_2, -0.5];
        assert!(validate(&base, &bad, &accum, &welford, &beauty).is_err());
        bad[1].sample.direction[0] = f32::NAN;
        assert!(validate(&base, &bad, &accum, &welford, &beauty).is_err());
        bad[1] = sample_reservoir(&lighting);
        bad[1].target_pdf = 0.001;
        assert!(validate(&base, &bad, &accum, &welford, &beauty).is_err());
        for (a, w, b) in [
            (&accum[1..], &welford[..], &beauty[..]),
            (&accum[..], &welford[1..], &beauty[..]),
            (&accum[..], &welford[..], &beauty[1..]),
        ] {
            assert!(validate(&base, &reservoirs, a, w, b).is_err());
        }
        assert!(validate(&base, &reservoirs[1..], &accum, &welford, &beauty).is_err());
        let mut missing_frames = accum.clone();
        missing_frames[3] = 0.0;
        assert!(validate(&base, &reservoirs, &missing_frames, &welford, &beauty).is_err());
        for invalid in [f32::NAN, f32::INFINITY, -1.0, 131_027.0] {
            let mut bad_accum = accum.clone();
            bad_accum[1] = invalid;
            assert!(validate(&base, &reservoirs, &bad_accum, &welford, &beauty).is_err());
        }
        for invalid in [
            f32::NAN,
            -1.0,
            // M2 whose estimator variance M2/(N*(N-1)) equals the threshold.
            desc.variance_threshold * desc.max_frames as f32 * (desc.max_frames - 1) as f32,
        ] {
            let mut bad_welford = welford.clone();
            bad_welford[1] = invalid;
            assert!(validate(&base, &reservoirs, &accum, &bad_welford, &beauty).is_err());
        }
        let mut bad_beauty = beauty.clone();
        bad_beauty[2..4].copy_from_slice(&f16::NAN.to_bits().to_le_bytes());
        assert!(validate(&base, &reservoirs, &accum, &welford, &bad_beauty).is_err());
    }

    #[test]
    fn terrain_statistics_layout_is_48_bytes() {
        assert_eq!(std::mem::size_of::<TerrainStatistics>(), 48);
    }

    #[test]
    fn parse_terrain_statistics_rejects_trace_error() {
        let mut stats = vec![TerrainStatistics::default(); 2];
        stats[1].x = 1.0;
        stats[1].y = 1.0;
        stats[1].trace_error = 1;
        let msg = parse_terrain_statistics(bytemuck::cast_slice(&stats))
            .unwrap_err()
            .to_string();
        assert!(msg.contains("did not converge"), "got: {msg}");
    }

    #[test]
    fn parse_terrain_statistics_rejects_overflow() {
        let mut stats = vec![TerrainStatistics::default(); 2];
        stats[0].overflow = 1;
        let msg = parse_terrain_statistics(bytemuck::cast_slice(&stats))
            .unwrap_err()
            .to_string();
        assert!(msg.contains("overflow"), "got: {msg}");
    }

    #[test]
    fn sdf_error_accumulates_across_frames_in_shader() {
        // trace_error must OR-accumulate so a failure on an earlier frame
        // survives until the host polls the buffer.
        let source = crate::shader_sources::hybrid_kernel();
        assert!(
            source.contains("wf.trace_error = wf.trace_error | hybrid_sdf_error;"),
            "per-frame write must OR-accumulate"
        );
        assert!(
            source.contains(
                "terrain_welford[pix].trace_error = terrain_welford[pix].trace_error | hybrid_sdf_error;"
            ),
            "post-AOV write must OR-accumulate"
        );
    }

    #[test]
    fn estimator_variance_exact_value() {
        // M2 = 2, N = 4 -> 2 / (4 * 3) = 1/6.
        let v = estimator_variance(2.0, 4).unwrap();
        assert!((v - 1.0 / 6.0).abs() < 1e-7, "{v}");
    }

    #[test]
    fn estimator_variance_rejects_invalid_moments() {
        assert!(estimator_variance(1.0, 1).is_err());
        assert!(estimator_variance(1.0, 0).is_err());
        assert!(estimator_variance(-1.0, 4).is_err());
        assert!(estimator_variance(f32::NAN, 4).is_err());
        assert!(estimator_variance(f32::INFINITY, 4).is_err());
        assert!(estimator_variance(f32::NEG_INFINITY, 4).is_err());
    }

    #[test]
    fn direct_bakes_intensity_restir_keeps_it_separate() {
        let i = 2.5f32;
        let c = [0.9f32, 0.5, 0.2];
        let (direct, restir_intensity, restir_color) = factor_sun_lighting(i, c);
        assert_eq!(direct, [i * c[0], i * c[1], i * c[2]]);
        assert_eq!(restir_color, c);
        assert_eq!(restir_intensity, i.max(1e-6));
        assert_ne!(restir_color, direct);
    }

    #[test]
    fn default_warm_white_matches_legacy_literals() {
        let i = 2.5f32;
        let (direct, restir_intensity, restir_color) = factor_sun_lighting(i, [1.0, 0.97, 0.92]);
        assert_eq!(direct, [i, i * 0.97, i * 0.92]);
        assert_eq!(restir_color, [1.0, 0.97, 0.92]);
        assert_eq!(restir_intensity, i.max(1e-6));
    }

    #[test]
    fn zero_colour_disables_sun_but_clamps_restir_intensity() {
        let (direct, restir_intensity, restir_color) = factor_sun_lighting(2.5, [0.0, 0.0, 0.0]);
        assert_eq!(direct, [0.0, 0.0, 0.0]);
        assert_eq!(restir_color, [0.0, 0.0, 0.0]);
        assert_eq!(restir_intensity, 2.5);
        assert!(!should_require_valid_sun_reservoirs(
            35.0,
            2.5,
            [0.0, 0.0, 0.0]
        ));
    }

    #[test]
    fn restir_intensity_still_clamps_when_light_intensity_is_zero() {
        let (_, restir_intensity, restir_color) = factor_sun_lighting(0.0, [1.0, 0.97, 0.92]);
        assert_eq!(restir_color, [1.0, 0.97, 0.92]);
        assert_eq!(restir_intensity, 1e-6);
    }

    #[test]
    fn nonzero_sun_energy_requires_valid_reservoirs() {
        assert!(should_require_valid_sun_reservoirs(
            35.0,
            2.5,
            [1.0, 0.97, 0.92]
        ));
        assert!(should_require_valid_sun_reservoirs(
            35.0,
            2.5,
            [0.0, 0.0, 0.1]
        ));
        assert!(!should_require_valid_sun_reservoirs(
            -1.0,
            2.5,
            [1.0, 0.97, 0.92]
        ));
    }
}
