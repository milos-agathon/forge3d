// src/path_tracing/inverse/mod.rs
// DIFFERENTIA: differentiable inverse path tracing — host orchestration.
//
// The solver is a genuine GPU reverse-mode pass over the REAL forward
// terrain renderer: every eval re-dispatches the verbatim `main_terrain`
// accumulation kernel (plus its publish twin, the one-shot G-buffer entry,
// and the standalone pt_restir_{temporal,spatial} reuse passes) for
// `desc.frames` frames, then runs the inverse kernels appended to the same
// module — the Reinhard-space loss adjoint, the stream-exact reverse shading
// pass, and the AOV-driven boundary estimator. There is no separate inverse
// primal: geometry, jitter, candidate generation, reservoir shading,
// atmosphere and tonemap are the forward path's own code.
//
// The optimizer is Adam on a mixed parameterization: albedo and sun
// intensity are stepped in log space (multiplicative positivity), sun
// direction on the unit sphere (tangent-projected gradient + renormalize),
// turbidity linearly in [1, 10]. Each evaluation computes the nonlinear
// display-space loss of the finite 16-replicate mean image. The expectation
// of that loss is the finite-replicate objective; the fixed seed set gives
// the optimizer a repeatable sample of it. During the albedo-frozen warmup a
// global per-channel albedo
// offset is optimized so the scalars are not pushed into the
// intensity/turbidity compensation basin just to fix an init level
// mismatch. An intensity line-search probe runs every
// 8th iteration; the loop has a true relative early stop
// (tol * |best_loss|), Polyak tail averaging over the last 30%, and a
// best-loss checkpoint. Host-visible memory stays under the 512 MiB budget:
// the only host-visible allocations are the per-eval staging readbacks,
// sampled live while each map is held.
//
// RELEVANT FILES: src/shaders/pt_inverse_*.wgsl, params.rs, grads.rs,
//                 pipeline.rs, src/py_functions/inverse.rs

mod geometry_cert;
pub mod grads;
pub mod params;
pub mod pipeline;

pub use params::{
    validate_solve_desc, AdamState, InverseParams, InverseParamsGpu, InverseSolveDesc,
    InverseSolveOutput, RestirUniforms, ALBEDO_MAX, ALBEDO_MIN, GRAD_SCALAR_SLOTS, SCORE_CLAMP,
    SUN_INTENSITY_MAX, SUN_INTENSITY_MIN, TURBIDITY_MAX, TURBIDITY_MIN,
};

use crate::core::error::RenderError;
use crate::core::memory_tracker::global_tracker;
use crate::core::resource_tracker::{
    abort_ledger_capture, begin_ledger_capture, finish_ledger_capture, AllocationOwner,
};
use crate::core::resource_tracker::{
    tracked_create_buffer, tracked_create_buffer_init, tracked_create_texture,
    tracked_host_allocation, ResourceHandle, TrackedBuffer, TrackedTexture,
};
use crate::path_tracing::compute_types::{Sphere, Uniforms};
use crate::path_tracing::hybrid_compute::terrain_heightfield::{
    EarthCurvatureUniforms, TerrainPtScene,
};
use crate::path_tracing::hybrid_compute::{HybridUniforms, LightingUniforms, TraversalMode};
use crate::path_tracing::lighting::{GpuAreaLight, GpuDirectionalLight};
use crate::path_tracing::restir::Reservoir;
#[cfg(test)]
use geometry_cert::Camera;
use geometry_cert::{
    certify_scene_slice, combine_smooth_and_validated_edge, decode_gpu_audit_bytes,
    gpu_edge_provenance, project_gpu_events, scale_validated_sun_gradient, validate_gpu_audits,
    Chart, Dem, DiscreteDraw, GpuAuditContext, GpuAuditNumericalBound, GpuEdgeProvenance,
    SceneSliceInput, ValidatedEdgeAccumulator,
};
use grads::{decode_scalars, read_buffer, read_texture, rgba16f_to_rgba8, InverseBuffers};
use pipeline::InversePipelines;

/// Aborts the ledger capture on early error return; a no-op once
/// `finish_ledger_capture` has consumed the capture.
struct CaptureGuard;
impl Drop for CaptureGuard {
    fn drop(&mut self) {
        abort_ledger_capture();
    }
}

/// Peak host-visible budget during solve() — the DIFFERENTIA contract.
pub const INVERSE_HOST_BUDGET: u64 = 512 * 1024 * 1024;

/// What one eval dispatches: `Render` = primal only (target synthesis +
/// recovered-beauty artifact), `LossOnly` = primal + loss (line-search
/// probes), `Full` = primal + loss + shade + edge (the gradient eval).
#[derive(Clone, Copy, PartialEq, Eq)]
enum EvalMode {
    Render,
    LossOnly,
    Full,
}

/// All GPU-resident solve state: the shared terrain scene (its albedo
/// texture IS the optimized parameter), the forward pixel buffers, and the
/// pre-built bind groups for the per-frame dispatch chain. Per-iteration
/// state goes through queue writes — bind groups are built once because
/// resource identity never changes across a solve.
#[allow(dead_code)]
struct InverseSceneGpu {
    terrain: TerrainPtScene,
    terrain_ubo: TrackedBuffer,
    /// Flat-Earth curvature uniform (disabled): the inverse model and its
    /// FD-verified gradients carry no curvature drop, so observations must
    /// be rendered with `earth_model='flat'`, `refraction_model='none'`.
    earth_curvature_ubo: TrackedBuffer,
    /// Per-frame base uniforms (frame_index + aov_flags baked); each frame's
    /// five dispatches bind `frame_ubos[f]` so the whole eval is one submit.
    frame_ubos: Vec<TrackedBuffer>,
    lighting_ubo: TrackedBuffer,
    hybrid_ubo: TrackedBuffer,
    inv_params_ubo: TrackedBuffer,
    dir_lights: TrackedBuffer,
    scene_buf: TrackedBuffer,
    mesh_v: TrackedBuffer,
    mesh_i: TrackedBuffer,
    mesh_bvh: TrackedBuffer,
    area_lights: TrackedBuffer,
    accum: TrackedBuffer,
    welford: TrackedBuffer,
    res_curr: TrackedBuffer,
    res_prev: TrackedBuffer,
    res_out: TrackedBuffer,
    gb_nr: TrackedBuffer,
    gb_pos: TrackedBuffer,
    out_tex: TrackedTexture,
    aov_tex: [TrackedTexture; 7],
    /// Bind groups for the main-layout pipelines, one per frame.
    g0_frames: Vec<wgpu::BindGroup>,
    /// Bind groups for the ReSTIR passes (group 0 = base uniforms only).
    restir_g0_frames: Vec<wgpu::BindGroup>,
    g1: wgpu::BindGroup,
    g2: wgpu::BindGroup,
    g3: wgpu::BindGroup,
    g3_edge: wgpu::BindGroup,
    g2_gbuffer: wgpu::BindGroup,
    g2_score_candidate: wgpu::BindGroup,
    g2_score_spatial: wgpu::BindGroup,
    restir_temporal_g2: wgpu::BindGroup,
    restir_spatial_g1: wgpu::BindGroup,
    restir_spatial_g2: wgpu::BindGroup,
    restir_empty_g1: wgpu::BindGroup,
    reservoir_bytes: u64,
    frames: u32,
    px_count: u64,
}

fn camera_frame(desc: &InverseSolveDesc) -> (glam::Vec3, glam::Vec3, glam::Vec3) {
    let origin = glam::Vec3::from(desc.cam_origin);
    let forward = (glam::Vec3::from(desc.cam_look_at) - origin).normalize();
    let right = forward.cross(glam::Vec3::from(desc.cam_up)).normalize();
    let up = right.cross(forward).normalize();
    (forward, right, up)
}

/// The shared base uniforms for one accumulation frame — identical layout
/// and values to the forward driver's `Uniforms` (the standalone ReSTIR
/// passes read the same fields at the same offsets). `seed` is the eval's
/// stream seed — solve() rewrites these per replicate (see GRAD_REPLICATES).
fn frame_uniforms(desc: &InverseSolveDesc, frame_index: u32, seed: u32) -> Uniforms {
    let (forward, right, up) = camera_frame(desc);
    Uniforms {
        width: desc.width,
        height: desc.height,
        frame_index,
        // Frame 0 writes the geometric AOVs from the unjittered center ray —
        // the same 0xFF the forward driver sets.
        aov_flags: if frame_index == 0 { 0xFF } else { 0 },
        cam_origin: desc.cam_origin,
        cam_fov_y: desc.fov_y_deg.to_radians(),
        cam_right: right.into(),
        cam_aspect: desc.width as f32 / desc.height as f32,
        cam_up: up.into(),
        cam_exposure: desc.exposure,
        cam_forward: forward.into(),
        seed_hi: seed,
        // decorrelate seed_hi ^ seed_lo in the kernel's xor mixing so
        // distinct seeds produce distinct streams (seed ^ C cancels).
        seed_lo: seed.wrapping_mul(0x9E37_79B9).wrapping_add(0x85EB_CA6B),
        // Single-pass pinhole: the full sensor is the render itself, so the
        // global-pixel ray and seed paths reduce exactly to the local ones.
        camera_model: 0,
        full_width: desc.width,
        full_height: desc.height,
        pixel_offset_x: 0,
        pixel_offset_y: 0,
        ortho_half_height: 1.0,
        camera_flags: 0,
        sensor_rect: [0.0, 0.0, 1.0, 1.0],
    }
}

// Separate proposal stream from the renderer's per-pixel beauty streams.
struct EdgeDrawRng(u64);
impl EdgeDrawRng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut x = self.0;
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        x ^ (x >> 31)
    }

    fn index(&mut self, n: u64) -> u64 {
        let threshold = n.wrapping_neg() % n;
        loop {
            let x = self.next();
            if x >= threshold {
                return x % n;
            }
        }
    }

    fn open_unit(&mut self) -> f64 {
        ((self.next() >> 11) as f64 + 0.5) * (1.0 / ((1u64 << 53) as f64))
    }
}

/// One independent discrete/continuous draw for the certified categorical
/// boundary estimator.  The host selects this before replay, but deliberately
/// does not construct a camera or light direction here: those are captured
/// from the actual f32 shader inputs by the preflight replay.
#[derive(Clone, Copy, Debug)]
struct EdgeDraw {
    selected_rep: u32,
    pixel: u32,
    frame: u32,
    sample: u32,
    chart: Chart,
    free: f64,
}

fn select_edge_draw(
    desc: &InverseSolveDesc,
    params: &InverseParams,
    reps: u32,
    frames: u32,
    terrain_origin_spacing: [f32; 4],
) -> Result<EdgeDraw, RenderError> {
    if reps == 0
        || frames == 0
        || desc.width == 0
        || desc.height == 0
        || desc.dem_width < 2
        || desc.dem_height < 2
    {
        return Err(RenderError::Render(
            "inverse edge draw has an empty discrete domain".into(),
        ));
    }
    let mut seed = u64::from(desc.seed) ^ 0xd1ff_e1e0_3ea5_7a11;
    for value in params.sun_dir {
        seed = seed.rotate_left(17) ^ u64::from(value.to_bits());
    }
    let mut rng = EdgeDrawRng(seed);
    let pixels = u64::from(desc.width) * u64::from(desc.height);
    if pixels == 0 || pixels > u64::from(u32::MAX) {
        return Err(RenderError::Render(
            "inverse event pixel index exceeds shader u32 range".into(),
        ));
    }
    let pixel = rng.index(pixels) as u32;
    let selected_rep = rng.index(u64::from(reps)) as u32;
    let frame = rng.index(u64::from(frames)) as u32;
    let sample = rng.index(u64::from(desc.spp.max(1))) as u32;
    let chart = if rng.index(2) == 0 {
        Chart::DependentX
    } else {
        Chart::DependentZ
    };
    let (free0, span) = match chart {
        Chart::DependentX => (
            f64::from(terrain_origin_spacing[1]),
            f64::from(terrain_origin_spacing[3]) * (desc.dem_height - 1) as f64,
        ),
        Chart::DependentZ => (
            f64::from(terrain_origin_spacing[0]),
            f64::from(terrain_origin_spacing[2]) * (desc.dem_width - 1) as f64,
        ),
    };
    Ok(EdgeDraw {
        selected_rep,
        pixel,
        frame,
        sample,
        chart,
        free: free0 + span * rng.open_unit(),
    })
}

const EDGE_EVENT_STRIDE: u64 = 64;
const EDGE_AUDIT_STRIDE: u64 = std::mem::size_of::<geometry_cert::GpuAuditRecord>() as u64;
const EDGE_SUM_BYTES: u64 = 3 * std::mem::size_of::<f32>() as u64;
const EDGE_PROBE_FLAG: u32 = 16;

/// GPU-resident state for one chunk of certified boundary events.  The
/// projected events themselves remain in the returned `TrackedVec`: the host
/// consumes that exact ideal-real metadata only after the GPU has produced an
/// actual-WGSL witness.
struct EdgeChunk {
    _events: TrackedBuffer,
    _params: TrackedBuffer,
    _audit: TrackedBuffer,
    _edge_sum: TrackedBuffer,
    g2: wgpu::BindGroup,
    g0_tiles: Vec<wgpu::BindGroup>,
    count: u32,
    event_range: std::ops::Range<usize>,
    struct_weight: f32,
    mean_image_weight: f32,
    replay_sample_weight: f32,
    jump_numerator: f32,
    jump_denominator: f32,
}

/// The final edge batch is valid only for the f32 source snapshot captured by
/// the dedicated replay preflight.  Keeping that token beside the projected
/// events makes it impossible to validate a later batch against a host-side
/// camera/light reconstruction.
struct EdgeWork {
    selected_rep: u32,
    provenance: GpuEdgeProvenance,
    projected: geometry_cert::TrackedVec<geometry_cert::ProjectedEvent>,
    chunks: Vec<EdgeChunk>,
}

/// Compact proof evidence retained by a full evaluation.  Each chunk's
/// componentwise atomic bound is checked before it contributes here; the
/// aggregate deliberately exposes only facts that remain meaningful across
/// independently accumulated chunks.
#[derive(Clone, Debug, Default)]
struct EdgeAuditEvidence {
    event_count: usize,
    accepted_camera_witness_count: usize,
    ibl_clear_count: usize,
    ibl_occluded_count: usize,
    max_camera_jitter_projection_error: f64,
    max_upload_weight_error: f64,
    max_ibl_direction_source_width: f64,
    max_source_radiance_width: f64,
    max_radiance_reconstruction_width: f64,
    max_loss_interval_width: f64,
    max_accumulation_rounding_bound: [f64; 3],
    max_atomic_sum_abs: [f64; 3],
    host_chunk_add_rounding_bound: [f64; 3],
    final_scalar_add_rounding_bound: [f64; 3],
    final_scalar_scale_rounding_bound: [f64; 3],
    audited_edge_chunk_count: usize,
    #[cfg(test)]
    audited_records: Vec<geometry_cert::GpuAuditRecord>,
    #[cfg(test)]
    audited_events: Vec<geometry_cert::ProjectedEvent>,
}

impl EdgeAuditEvidence {
    fn observe(&mut self, bound: GpuAuditNumericalBound) {
        self.event_count += bound.event_count;
        self.accepted_camera_witness_count += bound.accepted_camera_witness_count;
        self.ibl_clear_count += bound.ibl_clear_count;
        self.ibl_occluded_count += bound.ibl_occluded_count;
        self.max_camera_jitter_projection_error = self
            .max_camera_jitter_projection_error
            .max(bound.max_camera_jitter_projection_error);
        self.max_upload_weight_error = self
            .max_upload_weight_error
            .max(bound.max_upload_weight_error);
        self.max_ibl_direction_source_width = self
            .max_ibl_direction_source_width
            .max(bound.max_ibl_direction_source_width);
        self.max_source_radiance_width = self
            .max_source_radiance_width
            .max(bound.max_source_radiance_width);
        self.max_radiance_reconstruction_width = self
            .max_radiance_reconstruction_width
            .max(bound.max_radiance_reconstruction_width);
        self.max_loss_interval_width = self
            .max_loss_interval_width
            .max(bound.max_loss_interval_width);
        for k in 0..3 {
            self.max_accumulation_rounding_bound[k] =
                self.max_accumulation_rounding_bound[k].max(bound.accumulation_rounding_bound[k]);
            self.max_atomic_sum_abs[k] =
                self.max_atomic_sum_abs[k].max(f64::from(bound.atomic_sum_observed[k]).abs());
        }
    }

    fn observe_host_accumulator(&mut self, accumulator: &ValidatedEdgeAccumulator) {
        self.audited_edge_chunk_count = accumulator.chunk_count;
        self.host_chunk_add_rounding_bound = accumulator.host_add_rounding_bound;
    }

    #[cfg(test)]
    fn observe_audits(
        &mut self,
        events: &[geometry_cert::ProjectedEvent],
        audits: &[geometry_cert::GpuAuditRecord],
    ) {
        assert_eq!(
            events.len(),
            audits.len(),
            "production edge audit must retain its exact projected-event order"
        );
        self.audited_events.extend(events.iter().copied());
        self.audited_records.extend(audits.iter().copied());
    }
}

fn edge_record_bytes<T>(
    records: &[T],
    record_of: impl Fn(&T) -> geometry_cert::InvEdgeEvent,
) -> Result<(Vec<u8>, ResourceHandle), RenderError> {
    let byte_count = records
        .len()
        .checked_mul(64)
        .ok_or_else(|| RenderError::Render("inverse event byte size overflow".into()))?;
    let reservation = tracked_host_allocation(byte_count as u64, "inverse-event-upload-bytes")?;
    let mut bytes = Vec::with_capacity(byte_count);
    for event in records {
        let record = record_of(event);
        for value in record.ids {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for row in [record.receiver, record.normal, record.grad_weight] {
            for value in row {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
    }
    Ok((bytes, reservation))
}

/// Allocate the edge-only bindings for a known set of event records.  This is
/// shared by the source-provenance preflight and the final certified batch so
/// both execute the same entry point, bind-group layout, and replay inputs.
#[allow(clippy::too_many_arguments)]
fn create_edge_chunk<T>(
    device: &wgpu::Device,
    desc: &InverseSolveDesc,
    pipes: &InversePipelines,
    scene: &InverseSceneGpu,
    bufs: &InverseBuffers,
    params: &InverseParams,
    reps: u32,
    records: &[T],
    record_of: impl Fn(&T) -> geometry_cert::InvEdgeEvent,
    event_range: std::ops::Range<usize>,
    probe: bool,
) -> Result<EdgeChunk, RenderError> {
    let (bytes, bytes_reservation) = edge_record_bytes(records, record_of)?;
    let events_buf = tracked_create_buffer_init(
        device,
        &wgpu::util::BufferInitDescriptor {
            label: Some(if probe {
                "inv-certified-edge-probe-event"
            } else {
                "inv-certified-edge-events"
            }),
            contents: &bytes,
            usage: wgpu::BufferUsages::STORAGE,
        },
    )?;
    drop(bytes);
    drop(bytes_reservation);
    let count = u32::try_from(records.len())
        .map_err(|_| RenderError::Render("inverse event chunk count exceeds u32".into()))?;
    let audit_bytes = u64::from(count)
        .checked_mul(EDGE_AUDIT_STRIDE)
        .ok_or_else(|| RenderError::Render("inverse edge audit size overflow".into()))?;
    let audit = tracked_create_buffer(
        device,
        &wgpu::BufferDescriptor {
            label: Some(if probe {
                "inv-certified-edge-probe-audit"
            } else {
                "inv-certified-edge-audit"
            }),
            size: audit_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        },
    )?;
    let edge_sum = tracked_create_buffer(
        device,
        &wgpu::BufferDescriptor {
            label: Some("inv-certified-edge-sum"),
            size: EDGE_SUM_BYTES,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        },
    )?;
    let mut gpu_params = InverseParamsGpu::new(
        desc.sun_color,
        params.sun_intensity,
        (desc.dem_width, desc.dem_height),
        scene.frames,
        desc.spp,
        scene.px_count,
        desc.tile_size,
        desc.spatial_reuse,
        true,
        desc.score_correction,
    );
    gpu_params.ctrl[2] = count;
    if probe {
        gpu_params.ctrl[0] |= EDGE_PROBE_FLAG;
    }
    gpu_params.rsv0[0] = 1.0 / reps as f32;
    let struct_weight = gpu_params.fl[0];
    let mean_image_weight = gpu_params.rsv0[0];
    let replay_sample_weight = gpu_params.fl[3];
    let jump_numerator = gpu_params.fl[2];
    let jump_denominator = gpu_params.rsv0[0];
    let params_buf = tracked_create_buffer_init(
        device,
        &wgpu::util::BufferInitDescriptor {
            label: Some(if probe {
                "inv-certified-edge-probe-params"
            } else {
                "inv-certified-edge-params"
            }),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: wgpu::BufferUsages::UNIFORM,
        },
    )?;
    let height_view = scene
        .terrain
        .pyramid
        .height_texture
        .create_view(&Default::default());
    let minmax_view = scene
        .terrain
        .pyramid
        .minmax_texture
        .create_view(&Default::default());
    let env_view = scene.terrain.env_texture.create_view(&Default::default());
    let albedo_view = scene
        .terrain
        .albedo_texture
        .create_view(&Default::default());
    let g2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(if probe {
            "inv-g2-certified-edge-probe"
        } else {
            "inv-g2-certified-edge"
        }),
        layout: &pipes.inv_g2_edge,
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
                resource: scene.terrain_ubo.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: wgpu::BindingResource::TextureView(&env_view),
            },
            wgpu::BindGroupEntry {
                binding: 10,
                resource: scene.earth_curvature_ubo.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 11,
                resource: bufs.v4_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 12,
                resource: bufs.u32_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 13,
                resource: events_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 14,
                resource: audit.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 15,
                resource: edge_sum.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 16,
                resource: wgpu::BindingResource::TextureView(&albedo_view),
            },
        ],
    });
    let slots = scene.frames.max(1).min(desc.tile_size.max(1));
    let mut g0_tiles = Vec::new();
    let mut end = slots;
    while end < scene.frames {
        let ubo = &scene.frame_ubos[(end - 1) as usize];
        g0_tiles.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(if probe {
                "inv-g0-certified-edge-probe"
            } else {
                "inv-g0-certified-edge"
            }),
            layout: &pipes.inv_g0,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: ubo.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: scene.lighting_ubo.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        }));
        end += slots;
    }
    let ubo = &scene.frame_ubos[(scene.frames - 1) as usize];
    g0_tiles.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(if probe {
            "inv-g0-certified-edge-probe"
        } else {
            "inv-g0-certified-edge"
        }),
        layout: &pipes.inv_g0,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: ubo.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: scene.lighting_ubo.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buf.as_entire_binding(),
            },
        ],
    }));
    Ok(EdgeChunk {
        _events: events_buf,
        _params: params_buf,
        _audit: audit,
        _edge_sum: edge_sum,
        g2,
        g0_tiles,
        count,
        event_range,
        struct_weight,
        mean_image_weight,
        replay_sample_weight,
        jump_numerator,
        jump_denominator,
    })
}

fn prepare_edge_probe(
    device: &wgpu::Device,
    desc: &InverseSolveDesc,
    pipes: &InversePipelines,
    scene: &InverseSceneGpu,
    bufs: &InverseBuffers,
    params: &InverseParams,
    reps: u32,
    draw: EdgeDraw,
) -> Result<EdgeChunk, RenderError> {
    // The probe needs only the same discrete id checks as a final event. Its
    // receiver fields are intentionally unused under INV_FLAG_EDGE_PROBE.
    let record = geometry_cert::InvEdgeEvent {
        ids: [draw.pixel, draw.frame, draw.sample, 1],
        receiver: [0.0; 4],
        normal: [0.0; 4],
        grad_weight: [0.0; 4],
    };
    create_edge_chunk(
        device,
        desc,
        pipes,
        scene,
        bufs,
        params,
        reps,
        std::slice::from_ref(&record),
        |value| *value,
        0..0,
        true,
    )
}

fn read_edge_provenance(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    probe: &EdgeChunk,
) -> Result<GpuEdgeProvenance, RenderError> {
    let raw = read_buffer(device, queue, &probe._audit, 0, EDGE_AUDIT_STRIDE)?;
    let audit = decode_gpu_audit_bytes(&raw).map_err(|e| {
        RenderError::Render(format!("inverse edge provenance decode rejected: {e:?}"))
    })?;
    if audit.len() != 1 {
        return Err(RenderError::Render(
            "inverse edge provenance replay did not write exactly one audit".into(),
        ));
    }
    gpu_edge_provenance(&audit[0])
        .map_err(|e| RenderError::Render(format!("inverse edge provenance replay rejected: {e:?}")))
}

impl InverseSceneGpu {
    fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        desc: &InverseSolveDesc,
        pipes: &InversePipelines,
        bufs: &InverseBuffers,
    ) -> Result<Self, RenderError> {
        // The albedo map is the optimized parameter — construct the shared
        // scene WITH the map so the flag is raised and the DEM-sized
        // RGBA32F texture is the live binding at group 2 binding 16.
        let terrain = TerrainPtScene::new(
            device,
            queue,
            &desc.heights,
            desc.dem_width,
            desc.dem_height,
            desc.spacing,
            desc.exaggeration,
            [0.5, 0.5, 0.5], // fallback albedo — unused while the map is bound
            desc.env_map
                .as_ref()
                .map(|(d, w, h)| (d.as_slice(), *w, *h)),
            desc.env_intensity,
            Some((&desc.init_albedo, desc.dem_width, desc.dem_height)),
            desc.init.turbidity,
        )?;
        let terrain_uniforms = terrain.uniforms(desc.spp, 32);
        let terrain_ubo = tracked_create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("inv-terrain-ubo"),
                contents: bytemuck::bytes_of(&terrain_uniforms),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        )?;

        let earth_curvature = EarthCurvatureUniforms::new(
            crate::geo::refraction::EarthModel::Flat,
            crate::geo::refraction::RefractionModel::None,
            [0.0, 0.0],
            0.0,
        )?;
        let earth_curvature_ubo = tracked_create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("inv-earth-curvature-ubo"),
                contents: bytemuck::bytes_of(&earth_curvature),
                usage: wgpu::BufferUsages::UNIFORM,
            },
        )?;

        // One base-uniform buffer per frame: frame_index and aov_flags are
        // baked so the whole eval is a single submit (the standalone ReSTIR
        // passes read only the fields at identical offsets — same trick the
        // forward driver uses).
        let frames = desc.frames.max(1);
        let mut frame_ubos = Vec::with_capacity(frames as usize);
        for f in 0..frames {
            let u = frame_uniforms(desc, f, desc.seed);
            frame_ubos.push(tracked_create_buffer_init(
                device,
                &wgpu::util::BufferInitDescriptor {
                    label: Some("inv-frame-ubo"),
                    contents: bytemuck::bytes_of(&u),
                    // COPY_DST: eval_pass reseeds the stream per replicate.
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                },
            )?);
        }

        let lighting = LightingUniforms {
            light_dir: desc.init.sun_dir,
            lighting_type: 1,
            light_color: [
                desc.init.sun_intensity * desc.sun_color[0],
                desc.init.sun_intensity * desc.sun_color[1],
                desc.init.sun_intensity * desc.sun_color[2],
            ],
            shadows_enabled: 1,
            ambient_color: [0.0; 3],
            shadow_intensity: 1.0,
            hdri_intensity: desc.env_intensity,
            hdri_rotation: 0.0,
            specular_power: 32.0,
            _pad: [0; 5],
        };
        let lighting_ubo = tracked_create_buffer(
            device,
            &wgpu::BufferDescriptor {
                label: Some("inv-lighting-ubo"),
                size: std::mem::size_of::<LightingUniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            },
        )?;
        queue.write_buffer(&lighting_ubo, 0, bytemuck::bytes_of(&lighting));

        let hybrid = HybridUniforms {
            sdf_primitive_count: 0,
            sdf_node_count: 0,
            mesh_vertex_count: 0,
            mesh_index_count: 0,
            mesh_bvh_node_count: 0,
            traversal_mode: TraversalMode::TerrainOnly as u32,
            _pad: [0; 2],
        };
        let hybrid_ubo = tracked_create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("inv-hybrid-ubo"),
                contents: bytemuck::bytes_of(&hybrid),
                usage: wgpu::BufferUsages::UNIFORM,
            },
        )?;

        let inv_params = InverseParamsGpu::new(
            desc.sun_color,
            desc.init.sun_intensity,
            (desc.dem_width, desc.dem_height),
            frames,
            desc.spp,
            bufs.px_count,
            desc.tile_size,
            desc.spatial_reuse,
            desc.edge_term,
            desc.score_correction,
        );
        let inv_params_ubo = tracked_create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("inv-params-ubo"),
                contents: bytemuck::bytes_of(&inv_params),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        )?;

        // Scene + mesh dummies: one full element of the largest represented
        // WGSL type (traversal_mode = TerrainOnly never reads them).
        let scene_buf = tracked_create_buffer(
            device,
            &wgpu::BufferDescriptor {
                label: Some("inv-scene-dummy"),
                size: std::mem::size_of::<Sphere>() as u64,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            },
        )?;
        let mesh_v = tracked_create_buffer(
            device,
            &wgpu::BufferDescriptor {
                label: Some("inv-mesh-v-dummy"),
                size: 16,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            },
        )?;
        let mesh_i = tracked_create_buffer(
            device,
            &wgpu::BufferDescriptor {
                label: Some("inv-mesh-i-dummy"),
                size: 12,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            },
        )?;
        let mesh_bvh = tracked_create_buffer(
            device,
            &wgpu::BufferDescriptor {
                label: Some("inv-mesh-bvh-dummy"),
                size: 48,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            },
        )?;

        let px_count = bufs.px_count;
        let px = px_count as usize;
        let storage_rw = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let mk = |label: &str, size: u64| -> Result<TrackedBuffer, RenderError> {
            tracked_create_buffer(
                device,
                &wgpu::BufferDescriptor {
                    label: Some(label),
                    size: size.max(4),
                    usage: storage_rw,
                    mapped_at_creation: false,
                },
            )
        };
        let accum = mk("inv-accum", px_count * 16)?;
        let welford = mk(
            "inv-welford",
            px_count * std::mem::size_of::<u64>() as u64 * 6, // 48 B/px record
        )?;
        let reservoir_bytes = px_count * std::mem::size_of::<Reservoir>() as u64;
        let res_curr = mk("inv-res-curr", reservoir_bytes)?;
        let res_out = mk("inv-res-out", reservoir_bytes)?;
        let res_prev = tracked_create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("inv-res-prev"),
                contents: bytemuck::cast_slice(&vec![Reservoir::default(); px]),
                usage: storage_rw,
            },
        )?;
        let gb_nr = mk("inv-gb-nr", px_count * 16)?;
        let gb_pos = mk("inv-gb-pos", px_count * 16)?;

        // ReSTIR spatial-pass light tables — one directional entry following
        // the forward convention (negated direction, intensity floored).
        let sun_light = GpuDirectionalLight::new(
            [
                -desc.init.sun_dir[0],
                -desc.init.sun_dir[1],
                -desc.init.sun_dir[2],
            ],
            desc.init.sun_intensity.max(1e-6),
            desc.sun_color,
            1.0,
        );
        let dir_lights = tracked_create_buffer(
            device,
            &wgpu::BufferDescriptor {
                label: Some("inv-dir-lights"),
                size: std::mem::size_of::<GpuDirectionalLight>() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            },
        )?;
        queue.write_buffer(&dir_lights, 0, bytemuck::bytes_of(&sun_light));
        let area_light = <GpuAreaLight as bytemuck::Zeroable>::zeroed();
        let area_lights = tracked_create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("inv-area-lights"),
                contents: bytemuck::bytes_of(&area_light),
                usage: wgpu::BufferUsages::STORAGE,
            },
        )?;

        // ---- output + AOV targets (same formats the forward binds) ----
        let (w, h) = (desc.width, desc.height);
        let mktex = |label: &str,
                     format: wgpu::TextureFormat,
                     sampled: bool|
         -> Result<TrackedTexture, RenderError> {
            let mut usage = wgpu::TextureUsages::STORAGE_BINDING;
            if sampled {
                usage |= wgpu::TextureUsages::TEXTURE_BINDING;
            }
            tracked_create_texture(
                device,
                &wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d {
                        width: w,
                        height: h,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: usage | wgpu::TextureUsages::COPY_SRC,
                    view_formats: &[],
                },
            )
        };
        let out_tex = mktex("inv-out", wgpu::TextureFormat::Rgba16Float, false)?;
        // AOV order in the forward output layout: albedo, normal, depth,
        // direct, indirect, emission, visibility. Normal + depth get
        // TEXTURE_BINDING so the edge pass can sample them.
        let aov_tex = [
            mktex("inv-aov-albedo", wgpu::TextureFormat::Rgba16Float, false)?,
            mktex("inv-aov-normal", wgpu::TextureFormat::Rgba16Float, true)?,
            mktex("inv-aov-depth", wgpu::TextureFormat::R32Float, true)?,
            mktex("inv-aov-direct", wgpu::TextureFormat::Rgba16Float, false)?,
            mktex("inv-aov-indirect", wgpu::TextureFormat::Rgba16Float, false)?,
            mktex("inv-aov-emission", wgpu::TextureFormat::Rgba16Float, false)?,
            mktex("inv-aov-visibility", wgpu::TextureFormat::Rgba8Unorm, false)?,
        ];

        // ---- bind groups ----
        let height_view = terrain
            .pyramid
            .height_texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let minmax_view = terrain
            .pyramid
            .minmax_texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let env_view = terrain
            .env_texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let albedo_view = terrain
            .albedo_texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let target_view = bufs
            .target_tex
            .create_view(&wgpu::TextureViewDescriptor::default());
        let aov_views: Vec<wgpu::TextureView> = aov_tex
            .iter()
            .map(|t| t.create_view(&wgpu::TextureViewDescriptor::default()))
            .collect();
        let out_view = out_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let mut g0_frames = Vec::with_capacity(frames as usize);
        let mut restir_g0_frames = Vec::with_capacity(frames as usize);
        for ubo in &frame_ubos {
            g0_frames.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("inv-g0"),
                layout: &pipes.inv_g0,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: ubo.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: lighting_ubo.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: inv_params_ubo.as_entire_binding(),
                    },
                ],
            }));
            restir_g0_frames.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("inv-restir-g0"),
                layout: &pipes.restir_g0,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: ubo.as_entire_binding(),
                }],
            }));
        }

        let g1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("inv-g1"),
            layout: &pipes.inv_g1,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: scene_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: hybrid_ubo.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: mesh_v.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: mesh_i.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: mesh_bvh.as_entire_binding(),
                },
            ],
        });
        let g2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("inv-g2"),
            layout: &pipes.inv_g2,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: accum.as_entire_binding(),
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
                    resource: welford.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: res_curr.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(&env_view),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: res_prev.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: earth_curvature_ubo.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: bufs.v4_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: bufs.u32_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 16,
                    resource: wgpu::BindingResource::TextureView(&albedo_view),
                },
            ],
        });
        let g2_score_candidate = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("inv-g2-score-candidate"),
            layout: &pipes.inv_g2_score_spatial,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: terrain_ubo.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: res_curr.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: res_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: gb_pos.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: bufs.u32_buf.as_entire_binding(),
                },
            ],
        });
        let g2_score_spatial = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("inv-g2-score-spatial"),
            layout: &pipes.inv_g2_score_spatial,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: terrain_ubo.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: res_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: res_prev.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: gb_pos.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: bufs.u32_buf.as_entire_binding(),
                },
            ],
        });
        let g3 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("inv-g3"),
            layout: &pipes.inv_g3,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&out_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&aov_views[0]),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&aov_views[1]),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&aov_views[2]),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&aov_views[3]),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&aov_views[4]),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(&aov_views[5]),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(&aov_views[6]),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::TextureView(&target_view),
                },
            ],
        });
        let g3_edge = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("inv-g3-edge"),
            layout: &pipes.inv_g3_edge,
            entries: &[wgpu::BindGroupEntry {
                binding: 8,
                resource: wgpu::BindingResource::TextureView(&target_view),
            }],
        });
        let g2_gbuffer = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("inv-g2-gbuffer"),
            layout: &pipes.inv_g2_gbuffer,
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
                    resource: gb_nr.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: gb_pos.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: earth_curvature_ubo.as_entire_binding(),
                },
            ],
        });
        let restir_temporal_g2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("inv-restir-temporal-g2"),
            layout: &pipes.restir_temporal_g2,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: res_prev.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: res_curr.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: res_out.as_entire_binding(),
                },
            ],
        });
        let restir_spatial_g1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("inv-restir-spatial-g1"),
            layout: &pipes.restir_spatial_g1,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: area_lights.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: dir_lights.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: gb_nr.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: gb_pos.as_entire_binding(),
                },
            ],
        });
        let restir_spatial_g2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("inv-restir-spatial-g2"),
            layout: &pipes.restir_spatial_g2,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: res_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: res_prev.as_entire_binding(),
                },
            ],
        });
        let restir_empty_g1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("inv-restir-empty-g1"),
            layout: &pipes.restir_empty,
            entries: &[],
        });

        let scene = Self {
            terrain,
            terrain_ubo,
            earth_curvature_ubo,
            frame_ubos,
            lighting_ubo,
            hybrid_ubo,
            inv_params_ubo,
            dir_lights,
            scene_buf,
            mesh_v,
            mesh_i,
            mesh_bvh,
            area_lights,
            accum,
            welford,
            res_curr,
            res_prev,
            res_out,
            gb_nr,
            gb_pos,
            out_tex,
            aov_tex,
            g0_frames,
            restir_g0_frames,
            g1,
            g2,
            g3,
            g3_edge,
            g2_gbuffer,
            g2_score_candidate,
            g2_score_spatial,
            restir_temporal_g2,
            restir_spatial_g1,
            restir_spatial_g2,
            restir_empty_g1,
            reservoir_bytes,
            frames,
            px_count,
        };

        // The ReSTIR G-buffer is geometry+camera only — static across the
        // solve — so the forward's one-shot entry runs once here.
        let wg_x = desc.width.div_ceil(8);
        let wg_y = desc.height.div_ceil(8);
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("inv-gbuffer-enc"),
        });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("inv-gbuffer"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipes.terrain_gbuffer);
            pass.set_bind_group(0, &scene.g0_frames[0], &[]);
            pass.set_bind_group(1, &scene.g1, &[]);
            pass.set_bind_group(2, &scene.g2_gbuffer, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }
        queue.submit([enc.finish()]);
        Ok(scene)
    }

    /// Re-upload the per-iteration parameters: the albedo map (rgb -> rgba
    /// f32 into the shared terrain texture), the terrain uniform block
    /// (turbidity excess in albedo_pad.w), the lighting uniform (sun dir +
    /// color*intensity — the exact uniforms the primal reads), the ReSTIR
    /// directional-light table entry, and the inverse params block.
    fn upload_params(
        &self,
        queue: &wgpu::Queue,
        desc: &InverseSolveDesc,
        params: &InverseParams,
        albedo: &[f32],
    ) {
        let mut rgba = Vec::with_capacity(albedo.len() / 3 * 4);
        for c in albedo.chunks_exact(3) {
            rgba.extend_from_slice(&[c[0], c[1], c[2], 1.0]);
        }
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.terrain.albedo_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&rgba),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(desc.dem_width * 16),
                rows_per_image: Some(desc.dem_height),
            },
            wgpu::Extent3d {
                width: desc.dem_width,
                height: desc.dem_height,
                depth_or_array_layers: 1,
            },
        );

        let mut tu = self.terrain.uniforms(desc.spp, 32);
        tu.albedo_pad[3] = (params.turbidity - 1.0).max(0.0);
        queue.write_buffer(&self.terrain_ubo, 0, bytemuck::bytes_of(&tu));

        let len = (params.sun_dir[0] * params.sun_dir[0]
            + params.sun_dir[1] * params.sun_dir[1]
            + params.sun_dir[2] * params.sun_dir[2])
            .sqrt()
            .max(1e-8);
        let dir = [
            params.sun_dir[0] / len,
            params.sun_dir[1] / len,
            params.sun_dir[2] / len,
        ];
        let lighting = LightingUniforms {
            light_dir: dir,
            lighting_type: 1,
            light_color: [
                params.sun_intensity * desc.sun_color[0],
                params.sun_intensity * desc.sun_color[1],
                params.sun_intensity * desc.sun_color[2],
            ],
            shadows_enabled: 1,
            ambient_color: [0.0; 3],
            shadow_intensity: 1.0,
            hdri_intensity: desc.env_intensity,
            hdri_rotation: 0.0,
            specular_power: 32.0,
            _pad: [0; 5],
        };
        queue.write_buffer(&self.lighting_ubo, 0, bytemuck::bytes_of(&lighting));

        let sun_light = GpuDirectionalLight::new(
            [-dir[0], -dir[1], -dir[2]],
            params.sun_intensity.max(1e-6),
            desc.sun_color,
            1.0,
        );
        queue.write_buffer(&self.dir_lights, 0, bytemuck::bytes_of(&sun_light));

        self.upload_inv_params(queue, desc, params, 0, 1, false);
    }

    /// Write the inverse-params uniform for one phase of an eval.
    /// `rep_ord`/`reps` drive the replicate-mean accumulation (1/R weight,
    /// ordinal 0 overwrites); `mean_loss` makes the loss pass read the
    /// replicate-mean image instead of the live accumulator.
    fn upload_inv_params(
        &self,
        queue: &wgpu::Queue,
        desc: &InverseSolveDesc,
        params: &InverseParams,
        rep_ord: u32,
        reps: u32,
        mean_loss: bool,
    ) {
        let mut gpu_params = InverseParamsGpu::new(
            desc.sun_color,
            params.sun_intensity,
            (desc.dem_width, desc.dem_height),
            self.frames,
            desc.spp,
            self.px_count,
            desc.tile_size,
            desc.spatial_reuse,
            desc.edge_term,
            desc.score_correction,
        );
        gpu_params.rsv0[0] = 1.0 / reps.max(1) as f32;
        gpu_params.rsv0[1] = rep_ord as f32;
        if mean_loss {
            gpu_params.ctrl[0] |= 8; // INV_FLAG_MEAN_LOSS
        }
        queue.write_buffer(&self.inv_params_ubo, 0, bytemuck::bytes_of(&gpu_params));
    }
}

/// Certify one complete chart slice for one uniformly sampled discrete
/// beauty draw, then upload its events for replay in the selected replicate.
/// Every sampled slice is either complete or rejects the evaluation.
#[allow(clippy::too_many_arguments)]
fn prepare_edge_chunks(
    device: &wgpu::Device,
    desc: &InverseSolveDesc,
    pipes: &InversePipelines,
    scene: &InverseSceneGpu,
    bufs: &InverseBuffers,
    params: &InverseParams,
    reps: u32,
    draw: EdgeDraw,
    provenance: GpuEdgeProvenance,
) -> Result<
    (
        geometry_cert::TrackedVec<geometry_cert::ProjectedEvent>,
        Vec<EdgeChunk>,
    ),
    RenderError,
> {
    let height_host_bytes = (desc.heights.len() as u64)
        .checked_mul(8)
        .ok_or_else(|| RenderError::Render("inverse DEM certificate size overflow".into()))?;
    let _height_reservation =
        tracked_host_allocation(height_host_bytes, "inverse-edge-dem-certificate")?;
    let terrain = scene.terrain.uniforms(desc.spp, 32);
    let mut heights = Vec::with_capacity(desc.heights.len());
    heights.extend(
        desc.heights
            .iter()
            .map(|&h| f64::from(h * desc.exaggeration)),
    );
    let dem = Dem {
        width: desc.dem_width as usize,
        height: desc.dem_height as usize,
        x0: f64::from(terrain.origin_spacing[0]),
        z0: f64::from(terrain.origin_spacing[1]),
        sx: f64::from(terrain.origin_spacing[2]),
        sz: f64::from(terrain.origin_spacing[3]),
        heights: &heights,
    };
    let camera = provenance.camera;
    if camera.width != desc.width
        || camera.height != desc.height
        || camera.pixel_x != draw.pixel % desc.width
        || camera.pixel_y != draw.pixel / desc.width
    {
        return Err(RenderError::Render(
            "inverse edge preflight camera provenance did not match the selected draw".into(),
        ));
    }
    let omega = provenance.sun_direction.map(f64::from);
    let slice = SceneSliceInput {
        dem,
        chart: draw.chart,
        free: draw.free,
        omega,
        camera,
        tmin: 1e-3,
        tmax: 1e30,
        normal_offset: 1e-3,
    };
    let events = certify_scene_slice(&slice).map_err(|e| {
        RenderError::Render(format!(
            "inverse shadow-boundary certificate rejected: {e:?}"
        ))
    })?;
    let projected = project_gpu_events(
        &events,
        &slice.camera,
        DiscreteDraw {
            replicate_count: reps,
            frame_count: scene.frames,
            samples_per_frame: desc.spp.max(1),
            frame: draw.frame,
            sample: draw.sample,
        },
    )
    .map_err(|e| RenderError::Render(format!("inverse event projection rejected: {e:?}")))?;
    if projected.is_empty() {
        return Ok((projected, Vec::new()));
    }
    let max_records = (u64::from(device.limits().max_storage_buffer_binding_size)
        / EDGE_EVENT_STRIDE)
        .min(u64::from(device.limits().max_storage_buffer_binding_size) / EDGE_AUDIT_STRIDE)
        .min(u64::from(u32::MAX)) as usize;
    if max_records == 0 {
        return Err(RenderError::Render(
            "inverse event record exceeds adapter storage binding size".into(),
        ));
    }
    let mut chunks = Vec::new();
    let mut event_start = 0usize;
    for projected_chunk in projected.chunks(max_records) {
        let event_end = event_start
            .checked_add(projected_chunk.len())
            .ok_or_else(|| RenderError::Render("inverse event range overflow".into()))?;
        chunks.push(create_edge_chunk(
            device,
            desc,
            pipes,
            scene,
            bufs,
            params,
            reps,
            projected_chunk,
            |event| event.record,
            event_start..event_end,
            false,
        )?);
        event_start = event_end;
    }
    Ok((projected, chunks))
}

/// Test-only certificate inspection seam. Production evaluation never calls
/// this: it obtains its provenance solely from the GPU replay preflight above.
/// These unit fixtures inspect analytic event classes before their separate
/// full-evaluation GPU gate runs, so they construct the same UBO operands
/// locally and deliberately mark the synthetic history as a fresh sample.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn prepare_edge_chunks_for_test(
    device: &wgpu::Device,
    desc: &InverseSolveDesc,
    pipes: &InversePipelines,
    scene: &InverseSceneGpu,
    bufs: &InverseBuffers,
    params: &InverseParams,
    reps: u32,
) -> Result<
    (
        u32,
        geometry_cert::TrackedVec<geometry_cert::ProjectedEvent>,
        Vec<EdgeChunk>,
    ),
    RenderError,
> {
    let terrain = scene.terrain.uniforms(desc.spp, 32);
    let draw = select_edge_draw(desc, params, reps, scene.frames, terrain.origin_spacing)?;
    let ubo = frame_uniforms(desc, draw.frame, desc.seed);
    let half_h = (0.5 * ubo.cam_fov_y).tan();
    let half_w = ubo.cam_aspect * half_h;
    let n2 = params
        .sun_dir
        .iter()
        .map(|value| value * value)
        .sum::<f32>();
    if !n2.is_finite() || n2 <= 0.0 {
        return Err(RenderError::Render(
            "test edge certificate has no finite sun direction".into(),
        ));
    }
    let length = n2.sqrt();
    let light_direction = params.sun_dir.map(|value| value / length);
    let hash = |mut value: u32| {
        value = (value ^ (value >> 16)).wrapping_mul(0x7feb_352d);
        value = (value ^ (value >> 15)).wrapping_mul(0x846c_a68b);
        value ^ (value >> 16)
    };
    let mut keyed_state = hash(
        hash(ubo.seed_hi ^ ubo.seed_lo ^ hash(draw.pixel) ^ hash(draw.frame.wrapping_add(1)))
            ^ hash(draw.sample)
            ^ 0xa511_e9b3,
    );
    if keyed_state == 0 {
        keyed_state = 0x6d2b_79f5;
    }
    let next = |state: &mut u32| {
        *state ^= *state << 13;
        *state ^= *state >> 17;
        *state ^= *state << 5;
        *state as f32 / 4_294_967_296.0
    };
    let ibl_random = [next(&mut keyed_state), next(&mut keyed_state)];
    let provenance = GpuEdgeProvenance {
        camera: Camera {
            origin: ubo.cam_origin.map(f64::from),
            right: ubo.cam_right.map(f64::from),
            up: ubo.cam_up.map(f64::from),
            forward: ubo.cam_forward.map(f64::from),
            fov_y: f64::from(ubo.cam_fov_y),
            half_h: f64::from(half_h),
            half_w: f64::from(half_w),
            aspect: f64::from(ubo.cam_aspect),
            camera_exposure: f64::from(ubo.cam_exposure),
            width: ubo.width,
            height: ubo.height,
            pixel_x: draw.pixel % ubo.width,
            pixel_y: draw.pixel / ubo.width,
        },
        sun_direction: light_direction,
        reused_lighting: false,
        reservoir_wh: [0.0, 3.0],
        ibl_random,
        seed: [ubo.seed_hi, ubo.seed_lo],
        frame: draw.frame,
        sample: draw.sample,
        terrain_dimensions: [terrain.dims[0], terrain.dims[1]],
        terrain_flags: terrain.mips[1],
        env_width: terrain.mips[2],
        env_height: terrain.mips[3] as f32,
        light_direction,
        light_color: [
            params.sun_intensity * desc.sun_color[0],
            params.sun_intensity * desc.sun_color[1],
            params.sun_intensity * desc.sun_color[2],
        ],
        turbidity_excess: (params.turbidity - 1.0).max(0.0),
        env_intensity: terrain.h_params[3],
    };
    let (projected, chunks) = prepare_edge_chunks(
        device, desc, pipes, scene, bufs, params, reps, draw, provenance,
    )?;
    Ok((draw.selected_rep, projected, chunks))
}

/// One primal+adjoint evaluation over a set of independent Monte-Carlo
/// stream seeds ("replicates"). Three phases:
///
///   A. Per-replicate forward sweep (terrain -> temporal -> spatial|copy),
///      then fold the replicate's mean linear radiance into the shared
///      replicate-mean image with weight 1/R.
///   B. ONE loss pass on the replicate-mean image. Its expectation is the
///      finite-R objective E[loss((sum_r cur_r)/R)]. A finite replicate mean
///      does not equal the exact expected image inside a nonlinear loss.
///   C. Per-replicate adjoint: the seed-deterministic frame chain is
///      replayed per K-frame tile (gradient checkpointing) and shaded
///      against the shared mean-image adjoint; the chain-rule factor
///      ∂mean/∂cur_r = 1/R is applied when the per-replicate gradient
///      sums are averaged host-side. The categorical likelihood score uses
///      the whole nonlinear loss and is scaled separately to avoid 1/R.
///
/// Returns (mean-image loss, mean per-texel albedo gradient, mean scalar
/// gradients) — gradient sums are divided by the seed count.
#[allow(clippy::too_many_arguments)]
fn eval_seed_set(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    desc: &InverseSolveDesc,
    pipes: &InversePipelines,
    scene: &InverseSceneGpu,
    bufs: &InverseBuffers,
    params: &InverseParams,
    albedo: &[f32],
    mode: EvalMode,
    seeds: &[u32],
) -> Result<(f32, Vec<f32>, Vec<f32>, Option<EdgeAuditEvidence>), RenderError> {
    let reps = seeds.len().max(1) as u32;
    let wg_x = desc.width.div_ceil(8);
    let wg_y = desc.height.div_ceil(8);
    let px = scene.px_count;
    // Checkpointed history: the whist region holds K = min(tile_size,
    // frames) per-(frame, pixel) snapshots. The adjoint replay re-runs the
    // seed-deterministic frame chain once per K-frame tile — shading each
    // tile before the next replay overwrites its slots — so the retained
    // history stays bounded by K frames instead of all `frames` (spec step
    // 4c: checkpointed tiles).
    let slots = scene.frames.max(1).min(desc.tile_size.max(1));
    // 1-D dispatches fold past the 65535-workgroup x limit through gid.y:
    // x = min(wg, 65535), y = ceil(wg / 65535); the kernels reconstruct the
    // flat index as gid.y * (65535*256) + gid.x.
    let fold_1d = |items: u64| -> (u32, u32) {
        let wg = items.div_ceil(256);
        (wg.min(65535) as u32, wg.div_ceil(65535) as u32)
    };
    let (px_x, px_y) = fold_1d(px);
    let n_u32 = 8u64 + px + bufs.texels * 4;
    let (gclr_x, gclr_y) = fold_1d(n_u32);
    // Select the estimator draw now, but defer geometry certification until a
    // real f32 replay has captured the exact camera, projection and reservoir
    // lighting inputs for this draw.  A CPU reconstruction is not accepted as
    // evidence for the GPU branch.
    let edge_draw = if mode == EvalMode::Full && desc.edge_term {
        let terrain = scene.terrain.uniforms(desc.spp, 32);
        Some(select_edge_draw(
            desc,
            params,
            reps,
            scene.frames,
            terrain.origin_spacing,
        )?)
    } else {
        None
    };

    // Reseed the per-frame stream uniforms for this replicate. The bind
    // groups reference the same buffers, so a write per frame re-points the
    // whole eval at a different stream.
    let reseed = |seed: u32| {
        for (f, ubo) in scene.frame_ubos.iter().enumerate() {
            let u = frame_uniforms(desc, f as u32, seed);
            queue.write_buffer(ubo, 0, bytemuck::bytes_of(&u));
        }
    };

    // One frame of the real forward chain; `snap` records the reservoir
    // state into whist and `publish` refreshes out_tex (skipped outside
    // Render — the loss consumes accum_hdr/mean, not the display texel).
    let frame_chain = |enc: &mut wgpu::CommandEncoder,
                       f: u32,
                       snap: bool,
                       publish: bool,
                       score_gradients: bool| {
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("inv-terrain"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipes.terrain);
            pass.set_bind_group(0, &scene.g0_frames[f as usize], &[]);
            pass.set_bind_group(1, &scene.g1, &[]);
            pass.set_bind_group(2, &scene.g2, &[]);
            pass.set_bind_group(3, &scene.g3, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }
        if snap {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("inv-wsnap"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipes.wsnap);
            pass.set_bind_group(0, &scene.g0_frames[f as usize], &[]);
            pass.set_bind_group(1, &scene.g1, &[]);
            pass.set_bind_group(2, &scene.g2, &[]);
            pass.set_bind_group(3, &scene.g3, &[]);
            pass.dispatch_workgroups(px_x, px_y, 1);
        }
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("inv-restir-temporal"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipes.restir_temporal);
            pass.set_bind_group(0, &scene.restir_g0_frames[f as usize], &[]);
            pass.set_bind_group(1, &scene.restir_empty_g1, &[]);
            pass.set_bind_group(2, &scene.restir_temporal_g2, &[]);
            pass.dispatch_workgroups(px_x, px_y, 1);
        }
        if desc.spatial_reuse {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("inv-restir-spatial"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipes.restir_spatial);
            pass.set_bind_group(0, &scene.restir_g0_frames[f as usize], &[]);
            pass.set_bind_group(1, &scene.restir_spatial_g1, &[]);
            pass.set_bind_group(2, &scene.restir_spatial_g2, &[]);
            pass.dispatch_workgroups(px_x, px_y, 1);
            drop(pass);
            if score_gradients && desc.score_correction {
                // Spectral spatial reuse redraws a channel once per eligible
                // pixel. The resulting sample is in prev until the next frame.
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("inv-score-spatial"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipes.score_spatial);
                pass.set_bind_group(0, &scene.g0_frames[f as usize], &[]);
                pass.set_bind_group(1, &scene.g1, &[]);
                pass.set_bind_group(2, &scene.g2_score_spatial, &[]);
                pass.set_bind_group(3, &scene.g3, &[]);
                pass.dispatch_workgroups(px_x, px_y, 1);
            }
        } else {
            enc.copy_buffer_to_buffer(&scene.res_out, 0, &scene.res_prev, 0, scene.reservoir_bytes);
        }
        if score_gradients && desc.score_correction {
            // Both the fresh candidate and the temporal output remain live.
            // The score kernel omits candidates erased by spatial self-redraw.
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("inv-score-candidate"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipes.score_candidate);
            pass.set_bind_group(0, &scene.g0_frames[f as usize], &[]);
            pass.set_bind_group(1, &scene.g1, &[]);
            pass.set_bind_group(2, &scene.g2_score_candidate, &[]);
            pass.set_bind_group(3, &scene.g3, &[]);
            pass.dispatch_workgroups(px_x, px_y, 1);
        }
        if publish {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("inv-terrain-publish"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipes.terrain_publish);
            pass.set_bind_group(0, &scene.g0_frames[f as usize], &[]);
            pass.set_bind_group(1, &scene.g1, &[]);
            pass.set_bind_group(2, &scene.g2, &[]);
            pass.set_bind_group(3, &scene.g3, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }
    };
    let pixel_clear = |enc: &mut wgpu::CommandEncoder, label: &'static str| {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipes.clear);
        pass.set_bind_group(0, &scene.g0_frames[0], &[]);
        pass.set_bind_group(1, &scene.g1, &[]);
        pass.set_bind_group(2, &scene.g2, &[]);
        pass.set_bind_group(3, &scene.g3, &[]);
        pass.dispatch_workgroups(px_x, px_y, 1);
    };

    if mode == EvalMode::Render {
        scene.upload_params(queue, desc, params, albedo);
        reseed(seeds.first().copied().unwrap_or(desc.seed));
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("inv-render-enc"),
        });
        pixel_clear(&mut enc, "inv-clear");
        for f in 0..scene.frames {
            frame_chain(&mut enc, f, false, true, false);
        }
        queue.submit([enc.finish()]);
        return Ok((0.0, Vec::new(), Vec::new(), None));
    }

    // Phase 0: gradient state is cleared once — it accumulates across the
    // whole replicate set.
    scene.upload_params(queue, desc, params, albedo);
    {
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("inv-clear-grad-enc"),
        });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("inv-clear-grad"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipes.clear_grad);
            pass.set_bind_group(0, &scene.g0_frames[0], &[]);
            pass.set_bind_group(1, &scene.g1, &[]);
            pass.set_bind_group(2, &scene.g2, &[]);
            pass.set_bind_group(3, &scene.g3, &[]);
            pass.dispatch_workgroups(gclr_x, gclr_y, 1);
        }
        queue.submit([enc.finish()]);
    }

    // Phase A: per-replicate forward sweep + fold into the mean image.
    for (ord, &seed) in seeds.iter().enumerate() {
        scene.upload_inv_params(queue, desc, params, ord as u32, reps, true);
        reseed(seed);
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("inv-fwd-enc"),
        });
        pixel_clear(&mut enc, "inv-clear");
        for f in 0..scene.frames {
            frame_chain(&mut enc, f, false, false, false);
        }
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("inv-accum-mean"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipes.accum_mean);
            pass.set_bind_group(0, &scene.g0_frames[0], &[]);
            pass.set_bind_group(1, &scene.g1, &[]);
            pass.set_bind_group(2, &scene.g2, &[]);
            pass.set_bind_group(3, &scene.g3, &[]);
            pass.dispatch_workgroups(px_x, px_y, 1);
        }
        queue.submit([enc.finish()]);
    }

    // Phase B: one loss pass on the replicate-mean image.
    {
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("inv-loss-enc"),
        });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("inv-loss"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipes.loss);
            pass.set_bind_group(0, &scene.g0_frames[0], &[]);
            pass.set_bind_group(1, &scene.g1, &[]);
            pass.set_bind_group(2, &scene.g2, &[]);
            pass.set_bind_group(3, &scene.g3, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }
        queue.submit([enc.finish()]);
    }

    // Capture the source values actually consumed by the edge shader from an
    // otherwise identical replay of the selected replicate.  This pass ends
    // before receiver work, loss evaluation, or accumulation; its only
    // output is the f32 provenance token.  Phase C resets replay state before
    // producing the final audited events, and every final audit must match
    // this token bit-for-bit.
    let edge_work = if let Some(draw) = edge_draw {
        let probe = prepare_edge_probe(device, desc, pipes, scene, bufs, params, reps, draw)?;
        let selected_seed = seeds
            .get(draw.selected_rep as usize)
            .copied()
            .ok_or_else(|| {
                RenderError::Render("inverse edge selected replica is unavailable".into())
            })?;
        reseed(selected_seed);
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("inv-edge-provenance-enc"),
        });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("inv-edge-provenance-replay-clear"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipes.replay_clear);
            pass.set_bind_group(0, &scene.g0_frames[0], &[]);
            pass.set_bind_group(1, &scene.g1, &[]);
            pass.set_bind_group(2, &scene.g2, &[]);
            pass.set_bind_group(3, &scene.g3, &[]);
            pass.dispatch_workgroups(px_x, px_y, 1);
        }
        enc.clear_buffer(&probe._audit, 0, None);
        enc.clear_buffer(&probe._edge_sum, 0, None);
        let mut base = 0u32;
        let mut tile_index = 0usize;
        while base < scene.frames {
            let end = (base + slots).min(scene.frames);
            for frame in base..end {
                frame_chain(&mut enc, frame, true, false, false);
            }
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("inv-edge-provenance"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipes.edge);
            pass.set_bind_group(0, &probe.g0_tiles[tile_index], &[]);
            pass.set_bind_group(1, &scene.g1, &[]);
            pass.set_bind_group(2, &probe.g2, &[]);
            pass.set_bind_group(3, &scene.g3_edge, &[]);
            let (ex, ey) = fold_1d(u64::from(probe.count));
            pass.dispatch_workgroups(ex, ey, 1);
            tile_index += 1;
            base = end;
        }
        queue.submit([enc.finish()]);
        let provenance = read_edge_provenance(device, queue, &probe)?;
        let (projected, chunks) = prepare_edge_chunks(
            device, desc, pipes, scene, bufs, params, reps, draw, provenance,
        )?;
        Some(EdgeWork {
            selected_rep: draw.selected_rep,
            provenance,
            projected,
            chunks,
        })
    } else {
        None
    };

    // Phase C: per-replicate adjoint. Each replicate replays its frame
    // chain per K-frame tile (rebuilding reservoirs + whist snapshots) and
    // shades against the shared mean-image adjoint.
    if mode == EvalMode::Full {
        for (rep_ord, &seed) in seeds.iter().enumerate() {
            reseed(seed);
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("inv-adj-enc"),
            });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("inv-replay-clear"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipes.replay_clear);
                pass.set_bind_group(0, &scene.g0_frames[0], &[]);
                pass.set_bind_group(1, &scene.g1, &[]);
                pass.set_bind_group(2, &scene.g2, &[]);
                pass.set_bind_group(3, &scene.g3, &[]);
                pass.dispatch_workgroups(px_x, px_y, 1);
            }
            if let Some(work) = &edge_work {
                if rep_ord as u32 == work.selected_rep {
                    // Every event writes exactly once in its matching replay
                    // tile. Clear the persistent per-chunk witnesses before
                    // that replay so a missing write is a zero/non-finite
                    // audit rejection rather than stale GPU memory.
                    for chunk in &work.chunks {
                        enc.clear_buffer(&chunk._audit, 0, None);
                        enc.clear_buffer(&chunk._edge_sum, 0, None);
                    }
                }
            }
            let mut base = 0u32;
            let mut tile_index = 0usize;
            while base < scene.frames {
                let end = (base + slots).min(scene.frames);
                for f in base..end {
                    frame_chain(&mut enc, f, true, false, true);
                }
                {
                    // The tile's last frame UBO: the kernel derives the
                    // live whist window [f_end - f_end%K, f_end] from
                    // uniforms.frame_index.
                    let f_end = end - 1;
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("inv-shade"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&pipes.shade);
                    pass.set_bind_group(0, &scene.g0_frames[f_end as usize], &[]);
                    pass.set_bind_group(1, &scene.g1, &[]);
                    pass.set_bind_group(2, &scene.g2, &[]);
                    pass.set_bind_group(3, &scene.g3, &[]);
                    pass.dispatch_workgroups(wg_x, wg_y, 1);
                }
                if let Some(work) = &edge_work {
                    if rep_ord as u32 == work.selected_rep {
                        for chunk in &work.chunks {
                            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                                label: Some("inv-certified-edge"),
                                timestamp_writes: None,
                            });
                            pass.set_pipeline(&pipes.edge);
                            pass.set_bind_group(0, &chunk.g0_tiles[tile_index], &[]);
                            pass.set_bind_group(1, &scene.g1, &[]);
                            pass.set_bind_group(2, &chunk.g2, &[]);
                            pass.set_bind_group(3, &scene.g3_edge, &[]);
                            let (ex, ey) = fold_1d(u64::from(chunk.count));
                            pass.dispatch_workgroups(ex, ey, 1);
                        }
                    }
                }
                tile_index += 1;
                base = end;
            }
            queue.submit([enc.finish()]);
        }
    }

    // One staging readback covers the whole u32 region: scalars [0,32B),
    // loss [32B, 32B + px*4), dalbedo [32B + px*4, ...). The gradient
    // regions hold the SUM over replicates — divide by R for the mean
    // (the chain-rule 1/R of the replicate-mean image).
    let u32_bytes = 8 * 4 + px * 4 + bufs.texels * 16;
    let raw = read_buffer(device, queue, &bufs.u32_buf, 0, u32_bytes)?;
    let edge_rejection = u32::from_le_bytes(raw[28..32].try_into().unwrap());
    if edge_rejection != 0 {
        return Err(RenderError::Render(format!(
            "inverse certified edge replay rejected GPU event (flags {edge_rejection:#x})"
        )));
    }
    let (mut edge_audit_evidence, validated_edge_accumulator) = if let Some(work) = &edge_work {
        let terrain = scene.terrain.uniforms(desc.spp, 32);
        let mut evidence = EdgeAuditEvidence::default();
        let mut accumulator = ValidatedEdgeAccumulator::empty();
        for chunk in &work.chunks {
            let audit_bytes = u64::from(chunk.count)
                .checked_mul(EDGE_AUDIT_STRIDE)
                .ok_or_else(|| {
                    RenderError::Render("inverse edge audit readback size overflow".into())
                })?;
            let audit_raw = read_buffer(device, queue, &chunk._audit, 0, audit_bytes)?;
            let audits = decode_gpu_audit_bytes(&audit_raw).map_err(|e| {
                RenderError::Render(format!("inverse edge GPU audit decode rejected: {e:?}"))
            })?;
            #[cfg(test)]
            evidence.observe_audits(&work.projected[chunk.event_range.clone()], &audits);
            let edge_sum_raw = read_buffer(device, queue, &chunk._edge_sum, 0, EDGE_SUM_BYTES)?;
            let atomic_edge_sum = std::array::from_fn(|k| {
                let start = k * std::mem::size_of::<f32>();
                f32::from_le_bytes(
                    edge_sum_raw[start..start + std::mem::size_of::<f32>()]
                        .try_into()
                        .expect("fixed-size certified edge sum"),
                )
            });
            let selected_seed =
                seeds
                    .get(work.selected_rep as usize)
                    .copied()
                    .ok_or_else(|| {
                        RenderError::Render("inverse edge selected replica is unavailable".into())
                    })?;
            let source_ubo = frame_uniforms(desc, work.provenance.frame, selected_seed);
            let source_length = (params.sun_dir[0] * params.sun_dir[0]
                + params.sun_dir[1] * params.sun_dir[1]
                + params.sun_dir[2] * params.sun_dir[2])
                .sqrt()
                .max(1e-8);
            let source_light_direction = params.sun_dir.map(|value| value / source_length);
            let source_light_color = [
                params.sun_intensity * desc.sun_color[0],
                params.sun_intensity * desc.sun_color[1],
                params.sun_intensity * desc.sun_color[2],
            ];
            let ctx = GpuAuditContext {
                terrain_width: desc.dem_width as usize,
                terrain_height: desc.dem_height as usize,
                terrain_origin: [terrain.origin_spacing[0], terrain.origin_spacing[1]],
                terrain_spacing: [terrain.origin_spacing[2], terrain.origin_spacing[3]],
                height_exaggeration: terrain.h_params[2],
                heights: &desc.heights,
                albedo,
                env_map: desc
                    .env_map
                    .as_ref()
                    .map(|(data, width, height)| (data.as_slice(), *width, *height)),
                terrain_flags: terrain.mips[1],
                env_dimensions: [terrain.mips[2], terrain.mips[3]],
                light_direction: source_light_direction,
                light_color: source_light_color,
                turbidity_excess: (params.turbidity - 1.0).max(0.0),
                env_intensity: terrain.h_params[3],
                seed: [source_ubo.seed_hi, source_ubo.seed_lo],
                provenance: work.provenance,
                normal_offset: 1e-3,
                ray_tmin: 1e-3,
                ray_tmax: 1e30,
                struct_weight: chunk.struct_weight,
                mean_image_weight: chunk.mean_image_weight,
                replay_sample_weight: chunk.replay_sample_weight,
                jump_numerator: chunk.jump_numerator,
                jump_denominator: chunk.jump_denominator,
            };
            #[cfg(test)]
            geometry_cert::assert_gpu_audit_source_corruptions_reject(
                &ctx,
                &work.projected[chunk.event_range.clone()],
                &audits,
                atomic_edge_sum,
            );
            let bound = validate_gpu_audits(
                &ctx,
                &work.projected[chunk.event_range.clone()],
                &audits,
                atomic_edge_sum,
            )
            .map_err(|e| RenderError::Render(format!("inverse edge GPU audit rejected: {e:?}")))?;
            #[cfg(test)]
            eprintln!(
                "inverse certified edge chunk: atomic={:?} replay={:?} upload={} jump_diff={} ibl_direction_width={} source_radiance_width={} radiance_width={} loss_width={} accumulation={:?}",
                bound.atomic_sum_observed,
                bound.replay_order_sum_f32,
                bound.max_upload_weight_error,
                bound.max_jump_reference_difference,
                bound.max_ibl_direction_source_width,
                bound.max_source_radiance_width,
                bound.max_radiance_reconstruction_width,
                bound.max_loss_interval_width,
                bound.accumulation_rounding_bound,
            );
            accumulator.push_chunk(&bound).map_err(|e| {
                RenderError::Render(format!(
                    "inverse edge host accumulation rejected validated chunk: {e:?}"
                ))
            })?;
            evidence.observe(bound);
        }
        evidence.observe_host_accumulator(&accumulator);
        (Some(evidence), accumulator)
    } else {
        (None, ValidatedEdgeAccumulator::empty())
    };
    let mut scalars = decode_scalars(&raw[..(8 * 4)]);
    if !scalars[6].is_finite() || scalars[6] != 0.0 {
        return Err(RenderError::Render(format!(
            "inverse gradient rejected {} non-finite GPU contributions",
            scalars[6]
        )));
    }
    if edge_work.is_some() {
        let smooth_sun = [scalars[0], scalars[1], scalars[2]];
        let (combined_sun, rounding_bound) =
            combine_smooth_and_validated_edge(smooth_sun, &validated_edge_accumulator).map_err(
                |e| {
                    RenderError::Render(format!(
                        "inverse final scalar/validated-edge combination rejected: {e:?}"
                    ))
                },
            )?;
        scalars[..3].copy_from_slice(&combined_sun);
        if let Some(evidence) = &mut edge_audit_evidence {
            evidence.final_scalar_add_rounding_bound = rounding_bound;
        }
    }
    let loss_region = &raw[(8 * 4)..(8 * 4) + (px as usize) * 4];
    let mean_loss = loss_region
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64)
        .sum::<f64>()
        / px as f64;
    let dalbedo_region = &raw[(8 * 4) + (px as usize) * 4..];
    let inv_r = 1.0 / reps as f32;
    let d_albedo: Vec<f32> = dalbedo_region
        .chunks_exact(16)
        .flat_map(|t| {
            [
                f32::from_le_bytes([t[0], t[1], t[2], t[3]]) * inv_r,
                f32::from_le_bytes([t[4], t[5], t[6], t[7]]) * inv_r,
                f32::from_le_bytes([t[8], t[9], t[10], t[11]]) * inv_r,
            ]
        })
        .collect();
    let scalars: Vec<f32> = if edge_work.is_some() {
        let (scaled_sun, rounding_bound) =
            scale_validated_sun_gradient([scalars[0], scalars[1], scalars[2]], reps).map_err(
                |e| {
                    RenderError::Render(format!(
                        "inverse final validated-edge replicate scaling rejected: {e:?}"
                    ))
                },
            )?;
        if let Some(evidence) = &mut edge_audit_evidence {
            evidence.final_scalar_scale_rounding_bound = rounding_bound;
        }
        let mut result = Vec::with_capacity(scalars.len());
        result.extend(scaled_sun);
        result.extend(scalars[3..].iter().map(|value| value * inv_r));
        result
    } else {
        scalars.iter().map(|value| value * inv_r).collect()
    };
    Ok((mean_loss as f32, d_albedo, scalars, edge_audit_evidence))
}

/// Single-seed eval — convenience wrapper for tests and the Render path.
#[allow(clippy::too_many_arguments)]
fn eval_pass(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    desc: &InverseSolveDesc,
    pipes: &InversePipelines,
    scene: &InverseSceneGpu,
    bufs: &InverseBuffers,
    params: &InverseParams,
    albedo: &[f32],
    mode: EvalMode,
    seed: u32,
) -> Result<(f32, Vec<f32>, Vec<f32>), RenderError> {
    let (loss, albedo_gradient, scalar_gradient, _) = eval_seed_set(
        device,
        queue,
        desc,
        pipes,
        scene,
        bufs,
        params,
        albedo,
        mode,
        &[seed],
    )?;
    Ok((loss, albedo_gradient, scalar_gradient))
}

/// Test seam for the production edge-replay path.  It exposes only the
/// post-validation witness summary, never a bypass around the certificate.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn eval_pass_with_edge_evidence(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    desc: &InverseSolveDesc,
    pipes: &InversePipelines,
    scene: &InverseSceneGpu,
    bufs: &InverseBuffers,
    params: &InverseParams,
    albedo: &[f32],
    mode: EvalMode,
    seed: u32,
) -> Result<(f32, Vec<f32>, Vec<f32>, Option<EdgeAuditEvidence>), RenderError> {
    eval_seed_set(
        device,
        queue,
        desc,
        pipes,
        scene,
        bufs,
        params,
        albedo,
        mode,
        &[seed],
    )
}

/// Readback the forward beauty (out_tex — Reinhard-space rgba16f, quantized
/// to u8 exactly as the forward readback does) plus the linear mean
/// radiance (accum_hdr / frame count).
fn read_beauty(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    scene: &InverseSceneGpu,
) -> Result<(Vec<u8>, [f32; 3]), RenderError> {
    let raw = read_texture(device, queue, &scene.out_tex, 8)?;
    let rgba = rgba16f_to_rgba8(&raw);
    let acc_raw = read_buffer(device, queue, &scene.accum, 0, scene.px_count * 16)?;
    let frames = scene.frames.max(1) as f32;
    let mut mean = [0.0f64; 3];
    for px in acc_raw.chunks_exact(16) {
        for c in 0..3 {
            mean[c] += f32::from_le_bytes([px[c * 4], px[c * 4 + 1], px[c * 4 + 2], px[c * 4 + 3]])
                as f64
                / frames as f64;
        }
    }
    let n = scene.px_count as f64;
    Ok((
        rgba,
        [
            (mean[0] / n) as f32,
            (mean[1] / n) as f32,
            (mean[2] / n) as f32,
        ],
    ))
}

/// Independent Monte-Carlo stream replicates averaged per optimizer step —
/// the solver sees the expectation of the loss/gradient field instead of
/// one fixed-noise realization (whose residual bumps masquerade as shallow
/// basins along the intensity/turbidity/albedo valley).
const GRAD_REPLICATES: u32 = 16;

/// Bilinear upsample of a 3-channel (sw,sh) map to (dw,dh) texel centers.
/// Used by the coarse-albedo continuation stage: the latent lives on a
/// coarser grid and is expanded to the DEM texel map for upload.
fn upsample3(src: &[f32], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; dw * dh * 3];
    for iz in 0..dh {
        let tz = (iz as f32 + 0.5) * sh as f32 / dh as f32 - 0.5;
        let z0 = tz.floor().max(0.0) as usize;
        let z1 = (z0 + 1).min(sh - 1);
        let fz = (tz - z0 as f32).clamp(0.0, 1.0);
        for ix in 0..dw {
            let tx = (ix as f32 + 0.5) * sw as f32 / dw as f32 - 0.5;
            let x0 = tx.floor().max(0.0) as usize;
            let x1 = (x0 + 1).min(sw - 1);
            let fx = (tx - x0 as f32).clamp(0.0, 1.0);
            for c in 0..3 {
                let c00 = src[(z0 * sw + x0) * 3 + c];
                let c10 = src[(z0 * sw + x1) * 3 + c];
                let c01 = src[(z1 * sw + x0) * 3 + c];
                let c11 = src[(z1 * sw + x1) * 3 + c];
                out[(iz * dw + ix) * 3 + c] =
                    (c00 * (1.0 - fx) + c10 * fx) * (1.0 - fz) + (c01 * (1.0 - fx) + c11 * fx) * fz;
            }
        }
    }
    out
}

/// Adjoint of `upsample3`: gather the full-resolution log-albedo gradient
/// back onto the coarse latent with the same bilinear weights (the exact
/// transpose of the upsample operator, so the coarse Adam step sees the
/// true dL/d(latent)).
fn downsample3_adjoint(grad: &[f32], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; sw * sh * 3];
    for iz in 0..dh {
        let tz = (iz as f32 + 0.5) * sh as f32 / dh as f32 - 0.5;
        let z0 = tz.floor().max(0.0) as usize;
        let z1 = (z0 + 1).min(sh - 1);
        let fz = (tz - z0 as f32).clamp(0.0, 1.0);
        for ix in 0..dw {
            let tx = (ix as f32 + 0.5) * sw as f32 / dw as f32 - 0.5;
            let x0 = tx.floor().max(0.0) as usize;
            let x1 = (x0 + 1).min(sw - 1);
            let fx = (tx - x0 as f32).clamp(0.0, 1.0);
            let g = &grad[(iz * dw + ix) * 3..(iz * dw + ix) * 3 + 3];
            for c in 0..3 {
                out[(z0 * sw + x0) * 3 + c] += g[c] * (1.0 - fx) * (1.0 - fz);
                out[(z0 * sw + x1) * 3 + c] += g[c] * fx * (1.0 - fz);
                out[(z1 * sw + x0) * 3 + c] += g[c] * (1.0 - fx) * fz;
                out[(z1 * sw + x1) * 3 + c] += g[c] * fx * fz;
            }
        }
    }
    out
}

/// Project a unit-direction gradient onto the raw sun_dir parameter:
/// ∂L/∂ω_raw = (I − ω ωᵀ)/|ω| · ∂L/∂ω_unit (the kernel normalizes internally).
fn project_dir_grad(omega: [f32; 3], g: [f32; 3]) -> [f32; 3] {
    let len = (omega[0] * omega[0] + omega[1] * omega[1] + omega[2] * omega[2])
        .sqrt()
        .max(1e-8);
    let u = [omega[0] / len, omega[1] / len, omega[2] / len];
    let dot = u[0] * g[0] + u[1] * g[1] + u[2] * g[2];
    [
        (g[0] - u[0] * dot) / len,
        (g[1] - u[1] * dot) / len,
        (g[2] - u[2] * dot) / len,
    ]
}

/// Multi-replicate eval over well-separated deterministic stream seeds. The
/// loss is computed once on their mean image, and each replicate's adjoint
/// is shaded against that shared mean. This is a repeatable finite sample
/// of the expected finite-R loss, not an exact expectation. The seed set is
/// fixed across iterations to keep the optimization landscape deterministic.
#[allow(clippy::too_many_arguments)]
fn eval_mean(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    desc: &InverseSolveDesc,
    pipes: &InversePipelines,
    scene: &InverseSceneGpu,
    bufs: &InverseBuffers,
    params: &InverseParams,
    albedo: &[f32],
    mode: EvalMode,
    replicates: u32,
) -> Result<(f32, Vec<f32>, Vec<f32>), RenderError> {
    let seeds: Vec<u32> = (0..replicates.max(1))
        .map(|r| desc.seed.wrapping_add(r.wrapping_mul(0x9E37_79B9)))
        .collect();
    let (loss, albedo_gradient, scalar_gradient, _) = eval_seed_set(
        device, queue, desc, pipes, scene, bufs, params, albedo, mode, &seeds,
    )?;
    Ok((loss, albedo_gradient, scalar_gradient))
}

/// The DIFFERENTIA solver. Public entry used by `src/py_functions/inverse.rs`.
pub fn solve(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    desc: &InverseSolveDesc,
) -> Result<InverseSolveOutput, RenderError> {
    validate_solve_desc(desc)?;
    let tracker = global_tracker();
    // Worst-case transient host-visible footprint of one eval: the single
    // u32-region staging readback plus the beauty readbacks on the final
    // render. Checked up front under the registry's enforce policy; the
    // ledger capture below measures the realized solve-scoped peak.
    let px = desc.width as u64 * desc.height as u64;
    let texels = desc.dem_width as u64 * desc.dem_height as u64;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as u64;
    let padded_tex = (desc.width as u64 * 8).div_ceil(align) * align * desc.height as u64;
    let staging_estimate = (8 * 4 + px * 4 + texels * 16) + padded_tex + px * 16;
    tracker.check_budget_labeled(staging_estimate, "inverse-pt-solve")?;

    // Render-scoped allocation owner + ledger capture: every tracked
    // allocation in this solve (including transient staging readbacks and
    // certified-event CPU workspaces) is attributed to this owner, so the
    // captured peak includes their concurrent host-visible footprint.
    let allocation_owner = AllocationOwner::new();
    let _allocation_scope = allocation_owner.activate();
    begin_ledger_capture(&[allocation_owner.id()]);
    let _capture_guard = CaptureGuard;

    let pipes = InversePipelines::new(device)?;
    let bufs = InverseBuffers::new(
        device,
        queue,
        desc.width,
        desc.height,
        (desc.dem_width, desc.dem_height),
        desc.frames.max(1).min(desc.tile_size.max(1)),
        &desc.target_rgba,
    )?;
    let scene = InverseSceneGpu::new(device, queue, desc, &pipes, &bufs)?;

    // --- mixed parameterization ---
    // Albedo and intensity live in log space (positivity + multiplicative
    // geometry); sun direction on the unit sphere; turbidity linear.
    let mut log_alb: Vec<f32> = desc
        .init_albedo
        .iter()
        .map(|v| v.clamp(ALBEDO_MIN, ALBEDO_MAX).ln())
        .collect();
    let init_log_alb: Vec<f32> = log_alb.clone();
    let mut sun_dir = {
        let mut p = desc.init;
        p.clamp();
        p.sun_dir
    };
    let mut theta_i = desc.init.sun_intensity.clamp(1e-3, SUN_INTENSITY_MAX).ln();
    let mut tau = desc.init.turbidity.clamp(TURBIDITY_MIN, TURBIDITY_MAX);

    let mut adam_alb = AdamState::new(log_alb.len());
    let mut adam_sun = AdamState::new(4); // dir xyz + theta_i
    let mut adam_tau = AdamState::new(1);
    // Stage-A global albedo offset (per-channel log shift, 3 DOF).
    let mut alb_off = [0.0f32; 3];
    let mut adam_off = AdamState::new(3);
    // The scalar landscape is a narrow curved valley (sun direction,
    // intensity and turbidity trade through T(el, tau)); lower momentum
    // keeps the iterate inside the basin.
    adam_sun.beta1 = 0.5;
    adam_tau.beta1 = 0.5;
    // Albedo tracks the scalar trajectory quasi-statically: lower momentum
    // shortens the lag so the per-texel map stays equilibrated against the
    // current (sun, I, tau) instead of dragging a stale compensation.
    adam_alb.beta1 = 0.5;
    adam_off.beta1 = 0.5;

    let mut loss_history = Vec::with_capacity(desc.iters as usize);
    let mut best = f32::INFINITY;
    let mut stall = 0u32;
    let mut iters_run = 0u32;

    let tail_from = ((desc.iters as f32) * 0.7) as u32;
    let mut tail_n = 0u32;
    let mut alb_avg = vec![0.0f32; log_alb.len()];
    let mut dir_avg = [0.0f32; 3];
    let mut theta_avg = 0.0f32;
    let mut tau_avg = 0.0f32;
    let mut best_albedo: Option<Vec<f32>> = None;
    let mut best_params = desc.init;

    let mut params = InverseParams {
        sun_dir,
        sun_intensity: theta_i.exp(),
        turbidity: tau,
    };
    let mut albedo: Vec<f32> = log_alb.iter().map(|v| v.exp()).collect();

    // Albedo continuation schedule. Per-texel albedo can absorb a wrong
    // (sun, I, tau) almost exactly, which captures the joint descent in a
    // shallow wrong basin:
    //   * stage A [0, 15%):      albedo pattern frozen at init, but a
    //                            global per-channel offset is optimized —
    //                            it absorbs the init level mismatch while
    //                            the frozen pattern keeps the scalars
    //                            explaining the image structure (a fully
    //                            frozen map would force I/tau to compensate
    //                            for a wrong mean level, dragging them into
    //                            the high-intensity/high-turbidity basin);
    //   * stage B [15%, 60%):    albedo optimized on a coarse latent grid
    //                            (bilinear-upsampled to the DEM texel map)
    //                            — it can fit broad tone but NOT per-texel
    //                            compensation, so the scalars keep being
    //                            pushed toward the true basin;
    //   * stage C [60%, end):    full-resolution albedo, initialized from
    //                            the converged coarse latent.
    let warmup_iters = (desc.iters * 3) / 20;
    let coarse_until = (desc.iters * 3) / 5;
    // Coarse latent side: quarter of the DEM resolution, clamped to a
    // useful range; falls back to full-res behaviour on tiny DEMs.
    let coarse_side = (desc.dem_width.max(desc.dem_height) as usize / 4).clamp(2, 16);
    let full_alb = log_alb.len() == desc.dem_width as usize * desc.dem_height as usize * 3;
    let use_coarse =
        full_alb && coarse_side < desc.dem_width as usize && coarse_side < desc.dem_height as usize;
    let mut log_latent: Vec<f32> = {
        let m0 = log_alb.iter().copied().sum::<f32>() / log_alb.len() as f32;
        vec![m0; coarse_side * coarse_side * 3]
    };
    let mut adam_latent = AdamState::new(log_latent.len());
    adam_latent.beta1 = 0.5;

    for it in 0..desc.iters {
        iters_run = it + 1;
        let (loss, d_albedo, scalars) = eval_mean(
            device,
            queue,
            desc,
            &pipes,
            &scene,
            &bufs,
            &params,
            &albedo,
            EvalMode::Full,
            GRAD_REPLICATES,
        )?;
        loss_history.push(loss);
        if !loss.is_finite() {
            return Err(RenderError::Render(format!(
                "inverse solve produced non-finite loss at iteration {it}"
            )));
        }

        // Improvement + checkpoint bookkeeping BEFORE the Adam step — the
        // loss was measured at the current (pre-update) parameters.
        // True relative early stop: improvement must exceed tol * |best|.
        if it == 0 || best - loss > desc.early_stop_tol * best.abs().max(1e-30) {
            best = loss;
            best_albedo = Some(albedo.clone());
            best_params = params;
            stall = 0;
        } else {
            stall += 1;
        }

        let cos_decay =
            0.5 * (1.0 + (std::f32::consts::PI * it as f32 / desc.iters.max(1) as f32).cos());
        let lr_scale = 0.02 + 0.98 * cos_decay;
        // Scalar groups keep a higher lr floor: the (sun, I, tau) valley is
        // shallow and curved, so the slide toward its basin floor must
        // still be moving late in the budget rather than freezing at the
        // point where the cosine decay hits ~zero.
        let lr_scale_s = 0.10 + 0.90 * cos_decay;

        // Chain rule into the optimization space: dL/d(log a) = a * dL/da,
        // dL/d(log I) = I * dL/dI.
        let log_lo = ALBEDO_MIN.ln();
        let log_hi = ALBEDO_MAX.ln();
        let g_logalb: Vec<f32> = d_albedo
            .iter()
            .zip(albedo.iter())
            .map(|(g, a)| g * a)
            .collect();
        if it < warmup_iters {
            // Stage A: optimize only the global per-channel offset; the
            // frozen pattern keeps the scalars honest while the offset
            // absorbs any init level mismatch. d off_c = sum_t g_logalb.
            let mut g_off = [0.0f32; 3];
            for t in g_logalb.chunks_exact(3) {
                for (c, g) in t.iter().enumerate() {
                    g_off[c] += g;
                }
            }
            adam_off.step(&mut alb_off, &g_off, desc.lr_albedo * lr_scale);
            for (i, v) in log_alb.iter_mut().enumerate() {
                *v = (init_log_alb[i] + alb_off[i % 3]).clamp(log_lo, log_hi);
            }
        } else if use_coarse && it < coarse_until {
            if it == warmup_iters {
                // Stage B entry: reseed the flat latent at the level the
                // stage-A offset converged to (per-channel mean).
                let mut m = [0.0f64; 3];
                for t in log_alb.chunks_exact(3) {
                    for (c, v) in t.iter().enumerate() {
                        m[c] += *v as f64;
                    }
                }
                let nt = (log_alb.len() / 3).max(1) as f64;
                for (i, v) in log_latent.iter_mut().enumerate() {
                    *v = (m[i % 3] / nt) as f32;
                }
            }
            // Stage B: Adam step on the coarse latent, then expand.
            let g_lat = downsample3_adjoint(
                &g_logalb,
                coarse_side,
                coarse_side,
                desc.dem_width as usize,
                desc.dem_height as usize,
            );
            adam_latent.step(&mut log_latent, &g_lat, desc.lr_albedo * lr_scale);
            log_alb = upsample3(
                &log_latent,
                coarse_side,
                coarse_side,
                desc.dem_width as usize,
                desc.dem_height as usize,
            );
        } else {
            if use_coarse && it == coarse_until {
                // Stage B -> C transition: seed the full-resolution map
                // from the converged latent and switch Adam state over.
                log_alb = upsample3(
                    &log_latent,
                    coarse_side,
                    coarse_side,
                    desc.dem_width as usize,
                    desc.dem_height as usize,
                );
                adam_alb = AdamState::new(log_alb.len());
                adam_alb.beta1 = 0.5;
            }
            adam_alb.step(&mut log_alb, &g_logalb, desc.lr_albedo * lr_scale);
        }
        for v in log_alb.iter_mut() {
            *v = v.clamp(log_lo, log_hi);
        }
        // Clamp the latent too: during stage B it is the actual parameter,
        // and keeping it in range means the bilinear expansion can never
        // leave the albedo domain.
        for v in log_latent.iter_mut() {
            *v = v.clamp(log_lo, log_hi);
        }

        let g_dir = project_dir_grad(sun_dir, [scalars[0], scalars[1], scalars[2]]);
        let g_theta = scalars[3] * params.sun_intensity;
        let mut sun_vec = [sun_dir[0], sun_dir[1], sun_dir[2], theta_i];
        adam_sun.step(
            &mut sun_vec,
            &[g_dir[0], g_dir[1], g_dir[2], g_theta],
            desc.lr_sun * lr_scale_s,
        );
        sun_dir = [sun_vec[0], sun_vec[1], sun_vec[2]];
        theta_i = sun_vec[3].clamp(1e-3f32.ln(), SUN_INTENSITY_MAX.ln());
        // Re-normalize the direction after the tangent-space step.
        let dl = (sun_dir[0] * sun_dir[0] + sun_dir[1] * sun_dir[1] + sun_dir[2] * sun_dir[2])
            .sqrt()
            .max(1e-8);
        sun_dir = [sun_dir[0] / dl, sun_dir[1] / dl, sun_dir[2] / dl];

        let mut tau_vec = [tau];
        adam_tau.step(&mut tau_vec, &[scalars[4]], desc.lr_turbidity * lr_scale_s);
        tau = tau_vec[0].clamp(TURBIDITY_MIN, TURBIDITY_MAX);

        params = InverseParams {
            sun_dir,
            sun_intensity: theta_i.exp().clamp(SUN_INTENSITY_MIN, SUN_INTENSITY_MAX),
            turbidity: tau,
        };
        albedo = log_alb.iter().map(|v| v.exp()).collect();

        // Intensity line-search every 8th iteration: probe theta +/- 0.15
        // log units alongside the current point and step to the argmin —
        // a real 3-point search, not a heuristic nudge.
        if (it + 1) % 8 == 0 && it + 1 < desc.iters {
            let delta = 0.15f32;
            let mut best_theta = theta_i;
            let mut best_probe = f32::INFINITY;
            for cand in [theta_i - delta, theta_i, theta_i + delta] {
                let p = InverseParams {
                    sun_dir,
                    sun_intensity: cand.exp().clamp(1e-3, SUN_INTENSITY_MAX),
                    turbidity: tau,
                };
                let (l, _, _) = eval_mean(
                    device,
                    queue,
                    desc,
                    &pipes,
                    &scene,
                    &bufs,
                    &p,
                    &albedo,
                    EvalMode::LossOnly,
                    GRAD_REPLICATES,
                )?;
                if l.is_finite() && l < best_probe {
                    best_probe = l;
                    best_theta = cand;
                }
            }
            theta_i = best_theta;
            params.sun_intensity = theta_i.exp().clamp(SUN_INTENSITY_MIN, SUN_INTENSITY_MAX);
        }

        if it >= tail_from {
            for (a, s) in alb_avg.iter_mut().zip(log_alb.iter()) {
                *a += *s;
            }
            for k in 0..3 {
                dir_avg[k] += sun_dir[k];
            }
            theta_avg += theta_i;
            tau_avg += tau;
            tail_n += 1;
        }

        if stall >= desc.early_stop_patience.max(1) {
            break;
        }
    }

    // Tail-average candidate vs best-loss checkpoint — evaluate the average
    // once and keep whichever realizes the lower loss.
    if tail_n > 0 {
        let inv_n = 1.0 / tail_n as f32;
        let avg_albedo: Vec<f32> = alb_avg.iter().map(|v| (v * inv_n).exp()).collect();
        let dl = (dir_avg[0] * dir_avg[0] + dir_avg[1] * dir_avg[1] + dir_avg[2] * dir_avg[2])
            .sqrt()
            .max(1e-8);
        let avg_params = InverseParams {
            sun_dir: [dir_avg[0] / dl, dir_avg[1] / dl, dir_avg[2] / dl],
            sun_intensity: (theta_avg * inv_n)
                .exp()
                .clamp(SUN_INTENSITY_MIN, SUN_INTENSITY_MAX),
            turbidity: (tau_avg * inv_n).clamp(TURBIDITY_MIN, TURBIDITY_MAX),
        };
        let (avg_loss, _, _) = eval_mean(
            device,
            queue,
            desc,
            &pipes,
            &scene,
            &bufs,
            &avg_params,
            &avg_albedo,
            EvalMode::LossOnly,
            GRAD_REPLICATES,
        )?;
        if avg_loss.is_finite() && avg_loss <= best {
            albedo = avg_albedo;
            params = avg_params;
        } else if let Some(ba) = best_albedo {
            albedo = ba;
            params = best_params;
        }
    } else if let Some(ba) = best_albedo {
        albedo = ba;
        params = best_params;
    }

    // Recovered beauty: one primal render at the final parameters.
    eval_pass(
        device,
        queue,
        desc,
        &pipes,
        &scene,
        &bufs,
        &params,
        &albedo,
        EvalMode::Render,
        desc.seed,
    )?;
    let (rgba, _mean) = read_beauty(device, queue, &scene)?;

    // Solve-scoped peak host-visible bytes from the ledger capture: every
    // tracked allocation in this solve is owner-tagged. This includes
    // MAP_READ staging buffers and the certified-event CPU workspaces.
    let report = finish_ledger_capture();
    let peak = report.peak_host_visible_bytes;
    if peak > INVERSE_HOST_BUDGET {
        return Err(RenderError::Render(format!(
            "inverse solve exceeded the {INVERSE_HOST_BUDGET}-byte host-visible budget (peak {peak})"
        )));
    }

    Ok(InverseSolveOutput {
        albedo,
        sun_dir: params.sun_dir,
        sun_intensity: params.sun_intensity,
        turbidity: params.turbidity,
        loss_history,
        peak_host_visible_bytes: peak,
        iterations_run: iters_run,
        rgba,
    })
}

/// Primal-only render at explicit parameters — re-dispatches the same
/// forward chain the solver uses (identical to `hybrid_render_terrain_reference`
/// at matched frames/seed/spp). Returns (rgba u8, mean linear radiance rgb).
pub fn render_primal(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    desc: &InverseSolveDesc,
    params: &InverseParams,
    albedo: &[f32],
) -> Result<(Vec<u8>, Vec<f32>), RenderError> {
    validate_solve_desc(desc)?;
    let pipes = InversePipelines::new(device)?;
    let bufs = InverseBuffers::new(
        device,
        queue,
        desc.width,
        desc.height,
        (desc.dem_width, desc.dem_height),
        desc.frames.max(1).min(desc.tile_size.max(1)),
        &desc.target_rgba,
    )?;
    let scene = InverseSceneGpu::new(device, queue, desc, &pipes, &bufs)?;
    eval_pass(
        device,
        queue,
        desc,
        &pipes,
        &scene,
        &bufs,
        params,
        albedo,
        EvalMode::Render,
        desc.seed,
    )?;
    let (rgba, mean) = read_beauty(device, queue, &scene)?;
    Ok((rgba, mean.to_vec()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 4x4 flat DEM at height 0 (world y=0), 2 world units per texel.
    fn flat_desc() -> InverseSolveDesc {
        let dem_w = 4u32;
        let dem_h = 4u32;
        let heights = vec![0.0f32; (dem_w * dem_h) as usize];
        InverseSolveDesc {
            heights,
            dem_width: dem_w,
            dem_height: dem_h,
            spacing: (2.0, 2.0),
            exaggeration: 1.0,
            target_rgba: vec![0u8; 32 * 32 * 4],
            width: 32,
            height: 32,
            cam_origin: [0.0, 20.0, 0.0],
            cam_look_at: [0.0, 0.0, 0.0],
            cam_up: [0.0, 0.0, -1.0],
            fov_y_deg: 30.0,
            exposure: 1.0,
            init_albedo: vec![0.5f32; (dem_w * dem_h * 3) as usize],
            init: InverseParams {
                sun_dir: [0.5, 0.7, -0.5],
                sun_intensity: 2.5,
                turbidity: 2.0,
            },
            sun_color: [1.0, 0.97, 0.92],
            env_map: None,
            env_intensity: 0.35,
            iters: 1,
            spp: 4,
            frames: 2,
            tile_size: 8,
            seed: 7,
            lr_albedo: 0.02,
            lr_sun: 0.02,
            lr_turbidity: 0.02,
            early_stop_tol: 0.0,
            early_stop_patience: 8,
            spatial_reuse: false,
            edge_term: false,
            score_correction: false,
        }
    }

    fn gpu() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::default();
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))?;
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("inverse-test"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
            },
            None,
        ))
        .ok()?;
        Some((device, queue))
    }

    struct Fixture {
        device: wgpu::Device,
        queue: wgpu::Queue,
        desc: InverseSolveDesc,
        pipes: InversePipelines,
        bufs: InverseBuffers,
        scene: InverseSceneGpu,
    }

    /// Build a fixture and synthesize the target at `target_params` through
    /// the same forward chain the solve uses (the Rust-side FD tests measure
    /// the realized loss; the Python gate renders its target through the
    /// independent forward renderer instead).
    fn fixture(desc: &InverseSolveDesc, target_params: &InverseParams) -> Option<Fixture> {
        let (device, queue) = gpu()?;
        let pipes = InversePipelines::new(&device).expect("pipelines");
        let bufs = InverseBuffers::new(
            &device,
            &queue,
            desc.width,
            desc.height,
            (desc.dem_width, desc.dem_height),
            desc.frames.max(1).min(desc.tile_size.max(1)),
            &desc.target_rgba,
        )
        .expect("bufs");
        let scene = InverseSceneGpu::new(&device, &queue, desc, &pipes, &bufs).expect("scene");
        // Synthesize the observed beauty at the perturbed parameters, then
        // upload it as the target texture.
        eval_pass(
            &device,
            &queue,
            desc,
            &pipes,
            &scene,
            &bufs,
            target_params,
            &desc.init_albedo,
            EvalMode::Render,
            desc.seed,
        )
        .expect("target render");
        let (rgba, _) = read_beauty(&device, &queue, &scene).expect("read beauty");
        bufs.upload_target(&queue, desc.width, desc.height, &rgba);
        Some(Fixture {
            device,
            queue,
            desc: desc.clone(),
            pipes,
            bufs,
            scene,
        })
    }

    /// Re-evaluate a captured edge radiance through the ordinary loss entry
    /// point. This deliberately does not dispatch the edge kernel: it checks
    /// the live target texture, f32 accumulator input, and nonlinear loss
    /// pipeline independently of the event-replay entry point.
    fn independent_gpu_loss_from_linear(fx: &Fixture, pixel: u32, linear: [f32; 3]) -> f32 {
        assert!(pixel < fx.desc.width * fx.desc.height);
        fx.scene
            .upload_params(&fx.queue, &fx.desc, &fx.desc.init, &fx.desc.init_albedo);
        let accum = [linear[0], linear[1], linear[2], 1.0_f32];
        fx.queue.write_buffer(
            &fx.scene.accum,
            u64::from(pixel) * 16,
            bytemuck::bytes_of(&accum),
        );
        let mut enc = fx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("inverse-edge-independent-loss"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("inverse-edge-independent-loss"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&fx.pipes.loss);
            pass.set_bind_group(0, &fx.scene.g0_frames[0], &[]);
            pass.set_bind_group(1, &fx.scene.g1, &[]);
            pass.set_bind_group(2, &fx.scene.g2, &[]);
            pass.set_bind_group(3, &fx.scene.g3, &[]);
            pass.dispatch_workgroups(fx.desc.width.div_ceil(8), fx.desc.height.div_ceil(8), 1);
        }
        fx.queue.submit([enc.finish()]);
        let raw = read_buffer(
            &fx.device,
            &fx.queue,
            &fx.bufs.u32_buf,
            fx.bufs.loss_offset() + u64::from(pixel) * 4,
            4,
        )
        .expect("independent edge loss readback");
        f32::from_le_bytes(raw[..4].try_into().expect("single f32 loss"))
    }

    /// Cross-check the complete nonlinear loss through its ordinary GPU entry
    /// point, using the exact f32 radiances observed by certified edge
    /// replay.  This is intentionally distinct from interval validation: it
    /// independently executes `main_inverse_loss` against the live target
    /// texture for both sides of the IBL branch.
    fn assert_independent_gpu_losses_match_audit(
        fx: &Fixture,
        event: &geometry_cert::ProjectedEvent,
        audit: &geometry_cert::GpuAuditRecord,
    ) {
        let pixel = event.record.ids[0];
        let lit = independent_gpu_loss_from_linear(
            fx,
            pixel,
            [
                audit.lit_linear[0],
                audit.lit_linear[1],
                audit.lit_linear[2],
            ],
        );
        let shadow = independent_gpu_loss_from_linear(
            fx,
            pixel,
            [
                audit.shadow_linear[0],
                audit.shadow_linear[1],
                audit.shadow_linear[2],
            ],
        );
        assert_eq!(
            lit.to_bits(),
            audit.loss_jump[0].to_bits(),
            "independent loss disagrees with certified lit loss at pixel {pixel}"
        );
        assert_eq!(
            shadow.to_bits(),
            audit.loss_jump[1].to_bits(),
            "independent loss disagrees with certified shadow loss at pixel {pixel}"
        );
    }

    /// Finite-difference cross-check of the analytic gradient on a tiny
    /// scene — all four parameter channels (albedo texel, sun direction,
    /// intensity, turbidity) with the realized terms enabled (spatial reuse
    /// on, edge term on, score correction off — the score term is
    /// expectation-space and has no realized counterpart).
    #[test]
    fn inverse_gradient_matches_finite_difference() {
        let mut desc = flat_desc();
        desc.spatial_reuse = true;
        desc.edge_term = true;
        desc.score_correction = false;
        // Perturbed target so the residual is a smooth O(1) signal.
        let target_params = InverseParams {
            sun_dir: desc.init.sun_dir,
            sun_intensity: desc.init.sun_intensity + 0.6,
            turbidity: desc.init.turbidity + 0.4,
        };
        let Some(fx) = fixture(&desc, &target_params) else {
            return; // no GPU on this host
        };
        let (_, d_alb, scalars) = eval_pass(
            &fx.device,
            &fx.queue,
            &fx.desc,
            &fx.pipes,
            &fx.scene,
            &fx.bufs,
            &fx.desc.init,
            &fx.desc.init_albedo,
            EvalMode::Full,
            fx.desc.seed,
        )
        .expect("analytic");
        let loss_of = |p: &InverseParams, alb: &[f32]| -> f32 {
            eval_pass(
                &fx.device,
                &fx.queue,
                &fx.desc,
                &fx.pipes,
                &fx.scene,
                &fx.bufs,
                p,
                alb,
                EvalMode::LossOnly,
                fx.desc.seed,
            )
            .expect("eval")
            .0
        };

        let eps = 1e-3f32;
        let g_dir = project_dir_grad(fx.desc.init.sun_dir, [scalars[0], scalars[1], scalars[2]]);
        let check = |name: &str, analytic: f32, fd: f32| {
            assert!(fd.abs() > 1e-6, "{name}: FD too small to gate ({fd})");
            let tol = 0.05 * fd.abs() + 1e-6;
            assert!(
                (analytic - fd).abs() <= tol,
                "{name}: analytic {analytic} vs fd {fd} (tol {tol})"
            );
        };
        // Sun direction: FD each raw component — the parameter is normalized
        // on upload, so compare against the normalize-projected gradient.
        for comp in 0..3usize {
            let mut pp = fx.desc.init;
            let mut pm = fx.desc.init;
            pp.sun_dir[comp] += eps;
            pm.sun_dir[comp] -= eps;
            let fd = (loss_of(&pp, &fx.desc.init_albedo) - loss_of(&pm, &fx.desc.init_albedo))
                / (2.0 * eps);
            check(&format!("sun_dir[{comp}]"), g_dir[comp], fd);
        }
        // Intensity and turbidity scalars.
        for (name, analytic) in [("intensity", scalars[3]), ("turbidity", scalars[4])] {
            let mut pp = fx.desc.init;
            let mut pm = fx.desc.init;
            if name == "intensity" {
                pp.sun_intensity += eps;
                pm.sun_intensity -= eps;
            } else {
                pp.turbidity += eps;
                pm.turbidity -= eps;
            }
            let fd = (loss_of(&pp, &fx.desc.init_albedo) - loss_of(&pm, &fx.desc.init_albedo))
                / (2.0 * eps);
            check(name, analytic, fd);
        }

        // Albedo channel: perturb texel 0's red channel.
        let mut alb_p = fx.desc.init_albedo.clone();
        let mut alb_m = fx.desc.init_albedo.clone();
        alb_p[0] += eps;
        alb_m[0] -= eps;
        let fd = (loss_of(&fx.desc.init, &alb_p) - loss_of(&fx.desc.init, &alb_m)) / (2.0 * eps);
        check("albedo texel 0", d_alb[0], fd);
    }

    #[test]
    // Exercise the non-smooth perimeter, cutoff, and cell-entry classes with
    // both the certified replay and an independent forward branch crossing.
    // Exact nonlinear-loss equality is checked through `main_inverse_loss`;
    // finite differences remain a branch witness, never a proof by themselves.
    fn inverse_edge_term_exercises_boundary_classes_on_gpu() {
        let c = 3.0f32.sqrt() * 0.5;
        let cases: &[(&str, u32, u32, f32, &[f32], [f32; 3], u32, f32)] = &[
            (
                "perimeter x",
                3,
                2,
                1.0,
                &[0., 0., 1., 0., 0., 1.],
                [c, 0.5, 0.],
                4,
                0.1,
            ),
            // This is the same certified perimeter crossing, with the keyed
            // environment sample selected so the normal-offset IBL replay
            // enters the adjoining rising cell.  It supplies the occluded
            // branch witness for the exact loss oracle below.
            (
                "perimeter x IBL occluded",
                3,
                2,
                1.0,
                &[0., 0., 1., 0., 0., 1.],
                [c, 0.5, 0.],
                1,
                0.1,
            ),
            (
                "perimeter z",
                2,
                3,
                1.0,
                &[0., 0., 0., 0., 1., 1.],
                [0., 0.5, c],
                13,
                0.1,
            ),
            (
                "cutoff",
                7,
                2,
                0.0002,
                &[
                    0., 0., 0., 0., 0.004, 0.002, 0., 0., 0., 0., 0., 0.004, 0.002, 0.,
                ],
                [c, 0.5, 0.],
                4,
                0.0005,
            ),
            (
                "entry x",
                4,
                2,
                0.0006,
                &[0., 0., 0.004, 0.004, 0., 0., 0.004, 0.004],
                [c, 0.5, 0.],
                4,
                0.0005,
            ),
            (
                "entry z",
                2,
                4,
                0.0006,
                &[0., 0., 0., 0., 0.004, 0.004, 0.004, 0.004],
                [0., 0.5, c],
                13,
                0.0005,
            ),
        ];
        for &(name, dem_w, dem_h, spacing, heights, sun, seed, half_h) in cases {
            let mut desc = flat_desc();
            desc.dem_width = dem_w;
            desc.dem_height = dem_h;
            desc.heights = heights.to_vec();
            desc.init_albedo = vec![0.5; (dem_w * dem_h * 3) as usize];
            desc.spacing = (spacing, spacing);
            desc.width = 1;
            desc.height = 1;
            desc.target_rgba = vec![0; 4];
            let center_x = -(dem_w as f32 - 1.0) * spacing * 0.5;
            let center_z = -(dem_h as f32 - 1.0) * spacing * 0.5;
            desc.cam_origin = [center_x, 10.0, center_z];
            desc.cam_look_at = [center_x, 0.0, center_z];
            desc.cam_up = [0.0, 0.0, -1.0];
            desc.fov_y_deg = (2.0f32 * half_h.atan()).to_degrees();
            desc.init.sun_dir = sun;
            desc.env_intensity = if name == "perimeter x IBL occluded" {
                0.35
            } else {
                0.0
            };
            desc.frames = 1;
            desc.spp = 1;
            desc.seed = seed;
            desc.edge_term = true;
            desc.spatial_reuse = false;
            desc.score_correction = false;
            let Some(fx) = fixture(&desc, &desc.init) else {
                panic!("GPU required");
            };
            fx.bufs.upload_target(&fx.queue, 1, 1, &[0, 0, 0, 255]);
            let probe = prepare_edge_chunks_for_test(
                &fx.device,
                &fx.desc,
                &fx.pipes,
                &fx.scene,
                &fx.bufs,
                &fx.desc.init,
                1,
            )
            .expect("certificate probe");
            eprintln!(
                "{name}: edge chunks {:?}",
                probe.2.iter().map(|c| c.count).collect::<Vec<_>>()
            );
            let (_, _, with_edge, edge_evidence) = eval_pass_with_edge_evidence(
                &fx.device,
                &fx.queue,
                &fx.desc,
                &fx.pipes,
                &fx.scene,
                &fx.bufs,
                &fx.desc.init,
                &fx.desc.init_albedo,
                EvalMode::Full,
                fx.desc.seed,
            )
            .expect("certified edge evaluation");
            assert!(
                with_edge[5] > 0.0,
                "{name} event did not reach the GPU: {with_edge:?}"
            );
            let edge_evidence = edge_evidence.expect("certified edge audit");
            assert_eq!(
                edge_evidence.audited_events.len(),
                edge_evidence.audited_records.len(),
                "{name}: audited event/audit pairing drifted"
            );
            for (event, audit) in edge_evidence
                .audited_events
                .iter()
                .zip(&edge_evidence.audited_records)
            {
                assert_independent_gpu_losses_match_audit(&fx, event, audit);
            }
            if name == "perimeter x IBL occluded" {
                assert!(
                    edge_evidence
                        .audited_records
                        .iter()
                        .any(|audit| audit.ibl_ray[3].to_bits() == 1.0_f32.to_bits()),
                    "{name}: certified replay did not exercise the IBL-occluded branch"
                );
            }
            assert!(
                edge_evidence.audited_events.iter().any(|event| match name {
                    "perimeter x" | "perimeter x IBL occluded" => {
                        event.class == geometry_cert::EventClass::PerimeterX
                    }
                    "perimeter z" => event.class == geometry_cert::EventClass::PerimeterZ,
                    "cutoff" => event.class == geometry_cert::EventClass::Cutoff,
                    "entry x" => event.class == geometry_cert::EventClass::CellEntryX,
                    "entry z" => event.class == geometry_cert::EventClass::CellEntryZ,
                    _ => unreachable!(),
                }),
                "{name}: actual certified replay omitted requested event class"
            );
            let component = if sun[0] != 0.0 { 0 } else { 2 };
            let mut saw_requested_class = false;
            let mut saw_requested_forward_flip = false;
            for event in probe.1.iter() {
                let requested_class = match name {
                    "perimeter x" | "perimeter x IBL occluded" => {
                        event.class == geometry_cert::EventClass::PerimeterX
                    }
                    "perimeter z" => event.class == geometry_cert::EventClass::PerimeterZ,
                    "cutoff" => event.class == geometry_cert::EventClass::Cutoff,
                    "entry x" => event.class == geometry_cert::EventClass::CellEntryX,
                    "entry z" => event.class == geometry_cert::EventClass::CellEntryZ,
                    _ => unreachable!(),
                };
                saw_requested_class |= requested_class;
                let p = event.record.receiver;
                let mut oracle_desc = fx.desc.clone();
                oracle_desc.edge_term = false;
                oracle_desc.cam_origin = [p[0], p[1] + 10.0, p[2]];
                oracle_desc.cam_look_at = [p[0], p[1], p[2]];
                oracle_desc.fov_y_deg = 1e-6;
                oracle_desc.seed = 127;
                let oracle_fx = fixture(&oracle_desc, &oracle_desc.init).expect("oracle GPU");
                oracle_fx
                    .bufs
                    .upload_target(&oracle_fx.queue, 1, 1, &[0, 0, 0, 255]);
                for step in [0.01, 0.005] {
                    let mut plus = oracle_desc.init;
                    let mut minus = oracle_desc.init;
                    plus.sun_dir[component] += step;
                    minus.sun_dir[component] -= step;
                    let lp = eval_pass(
                        &oracle_fx.device,
                        &oracle_fx.queue,
                        &oracle_fx.desc,
                        &oracle_fx.pipes,
                        &oracle_fx.scene,
                        &oracle_fx.bufs,
                        &plus,
                        &oracle_fx.desc.init_albedo,
                        EvalMode::LossOnly,
                        oracle_fx.desc.seed,
                    )
                    .expect("one-sided plus")
                    .0;
                    let lm = eval_pass(
                        &oracle_fx.device,
                        &oracle_fx.queue,
                        &oracle_fx.desc,
                        &oracle_fx.pipes,
                        &oracle_fx.scene,
                        &oracle_fx.bufs,
                        &minus,
                        &oracle_fx.desc.init_albedo,
                        EvalMode::LossOnly,
                        oracle_fx.desc.seed,
                    )
                    .expect("one-sided minus")
                    .0;
                    let jump = f64::from((lp - lm).abs());
                    assert!(jump.is_finite());
                    if requested_class {
                        assert!(
                            jump > 0.0,
                            "{name} boundary did not flip in independent forward GPU evaluation"
                        );
                        saw_requested_forward_flip = true;
                    }
                }
            }
            assert!(saw_requested_class, "{name} event class absent");
            assert!(
                saw_requested_forward_flip,
                "{name} requested independent forward boundary witness was absent"
            );
        }
    }

    /// Independent GPU witness for the non-fallback environment source path.
    /// The event is deliberately IBL-occluded: `environment_effective` is
    /// captured before the visibility multiplier, so this proves that the
    /// host checks the map's actual direction/texel chain instead of merely
    /// accepting a zero indirect radiance.
    #[test]
    fn inverse_edge_term_replays_nonuniform_environment_map_on_gpu() {
        let c = 3.0f32.sqrt() * 0.5;
        let mut desc = flat_desc();
        desc.dem_width = 3;
        desc.dem_height = 2;
        desc.heights = vec![0., 0., 1., 0., 0., 1.];
        desc.init_albedo = vec![0.5; 3 * 2 * 3];
        desc.spacing = (1.0, 1.0);
        desc.width = 1;
        desc.height = 1;
        desc.target_rgba = vec![0; 4];
        desc.cam_origin = [-1.0, 10.0, -0.5];
        desc.cam_look_at = [-1.0, 0.0, -0.5];
        desc.cam_up = [0.0, 0.0, -1.0];
        desc.fov_y_deg = (2.0f32 * 0.1f32.atan()).to_degrees();
        desc.init.sun_dir = [c, 0.5, 0.0];
        desc.env_intensity = 0.35;
        desc.env_map = Some((
            vec![
                0.15, 0.05, 0.30, // (0, 0)
                0.40, 0.12, 0.08, // (1, 0)
                0.05, 0.35, 0.16, // (0, 1)
                0.22, 0.44, 0.10, // (1, 1)
            ],
            2,
            2,
        ));
        desc.frames = 1;
        desc.spp = 1;
        desc.seed = 1;
        desc.edge_term = true;
        desc.spatial_reuse = false;
        desc.score_correction = false;
        let Some(fx) = fixture(&desc, &desc.init) else {
            panic!("GPU required");
        };
        fx.bufs.upload_target(&fx.queue, 1, 1, &[0, 0, 0, 255]);
        let (_, _, with_edge, evidence) = eval_pass_with_edge_evidence(
            &fx.device,
            &fx.queue,
            &fx.desc,
            &fx.pipes,
            &fx.scene,
            &fx.bufs,
            &fx.desc.init,
            &fx.desc.init_albedo,
            EvalMode::Full,
            fx.desc.seed,
        )
        .expect("mapped-environment certified edge evaluation");
        assert!(
            with_edge[5] > 0.0,
            "mapped environment event did not reach GPU"
        );
        let evidence = evidence.expect("mapped environment edge audit");
        assert!(
            evidence.audited_records.iter().any(|audit| {
                audit.environment_lookup[2..] == [2, 2]
                    && audit.environment_effective[..3]
                        .iter()
                        .any(|value| *value > 0.0)
            }),
            "mapped environment replay omitted the actual angular map source: {:?}",
            evidence.audited_records
        );
    }

    #[test]
    fn inverse_gaussian_smooth_and_crease_gpu_probe() {
        let mut desc = flat_desc();
        desc.dem_width = 16;
        desc.dem_height = 16;
        desc.spacing = (32.0 / 15.0, 32.0 / 15.0);
        desc.heights.clear();
        for z in 0..16 {
            for x in 0..16 {
                let xx = -1.0 + 2.0 * x as f64 / 15.0;
                let zz = -1.0 + 2.0 * z as f64 / 15.0;
                let h = 8.0 * (-((xx - 0.35).powi(2) + (zz + 0.2).powi(2)) / 0.08).exp()
                    + 6.0 * (-((xx + 0.45).powi(2) + (zz - 0.4).powi(2)) / 0.06).exp()
                    + 4.0 * (-((xx + 0.1).powi(2) + (zz + 0.55).powi(2)) / 0.05).exp()
                    + 1.2 * (0.5 + 0.5 * xx);
                desc.heights.push(h as f32);
            }
        }
        desc.init_albedo = vec![0.5; 16 * 16 * 3];
        desc.width = 96;
        desc.height = 96;
        desc.target_rgba = vec![0; 96 * 96 * 4];
        desc.cam_up = [0.0, 0.0, -1.0];
        desc.fov_y_deg = 38.0;
        desc.cam_origin = [0.0, 42.0, 0.001];
        desc.cam_look_at = [0.0, 0.0, 0.001];
        let az = 215.0f32.to_radians();
        let el = 30.0f32.to_radians();
        desc.init.sun_dir = [az.cos() * el.cos(), el.sin(), az.sin() * el.cos()];
        // Keep IBL live: this is the independent GPU witness gate for both
        // the clear and terrain-occluded normal-offset visibility branches.
        desc.env_intensity = 0.35;
        desc.frames = 1;
        desc.spp = 1;
        desc.edge_term = true;
        desc.spatial_reuse = false;
        desc.score_correction = false;
        let mut saw_ibl_clear = false;
        for (seed, class) in [
            (15268445u32, geometry_cert::EventClass::SmoothGrazing),
            (20121081u32, geometry_cert::EventClass::CreaseX),
        ] {
            desc.seed = seed;
            let fx = fixture(&desc, &desc.init).expect("GPU required");
            let probe = prepare_edge_chunks_for_test(
                &fx.device,
                &fx.desc,
                &fx.pipes,
                &fx.scene,
                &fx.bufs,
                &fx.desc.init,
                1,
            );
            let (_, projected, _) = probe.expect("Gaussian certificate");
            assert!(projected.iter().any(|e| e.class == class));
            let mut black_target = vec![0u8; (desc.width * desc.height * 4) as usize];
            for rgba in black_target.chunks_exact_mut(4) {
                rgba[3] = 255;
            }
            fx.bufs
                .upload_target(&fx.queue, desc.width, desc.height, &black_target);
            let (_, _, _with_edge, edge_evidence) = eval_pass_with_edge_evidence(
                &fx.device,
                &fx.queue,
                &fx.desc,
                &fx.pipes,
                &fx.scene,
                &fx.bufs,
                &fx.desc.init,
                &fx.desc.init_albedo,
                EvalMode::Full,
                fx.desc.seed,
            )
            .expect("Gaussian full gradient");
            let edge_evidence = edge_evidence.expect("Gaussian edge witness");
            assert!(
                edge_evidence.event_count > 0
                    && edge_evidence.event_count == edge_evidence.accepted_camera_witness_count,
                "Gaussian {seed}: every GPU event must carry an accepted camera witness: {edge_evidence:?}"
            );
            assert_eq!(
                edge_evidence.event_count,
                edge_evidence.ibl_clear_count + edge_evidence.ibl_occluded_count,
                "Gaussian {seed}: every GPU event must resolve exactly one IBL branch: {edge_evidence:?}"
            );
            assert!(
                edge_evidence.max_camera_jitter_projection_error.is_finite()
                    && edge_evidence.max_upload_weight_error.is_finite()
                    && edge_evidence.max_ibl_direction_source_width.is_finite()
                    && edge_evidence.max_source_radiance_width.is_finite()
                    && edge_evidence.max_radiance_reconstruction_width.is_finite()
                    && edge_evidence.max_loss_interval_width.is_finite()
                    && edge_evidence
                        .max_accumulation_rounding_bound
                        .iter()
                        .copied()
                        .all(f64::is_finite)
                    && edge_evidence
                        .max_atomic_sum_abs
                        .iter()
                        .copied()
                        .all(f64::is_finite)
                    && edge_evidence
                        .host_chunk_add_rounding_bound
                        .iter()
                        .copied()
                        .all(f64::is_finite)
                    && edge_evidence
                        .final_scalar_add_rounding_bound
                        .iter()
                        .copied()
                        .all(f64::is_finite)
                    && edge_evidence
                        .final_scalar_scale_rounding_bound
                        .iter()
                        .copied()
                        .all(f64::is_finite),
                "Gaussian {seed}: witness numerical interval is not finite: {edge_evidence:?}"
            );
            assert_eq!(
                edge_evidence.audited_events.len(),
                edge_evidence.audited_records.len(),
                "Gaussian {seed}: audited event/audit pairing drifted"
            );
            assert_eq!(
                edge_evidence.event_count,
                edge_evidence.audited_records.len(),
                "Gaussian {seed}: every accepted event must retain its exact audit"
            );
            let mut saw_requested_audit = false;
            for (event, audit) in edge_evidence
                .audited_events
                .iter()
                .zip(&edge_evidence.audited_records)
            {
                assert_independent_gpu_losses_match_audit(&fx, event, audit);
                saw_requested_audit |= event.class == class;
                saw_ibl_clear |= audit.ibl_ray[3].to_bits() == 0.0_f32.to_bits();
            }
            assert!(
                saw_requested_audit,
                "Gaussian {seed}: actual audited replay omitted requested {class:?} event"
            );

            // The edge replay intentionally uses an independent keyed IBL
            // draw, while a stand-alone forward fixture has its own beauty
            // stream. The live run above therefore proves the nonlinear-loss
            // value through a second GPU entry point. Keep this no-IBL
            // fixture as a separate forward boundary-crossing witness; it
            // must not be used as a magnitude oracle for another baseline.
            let mut numeric_desc = fx.desc.clone();
            numeric_desc.env_intensity = 0.0;
            let numeric_fx = fixture(&numeric_desc, &numeric_desc.init)
                .expect("Gaussian no-IBL oracle fixture GPU");
            numeric_fx.bufs.upload_target(
                &numeric_fx.queue,
                numeric_desc.width,
                numeric_desc.height,
                &black_target,
            );
            let (_, numeric_projected, _) = prepare_edge_chunks_for_test(
                &numeric_fx.device,
                &numeric_fx.desc,
                &numeric_fx.pipes,
                &numeric_fx.scene,
                &numeric_fx.bufs,
                &numeric_fx.desc.init,
                1,
            )
            .expect("Gaussian no-IBL certificate");
            assert!(
                numeric_projected.iter().any(|event| event.class == class),
                "Gaussian {seed}: no-IBL certificate omitted {class:?}"
            );
            let (_, _, _numeric_with_edge, numeric_evidence) = eval_pass_with_edge_evidence(
                &numeric_fx.device,
                &numeric_fx.queue,
                &numeric_fx.desc,
                &numeric_fx.pipes,
                &numeric_fx.scene,
                &numeric_fx.bufs,
                &numeric_fx.desc.init,
                &numeric_fx.desc.init_albedo,
                EvalMode::Full,
                numeric_fx.desc.seed,
            )
            .expect("Gaussian no-IBL full gradient");
            let numeric_evidence = numeric_evidence.expect("Gaussian no-IBL edge witness");
            assert_eq!(
                numeric_evidence.event_count, numeric_evidence.accepted_camera_witness_count,
                "Gaussian {seed}: no-IBL event failed the camera witness: {numeric_evidence:?}"
            );
            let mut oracle_unit = [[0.0f64; 3]; 3];
            let mut saw_forward_flip = false;
            for event in numeric_projected.iter() {
                let p = event.record.receiver;
                let mut oracle_desc = numeric_fx.desc.clone();
                oracle_desc.width = 1;
                oracle_desc.height = 1;
                oracle_desc.target_rgba = vec![0, 0, 0, 255];
                oracle_desc.edge_term = false;
                oracle_desc.cam_origin = [p[0], p[1] + 10.0, p[2]];
                oracle_desc.cam_look_at = [p[0], p[1], p[2]];
                oracle_desc.fov_y_deg = 1e-6;
                oracle_desc.seed = 127;
                let oracle_fx =
                    fixture(&oracle_desc, &oracle_desc.init).expect("Gaussian oracle GPU");
                oracle_fx
                    .bufs
                    .upload_target(&oracle_fx.queue, 1, 1, &[0, 0, 0, 255]);
                for (slot, step) in [0.01, 0.005, 0.0025].into_iter().enumerate() {
                    let mut plus = oracle_fx.desc.init;
                    let mut minus = oracle_fx.desc.init;
                    plus.sun_dir[0] += step;
                    minus.sun_dir[0] -= step;
                    let lp = eval_pass(
                        &oracle_fx.device,
                        &oracle_fx.queue,
                        &oracle_fx.desc,
                        &oracle_fx.pipes,
                        &oracle_fx.scene,
                        &oracle_fx.bufs,
                        &plus,
                        &oracle_fx.desc.init_albedo,
                        EvalMode::LossOnly,
                        oracle_fx.desc.seed,
                    )
                    .expect("Gaussian plus")
                    .0;
                    let lm = eval_pass(
                        &oracle_fx.device,
                        &oracle_fx.queue,
                        &oracle_fx.desc,
                        &oracle_fx.pipes,
                        &oracle_fx.scene,
                        &oracle_fx.bufs,
                        &minus,
                        &oracle_fx.desc.init_albedo,
                        EvalMode::LossOnly,
                        oracle_fx.desc.seed,
                    )
                    .expect("Gaussian minus")
                    .0;
                    let jump = f64::from((lp - lm).abs());
                    assert!(
                        jump.is_finite() && jump > 0.0,
                        "Gaussian {class:?}: independent forward GPU boundary did not flip"
                    );
                    saw_forward_flip = true;
                    for k in 0..3 {
                        oracle_unit[slot][k] += f64::from(event.record.grad_weight[k]) * jump
                            / f64::from(numeric_desc.width * numeric_desc.height);
                    }
                }
            }
            let oracle = oracle_unit.map(|unit| {
                f64::from(
                    project_dir_grad(numeric_fx.desc.init.sun_dir, unit.map(|value| value as f32))
                        [0],
                )
            });
            assert!(saw_forward_flip);
            assert!(
                oracle.iter().copied().all(f64::is_finite),
                "Gaussian {class:?}: forward boundary witness is non-finite: {oracle:?}"
            );
            eprintln!(
                "Gaussian {seed} {class:?}: independent forward boundary witnesses {:?}; exact nonlinear-loss equality was checked through main_inverse_loss",
                oracle
            );
        }
        assert!(
            saw_ibl_clear,
            "GPU witness did not exercise an IBL-clear event"
        );
    }

    #[test]
    fn inverse_reservoir_boundary_selection_probe() {
        let c = 3.0f32.sqrt() * 0.5;
        let mut desc = flat_desc();
        desc.dem_width = 3;
        desc.dem_height = 2;
        desc.heights = vec![0., 0., 1., 0., 0., 1.];
        desc.init_albedo = vec![0.5; 3 * 2 * 3];
        desc.spacing = (1.0, 1.0);
        desc.width = 1;
        desc.height = 1;
        desc.target_rgba = vec![0, 0, 0, 255];
        desc.cam_origin = [-1.0, 10.0, -0.5];
        desc.cam_look_at = [-1.0, 0.0, -0.5];
        desc.cam_up = [0.0, 0.0, -1.0];
        desc.fov_y_deg = (2.0f32 * 0.1f32.atan()).to_degrees();
        desc.init.sun_dir = [c, 0.5, 0.0];
        desc.env_intensity = 0.0;
        desc.frames = 2;
        desc.spp = 1;
        desc.edge_term = true;
        let mut selected_channels = [false; 3];
        for seed in [4, 9, 18] {
            desc.seed = seed;
            let fx = fixture(&desc, &desc.init).expect("GPU required");
            fx.bufs.upload_target(&fx.queue, 1, 1, &[0, 0, 0, 255]);
            let probe = prepare_edge_chunks_for_test(
                &fx.device,
                &fx.desc,
                &fx.pipes,
                &fx.scene,
                &fx.bufs,
                &fx.desc.init,
                1,
            );
            eprintln!(
                "reservoir seed {seed}: certificate {:?}",
                probe
                    .as_ref()
                    .map(|(_, _, c)| c.iter().map(|v| v.count).collect::<Vec<_>>())
            );
            let (_, projected, _) = probe.expect("reservoir certificate");
            let event = projected
                .iter()
                .find(|e| e.class == geometry_cert::EventClass::PerimeterX)
                .expect("perimeter event with reservoir");
            assert_eq!(event.record.ids[1], 1, "second frame must be selected");
            let (_, _, scalars) = eval_pass(
                &fx.device,
                &fx.queue,
                &fx.desc,
                &fx.pipes,
                &fx.scene,
                &fx.bufs,
                &fx.desc.init,
                &fx.desc.init_albedo,
                EvalMode::Full,
                fx.desc.seed,
            )
            .expect("reservoir edge");
            let mut no_edge = fx.desc.clone();
            no_edge.edge_term = false;
            let (_, _, smooth) = eval_pass(
                &fx.device,
                &fx.queue,
                &no_edge,
                &fx.pipes,
                &fx.scene,
                &fx.bufs,
                &fx.desc.init,
                &fx.desc.init_albedo,
                EvalMode::Full,
                fx.desc.seed,
            )
            .expect("reservoir smooth");
            let gpu_edge = scalars[0] - smooth[0];
            let wh = read_buffer(&fx.device, &fx.queue, &fx.bufs.v4_buf, 32, 16)
                .expect("reservoir history");
            let channel = f32::from_le_bytes(wh[4..8].try_into().unwrap()) as usize;
            assert!(channel < 3, "selected reservoir has no spectral channel");
            selected_channels[channel] = true;
            eprintln!(
                "reservoir seed {seed}: weight {}, channel {}, edge mass {}",
                f32::from_le_bytes(wh[0..4].try_into().unwrap()),
                f32::from_le_bytes(wh[4..8].try_into().unwrap()),
                scalars[5]
            );

            let hash = |mut x: u32| {
                x = (x ^ (x >> 16)).wrapping_mul(0x7feb_352d);
                x = (x ^ (x >> 15)).wrapping_mul(0x846c_a68b);
                x ^ (x >> 16)
            };
            let seed_lo = seed.wrapping_mul(0x9e37_79b9).wrapping_add(0x85eb_ca6b);
            let mut state = hash(seed ^ seed_lo ^ hash(0) ^ hash(2));
            let next = |st: &mut u32| {
                *st ^= *st << 13;
                *st ^= *st >> 17;
                *st ^= *st << 5;
                *st as f32 / 4294967296.0
            };
            let tent = |u: f32| {
                if u < 0.5 {
                    (2.0 * u).sqrt() - 1.0
                } else {
                    1.0 - (2.0 * (1.0 - u)).sqrt()
                }
            };
            let jx = tent(next(&mut state)) * 0.5;
            let jy = tent(next(&mut state)) * 0.5;
            let p = event.record.receiver;
            eprintln!(
                "reservoir seed {seed}: event receiver {:?}, gradient weight {}",
                p, event.record.grad_weight[0]
            );
            let mut oracle_desc = fx.desc.clone();
            oracle_desc.edge_term = false;
            oracle_desc.fov_y_deg = (2.0f32 * 0.1f32.atan()).to_degrees();
            oracle_desc.cam_origin = [p[0] - 2.0 * jx, p[1] + 10.0, p[2] - 2.0 * jy];
            oracle_desc.cam_look_at = [oracle_desc.cam_origin[0], p[1], oracle_desc.cam_origin[2]];
            // Independent forward GPU oracle: retain the real frame-0
            // radiance and selected reservoir, then move only the frame-1
            // camera ray to the certified event. This keeps the nonlinear
            // companion sample and spectral selection identical to replay.
            fx.scene
                .upload_params(&fx.queue, &fx.desc, &fx.desc.init, &fx.desc.init_albedo);
            fx.queue.write_buffer(
                &fx.scene.frame_ubos[0],
                0,
                bytemuck::bytes_of(&frame_uniforms(&fx.desc, 0, seed)),
            );
            let mut enc = fx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("inverse-boundary-oracle-frame0"),
                });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("inverse-boundary-oracle-clear"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&fx.pipes.clear);
                pass.set_bind_group(0, &fx.scene.g0_frames[0], &[]);
                pass.set_bind_group(1, &fx.scene.g1, &[]);
                pass.set_bind_group(2, &fx.scene.g2, &[]);
                pass.set_bind_group(3, &fx.scene.g3, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("inverse-boundary-oracle-terrain0"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&fx.pipes.terrain);
                pass.set_bind_group(0, &fx.scene.g0_frames[0], &[]);
                pass.set_bind_group(1, &fx.scene.g1, &[]);
                pass.set_bind_group(2, &fx.scene.g2, &[]);
                pass.set_bind_group(3, &fx.scene.g3, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("inverse-boundary-oracle-temporal0"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&fx.pipes.restir_temporal);
                pass.set_bind_group(0, &fx.scene.restir_g0_frames[0], &[]);
                pass.set_bind_group(1, &fx.scene.restir_empty_g1, &[]);
                pass.set_bind_group(2, &fx.scene.restir_temporal_g2, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            enc.copy_buffer_to_buffer(
                &fx.scene.res_out,
                0,
                &fx.scene.res_prev,
                0,
                fx.scene.reservoir_bytes,
            );
            fx.queue.submit([enc.finish()]);
            let frame0 = read_buffer(&fx.device, &fx.queue, &fx.scene.accum, 0, 16)
                .expect("oracle frame-0 radiance");
            let selected = read_buffer(
                &fx.device,
                &fx.queue,
                &fx.scene.res_prev,
                0,
                fx.scene.reservoir_bytes,
            )
            .expect("oracle selected reservoir");
            let original_reservoir: Reservoir = bytemuck::pod_read_unaligned(&selected);
            assert_eq!(original_reservoir.sample.light_index as usize, channel);
            assert_eq!(
                original_reservoir.weight,
                f32::from_le_bytes(wh[0..4].try_into().unwrap())
            );
            fx.queue.write_buffer(
                &fx.scene.frame_ubos[1],
                0,
                bytemuck::bytes_of(&frame_uniforms(&oracle_desc, 1, seed)),
            );

            let mut oracle = [0.0f64; 3];
            for (i, step) in [0.01, 0.005, 0.0025].into_iter().enumerate() {
                let mut sides = [0.0f32; 2];
                for (side, sign) in [-1.0f32, 1.0].into_iter().enumerate() {
                    let mut perturbed = fx.desc.init;
                    perturbed.sun_dir[0] += sign * step;
                    let len = perturbed.sun_dir.iter().map(|x| x * x).sum::<f32>().sqrt();
                    let dir = perturbed.sun_dir.map(|x| x / len);
                    let mut reservoir = original_reservoir;
                    reservoir.sample.direction = dir;
                    fx.scene
                        .upload_params(&fx.queue, &fx.desc, &perturbed, &fx.desc.init_albedo);
                    fx.queue
                        .write_buffer(&fx.scene.res_prev, 0, bytemuck::bytes_of(&reservoir));
                    fx.queue.write_buffer(&fx.scene.accum, 0, &frame0);
                    let mut enc =
                        fx.device
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("inverse-boundary-oracle-side"),
                            });
                    {
                        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("inverse-boundary-oracle-terrain1"),
                            timestamp_writes: None,
                        });
                        pass.set_pipeline(&fx.pipes.terrain);
                        pass.set_bind_group(0, &fx.scene.g0_frames[1], &[]);
                        pass.set_bind_group(1, &fx.scene.g1, &[]);
                        pass.set_bind_group(2, &fx.scene.g2, &[]);
                        pass.set_bind_group(3, &fx.scene.g3, &[]);
                        pass.dispatch_workgroups(1, 1, 1);
                    }
                    {
                        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("inverse-boundary-oracle-loss"),
                            timestamp_writes: None,
                        });
                        pass.set_pipeline(&fx.pipes.loss);
                        pass.set_bind_group(0, &fx.scene.g0_frames[0], &[]);
                        pass.set_bind_group(1, &fx.scene.g1, &[]);
                        pass.set_bind_group(2, &fx.scene.g2, &[]);
                        pass.set_bind_group(3, &fx.scene.g3, &[]);
                        pass.dispatch_workgroups(1, 1, 1);
                    }
                    fx.queue.submit([enc.finish()]);
                    let raw = read_buffer(
                        &fx.device,
                        &fx.queue,
                        &fx.bufs.u32_buf,
                        fx.bufs.loss_offset(),
                        4,
                    )
                    .expect("oracle loss");
                    sides[side] = f32::from_le_bytes(raw[..4].try_into().unwrap());
                }
                let jump = f64::from((sides[0] - sides[1]).abs());
                assert!(jump > 0.0, "selected reservoir event did not flip");
                oracle[i] = f64::from(event.record.grad_weight[0]) * jump;
                eprintln!(
                    "reservoir seed {seed}, step {step}: frozen-history forward sides {:?}",
                    sides
                );
            }
            let extrapolated = 2.0 * oracle[2] - oracle[1];
            let coarse = 2.0 * oracle[1] - oracle[0];
            let error = (f64::from(gpu_edge) - extrapolated).abs();
            let allowance =
                (extrapolated - coarse).abs() + 32.0 * f64::from(f32::EPSILON) * extrapolated.abs();
            eprintln!("reservoir seed {seed}, channel {channel}: GPU edge {gpu_edge}, oracle {:?}, error {error}, allowance {allowance}", oracle);
            assert!(error <= allowance, "selected reservoir boundary gradient disagrees with frozen-history forward GPU limit");
        }
        assert!(
            selected_channels.into_iter().all(|v| v),
            "missing spectral reservoir selection"
        );
    }

    /// A finite categorical expectation with a nonlinear loss requires the
    /// whole-loss likelihood score in addition to the pathwise 1/q term.
    /// Binary-rational inputs keep this identity exact in f64, so the check
    /// does not depend on Monte Carlo sampling or an arbitrary FD tolerance.
    #[test]
    fn inverse_nonlinear_score_identity() {
        let q: [f64; 3] = [0.25, 0.25, 0.5];
        let dq: [f64; 3] = [0.125, -0.0625, -0.0625];
        let residual: [f64; 3] = [0.25, 0.5, 1.0];
        let mut pathwise = 0.0;
        let mut score = 0.0;
        let mut exact = 0.0;
        for c in 0..3 {
            let y = residual[c] / q[c];
            let dy = -residual[c] * dq[c] / (q[c] * q[c]);
            let loss = y * y;
            pathwise += q[c] * 2.0 * y * dy;
            score += q[c] * loss * dq[c] / q[c];
            exact -= residual[c] * residual[c] * dq[c] / (q[c] * q[c]);
        }
        assert_ne!(pathwise, exact);
        assert_eq!(pathwise + score, exact);
    }

    /// The score-function correction completes the categorical spectral
    /// reservoir estimator in expectation. This test exercises that the term
    /// is live: gradients with and without it must differ and stay finite.
    #[test]
    fn inverse_score_correction_is_exercised() {
        let mut desc = flat_desc();
        for (i, h) in desc.heights.iter_mut().enumerate() {
            let x = (i % desc.dem_width as usize) as f32 - 1.5;
            let y = (i / desc.dem_width as usize) as f32 - 1.5;
            *h = 2.5 * (-(x * x + y * y) / 0.8).exp();
        }
        desc.spatial_reuse = true;
        desc.edge_term = false;
        let target_params = InverseParams {
            sun_dir: desc.init.sun_dir,
            sun_intensity: desc.init.sun_intensity + 0.6,
            turbidity: desc.init.turbidity + 0.4,
        };
        let Some(fx) = fixture(&desc, &target_params) else {
            return;
        };
        let eval_scalars = |score: bool| -> Vec<f32> {
            let mut d = fx.desc.clone();
            d.score_correction = score;
            eval_pass(
                &fx.device,
                &fx.queue,
                &d,
                &fx.pipes,
                &fx.scene,
                &fx.bufs,
                &fx.desc.init,
                &fx.desc.init_albedo,
                EvalMode::Full,
                d.seed,
            )
            .expect("eval")
            .2
        };
        let on = eval_scalars(true);
        let off = eval_scalars(false);
        assert!(on.iter().all(|v| v.is_finite()));
        assert!(off.iter().all(|v| v.is_finite()));
        let differs = on[..5]
            .iter()
            .zip(off[..5].iter())
            .any(|(a, b)| (a - b).abs() > 0.0);
        assert!(
            differs,
            "score correction produced identical gradients — term not exercised"
        );
    }

    /// Selected non-flat GPU finite-difference smoke for the pathwise,
    /// visibility, and discrete-selection terms. Eight sampled seeds cannot
    /// prove an expectation-space unbiasedness claim; the separate source
    /// contract and boundary completeness evidence are required for that.
    #[test]
    fn inverse_gradient_matches_fd_nonflat_all_terms() {
        let mut desc = flat_desc();
        // Gaussian bump — varying normals exercise the direct and boundary
        // channels while the spectral reservoir exercises through-W and
        // score-function terms.
        for (i, h) in desc.heights.iter_mut().enumerate() {
            let x = (i % desc.dem_width as usize) as f32 - 1.5;
            let y = (i / desc.dem_width as usize) as f32 - 1.5;
            *h = 2.5 * (-(x * x + y * y) / 0.8).exp();
        }
        desc.spatial_reuse = true;
        desc.edge_term = true;
        desc.spp = 16; // denser sampling tightens the FD staircase noise
        let target_params = InverseParams {
            sun_dir: desc.init.sun_dir,
            sun_intensity: desc.init.sun_intensity + 0.6,
            turbidity: desc.init.turbidity + 0.4,
        };
        let eps = 1e-3f32;
        let n_seeds = 8u32;

        // Multi-seed mean analytic gradients with the score correction on.
        let mut an_on = [0.0f32; 3];
        let mut fd = [0.0f32; 3];
        let mut ran = false;
        for s in 0..n_seeds {
            let mut d = desc.clone();
            d.seed = desc.seed.wrapping_add(0x9e3779b9u32.wrapping_mul(s + 1));
            let Some(fx_s) = fixture(&d, &target_params) else {
                if !ran {
                    return; // no GPU on this host
                }
                break;
            };
            ran = true;
            for (k, ch) in [0usize, 3usize, 4usize].iter().enumerate() {
                let mut pp = fx_s.desc.init;
                let mut pm = fx_s.desc.init;
                match *ch {
                    0 => {
                        pp.sun_dir[0] += eps;
                        pm.sun_dir[0] -= eps;
                    }
                    3 => {
                        pp.sun_intensity += eps;
                        pm.sun_intensity -= eps;
                    }
                    _ => {
                        pp.turbidity += eps;
                        pm.turbidity -= eps;
                    }
                }
                let lp = eval_pass(
                    &fx_s.device,
                    &fx_s.queue,
                    &fx_s.desc,
                    &fx_s.pipes,
                    &fx_s.scene,
                    &fx_s.bufs,
                    &pp,
                    &fx_s.desc.init_albedo,
                    EvalMode::LossOnly,
                    fx_s.desc.seed,
                )
                .expect("eval+")
                .0;
                let lm = eval_pass(
                    &fx_s.device,
                    &fx_s.queue,
                    &fx_s.desc,
                    &fx_s.pipes,
                    &fx_s.scene,
                    &fx_s.bufs,
                    &pm,
                    &fx_s.desc.init_albedo,
                    EvalMode::LossOnly,
                    fx_s.desc.seed,
                )
                .expect("eval-")
                .0;
                fd[k] += (lp - lm) / (2.0 * eps);
            }
            let mut ds = fx_s.desc.clone();
            ds.score_correction = true;
            let sc = eval_pass(
                &fx_s.device,
                &fx_s.queue,
                &ds,
                &fx_s.pipes,
                &fx_s.scene,
                &fx_s.bufs,
                &fx_s.desc.init,
                &fx_s.desc.init_albedo,
                EvalMode::Full,
                fx_s.desc.seed,
            )
            .expect("analytic");
            let g_dir = project_dir_grad(fx_s.desc.init.sun_dir, [sc.2[0], sc.2[1], sc.2[2]]);
            let row = [g_dir[0], sc.2[3], sc.2[4]];
            for (k, v) in row.iter().enumerate() {
                an_on[k] += v;
            }
        }
        let n = n_seeds as f32;
        for k in 0..3 {
            fd[k] /= n;
            an_on[k] /= n;
        }
        let names = ["sun_dir[x] (proj)", "intensity", "turbidity"];
        for k in 0..3 {
            assert!(
                fd[k].is_finite() && an_on[k].is_finite(),
                "{}: non-finite channel (fd {}, analytic {})",
                names[k],
                fd[k],
                an_on[k]
            );
        }
        // A particular finite-seed FD sample need not be closer when the
        // likelihood score is enabled: its expectation, not each realization,
        // is the target. Retain the existing numerical error smoke and check
        // score liveness separately in inverse_score_correction_is_exercised.
        for k in 0..3 {
            if fd[k].abs() < 1e-6 {
                continue; // channel too weak to gate
            }
            let err_on = (an_on[k] - fd[k]).abs();
            let tolerance = 0.05 * fd[k].abs() + 1e-6;
            assert!(
                err_on <= tolerance,
                "{}: score-corrected analytic {} vs FD {} (tol {})",
                names[k],
                an_on[k],
                fd[k],
                tolerance
            );
        }
    }

    /// Checkpointed-tile replay: with frames > tile_size the driver must
    /// reproduce the same gradients as the fully-cached run — the replayed
    /// seed-deterministic frame chain refills each tile's whist slots
    /// identically, so the results agree to CAS ordering (last-ulp).
    #[test]
    fn inverse_checkpointed_replay_matches_untiled() {
        let mut desc = flat_desc();
        for (i, h) in desc.heights.iter_mut().enumerate() {
            let x = (i % desc.dem_width as usize) as f32 - 1.5;
            let y = (i / desc.dem_width as usize) as f32 - 1.5;
            *h = 2.5 * (-(x * x + y * y) / 0.8).exp();
        }
        desc.frames = 12;
        desc.tile_size = 8; // K=8 < frames=12 -> two replay tiles
        desc.spatial_reuse = true;
        let target_params = InverseParams {
            sun_dir: desc.init.sun_dir,
            sun_intensity: desc.init.sun_intensity + 0.6,
            turbidity: desc.init.turbidity + 0.4,
        };
        // Each run builds a full fixture — the scene's bind groups hold the
        // whist buffer sized for that run's slot count.
        let run = |tile_size: u32| -> Option<(Vec<f32>, Vec<f32>)> {
            let mut d = desc.clone();
            d.tile_size = tile_size;
            let fx = fixture(&d, &target_params)?;
            let (_l, a, s) = eval_pass(
                &fx.device,
                &fx.queue,
                &fx.desc,
                &fx.pipes,
                &fx.scene,
                &fx.bufs,
                &fx.desc.init,
                &fx.desc.init_albedo,
                EvalMode::Full,
                fx.desc.seed,
            )
            .expect("eval");
            Some((a, s))
        };
        // Untiled: K >= frames caches every frame's snapshot; tiled: K=8.
        let Some((alb_full, sc_full)) = run(64) else {
            return;
        };
        let Some((alb_tile, sc_tile)) = run(8) else {
            return;
        };
        for (i, (a, b)) in alb_full.iter().zip(alb_tile.iter()).enumerate() {
            assert!(
                (a - b).abs() <= 1e-5 * a.abs().max(1.0),
                "albedo grad texel {i}: untiled {a} vs tiled {b}"
            );
        }
        for (i, (a, b)) in sc_full.iter().zip(sc_tile.iter()).enumerate() {
            assert!(
                (a - b).abs() <= 1e-5 * a.abs().max(1.0),
                "scalar grad {i}: untiled {a} vs tiled {b}"
            );
        }
    }
}
