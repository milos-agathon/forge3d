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
// turbidity linearly in [1, 10]. The optimized objective is the nonlinear
// display-space loss of the 16-replicate MEAN image — evaluating the loss
// per replicate instead (E[loss(cur)]) would embed a Var(cur) term whose
// gradient pulls descent toward display-space compression (high intensity/
// turbidity). During the albedo-frozen warmup a global per-channel albedo
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
    tracked_create_buffer, tracked_create_buffer_init, tracked_create_texture, TrackedBuffer,
    TrackedTexture,
};
use crate::path_tracing::compute_types::{Sphere, Uniforms};
use crate::path_tracing::hybrid_compute::terrain_heightfield::{
    EarthCurvatureUniforms, TerrainPtScene,
};
use crate::path_tracing::hybrid_compute::{HybridUniforms, LightingUniforms, TraversalMode};
use crate::path_tracing::lighting::{GpuAreaLight, GpuDirectionalLight};
use crate::path_tracing::restir::Reservoir;
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
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: wgpu::BindingResource::TextureView(&aov_views[1]),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: wgpu::BindingResource::TextureView(&aov_views[2]),
                },
            ],
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

/// One primal+adjoint evaluation over a set of independent Monte-Carlo
/// stream seeds ("replicates"). Three phases:
///
///   A. Per-replicate forward sweep (terrain -> temporal -> spatial|copy),
///      then fold the replicate's mean linear radiance into the shared
///      replicate-mean image with weight 1/R.
///   B. ONE loss pass on the replicate-mean image. The optimized objective
///      is loss(E[cur]) — the nonlinear display metric applied to the
///      R-replicate mean — so estimator variance enters at 1/R. Evaluating
///      per-replicate losses instead (E[loss(cur)]) embeds a Var(cur) term
///      in the objective whose gradient systematically pulls the descent
///      toward display-space compression (higher intensity/turbidity).
///   C. Per-replicate adjoint: the seed-deterministic frame chain is
///      replayed per K-frame tile (gradient checkpointing) and shaded
///      against the shared mean-image adjoint; the chain-rule factor
///      ∂mean/∂cur_r = 1/R is applied when the per-replicate gradient
///      sums are averaged host-side.
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
) -> Result<(f32, Vec<f32>, Vec<f32>), RenderError> {
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
    let frame_chain = |enc: &mut wgpu::CommandEncoder, f: u32, snap: bool, publish: bool| {
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
        } else {
            enc.copy_buffer_to_buffer(&scene.res_out, 0, &scene.res_prev, 0, scene.reservoir_bytes);
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
            frame_chain(&mut enc, f, false, true);
        }
        queue.submit([enc.finish()]);
        return Ok((0.0, Vec::new(), Vec::new()));
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
            frame_chain(&mut enc, f, false, false);
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

    // Phase C: per-replicate adjoint. Each replicate replays its frame
    // chain per K-frame tile (rebuilding reservoirs + whist snapshots) and
    // shades against the shared mean-image adjoint.
    if mode == EvalMode::Full {
        for &seed in seeds {
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
            let mut base = 0u32;
            while base < scene.frames {
                let end = (base + slots).min(scene.frames);
                for f in base..end {
                    frame_chain(&mut enc, f, true, false);
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
                base = end;
            }
            if desc.edge_term {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("inv-edge"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipes.edge);
                pass.set_bind_group(0, &scene.g0_frames[0], &[]);
                pass.set_bind_group(1, &scene.g1, &[]);
                pass.set_bind_group(2, &scene.g2, &[]);
                pass.set_bind_group(3, &scene.g3_edge, &[]);
                let (ex, ey) = fold_1d(2u64 * px);
                pass.dispatch_workgroups(ex, ey, 1);
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
    let scalars = decode_scalars(&raw[..(8 * 4)]);
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
    let scalars: Vec<f32> = scalars.iter().map(|v| v * inv_r).collect();
    Ok((mean_loss as f32, d_albedo, scalars))
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

/// Multi-replicate eval over well-separated stream seeds. Each replicate is
/// an independent unbiased estimate of the same image; the loss is computed
/// once on the replicate-mean image (loss(E[cur]), variance entering at
/// 1/R) and each replicate's adjoint is shaded against the shared mean-
/// image upstream — the optimizer walks the EXPECTED landscape instead of
/// one fixed-noise realization or a variance-inflated E[loss(cur)]. The
/// seed set is fixed across iterations so the landscape stays
/// deterministic for the optimizer.
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
    eval_seed_set(
        device, queue, desc, pipes, scene, bufs, params, albedo, mode, &seeds,
    )
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
    // allocation in this solve (including the transient staging readbacks,
    // the only host-visible ones) is attributed to this owner, so the
    // captured peak is the solve's true concurrent host-visible footprint.
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
    // tracked allocation in this solve is owner-tagged, and the transient
    // MAP_READ staging buffers are the only host-visible ones — the
    // captured peak is the solve's true concurrent host-visible footprint.
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

    /// Edge-term strict improvement: on a scene with real cast-shadow
    /// boundaries (a mesa shadow band terminating on lit flat ground) the
    /// analytic sun-direction gradient with the boundary term must be
    /// strictly closer to FD than without it — a zero or sign-flipped edge
    /// term fails this gate.
    #[test]
    fn inverse_edge_term_strictly_improves_fd() {
        let mut desc = flat_desc();
        // A steep mesa on a wide flat apron: the anti-sun slope exceeds the
        // grazing angle (90° - sun_elevation ~ 45°) so the feature both
        // forms a terminator AND casts a shadow that lands on flat ground —
        // only the cast-shadow boundary is a true image-space discontinuity
        // (a terminator's image function is continuous through n.omega = 0),
        // so the edge term needs lit pixels adjacent to occluded flat
        // ground. 5 units high over a 3-unit texel step is a ~59° slope; the
        // shadow runs ~5 units across the ~3-unit apron.
        desc.spacing = (3.0, 3.0);
        // The mesa sits in the +x/-z (sun-side) quadrant so its cast shadow
        // falls across the wide -x/+z apron and TERMINATES on flat ground —
        // a shadow band whose lit/shadowed outer edge is inside the frame
        // produces real cast-shadow pixel pairs.
        for (i, h) in desc.heights.iter_mut().enumerate() {
            let x = (i % desc.dem_width as usize) as f32 - 1.5;
            let y = (i / desc.dem_width as usize) as f32 - 1.5;
            *h = if x >= 0.5 && y <= -0.5 { 5.0 } else { 0.0 };
        }
        // Dense per-pixel sampling: each boundary pixel's theta-response is a
        // staircase of `spp` sub-pixel flips, so the realized FD concentrates
        // toward the boundary term's expectation instead of a Bernoulli draw.
        desc.spp = 64;
        desc.spatial_reuse = false;
        desc.score_correction = false;
        let target_params = InverseParams {
            sun_dir: [
                desc.init.sun_dir[0] + 0.06,
                desc.init.sun_dir[1],
                desc.init.sun_dir[2],
            ],
            sun_intensity: desc.init.sun_intensity,
            turbidity: desc.init.turbidity,
        };
        // FD expectation: average the x-component central difference over
        // several eval seeds — the single-draw FD is a jittered staircase
        // whose expectation is what the boundary estimator approximates.
        let eps = 1e-3f32;
        let mut fd_mean = 0.0f32;
        let n_seeds = 4u32;
        for s in 0..n_seeds {
            let mut d = desc.clone();
            d.seed = desc.seed.wrapping_add(0x9e3779b9u32.wrapping_mul(s + 1));
            let Some(fx_s) = fixture(&d, &target_params) else {
                return;
            };
            let mut pp = fx_s.desc.init;
            let mut pm = fx_s.desc.init;
            pp.sun_dir[0] += eps;
            pm.sun_dir[0] -= eps;
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
            let fdx = (lp - lm) / (2.0 * eps);
            fd_mean += fdx;
        }
        fd_mean /= n_seeds as f32;

        let Some(fx) = fixture(&desc, &target_params) else {
            return;
        };
        let eval_scalars = |edge: bool| -> Vec<f32> {
            let mut d = fx.desc.clone();
            d.edge_term = edge;
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
        let sc_edge = eval_scalars(true);
        let sc_smooth = eval_scalars(false);
        let an_edge = project_dir_grad(fx.desc.init.sun_dir, [sc_edge[0], sc_edge[1], sc_edge[2]]);
        let an_smooth = project_dir_grad(
            fx.desc.init.sun_dir,
            [sc_smooth[0], sc_smooth[1], sc_smooth[2]],
        );
        let err_edge = (an_edge[0] - fd_mean).abs();
        let err_smooth = (an_smooth[0] - fd_mean).abs();
        assert!(
            err_edge < err_smooth,
            "edge term did not strictly improve FD agreement: with-edge {} \
             (err {err_edge}) vs without {} (err {err_smooth}) vs fd {fd_mean}",
            an_edge[0],
            an_smooth[0]
        );
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

    /// Non-flat verification of the full estimator — differentiable spectral
    /// reservoir weight, reparameterized boundary term, and score-function
    /// correction — against the realized finite-difference expectation. The
    /// realized FD sees the discrete reservoir-selection channel too (a
    /// shifted sun re-picks channels), so the multi-seed FD mean is the
    /// expectation the FULL analytic estimator (all terms on) must match.
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

        // Multi-seed mean analytic gradients, paired per seed, score
        // correction on vs off.
        let mut an_on = [0.0f32; 3];
        let mut an_off = [0.0f32; 3];
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
            for score in [true, false] {
                let mut ds = fx_s.desc.clone();
                ds.score_correction = score;
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
                    if score {
                        an_on[k] += v;
                    } else {
                        an_off[k] += v;
                    }
                }
            }
        }
        let n = n_seeds as f32;
        for k in 0..3 {
            fd[k] /= n;
            an_on[k] /= n;
            an_off[k] /= n;
        }
        let names = ["sun_dir[x] (proj)", "intensity", "turbidity"];
        for k in 0..3 {
            assert!(
                fd[k].is_finite() && an_on[k].is_finite() && an_off[k].is_finite(),
                "{}: non-finite channel (fd {}, on {}, off {})",
                names[k],
                fd[k],
                an_on[k],
                an_off[k]
            );
        }
        // The score correction is the expectation-space completion of the
        // detached selection: with it, the multi-seed analytic must sit at
        // least as close to the FD expectation as without, and on the
        // channel where the correction is significant, strictly closer.
        let mut improved_any = false;
        for k in 0..3 {
            if fd[k].abs() < 1e-6 {
                continue; // channel too weak to gate
            }
            let err_on = (an_on[k] - fd[k]).abs();
            let err_off = (an_off[k] - fd[k]).abs();
            let tolerance = 0.05 * fd[k].abs() + 1e-6;
            assert!(
                err_on <= tolerance,
                "{}: score-corrected analytic {} vs FD {} (tol {})",
                names[k],
                an_on[k],
                fd[k],
                tolerance
            );
            assert!(
                err_on <= err_off * 1.5 + 0.1 * fd[k].abs(),
                "{}: score-corrected analytic {} farther from FD {} than \
                 uncorrected {} — the expectation-space term is wrong-signed",
                names[k],
                an_on[k],
                fd[k],
                an_off[k]
            );
            improved_any |= err_on < err_off;
        }
        assert!(
            improved_any,
            "score correction never improved FD agreement on non-flat terrain: fd {fd:?}, on {an_on:?}, off {an_off:?}"
        );
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
