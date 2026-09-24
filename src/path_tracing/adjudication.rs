// src/path_tracing/adjudication.rs
// AEQUITAS path-traced ground truth: drives the existing WavefrontScheduler
// over the analytic ReferenceSceneDesc at high accumulated spp. This is a
// genuine multi-bounce wavefront path trace (NEE at every hit, BSDF-sampled
// continuation rays up to the scheduler depth cap, Russian roulette from
// depth 4) — every frame must execute at least two wavefront iterations
// (primary wave + a bounce wave) or rendering fails.
// ReSTIR spatial reuse runs on current-frame primary geometry with real sun
// samples. Temporal reuse stays disabled, and each frame must dispatch the
// scene-bound spatial pass before shading from its resolved reservoir.
// RELEVANT FILES: src/path_tracing/wavefront/render.rs, src/path_tracing/reference_scene.rs

use crate::core::error::RenderError;
use crate::core::resource_tracker::{
    tracked_create_buffer, tracked_create_buffer_init, TrackedBuffer,
};
use crate::path_tracing::reference_scene::ReferenceSceneDesc;
use crate::path_tracing::wavefront::WavefrontScheduler;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use wgpu::{Device, Queue};

/// Uniforms layout shared by every wavefront kernel (pt_raygen.wgsl et al.).
/// 96 bytes; identical field placement to `compute_types::Uniforms`, but the
/// 4th u32 is `spp` in the wavefront kernels (aov_flags in the megakernel).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct WavefrontUniforms {
    width: u32,
    height: u32,
    frame_index: u32,
    spp: u32,
    cam_origin: [f32; 3],
    cam_fov_y: f32,
    cam_right: [f32; 3],
    cam_aspect: f32,
    cam_up: [f32; 3],
    cam_exposure: f32,
    cam_forward: [f32; 3],
    seed_hi: u32,
    seed_lo: u32,
    _pad_end: [u32; 3],
}

fn storage_buffer(
    device: &Device,
    label: &str,
    contents: &[u8],
) -> Result<TrackedBuffer, RenderError> {
    tracked_create_buffer_init(
        device,
        &wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        },
    )
}

/// Per-frame seed decorrelation. pt_raygen derives
/// rng_hi = seed_hi ^ (pixel * 9781) ^ (frame * 6271) and pt_shade re-hashes
/// with distinct odd constants (26699/30977), so samples decorrelate per pixel
/// AND per frame. Hashing fresh per-frame seeds (splitmix32-style) additionally
/// decorrelates the raygen jitter and Russian-roulette streams across frames,
/// so the accumulated per-pixel mean converges to the scene's full
/// light-transport integral.
fn splitmix32(mut x: u32) -> u32 {
    x = x.wrapping_add(0x9E37_79B9);
    let mut z = x;
    z = (z ^ (z >> 16)).wrapping_mul(0x21F0_AAAD);
    z = (z ^ (z >> 15)).wrapping_mul(0x735A_2D97);
    z ^ (z >> 15)
}

/// The existing `WavefrontScheduler` with its scene, accumulation and camera
/// bindings populated from one `ReferenceSceneDesc`. Frames accumulate into a
/// linear-HDR sum buffer; `read_mean_hdr` resolves the per-pixel mean.
struct ReferenceWavefront {
    scheduler: WavefrontScheduler,
    scene_bind_group: wgpu::BindGroup,
    accum_bind_group: wgpu::BindGroup,
    accum_buffer: TrackedBuffer,
    accum_bytes: u64,
    uniforms: WavefrontUniforms,
    uniforms_buffer: TrackedBuffer,
    seed_hi: u32,
    seed_lo: u32,
    // Scene storage referenced by the bind groups; held so the memory tracker
    // ledger releases it only when the reference is dropped.
    _scene_buffers: Vec<TrackedBuffer>,
}

impl ReferenceWavefront {
    fn new(
        device: &Arc<Device>,
        queue: &Arc<Queue>,
        desc: &ReferenceSceneDesc,
        width: u32,
        height: u32,
    ) -> Result<Self, RenderError> {
        let mut scheduler =
            WavefrontScheduler::new(device.clone(), queue.clone(), width, height)
                .map_err(|e| RenderError::Render(format!("wavefront scheduler init: {e}")))?;
        // Spatial reuse uses current-frame geometry, without temporal history.
        scheduler.set_restir_enabled(true);
        scheduler.set_restir_spatial_enabled(true);
        scheduler.set_restir_temporal_enabled(false);
        scheduler.set_environment_params(&desc.environment_raw());

        // --- Scene buffers from the single ReferenceSceneDesc ---
        let spheres = desc.wavefront_spheres();
        let spheres_buffer = storage_buffer(
            device,
            "adjudication-spheres",
            bytemuck::cast_slice(&spheres),
        )?;
        let area_lights = desc.area_lights();
        let area_lights_buffer = storage_buffer(
            device,
            "adjudication-area-lights",
            bytemuck::cast_slice(&area_lights),
        )?;
        let dir_lights = desc.directional_lights();
        let dir_lights_buffer = storage_buffer(
            device,
            "adjudication-dir-lights",
            bytemuck::cast_slice(&dir_lights),
        )?;
        let (light_samples, alias_entries, light_probs) =
            crate::path_tracing::restir::build_light_samples_and_alias(device, &[], &dir_lights)?;
        scheduler.set_restir_light_data(light_samples, alias_entries);
        scheduler.set_restir_light_probs(light_probs);
        let importance = desc.object_importance();
        let importance_buffer = storage_buffer(
            device,
            "adjudication-importance",
            bytemuck::cast_slice(&importance),
        )?;

        // Ground plane as a real mesh BLAS so pt_intersect/pt_shadow see the
        // exact same two triangles the raster path draws.
        let plane = desc.plane_mesh();
        let bvh = crate::accel::cpu_bvh::build_bvh_cpu(&plane, &Default::default())
            .map_err(|e| RenderError::Render(format!("plane BVH build: {e}")))?;
        let atlas_items = [(plane, bvh)];
        let atlas = crate::path_tracing::mesh::build_mesh_atlas(device, &atlas_items)
            .map_err(|e| RenderError::Render(format!("plane mesh atlas: {e}")))?;

        let ident: [f32; 16] = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let plane_instance = crate::accel::instancing::InstanceData {
            transform: ident,
            inv_transform: ident,
            blas_index: 0,
            material_id: 3, // plane material slot in the spheres array
            _padding: [0; 2],
        };
        scheduler.set_instances_buffer(storage_buffer(
            device,
            "adjudication-instances",
            bytemuck::cast_slice(&[plane_instance]),
        )?);
        scheduler.set_blas_descs_buffer(atlas.descs_buffer);

        // Fully bind the ReSTIR scene-spatial group and confirm it is Some,
        // per the adjudication contract (spatial resampling must never
        // early-return for a missing scene binding).
        scheduler
            .init_restir_scene_spatial_bind_group(&area_lights_buffer, &dir_lights_buffer)
            .map_err(|e| RenderError::Render(format!("restir scene-spatial bind group: {e}")))?;
        if !scheduler.restir_scene_bound() {
            return Err(RenderError::Render(
                "restir scene-spatial bind group not bound".into(),
            ));
        }

        let scene_bind_group = scheduler
            .create_scene_bind_group(
                &spheres_buffer,
                &atlas.vertex_buffer,
                &atlas.index_buffer,
                &atlas.bvh_buffer,
                &area_lights_buffer,
                &dir_lights_buffer,
                &importance_buffer,
            )
            .map_err(|e| RenderError::Render(format!("scene bind group: {e}")))?;

        // --- Accumulation target (vec4<f32> per pixel, zero-initialized) ---
        let px_count = (width as usize) * (height as usize);
        let accum_bytes = (px_count * std::mem::size_of::<[f32; 4]>()) as u64;
        let accum_buffer = tracked_create_buffer(
            device,
            &wgpu::BufferDescriptor {
                label: Some("adjudication-accum-hdr"),
                size: accum_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            },
        )?;
        let accum_bind_group = scheduler.create_accum_bind_group(&accum_buffer);

        // --- Camera uniforms exactly as pt_raygen.wgsl consumes them ---
        let (origin, forward, right, up) = desc.camera_basis();
        let uniforms = WavefrontUniforms {
            width,
            height,
            frame_index: 0,
            spp: 1,
            cam_origin: origin.into(),
            cam_fov_y: desc.fov_y_rad(),
            cam_right: right.into(),
            cam_aspect: width as f32 / height as f32,
            cam_up: up.into(),
            cam_exposure: desc.exposure,
            cam_forward: forward.into(),
            seed_hi: desc.seed_hi,
            seed_lo: desc.seed_lo,
            _pad_end: [0; 3],
        };
        let uniforms_buffer = tracked_create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("adjudication-uniforms"),
                contents: bytemuck::bytes_of(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        )?;

        Ok(Self {
            scheduler,
            scene_bind_group,
            accum_bind_group,
            accum_buffer,
            accum_bytes,
            uniforms,
            uniforms_buffer,
            seed_hi: desc.seed_hi,
            seed_lo: desc.seed_lo,
            _scene_buffers: vec![
                spheres_buffer,
                area_lights_buffer,
                dir_lights_buffer,
                importance_buffer,
                atlas.vertex_buffer,
                atlas.index_buffer,
                atlas.bvh_buffer,
            ],
        })
    }

    /// Trace and accumulate one camera sample per pixel. Returns the number of
    /// wavefront iterations executed; fails unless the frame ran the primary
    /// wave AND at least one bounce wave (multi-bounce contract).
    fn render_frame(
        &mut self,
        queue: &Queue,
        frame: u32,
        timing: Option<(&mut crate::core::gpu_timing::OneShotTiming, &str, u32)>,
    ) -> Result<u32, RenderError> {
        self.uniforms.frame_index = frame;
        self.uniforms.seed_hi = splitmix32(self.seed_hi ^ frame);
        self.uniforms.seed_lo = splitmix32(self.seed_lo ^ frame.wrapping_mul(0x0000_9E3D));
        queue.write_buffer(&self.uniforms_buffer, 0, bytemuck::bytes_of(&self.uniforms));
        let iterations = self
            .scheduler
            .render_frame_simple(
                &self.uniforms_buffer,
                &self.scene_bind_group,
                &self.accum_bind_group,
                timing,
            )
            .map_err(|e| RenderError::Render(format!("wavefront frame {frame}: {e}")))?;
        // Multi-bounce contract: a path-traced reference frame of this scene
        // always consumes the primary wave AND at least one bounce wave. One
        // iteration means the tracer degenerated into a primary-only
        // (direct-lighting) shortcut.
        if iterations < 2 {
            return Err(RenderError::Render(format!(
                "adjudication PT frame {frame} executed {iterations} wavefront iteration(s); \
                 a multi-bounce path-traced reference requires >= 2"
            )));
        }
        Ok(iterations)
    }

    /// Per-pixel mean radiance over `frames` accumulated frames, read back
    /// through a tracked host-visible staging buffer. Alpha is forced to 1
    /// (the accum alpha channel is a ReSTIR diagnostic, not coverage).
    fn read_mean_hdr(
        &self,
        device: &Device,
        queue: &Queue,
        frames: u32,
    ) -> Result<Vec<f32>, RenderError> {
        // `tracked_create_buffer` records the host-visible allocation in the
        // global ledger and releases it when `staging` is dropped below.
        let staging = tracked_create_buffer(
            device,
            &wgpu::BufferDescriptor {
                label: Some("adjudication-accum-readback"),
                size: self.accum_bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            },
        )?;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("adjudication-accum-copy"),
        });
        encoder.copy_buffer_to_buffer(&self.accum_buffer, 0, &staging, 0, self.accum_bytes);
        queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);
        let mut hdr: Vec<f32> = {
            let data = slice.get_mapped_range();
            bytemuck::cast_slice::<u8, f32>(&data).to_vec()
        };
        staging.unmap();
        drop(staging);

        let inv = 1.0 / frames as f32;
        for px in hdr.chunks_exact_mut(4) {
            px[0] *= inv;
            px[1] *= inv;
            px[2] *= inv;
            px[3] = 1.0;
        }
        Ok(hdr)
    }
}

/// Render the linear-HDR path-traced reference: mean radiance over
/// `spp_frames` accumulated frames of the full wavefront path tracer
/// (multi-bounce: NEE at every hit, BSDF-sampled continuation rays up to the
/// scheduler's depth cap, Russian roulette from depth 4). One camera sample
/// per pixel per frame; the per-pixel mean over `spp_frames` converges to the
/// full light-transport integral of the reference scene, not a direct-lighting
/// proxy. Each frame must execute at least two wavefront iterations
/// (primary wave + at least one bounce wave) or rendering fails.
/// Returns RGBA f32, row-major, `width * height * 4` values, alpha = 1.
///
/// `timing` (CENSOR F-04): when supplied, the first frame's primary wavefront
/// wave is timed under the "adjudication.path_trace" certificate label (see
/// `render_frame_simple` — a whole frame cannot be bracketed on one encoder).
/// The `OneShotTiming` MUST live on the same wgpu device as `device`; pass
/// `None` when driving a standalone device (tests).
pub fn render_pt_reference(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    desc: &ReferenceSceneDesc,
    width: u32,
    height: u32,
    spp_frames: u32,
    mut timing: Option<&mut crate::core::gpu_timing::OneShotTiming>,
) -> Result<Vec<f32>, RenderError> {
    if width == 0 || height == 0 || spp_frames == 0 {
        return Err(RenderError::Render(
            "adjudication PT reference requires non-zero width/height/spp".into(),
        ));
    }

    let mut reference = ReferenceWavefront::new(device, queue, desc, width, height)?;
    for frame in 0..spp_frames {
        // Only frame 0 carries the timing scope so the certificate records the
        // "adjudication.path_trace" pass exactly once.
        let frame_timing = if frame == 0 {
            timing
                .as_mut()
                .map(|t| (&mut **t, "adjudication.path_trace", spp_frames))
        } else {
            None
        };
        reference.render_frame(queue, frame, frame_timing)?;
        if frame % 64 == 63 {
            device.poll(wgpu::Maintain::Wait);
        }
    }

    if reference.scheduler.restir_spatial_dispatches() != spp_frames {
        return Err(RenderError::Render(
            "adjudication requires one spatial dispatch per frame".into(),
        ));
    }

    reference.read_mean_hdr(device, queue, spp_frames)
}

/// Device for the AEQUITAS GPU unit tests, or `None` (with the reason on
/// stderr) when this host cannot meaningfully run them — the same
/// hardware-only convention the Python adjudication gate follows:
/// - software adapters (WARP, lavapipe, llvmpipe, SwiftShader) are declined;
/// - with `wavefront`, Metal is declined: naga 0.19's MSL backend does not
///   implement `atomicCompareExchange`, which pt_shadow.wgsl's float
///   accumulation needs, so the wavefront shadow pipeline cannot be built.
#[cfg(test)]
pub(crate) fn adjudication_test_device(wavefront: bool) -> Option<(Arc<Device>, Arc<Queue>)> {
    let (device, queue, info) = crate::core::gpu::create_device_queue_and_info_for_test()?;
    let name = info.name.to_lowercase();
    let software = info.device_type == wgpu::DeviceType::Cpu
        || [
            "basic render driver",
            "warp",
            "lavapipe",
            "llvmpipe",
            "swiftshader",
        ]
        .iter()
        .any(|token| name.contains(token));
    if software {
        eprintln!(
            "skipping AEQUITAS GPU test: software adapter '{}'",
            info.name
        );
        return None;
    }
    if wavefront && info.backend == wgpu::Backend::Metal {
        eprintln!(
            "skipping AEQUITAS wavefront test on Metal: naga 0.19 MSL lacks atomicCompareExchange"
        );
        return None;
    }
    Some((Arc::new(device), Arc::new(queue)))
}

#[cfg(test)]
mod tests {
    #[test]
    fn adjudication_driver_uses_wavefront_active_count_loop_not_zero_stub() {
        let driver = include_str!("adjudication.rs");
        let render = include_str!("wavefront/render.rs");
        let queues = include_str!("wavefront/queues/types.rs");

        assert!(driver.contains(".render_frame_simple("));
        // The loop terminates on the queue header's in_count - out_count
        // (the same value get_active_ray_count reports); the behavioural
        // proof is assert_active_count_tracks_bounce_waves below.
        assert!(render.contains("header.active_count()"));
        assert!(queues.contains(".active_count())"));
        assert!(!queues.contains("Ok(0)"));
        // Gap 2: the one-iteration escape hatch must stay dead.
        assert!(
            !render.contains("did_any"),
            "render_frame_simple must not reintroduce the did_any one-iteration fallback"
        );
        // Gap 1: the driver must enforce the multi-bounce contract per frame.
        assert!(
            driver.contains("iterations < 2"),
            "render_pt_reference must reject frames that ran fewer than two wavefront iterations"
        );
    }

    #[test]
    fn adjudication_requires_spatial_reuse_without_temporal_history() {
        let driver = include_str!("adjudication.rs")
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        assert!(driver.contains("scheduler.set_restir_enabled(true)"));
        assert!(driver.contains("scheduler.set_restir_spatial_enabled(true)"));
        assert!(driver.contains("scheduler.set_restir_temporal_enabled(false)"));
    }

    fn assert_shadow_accumulation(device: &wgpu::Device, queue: &wgpu::Queue) {
        use crate::core::resource_tracker::tracked_create_buffer;
        let source = include_str!("../shaders/pt_shadow.wgsl").to_owned()
            + r#"
@compute @workgroup_size(256)
fn stress_accumulation() {
    accumulate_shadow(0u, vec3<f32>(1.0, 2.0, 3.0));
}
"#;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shadow-accumulation-stress"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = crate::core::shader_registry::create_compute_pipeline_scoped(
            device,
            &wgpu::ComputePipelineDescriptor {
                label: Some("shadow-accumulation-stress"),
                layout: None,
                module: &shader,
                entry_point: "stress_accumulation",
            },
        );
        let result = tracked_create_buffer(
            device,
            &wgpu::BufferDescriptor {
                label: Some("shadow-stress-result"),
                size: 16,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            },
        )
        .unwrap();
        let readback = tracked_create_buffer(
            device,
            &wgpu::BufferDescriptor {
                label: Some("shadow-stress-readback"),
                size: 16,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            },
        )
        .unwrap();
        let groups: Vec<_> = (0..4)
            .map(|group| {
                let entries = [wgpu::BindGroupEntry {
                    binding: 0,
                    resource: result.as_entire_binding(),
                }];
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &pipeline.get_bind_group_layout(group),
                    entries: if group == 3 { &entries } else { &[] },
                })
            })
            .collect();
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            for (index, group) in groups.iter().enumerate() {
                pass.set_bind_group(index as u32, group, &[]);
            }
            pass.dispatch_workgroups(2, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&result, 0, &readback, 0, 16);
        queue.submit(Some(encoder.finish()));
        let (tx, rx) = std::sync::mpsc::channel();
        readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| tx.send(result).unwrap());
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let data = readback.slice(..).get_mapped_range();
        assert_eq!(
            bytemuck::cast_slice::<u8, f32>(&data),
            &[512.0, 1024.0, 1536.0, 0.0]
        );
        drop(data);
        readback.unmap();
    }

    fn assert_transport_response(device: &wgpu::Device, queue: &wgpu::Queue) {
        let shade = include_str!("../shaders/pt_shade.wgsl").to_owned()
            + r#"
@compute @workgroup_size(1)
fn transport_probe() {
    let n = vec3<f32>(0.0, 1.0, 0.0);
    let brdf = bsdf_eval_pdf(n, n, n, vec3<f32>(0.0), 0.0, 1.0, 1.0, 1.0);
    accum_hdr[0] = vec4<f32>(sampled_brdf_weight(brdf, 1.0), 1.0);
}
"#;
        let scatter = include_str!("../shaders/pt_scatter.wgsl").to_owned()
            + r#"
@compute @workgroup_size(1)
fn transport_probe() {
    accum_hdr[0] = vec4<f32>(environment_escape_weight(0u, 1.0),
        environment_escape_weight(1u, 0.5), environment_escape_weight(1u, 0.0), 1.0);
}
"#;
        for (source, expected) in [
            (shade, [0.01, 0.01, 0.01, 1.0]),
            (scatter, [1.0, 0.0, 1.0, 1.0]),
        ] {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("adjudication-transport-probe"),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
            let pipeline = crate::core::shader_registry::create_compute_pipeline_scoped(
                device,
                &wgpu::ComputePipelineDescriptor {
                    label: Some("adjudication-transport-probe"),
                    layout: None,
                    module: &shader,
                    entry_point: "transport_probe",
                },
            );
            let output = super::tracked_create_buffer(
                device,
                &wgpu::BufferDescriptor {
                    label: Some("transport-probe-output"),
                    size: 16,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                },
            )
            .unwrap();
            let readback = super::tracked_create_buffer(
                device,
                &wgpu::BufferDescriptor {
                    label: Some("transport-probe-readback"),
                    size: 16,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                },
            )
            .unwrap();
            let groups: Vec<_> = (0..4)
                .map(|index| {
                    let entries = [wgpu::BindGroupEntry {
                        binding: 0,
                        resource: output.as_entire_binding(),
                    }];
                    device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &pipeline.get_bind_group_layout(index),
                        entries: if index == 3 { &entries } else { &[] },
                    })
                })
                .collect();
            let mut encoder = device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                for (index, group) in groups.iter().enumerate() {
                    pass.set_bind_group(index as u32, group, &[]);
                }
                pass.dispatch_workgroups(1, 1, 1);
            }
            encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, 16);
            queue.submit(Some(encoder.finish()));
            let (tx, rx) = std::sync::mpsc::channel();
            readback
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| tx.send(result).unwrap());
            device.poll(wgpu::Maintain::Wait);
            rx.recv().unwrap().unwrap();
            let data = readback.slice(..).get_mapped_range();
            for (got, expected) in bytemuck::cast_slice::<u8, f32>(&data).iter().zip(expected) {
                assert!(
                    (got - expected).abs() <= f32::EPSILON,
                    "{got} != {expected}"
                );
            }
            drop(data);
            readback.unmap();
        }
    }

    fn assert_constant_environment_plane(
        device: &std::sync::Arc<wgpu::Device>,
        queue: &std::sync::Arc<wgpu::Queue>,
    ) {
        let mut desc = crate::path_tracing::reference_scene::adjudication_scene();
        desc.cam_origin = [0.0, 1.0, 0.0];
        desc.cam_look_at = [0.0, 0.0, 0.0];
        desc.cam_up = [0.0, 0.0, -1.0];
        desc.fov_y_deg = 0.0;
        desc.sun_intensity = 0.0;
        desc.ambient_color = [1.0; 3];
        desc.sky_color = [0.0; 3];
        for sphere in desc.spheres.iter_mut().take(3) {
            sphere.radius = 0.0;
        }
        desc.spheres[3].albedo = [0.25, 0.5, 0.75];
        desc.spheres[3].roughness = 1.0;
        let a = 1.0f64 / 2.0f64.sqrt();
        let mut diffuse_fresnel_integral = 0.0;
        let mut specular_fresnel_integral = 0.0;
        for (power, coefficient) in [1.0, -5.0, 10.0, -10.0, 5.0, -1.0].into_iter().enumerate() {
            let k = power as i32;
            diffuse_fresnel_integral += coefficient
                * (8.0 * (1.0 - a.powi(k + 4)) / f64::from(k + 4)
                    - 4.0 * (1.0 - a.powi(k + 2)) / f64::from(k + 2));
            let inverse_term = if k == 0 {
                -2.0 * a.ln()
            } else {
                2.0 * (1.0 - a.powi(k)) / f64::from(k)
            };
            specular_fresnel_integral +=
                coefficient * (4.0 * (1.0 - a.powi(k + 2)) / f64::from(k + 2) - inverse_term);
        }
        let f0 = f64::from(crate::path_tracing::reference_scene::REFERENCE_DIELECTRIC_F0);
        let diffuse = (1.0 - f0) - 2.0 * (1.0 - f0) * diffuse_fresnel_integral;
        let specular = f0 * (1.0 - 2.0f64.ln()) + (1.0 - f0) * specular_fresnel_integral;
        let mut expected_hdr = [0.0f32; 4];
        for (channel, albedo) in desc.spheres[3].albedo.iter().enumerate() {
            expected_hdr[channel] = (f64::from(*albedo) * diffuse + specular) as f32;
        }
        expected_hdr[3] = 1.0;
        let hdr = super::render_pt_reference(device, queue, &desc, 32, 32, 4096, None).unwrap();
        let actual = crate::core::tonemap::resolve_reference_hdr_to_rgba8(&hdr, 1.0);
        let expected = crate::core::tonemap::resolve_reference_hdr_to_rgba8(&expected_hdr, 1.0);
        let mean_absolute_error = actual
            .chunks_exact(4)
            .map(|pixel| {
                (0..3)
                    .map(|channel| (f64::from(pixel[channel]) - f64::from(expected[channel])).abs())
                    .sum::<f64>()
            })
            .sum::<f64>()
            / (32.0 * 32.0 * 3.0);
        assert!(mean_absolute_error <= 2.0, "constant-environment plane disagrees with analytic Lambert/GGX integral: {mean_absolute_error}");
    }

    /// Drives the reference scheduler frame by frame and checks the ray-queue
    /// accounting the wavefront loop terminates on: the active count must
    /// report the bounce rays the scatter pass pushed (a zero stub, or the
    /// old failed-pop over-increment of out_count, reports 0 and ends the
    /// frame after the primary wave), and every frame must drain exactly.
    fn assert_active_count_tracks_bounce_waves(
        device: &std::sync::Arc<wgpu::Device>,
        queue: &std::sync::Arc<wgpu::Queue>,
    ) {
        let desc = crate::path_tracing::reference_scene::adjudication_scene();
        let (width, height) = (32u32, 32u32);
        let mut reference =
            super::ReferenceWavefront::new(device, queue, &desc, width, height).unwrap();
        for frame in 0..4 {
            let iterations = reference.render_frame(queue, frame, None).unwrap();
            let waves = reference.scheduler.last_frame_wave_sizes().to_vec();
            assert_eq!(waves.len() as u32, iterations);
            // Raygen pushes exactly one camera ray per pixel.
            assert_eq!(waves[0], width * height, "primary wave: {waves:?}");
            // The plane fills the lower image, so the primary wave always
            // scatters bounce rays and the active count must see them.
            assert!(waves[1] > 0, "bounce wave reported empty: {waves:?}");
            // Waves shrink as paths terminate (miss, depth cap, roulette).
            assert!(waves.windows(2).all(|w| w[1] <= w[0]), "{waves:?}");
            let header = reference.scheduler.read_ray_queue_header().unwrap();
            assert_eq!(
                header.in_count,
                waves.iter().sum::<u32>(),
                "every pushed ray must have been reported active in exactly one wave"
            );
            assert_eq!(
                header.out_count, header.in_count,
                "queue did not drain exactly"
            );
            assert_eq!(header.active_count(), 0);
        }
    }

    #[test]
    fn pt_reference_renders_multibounce_frames() {
        // Fails if the adjudication PT path is ever rewired as a
        // one-iteration/direct-lighting shortcut: render_pt_reference errors
        // when any frame executes fewer than two wavefront iterations.
        let Some((device, queue)) = super::adjudication_test_device(true) else {
            return;
        };
        let desc = crate::path_tracing::reference_scene::adjudication_scene();
        // timing = None: this test drives a standalone device, and a timing
        // manager from the global context would live on a different device.
        let hdr = super::render_pt_reference(&device, &queue, &desc, 32, 32, 2, None)
            .expect("multi-bounce PT reference render");
        assert_eq!(hdr.len(), 32 * 32 * 4);
        assert!(hdr.iter().any(|&v| v > 0.0), "PT reference is all black");
        assert_active_count_tracks_bounce_waves(&device, &queue);
        assert_shadow_accumulation(&device, &queue);
        assert_transport_response(&device, &queue);
        assert_constant_environment_plane(&device, &queue);
    }
}
