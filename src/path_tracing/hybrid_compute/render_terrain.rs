// src/path_tracing/hybrid_compute/render_terrain.rs
// PROMETHEUS accumulation driver: renders a converged path-traced reference
// of a real DEM under sun + IBL through the `main_terrain` kernel entry,
// optionally mixed with mesh geometry through the shared HybridScene seam.
// Each frame dispatches the terrain kernel (spp jittered samples + canonical
// ReSTIR candidate generation), then the pt_restir_temporal and
// pt_restir_spatial reuse passes over the canonical 80-byte Reservoir layout.
// Convergence gates on the per-pixel luminance variance of the running mean
// across the last WELFORD_WINDOW frames (hard error on cap miss — no silent
// fake convergence). Every GPU allocation is registered with the global
// memory tracker through a drop-guard so error paths cannot leak metrics.
// RELEVANT FILES: src/shaders/hybrid_terrain_traversal.wgsl,
//                 src/path_tracing/hybrid_compute/terrain_heightfield.rs

use super::terrain_heightfield::TerrainPtScene;
use super::*;
use crate::core::memory_tracker::global_tracker;
use crate::path_tracing::lighting::{GpuAreaLight, GpuDirectionalLight};
use crate::path_tracing::restir::{
    create_reservoir_buffer, create_restir_gbuffer, create_restir_gbuffer_pos, Reservoir,
};
use wgpu::util::DeviceExt;

/// Frames per convergence window: the gate measures the variance of the
/// accumulated mean luminance across the last N frames (02-prometheus DoD).
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
    /// Optional equirect environment map (RGB f32 rows) + dims; None uses the
    /// constant-white fallback scaled by `env_intensity`.
    pub env_map: Option<(Vec<f32>, u32, u32)>,
    pub env_intensity: f32,
    /// Optional mesh mixed into the scene: flat [x,y,z] vertices + triangle
    /// indices, traversed alongside the heightfield (TraversalMode::Hybrid).
    pub mesh: Option<(Vec<f32>, Vec<u32>)>,
    pub width: u32,
    pub height: u32,
    pub seed: u32,
    /// Camera samples per accumulation frame (min-max descent keeps the
    /// per-sample texture reads O(log mips), so cost scales ~linearly).
    pub spp: u32,
    /// Hard frame cap; convergence earlier is allowed.
    pub max_frames: u32,
    /// Frames rendered before the first convergence check.
    pub min_frames: u32,
    /// Converged when the max per-pixel luminance variance of the running
    /// mean across the last WELFORD_WINDOW frames < this.
    pub variance_threshold: f32,
}

/// Converged reference output.
pub struct TerrainReferenceOutput {
    pub rgba: Vec<u8>,
    pub albedo: Vec<f32>,
    pub normal: Vec<f32>,
    pub depth: Vec<f32>,
    pub frames: u32,
    pub variance: f32,
    pub converged: bool,
    pub peak_host_visible_bytes: u64,
    pub minmax_pyramid_bytes: u64,
    /// Sum of every GPU resource this render registered with the memory
    /// tracker (pyramid, env, accum, Welford, reservoirs, G-buffer, UBOs,
    /// output + AOV textures, mesh buffers).
    pub gpu_resource_bytes: u64,
}

/// Registers every allocation with the global memory tracker and frees the
/// entries on drop, so `?`/early-return paths cannot pollute the metrics.
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
        let size = buf.size();
        global_tracker().track_buffer_allocation(size, false);
        self.buffers.push(size);
    }

    fn texture(&mut self, width: u32, height: u32, format: wgpu::TextureFormat) {
        let bpp: u64 = match format {
            wgpu::TextureFormat::Rgba32Float => 16,
            wgpu::TextureFormat::Rgba16Float => 8,
            wgpu::TextureFormat::Rg32Float => 8,
            wgpu::TextureFormat::R32Float | wgpu::TextureFormat::Rgba8Unorm => 4,
            _ => 4,
        };
        global_tracker().track_texture_allocation(width, height, format);
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

impl Drop for TrackedGpu {
    fn drop(&mut self) {
        for size in self.buffers.drain(..) {
            global_tracker().free_buffer_allocation(size, false);
        }
        for (w, h, fmt, _) in self.textures.drain(..) {
            global_tracker().free_texture_allocation(w, h, fmt);
        }
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
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("hybrid-pt-terrain-readback"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    global_tracker().track_buffer_allocation(size, true);
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
    global_tracker().free_buffer_allocation(size, true);
    Ok(out)
}

fn read_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    size: u64,
) -> Result<Vec<u8>, RenderError> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("hybrid-pt-terrain-buf-readback"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    global_tracker().track_buffer_allocation(size, true);
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
    global_tracker().free_buffer_allocation(size, true);
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
    pub fn render_terrain_reference(
        &self,
        desc: &TerrainReferenceDesc,
    ) -> Result<TerrainReferenceOutput, RenderError> {
        let device = &try_ctx()?.device;
        let queue = &try_ctx()?.queue;
        let (width, height) = (desc.width, desc.height);
        validate_desc(desc)?;
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
            desc.env_intensity,
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
        let light_color = [
            desc.sun_intensity,
            desc.sun_intensity * 0.97,
            desc.sun_intensity * 0.92,
        ];

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
            cam_exposure: desc.exposure,
            cam_forward: forward.into(),
            seed_hi: desc.seed,
            seed_lo: desc.seed ^ 0x85EB_CA6B,
            _pad_end: [0; 3],
        };
        let base_ubo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("hybrid-pt-terrain-base-ubo"),
            contents: bytemuck::bytes_of(&base),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        tracked.buffer(&base_ubo);
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
            traversal_mode: if desc.mesh.is_some() {
                TraversalMode::Hybrid as u32
            } else {
                TraversalMode::TerrainOnly as u32
            },
            _pad: [0; 2],
        };
        let hybrid_ubo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("hybrid-pt-terrain-hybrid-ubo"),
            contents: bytemuck::bytes_of(&hybrid_uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        tracked.buffer(&hybrid_ubo);
        let lighting = LightingUniforms {
            light_dir,
            lighting_type: 1,
            light_color,
            shadows_enabled: 1,
            ambient_color: [0.0, 0.0, 0.0],
            shadow_intensity: 1.0,
            hdri_intensity: desc.env_intensity,
            hdri_rotation: 0.0,
            specular_power: 32.0,
            _pad: [0; 5],
        };
        let lighting_ubo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("hybrid-pt-terrain-lighting-ubo"),
            contents: bytemuck::bytes_of(&lighting),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        tracked.buffer(&lighting_ubo);
        let terrain_uniforms = terrain_scene.uniforms(desc.spp, WELFORD_WINDOW);
        let terrain_ubo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("hybrid-pt-terrain-ubo"),
            contents: bytemuck::bytes_of(&terrain_uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        tracked.buffer(&terrain_ubo);

        // --- Scene buffers (the spheres slot needs 1 dummy element) ---
        let scene_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hybrid-pt-terrain-scene"),
            size: std::mem::size_of::<Sphere>() as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        tracked.buffer(&scene_buf);

        // --- Directional-light table for the ReSTIR spatial pass ---
        let sun_light = GpuDirectionalLight::new(
            [-light_dir[0], -light_dir[1], -light_dir[2]],
            desc.sun_intensity.max(1e-6),
            [1.0, 0.97, 0.92],
            1.0,
        );
        let dir_lights_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("hybrid-pt-terrain-dir-lights"),
            contents: bytemuck::bytes_of(&sun_light),
            usage: wgpu::BufferUsages::STORAGE,
        });
        tracked.buffer(&dir_lights_buf);
        let area_light = <GpuAreaLight as bytemuck::Zeroable>::zeroed();
        let area_lights_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("hybrid-pt-terrain-area-lights"),
            contents: bytemuck::bytes_of(&area_light),
            usage: wgpu::BufferUsages::STORAGE,
        });
        tracked.buffer(&area_lights_buf);

        // --- Accumulation (same shape as render.rs "hybrid-pt-accum"),
        // Welford variance, canonical ReSTIR reservoirs and G-buffer ---
        let px_count = (width as u64) * (height as u64);
        let px_usize = px_count as usize;
        let accum_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hybrid-pt-accum"),
            size: px_count * 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        tracked.buffer(&accum_buf);
        let welford_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hybrid-pt-terrain-welford"),
            size: px_count * 8,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        tracked.buffer(&welford_buf);
        // Canonical Reservoir buffers (80-byte stride, matching
        // src/path_tracing/restir/types.rs and the pt_restir_* passes):
        // curr = fresh candidates, out = temporal merge, prev = final merged
        // history (spatial writes back into prev; the kernel shades from it).
        let reservoir_curr = create_reservoir_buffer(device, px_usize);
        let reservoir_out = create_reservoir_buffer(device, px_usize);
        let prev_init = vec![Reservoir::default(); px_usize];
        let reservoir_prev = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("hybrid-pt-terrain-reservoir-prev"),
            contents: bytemuck::cast_slice(&prev_init),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        tracked.buffer(&reservoir_curr);
        tracked.buffer(&reservoir_out);
        tracked.buffer(&reservoir_prev);
        let gbuffer_nr = create_restir_gbuffer(device, px_usize);
        let gbuffer_pos = create_restir_gbuffer_pos(device, px_usize);
        tracked.buffer(&gbuffer_nr);
        tracked.buffer(&gbuffer_pos);

        // --- Output + AOV targets (matching _pt_render_gpu_mesh's plumbing) ---
        let out_tex = device.create_texture(&wgpu::TextureDescriptor {
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
        });
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
        let aov_frames = AovFrames::new(device, width, height, &aovs_all);
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
        let gpu_resource_bytes = tracked.bytes() + terrain_scene.byte_size();
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
        {
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("hybrid-pt-terrain-gbuffer-enc"),
            });
            {
                let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("hybrid-pt-terrain-gbuffer-cpass"),
                    ..Default::default()
                });
                cpass.set_pipeline(&self.pipeline_terrain_gbuffer);
                cpass.set_bind_group(0, &bg0, &[]);
                cpass.set_bind_group(1, &bg1, &[]);
                cpass.set_bind_group(2, &bg_gbuffer, &[]);
                cpass.dispatch_workgroups(wg_x, wg_y, 1);
            }
            queue.submit([enc.finish()]);
        }

        // --- Accumulate until converged (windowed variance) or capped ---
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
            {
                let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("hybrid-pt-terrain-cpass"),
                    ..Default::default()
                });
                cpass.set_pipeline(&self.pipeline_terrain);
                cpass.set_bind_group(0, &bg0, &[]);
                cpass.set_bind_group(1, &bg1, &[]);
                cpass.set_bind_group(2, &bg2, &[]);
                cpass.set_bind_group(3, &bg3, &[]);
                cpass.dispatch_workgroups(wg_x, wg_y, 1);
            }
            {
                let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("hybrid-pt-restir-temporal-cpass"),
                    ..Default::default()
                });
                cpass.set_pipeline(&self.pipeline_restir_temporal);
                cpass.set_bind_group(0, &bg0, &[]);
                cpass.set_bind_group(1, &bg_empty, &[]);
                cpass.set_bind_group(2, &bg_temporal, &[]);
                cpass.dispatch_workgroups(px_workgroups, 1, 1);
            }
            {
                let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("hybrid-pt-restir-spatial-cpass"),
                    ..Default::default()
                });
                cpass.set_pipeline(&self.pipeline_restir_spatial);
                cpass.set_bind_group(0, &bg0, &[]);
                cpass.set_bind_group(1, &bg_spatial_scene, &[]);
                cpass.set_bind_group(2, &bg_spatial_reuse, &[]);
                cpass.dispatch_workgroups(px_workgroups, 1, 1);
            }
            queue.submit([enc.finish()]);
            frames += 1;

            let window_full = frames.is_multiple_of(WELFORD_WINDOW);
            if window_full || frames == desc.max_frames {
                let n_window = ((frames - 1) % WELFORD_WINDOW) + 1;
                if n_window >= 2 {
                    device.poll(wgpu::Maintain::Wait);
                    let wf = read_buffer(device, queue, &welford_buf, px_count * 8)?;
                    let wf: &[f32] = bytemuck::cast_slice(&wf);
                    let n = n_window as f32;
                    let mut vmax = 0.0f32;
                    for px in wf.chunks_exact(2) {
                        let m2 = px[1];
                        if !m2.is_finite() {
                            return Err(RenderError::Render(
                                "terrain PT produced non-finite variance (NaN in accumulation)"
                                    .into(),
                            ));
                        }
                        // Sample variance of the running-mean luminance across
                        // the last n frames.
                        vmax = vmax.max(m2 / (n - 1.0));
                    }
                    variance = vmax;
                    if frames >= desc.min_frames && variance < desc.variance_threshold {
                        converged = true;
                        break;
                    }
                }
            }
        }
        device.poll(wgpu::Maintain::Wait);
        if !converged {
            return Err(RenderError::Render(format!(
                "terrain PT did not converge: per-pixel luminance variance {variance:.3e} \
                 over the last {WELFORD_WINDOW}-frame window after {frames} frames \
                 (threshold {:.1e}); raise max_frames or simplify the scene — refusing \
                 to return a fake reference",
                desc.variance_threshold
            )));
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
        if desc.sun_elevation_deg > 0.0 && desc.sun_intensity > 0.0 && !any_valid {
            return Err(RenderError::Render(
                "terrain PT ReSTIR reuse chain produced no valid reservoirs for a sun-lit \
                 scene — temporal/spatial reuse is broken"
                    .into(),
            ));
        }

        // --- Readbacks ---
        let beauty = read_texture_pixels(device, queue, &out_tex, width, height, 8)?;
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

        Ok(TerrainReferenceOutput {
            rgba,
            albedo,
            normal,
            depth,
            frames,
            variance,
            converged,
            peak_host_visible_bytes: peak,
            minmax_pyramid_bytes: terrain_scene.pyramid.byte_size,
            gpu_resource_bytes,
        })
    }
}
