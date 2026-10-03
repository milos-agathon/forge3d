// src/path_tracing/fused_reference.rs
// SPLAT-FUSED occlusion reference: the fused scene re-rendered by the
// AEQUITAS wavefront path tracer (src/path_tracing/adjudication.rs drives the
// same scheduler) with every soft primitive replaced by the HARD surface the
// unified occlusion model binarizes to:
//
//   * a splat      -> the iso-ellipsoid where T = exp(-kappa * rho) crosses 1/2
//                     (squared Mahalanobis level 2 ln(alpha kappa / ln 2))
//   * a LiDAR point -> the sphere of radius r sqrt(1 - 1/(2 opacity)) where
//                     its coverage crosses 1/2
//   * the terrain  -> the heightfield as a triangle mesh
//
// This shares no traversal, shading or paging code with the fused
// integrator: it is triangle BVHs + next-event estimation in the wavefront
// kernels, so agreement of the two shadow masks is evidence, not tautology.
// The mask helpers at the bottom turn both renders into binary shadow masks
// and score their intersection-over-union.
// RELEVANT FILES: src/path_tracing/adjudication.rs, src/splat/kernel.rs,
//                 tests/test_splat_fusion_occlusion.rs

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use wgpu::{Device, Queue};

use crate::accel::cpu_bvh::{Aabb as CpuAabb, BuildStats, BvhCPU, BvhNode, MeshCPU};
use crate::accel::instancing::InstanceData;
use crate::core::error::RenderError;
use crate::core::resource_tracker::{
    tracked_create_buffer, tracked_create_buffer_init, TrackedBuffer,
};
use crate::path_tracing::hybrid_compute::FusedTerrainDesc;
use crate::path_tracing::lighting::{GpuAreaLight, GpuDirectionalLight};
use crate::path_tracing::reference_scene::{ReferenceEnvironmentRaw, WavefrontGpuSphere};
use crate::path_tracing::wavefront::WavefrontScheduler;
use crate::splat::bvh::{build_bvh, Aabb};
use crate::splat::fusion::FusedScene;
use crate::splat::kernel::{hard_proxy_level, sphelet_proxy_radius};
use crate::splat::num::f32_from_u32;
use crate::splat::quat_to_mat3;
use crate::splat::stream::PagePayload;

/// Material slots (and therefore albedo AOV values) of the three classes.
pub const REFERENCE_ALBEDO_TERRAIN: [f32; 3] = [0.5, 0.5, 0.5];
pub const REFERENCE_ALBEDO_SPLAT: [f32; 3] = [0.6, 0.3, 0.3];
pub const REFERENCE_ALBEDO_LIDAR: [f32; 3] = [0.3, 0.6, 0.3];

/// Receiver class of a reference pixel, recovered from the albedo AOV.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReceiverClass {
    Miss,
    Terrain,
    Splat,
    Lidar,
}

/// Description of the reference render.
pub struct FusedReferenceDesc<'a> {
    pub scene: &'a FusedScene,
    pub terrain: Option<&'a FusedTerrainDesc>,
    /// Only pages whose box intersects this region are converted to proxies
    /// (the far field of a paged scene is outside the region of interest).
    pub region: Aabb,
    pub cam_origin: [f32; 3],
    pub cam_look_at: [f32; 3],
    pub cam_up: [f32; 3],
    pub fov_y_deg: f32,
    pub sun_azimuth_deg: f32,
    pub sun_elevation_deg: f32,
    pub sun_intensity: f32,
    pub sun_color: [f32; 3],
    pub width: u32,
    pub height: u32,
    /// Accumulated frames (one camera sample per pixel per frame).
    pub spp_frames: u32,
    pub seed: u32,
}

/// Output of the reference render.
#[derive(Clone, Debug)]
pub struct FusedReferenceOutput {
    pub width: u32,
    pub height: u32,
    /// Mean linear radiance, RGB per pixel (sun only: black environment).
    pub radiance: Vec<f32>,
    /// Primary-hit albedo / normal (RGB per pixel) and depth.
    pub albedo: Vec<f32>,
    pub normal: Vec<f32>,
    pub depth: Vec<f32>,
    pub terrain_triangles: u32,
    pub splat_proxies: u32,
    pub lidar_proxies: u32,
}

/// Uniforms layout shared by the wavefront kernels (see adjudication.rs).
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
    camera_model: u32,
    full_width: u32,
    full_height: u32,
    pixel_offset_x: u32,
    pixel_offset_y: u32,
    ortho_half_height: f32,
    camera_flags: u32,
    sensor_rect: [f32; 4],
}

/// Unit icosphere with outward (counter-clockwise) winding.
fn icosphere(subdivisions: u32) -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
    let t = (1.0 + 5.0f32.sqrt()) / 2.0;
    let mut vertices: Vec<glam::Vec3> = [
        [-1.0, t, 0.0],
        [1.0, t, 0.0],
        [-1.0, -t, 0.0],
        [1.0, -t, 0.0],
        [0.0, -1.0, t],
        [0.0, 1.0, t],
        [0.0, -1.0, -t],
        [0.0, 1.0, -t],
        [t, 0.0, -1.0],
        [t, 0.0, 1.0],
        [-t, 0.0, -1.0],
        [-t, 0.0, 1.0],
    ]
    .into_iter()
    .map(|v| glam::Vec3::from(v).normalize())
    .collect();
    let mut faces: Vec<[u32; 3]> = vec![
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ];
    for _ in 0..subdivisions {
        let mut midpoints = std::collections::HashMap::new();
        let mut next = Vec::with_capacity(faces.len() * 4);
        for face in &faces {
            let mut mid = [0u32; 3];
            for edge in 0..3 {
                let (a, b) = (face[edge], face[(edge + 1) % 3]);
                let key = (a.min(b), a.max(b));
                mid[edge] = *midpoints.entry(key).or_insert_with(|| {
                    vertices
                        .push(((vertices[a as usize] + vertices[b as usize]) * 0.5).normalize());
                    (vertices.len() - 1) as u32
                });
            }
            next.push([face[0], mid[0], mid[2]]);
            next.push([face[1], mid[1], mid[0]]);
            next.push([face[2], mid[2], mid[1]]);
            next.push(mid);
        }
        faces = next;
    }
    for face in &mut faces {
        let [a, b, c] = face.map(|i| vertices[i as usize]);
        if (b - a).cross(c - a).dot(a + b + c) < 0.0 {
            face.swap(1, 2);
        }
    }
    (vertices.into_iter().map(|v| v.to_array()).collect(), faces)
}

/// Scale that puts a polyhedron's silhouette midway between its inscribed
/// and circumscribed spheres, so the proxy neither shrinks nor grows the
/// analytic shape it stands in for.
fn silhouette_scale(vertices: &[[f32; 3]], faces: &[[u32; 3]]) -> f32 {
    let inradius = faces
        .iter()
        .map(|face| {
            let [a, b, c] = face.map(|i| glam::Vec3::from(vertices[i as usize]));
            ((a + b + c) / 3.0).length()
        })
        .fold(f32::INFINITY, f32::min);
    2.0 / (1.0 + inradius)
}

struct ProxyMesh {
    vertices: Vec<[f32; 3]>,
    indices: Vec<[u32; 3]>,
    count: u32,
}

impl ProxyMesh {
    fn new() -> Self {
        Self {
            vertices: Vec::new(),
            indices: Vec::new(),
            count: 0,
        }
    }

    fn push(&mut self, template: &(Vec<[f32; 3]>, Vec<[u32; 3]>), map: impl Fn([f32; 3]) -> [f32; 3]) {
        let base = self.vertices.len() as u32;
        self.vertices.extend(template.0.iter().map(|v| map(*v)));
        self.indices
            .extend(template.1.iter().map(|f| f.map(|i| i + base)));
        self.count += 1;
    }
}

fn boxes_overlap(a: Aabb, b: Aabb) -> bool {
    (0..3).all(|axis| a.min[axis] <= b.max[axis] && a.max[axis] >= b.min[axis])
}

fn storage(device: &Device, label: &str, contents: &[u8]) -> Result<TrackedBuffer, RenderError> {
    tracked_create_buffer_init(
        device,
        &wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        },
    )
}

/// Triangle BVH with the ROOT AT NODE 0, the order the wavefront kernels
/// traverse (they start at `desc.node_offset`). `accel::cpu_bvh::build_bvh_cpu`
/// emits its nodes post-order — root last — which those kernels only handle
/// for single-leaf meshes, so the reference builds its trees here.
fn root_first_bvh(mesh: &MeshCPU) -> BvhCPU {
    let boxes: Vec<Aabb> = mesh
        .indices
        .iter()
        .map(|tri| {
            tri.iter().fold(Aabb::EMPTY, |acc, &i| {
                let v = mesh.vertices[i as usize];
                acc.union(Aabb::new(v, v))
            })
        })
        .collect();
    let bvh = build_bvh(&boxes, 4);
    let world = bvh.bounds();
    let nodes: Vec<BvhNode> = bvh
        .nodes
        .iter()
        .map(|node| {
            let aabb = CpuAabb::new(node.aabb_min, node.aabb_max);
            if node.is_leaf() {
                BvhNode::leaf(aabb, node.a, node.leaf_count())
            } else {
                BvhNode::internal(aabb, node.a, node.b)
            }
        })
        .collect();
    BvhCPU {
        build_stats: BuildStats {
            triangle_count: mesh.triangle_count(),
            node_count: nodes.len() as u32,
            max_depth: bvh.depth(),
            ..BuildStats::default()
        },
        nodes,
        tri_indices: bvh.order,
        world_aabb: CpuAabb::new(world.min, world.max),
    }
}

fn terrain_mesh(terrain: &FusedTerrainDesc) -> MeshCPU {
    let (w, h) = (terrain.width, terrain.height);
    let (sx, sz) = terrain.spacing;
    let ox = -0.5 * (f32_from_u32(w) - 1.0) * sx;
    let oz = -0.5 * (f32_from_u32(h) - 1.0) * sz;
    let mut vertices = Vec::with_capacity((w * h) as usize);
    for row in 0..h {
        for col in 0..w {
            vertices.push([
                ox + f32_from_u32(col) * sx,
                terrain.heights[(row * w + col) as usize] * terrain.exaggeration,
                oz + f32_from_u32(row) * sz,
            ]);
        }
    }
    let mut indices = Vec::with_capacity(((w - 1) * (h - 1) * 2) as usize);
    for row in 0..h - 1 {
        for col in 0..w - 1 {
            let v00 = row * w + col;
            let (v10, v01, v11) = (v00 + 1, v00 + w, v00 + w + 1);
            // Wound so cross(e1, e2) points up (+y).
            indices.push([v00, v01, v10]);
            indices.push([v10, v01, v11]);
        }
    }
    MeshCPU::new(vertices, indices)
}

/// Render the hard-proxy reference with the wavefront path tracer.
/// Certificate pass label of the reference path trace.
const REFERENCE_PASS_LABEL: &str = "fused_reference.path_trace";

pub fn render_fused_reference(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    desc: &FusedReferenceDesc,
) -> Result<FusedReferenceOutput, RenderError> {
    let (width, height) = (desc.width, desc.height);
    if width == 0 || height == 0 || desc.spp_frames == 0 {
        return Err(RenderError::Render(
            "fused reference requires non-zero width, height and frames".into(),
        ));
    }
    let params = desc.scene.params();

    // --- Hard proxies of every page in the region ---
    let ellipsoid = icosphere(1);
    let ellipsoid_scale = silhouette_scale(&ellipsoid.0, &ellipsoid.1);
    let sphere = icosphere(0);
    let sphere_scale = silhouette_scale(&sphere.0, &sphere.1);
    let lidar_radius = sphelet_proxy_radius(params.lidar_radius, params.lidar_opacity);
    let mut splat_mesh = ProxyMesh::new();
    let mut lidar_mesh = ProxyMesh::new();
    for page in 0..desc.scene.page_count() {
        if !boxes_overlap(desc.scene.page_aabb(page), desc.region) {
            continue;
        }
        match desc.scene.load_page(page)? {
            PagePayload::Splats(chunk) => {
                for i in 0..chunk.len() {
                    let Some(level) = hard_proxy_level(chunk.opacities[i], params.kappa) else {
                        continue; // too faint to ever cast a binary shadow
                    };
                    let r = quat_to_mat3(chunk.rotations[i]);
                    let axes = chunk.scales[i].map(|s| s * level.sqrt() * ellipsoid_scale);
                    let p = chunk.positions[i];
                    splat_mesh.push(&ellipsoid, |v| {
                        let local = [v[0] * axes[0], v[1] * axes[1], v[2] * axes[2]];
                        [
                            p[0] + r[0][0] * local[0] + r[0][1] * local[1] + r[0][2] * local[2],
                            p[1] + r[1][0] * local[0] + r[1][1] * local[1] + r[1][2] * local[2],
                            p[2] + r[2][0] * local[0] + r[2][1] * local[1] + r[2][2] * local[2],
                        ]
                    });
                }
            }
            PagePayload::Points { positions, .. } => {
                let Some(radius) = lidar_radius else {
                    continue;
                };
                let radius = radius * sphere_scale;
                for p in positions {
                    lidar_mesh.push(&sphere, |v| {
                        [
                            p[0] + radius * v[0],
                            p[1] + radius * v[1],
                            p[2] + radius * v[2],
                        ]
                    });
                }
            }
        }
    }

    // One BLAS per class so each instance carries its class material.
    let mut items = Vec::new();
    let mut materials = Vec::new();
    let mut terrain_triangles = 0u32;
    let mut add = |mesh: MeshCPU, albedo: [f32; 3]| {
        let bvh = root_first_bvh(&mesh);
        items.push((mesh, bvh));
        materials.push(albedo);
    };
    if let Some(terrain) = desc.terrain {
        let mesh = terrain_mesh(terrain);
        terrain_triangles = mesh.triangle_count();
        add(mesh, REFERENCE_ALBEDO_TERRAIN);
    }
    if splat_mesh.count > 0 {
        add(
            MeshCPU::new(splat_mesh.vertices, splat_mesh.indices),
            REFERENCE_ALBEDO_SPLAT,
        );
    }
    if lidar_mesh.count > 0 {
        add(
            MeshCPU::new(lidar_mesh.vertices, lidar_mesh.indices),
            REFERENCE_ALBEDO_LIDAR,
        );
    }
    if items.is_empty() {
        return Err(RenderError::Render(
            "fused reference has no geometry inside the requested region".into(),
        ));
    }

    let mut scheduler = WavefrontScheduler::new(device.clone(), queue.clone(), width, height)
        .map_err(|e| RenderError::Render(format!("wavefront scheduler init: {e}")))?;
    scheduler.set_restir_enabled(true);
    scheduler.set_restir_spatial_enabled(true);
    scheduler.set_restir_temporal_enabled(false);
    // Sun only: a black environment makes "lit by the sun" directly readable
    // from the radiance.
    scheduler.set_environment_params(&ReferenceEnvironmentRaw {
        env_ground: [0.0; 4],
        env_sky: [0.0; 4],
        miss_ground: [0.0; 4],
        miss_sky: [0.0; 4],
    });

    // Material carriers: radius-0 spheres are never intersected.
    let spheres: Vec<WavefrontGpuSphere> = materials
        .iter()
        .map(|albedo| WavefrontGpuSphere {
            center: [0.0, -1.0e6, 0.0],
            radius: 0.0,
            albedo: *albedo,
            metallic: 0.0,
            roughness: 1.0,
            ior: 1.0,
            _pad0: [0.0; 2],
            emissive: [0.0; 3],
            ax: 0.0,
            ay: 0.0,
            _pad1: [0.0; 3],
        })
        .collect();
    let spheres_buffer = storage(device, "fused-reference-materials", bytemuck::cast_slice(&spheres))?;
    let az = desc.sun_azimuth_deg.to_radians();
    let el = desc.sun_elevation_deg.to_radians();
    let toward_sun = [az.cos() * el.cos(), el.sin(), az.sin() * el.cos()];
    let dir_lights = [GpuDirectionalLight::new(
        toward_sun.map(|c| -c),
        desc.sun_intensity,
        desc.sun_color,
        1.0,
    )];
    let dir_lights_buffer = storage(device, "fused-reference-dir-lights", bytemuck::cast_slice(&dir_lights))?;
    let area_lights = [GpuAreaLight::disc(
        [0.0, -1.0e6, 0.0],
        [0.0, -1.0, 0.0],
        1.0e-6,
        0.0,
        [0.0; 3],
        0.0,
    )];
    let area_lights_buffer = storage(device, "fused-reference-area-lights", bytemuck::cast_slice(&area_lights))?;
    let (light_samples, alias_entries, light_probs) =
        crate::path_tracing::restir::build_light_samples_and_alias(device, &[], &dir_lights)?;
    scheduler.set_restir_light_data(light_samples, alias_entries);
    scheduler.set_restir_light_probs(light_probs);
    let importance = vec![1.0f32; materials.len().max(4)];
    let importance_buffer = storage(device, "fused-reference-importance", bytemuck::cast_slice(&importance))?;

    let atlas = crate::path_tracing::mesh::build_mesh_atlas(device, &items)
        .map_err(|e| RenderError::Render(format!("fused reference mesh atlas: {e}")))?;
    let identity = glam::Mat4::IDENTITY.to_cols_array();
    let instances: Vec<InstanceData> = (0..items.len() as u32)
        .map(|index| InstanceData {
            transform: identity,
            inv_transform: identity,
            blas_index: index,
            material_id: index,
            _padding: [0; 2],
        })
        .collect();
    drop(items);
    scheduler.set_instances_buffer(storage(
        device,
        "fused-reference-instances",
        bytemuck::cast_slice(&instances),
    )?);
    scheduler.set_blas_descs_buffer(atlas.descs_buffer);
    scheduler
        .init_restir_scene_spatial_bind_group(&area_lights_buffer, &dir_lights_buffer)
        .map_err(|e| RenderError::Render(format!("restir scene-spatial bind group: {e}")))?;
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

    let px_count = (width as usize) * (height as usize);
    let accum_bytes = (px_count * 16) as u64;
    let accum_buffer = tracked_create_buffer(
        device,
        &wgpu::BufferDescriptor {
            label: Some("fused-reference-accum"),
            size: accum_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        },
    )?;
    let accum_bind_group = scheduler.create_accum_bind_group(&accum_buffer);

    let origin = glam::Vec3::from(desc.cam_origin);
    let forward = (glam::Vec3::from(desc.cam_look_at) - origin).normalize();
    let right = forward.cross(glam::Vec3::from(desc.cam_up)).normalize();
    let up = right.cross(forward).normalize();
    let mut uniforms = WavefrontUniforms {
        width,
        height,
        frame_index: 0,
        spp: 1,
        cam_origin: desc.cam_origin,
        cam_fov_y: desc.fov_y_deg.to_radians(),
        cam_right: right.into(),
        cam_aspect: f32_from_u32(width) / f32_from_u32(height),
        cam_up: up.into(),
        cam_exposure: 1.0,
        cam_forward: forward.into(),
        seed_hi: desc.seed,
        seed_lo: desc.seed.wrapping_mul(0x9E37_79B9).wrapping_add(0x85EB_CA6B),
        camera_model: 0,
        full_width: width,
        full_height: height,
        pixel_offset_x: 0,
        pixel_offset_y: 0,
        ortho_half_height: 1.0,
        camera_flags: 0,
        sensor_rect: [0.0, 0.0, 1.0, 1.0],
    };
    let uniforms_buffer = tracked_create_buffer_init(
        device,
        &wgpu::util::BufferInitDescriptor {
            label: Some("fused-reference-uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        },
    )?;

    fn splitmix32(mut x: u32) -> u32 {
        x = x.wrapping_add(0x9E37_79B9);
        let mut z = x;
        z = (z ^ (z >> 16)).wrapping_mul(0x21F0_AAAD);
        z = (z ^ (z >> 15)).wrapping_mul(0x735A_2D97);
        z ^ (z >> 15)
    }
    // CENSOR F-04: frame 0's primary wavefront iteration is the timed scope
    // (a whole frame spans several encoders, see `render_frame_simple`).
    let mut timing = crate::core::gpu_timing::OneShotTiming::for_device(
        Arc::clone(device),
        Arc::clone(queue),
    );
    for frame in 0..desc.spp_frames {
        uniforms.frame_index = frame;
        uniforms.seed_hi = splitmix32(desc.seed ^ frame);
        uniforms.seed_lo = splitmix32(desc.seed.rotate_left(13) ^ frame.wrapping_mul(0x0000_9E3D));
        queue.write_buffer(&uniforms_buffer, 0, bytemuck::bytes_of(&uniforms));
        let frame_timing = if frame == 0 {
            Some((&mut timing, REFERENCE_PASS_LABEL, 1))
        } else {
            None
        };
        scheduler
            .render_frame_simple(
                &uniforms_buffer,
                &scene_bind_group,
                &accum_bind_group,
                frame_timing,
            )
            .map_err(|e| RenderError::Render(format!("fused reference frame {frame}: {e}")))?;
        if frame % 16 == 15 {
            device.poll(wgpu::Maintain::Wait);
        }
    }
    device.poll(wgpu::Maintain::Wait);
    if !timing.record_into_certificate() {
        crate::core::certificate::record_pass(REFERENCE_PASS_LABEL, 0.0, desc.spp_frames);
    }

    // --- Readbacks: accumulated radiance + primary-hit AOVs ---
    let staging = tracked_create_buffer(
        device,
        &wgpu::BufferDescriptor {
            label: Some("fused-reference-readback"),
            size: accum_bytes * 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        },
    )?;
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("fused-reference-readback"),
    });
    encoder.copy_buffer_to_buffer(&accum_buffer, 0, &staging, 0, accum_bytes);
    for (slot, source) in [
        scheduler.aov_albedo_buffer(),
        scheduler.aov_normal_buffer(),
        scheduler.aov_depth_buffer(),
    ]
    .into_iter()
    .enumerate()
    {
        encoder.copy_buffer_to_buffer(source, 0, &staging, accum_bytes * (slot as u64 + 1), accum_bytes);
    }
    queue.submit([encoder.finish()]);
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|_| RenderError::Readback("fused reference map channel closed".into()))?
        .map_err(|e| RenderError::Readback(format!("fused reference map failed: {e:?}")))?;
    let all: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&slice.get_mapped_range()).to_vec();
    staging.unmap();

    let plane = |index: usize| &all[index * px_count * 4..(index + 1) * px_count * 4];
    let inverse = 1.0 / f32_from_u32(desc.spp_frames);
    let rgb = |values: &[f32], scale: f32| -> Vec<f32> {
        values
            .chunks_exact(4)
            .flat_map(|px| [px[0] * scale, px[1] * scale, px[2] * scale])
            .collect()
    };
    let radiance = rgb(plane(0), inverse);
    if radiance.iter().any(|v| !v.is_finite()) {
        return Err(RenderError::Render(
            "fused reference produced non-finite radiance".into(),
        ));
    }
    Ok(FusedReferenceOutput {
        width,
        height,
        radiance,
        albedo: rgb(plane(1), 1.0),
        normal: rgb(plane(2), 1.0),
        depth: plane(3).chunks_exact(4).map(|px| px[0]).collect(),
        terrain_triangles,
        splat_proxies: splat_mesh.count,
        lidar_proxies: lidar_mesh.count,
    })
}

impl FusedReferenceOutput {
    /// Receiver class of pixel `i` from the albedo AOV (nearest class colour).
    pub fn class(&self, i: usize) -> ReceiverClass {
        let a = &self.albedo[3 * i..3 * i + 3];
        if self.depth[i].is_nan() || self.depth[i] <= 0.0 || a.iter().all(|c| *c == 0.0) {
            return ReceiverClass::Miss;
        }
        let distance = |c: [f32; 3]| (0..3).map(|k| (a[k] - c[k]).powi(2)).sum::<f32>();
        let candidates = [
            (distance(REFERENCE_ALBEDO_TERRAIN), ReceiverClass::Terrain),
            (distance(REFERENCE_ALBEDO_SPLAT), ReceiverClass::Splat),
            (distance(REFERENCE_ALBEDO_LIDAR), ReceiverClass::Lidar),
        ];
        candidates
            .into_iter()
            .min_by(|x, y| x.0.total_cmp(&y.0))
            .map(|(_, class)| class)
            .unwrap_or(ReceiverClass::Miss)
    }

    /// N.L of pixel `i` toward the sun (unit `toward_sun`).
    pub fn sun_cosine(&self, i: usize, toward_sun: [f32; 3]) -> f32 {
        let n = &self.normal[3 * i..3 * i + 3];
        n[0] * toward_sun[0] + n[1] * toward_sun[1] + n[2] * toward_sun[2]
    }

    /// Per-pixel sun response `luminance / (albedo luminance * N.L)`: constant
    /// (plus a little bounce light) wherever the sun reaches the surface and
    /// near zero in shadow, independent of the receiver's albedo and tilt.
    /// `None` for misses and surfaces facing away by more than `min_cosine`.
    pub fn sun_response(&self, toward_sun: [f32; 3], min_cosine: f32) -> Vec<Option<f32>> {
        let luminance = |c: &[f32]| 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
        (0..self.depth.len())
            .map(|i| {
                if self.class(i) == ReceiverClass::Miss {
                    return None;
                }
                let cosine = self.sun_cosine(i, toward_sun);
                let albedo = luminance(&self.albedo[3 * i..3 * i + 3]);
                (cosine > min_cosine && albedo > 0.0)
                    .then(|| luminance(&self.radiance[3 * i..3 * i + 3]) / (albedo * cosine))
            })
            .collect()
    }

    /// Binary shadow mask of the path-traced reference: a pixel is shadowed
    /// when its sun response is below half the lit level, where the lit
    /// level is a high percentile of the response over all facing pixels.
    /// `None` marks pixels that cannot be classified (miss / facing away).
    pub fn shadow_mask(&self, toward_sun: [f32; 3], min_cosine: f32) -> Vec<Option<bool>> {
        let response = self.sun_response(toward_sun, min_cosine);
        let mut sorted: Vec<f32> = response.iter().flatten().copied().collect();
        if sorted.is_empty() {
            return vec![None; response.len()];
        }
        sorted.sort_by(f32::total_cmp);
        let lit_level = sorted[(sorted.len() * 9 / 10).min(sorted.len() - 1)];
        response
            .into_iter()
            .map(|value| value.map(|v| v < 0.5 * lit_level))
            .collect()
    }
}

/// Intersection-over-union of two binary masks over the `valid` pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShadowIou {
    pub valid_pixels: u32,
    pub fused_shadow: u32,
    pub reference_shadow: u32,
    pub intersection: u32,
    pub union: u32,
    pub iou: f32,
}

/// Score `fused` against `reference` where `valid` is set.
pub fn shadow_iou(fused: &[bool], reference: &[bool], valid: &[bool]) -> ShadowIou {
    let mut score = ShadowIou {
        valid_pixels: 0,
        fused_shadow: 0,
        reference_shadow: 0,
        intersection: 0,
        union: 0,
        iou: 0.0,
    };
    for i in 0..valid.len() {
        if !valid[i] {
            continue;
        }
        score.valid_pixels += 1;
        score.fused_shadow += u32::from(fused[i]);
        score.reference_shadow += u32::from(reference[i]);
        score.intersection += u32::from(fused[i] && reference[i]);
        score.union += u32::from(fused[i] || reference[i]);
    }
    if score.union > 0 {
        score.iou = f32_from_u32(score.intersection) / f32_from_u32(score.union);
    }
    score
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn icosphere_is_closed_outward_and_unit() {
        for (level, vertices, faces) in [(0u32, 12usize, 20usize), (1, 42, 80)] {
            let (v, f) = icosphere(level);
            assert_eq!((v.len(), f.len()), (vertices, faces));
            let mut edges = std::collections::HashMap::new();
            for face in &f {
                let [a, b, c] = face.map(|i| glam::Vec3::from(v[i as usize]));
                assert!((a.length() - 1.0).abs() < 1e-5);
                assert!((b - a).cross(c - a).dot(a + b + c) > 0.0, "inward face");
                for edge in 0..3 {
                    let (p, q) = (face[edge], face[(edge + 1) % 3]);
                    *edges.entry((p.min(q), p.max(q))).or_insert(0u32) += 1;
                }
            }
            // Watertight: every edge is shared by exactly two faces.
            assert!(edges.values().all(|count| *count == 2));
            let scale = silhouette_scale(&v, &f);
            assert!(scale > 1.0 && scale < 1.2, "{scale}");
        }
    }

    #[test]
    fn terrain_mesh_faces_up() {
        let terrain = FusedTerrainDesc {
            heights: vec![0.0, 1.0, 0.5, 0.25, 0.0, 2.0],
            width: 3,
            height: 2,
            spacing: (2.0, 3.0),
            exaggeration: 1.5,
            albedo: [0.5; 3],
        };
        let mesh = terrain_mesh(&terrain);
        assert_eq!(mesh.triangle_count(), 4);
        assert_eq!(mesh.vertices[1], [0.0, 1.5, -1.5]);
        for tri in &mesh.indices {
            let [a, b, c] = tri.map(|i| glam::Vec3::from(mesh.vertices[i as usize]));
            assert!((b - a).cross(c - a).y > 0.0);
        }
    }

    #[test]
    fn reference_bvh_puts_the_root_first() {
        let terrain = FusedTerrainDesc {
            heights: (0..81).map(|i| ((i * 7) % 5) as f32).collect(),
            width: 9,
            height: 9,
            spacing: (1.0, 1.0),
            exaggeration: 1.0,
            albedo: [0.5; 3],
        };
        let mesh = terrain_mesh(&terrain);
        let bvh = root_first_bvh(&mesh);
        assert!(bvh.nodes.len() > 1);
        // Node 0 bounds the whole mesh and is an interior node.
        assert_eq!(bvh.nodes[0].aabb_min, bvh.world_aabb.min);
        assert_eq!(bvh.nodes[0].aabb_max, bvh.world_aabb.max);
        assert!(bvh.nodes[0].flags & 1 == 0);
        // Leaves cover every triangle exactly once.
        let mut seen = vec![false; mesh.indices.len()];
        for node in bvh.nodes.iter().filter(|node| node.flags & 1 != 0) {
            for k in 0..node.right {
                let tri = bvh.tri_indices[(node.left + k) as usize] as usize;
                assert!(!seen[tri]);
                seen[tri] = true;
            }
        }
        assert!(seen.iter().all(|s| *s));
    }

    #[test]
    fn iou_counts_only_valid_pixels() {
        let fused = [true, true, false, false, true];
        let reference = [true, false, false, true, true];
        let valid = [true, true, true, true, false];
        let score = shadow_iou(&fused, &reference, &valid);
        assert_eq!(score.valid_pixels, 4);
        assert_eq!((score.intersection, score.union), (1, 3));
        assert!((score.iou - 1.0 / 3.0).abs() < 1e-6);
        assert_eq!(shadow_iou(&[false], &[false], &[true]).iou, 0.0);
    }

    #[test]
    fn reference_mask_separates_lit_from_shadowed_response() {
        // Two terrain pixels facing the sun (one lit, one dark), one facing
        // away, one miss.
        let out = FusedReferenceOutput {
            width: 4,
            height: 1,
            radiance: vec![0.8, 0.8, 0.8, 0.05, 0.05, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            albedo: [REFERENCE_ALBEDO_TERRAIN; 3]
                .into_iter()
                .flatten()
                .chain([0.0; 3])
                .collect(),
            normal: vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
            depth: vec![1.0, 1.0, 1.0, 0.0],
            terrain_triangles: 2,
            splat_proxies: 0,
            lidar_proxies: 0,
        };
        assert_eq!(out.class(0), ReceiverClass::Terrain);
        assert_eq!(out.class(3), ReceiverClass::Miss);
        let mask = out.shadow_mask([0.0, 1.0, 0.0], 0.15);
        assert_eq!(mask, vec![Some(false), Some(true), None, None]);
    }
}
