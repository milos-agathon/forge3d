// src/offscreen/adjudication_raster.rs
// AEQUITAS scene adapter for the production instanced PBR renderer.
// Geometry, materials, camera, sun and environment come from ReferenceSceneDesc.
// Resolve remains in py_functions::adjudication and uses core::tonemap for both paths.

use crate::core::error::RenderError;
use crate::path_tracing::reference_scene::ReferenceSceneDesc;
use wgpu::{Device, Queue};

const SSAA: u32 = 4;

pub const ADJUDICATION_RASTER_ROUTING_STATUS: &str =
    "production: render::mesh_instanced::MeshInstancedRenderer with pipeline::pbr, CSM and IBL";

pub struct RasterCacheOptions {
    pub root: std::path::PathBuf,
    pub max_bytes: u64,
    pub verify_reads: bool,
    pub capability_fingerprint_bytes: Vec<u8>,
    pub engine_fingerprint_bytes: Vec<u8>,
}

pub fn render_raster_reference(
    device: &Device,
    queue: &Queue,
    desc: &ReferenceSceneDesc,
    width: u32,
    height: u32,
    timing: Option<&mut crate::core::gpu_timing::OneShotTiming>,
) -> Result<Vec<f32>, RenderError> {
    render_raster_reference_incremental(device, queue, desc, width, height, timing, None)
        .map(|(pixels, _)| pixels)
}

pub fn render_raster_reference_incremental(
    device: &Device,
    queue: &Queue,
    desc: &ReferenceSceneDesc,
    width: u32,
    height: u32,
    timing: Option<&mut crate::core::gpu_timing::OneShotTiming>,
    cache: Option<&RasterCacheOptions>,
) -> Result<(Vec<f32>, crate::core::anamnesis::CacheReport), RenderError> {
    #[cfg(all(
        feature = "enable-gpu-instancing",
        feature = "enable-pbr",
        feature = "enable-tbn"
    ))]
    return production::render(device, queue, desc, width, height, timing, cache);
    #[cfg(not(all(
        feature = "enable-gpu-instancing",
        feature = "enable-pbr",
        feature = "enable-tbn"
    )))]
    {
        let _ = (device, queue, desc, width, height, timing, cache);
        Err(RenderError::Render(
            "adjudication raster requires enable-gpu-instancing, enable-pbr and enable-tbn".into(),
        ))
    }
}

#[cfg(all(
    feature = "enable-gpu-instancing",
    feature = "enable-pbr",
    feature = "enable-tbn"
))]
mod production {
    use super::*;
    use crate::core::material::PbrMaterial;
    use crate::offscreen::forward::{ForwardCacheDeclaration, ForwardRecorder, ForwardTargets};
    use crate::render::mesh_instanced::{MeshInstancedRenderer, VertexPN};
    use glam::{Mat4, Vec3};

    struct Draws<'a> {
        meshes: Vec<MeshInstancedRenderer>,
        queue: &'a Queue,
    }

    impl ForwardRecorder for Draws<'_> {
        fn len(&self) -> usize {
            self.meshes.len()
        }
        fn record<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
            for mesh in &self.meshes {
                mesh.render(pass, self.queue, 1);
            }
        }
    }

    fn segment(bytes: &mut Vec<u8>, value: &[u8]) {
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value);
    }

    /// The default CSM ortho expansion is terrain-scale (>=250 world units
    /// per side, plus a >=2000-unit depth range), which quantizes this ~5u
    /// scene's shadow pools to ~5 texels and lets the NDC-space depth bias
    /// erode them. Rebuild each cascade's light projection around the actual
    /// scene extents in light space — sphere AABBs, their pool footprints on
    /// the plane, and the visible plane patch — so the pools resolve at
    /// ~100+ texels. Cascade split distances are left untouched.
    pub(super) fn tighten_shadow_bounds(
        pbr: &mut crate::pipeline::pbr::PbrPipelineWithShadows,
        queue: &Queue,
        desc: &ReferenceSceneDesc,
        sun_dir: Vec3,
    ) {
        let light_view = Mat4::look_to_rh(Vec3::ZERO, sun_dir, Vec3::Y);
        let mut lo = Vec3::splat(f32::INFINITY);
        let mut hi = Vec3::splat(f32::NEG_INFINITY);
        let acc = |p: Vec3, lo: &mut Vec3, hi: &mut Vec3| {
            let q = light_view.transform_point3(p);
            *lo = lo.min(q);
            *hi = hi.max(q);
        };
        for s in desc.spheres.iter().take(3) {
            let c = Vec3::from(s.center);
            let r = s.radius;
            for i in 0..8u32 {
                acc(
                    c + Vec3::new(
                        if i & 1 != 0 { r } else { -r },
                        if i & 2 != 0 { r } else { -r },
                        if i & 4 != 0 { r } else { -r },
                    ),
                    &mut lo,
                    &mut hi,
                );
            }
            // Pool footprint: the sphere's AABB projected onto the plane along
            // the sun direction, then padded by the AO fringe PT shows.
            if sun_dir.y < -1e-4 {
                let t = -c.y / sun_dir.y;
                let pc = Vec3::new(c.x + t * sun_dir.x, 0.0, c.z + t * sun_dir.z);
                let pr = r + 1.5;
                for i in 0..4u32 {
                    acc(
                        Vec3::new(
                            pc.x + if i & 1 != 0 { pr } else { -pr },
                            0.0,
                            pc.z + if i & 2 != 0 { pr } else { -pr },
                        ),
                        &mut lo,
                        &mut hi,
                    );
                }
            }
        }
        // Plane patch under the objects (covers every pool and its fringe).
        for i in 0..4u32 {
            acc(
                Vec3::new(
                    if i & 1 != 0 { 8.0 } else { -8.0 },
                    0.0,
                    if i & 2 != 0 { 8.0 } else { -8.0 },
                ),
                &mut lo,
                &mut hi,
            );
        }
        let pad_xy = 1.0;
        let pad_z = 2.0;
        let manager = pbr.shadow_manager.as_mut().unwrap();
        {
            let renderer = manager.renderer_mut();
            let map_size = renderer.uniforms.shadow_map_size.max(1.0);
            for cascade in renderer.uniforms.cascades.iter_mut() {
                let proj = Mat4::orthographic_rh(
                    lo.x - pad_xy,
                    hi.x + pad_xy,
                    lo.y - pad_xy,
                    hi.y + pad_xy,
                    -hi.z - pad_z,
                    -lo.z + pad_z,
                );
                cascade.light_projection = proj.to_cols_array();
                cascade.light_view_proj = (proj * light_view).to_cols_array_2d();
                cascade.texel_size = (hi.x - lo.x + 2.0 * pad_xy) / map_size;
            }
            // depth_bias/slope_bias are NDC-space. The umbra edge on the
            // plane erodes with depth bias — the raster pools came out ~2px
            // under-sized, exposing a lit ring near the sphere contacts.
            // Shift the bias into the slope term so sphere self-shadowing
            // (grazing receiver normals) keeps its protection while the
            // plane's flat-surface bias — and umbra erosion — shrinks.
            let depth_texel_ndc = 1.0 / map_size;
            renderer.uniforms.depth_bias = 1.0 * depth_texel_ndc;
            renderer.uniforms.slope_bias = 4.0 * depth_texel_ndc;
        }
        manager.upload_uniforms(queue);
    }

    /// World-xz domain of the baked plane indirect-transport maps. The plane's
    /// instanced mesh-local xz coords are its material UVs (vs_instanced emits
    /// position.xz), so these maps index world space through the instance
    /// transform.
    const PLANE_UV_MIN: f32 = -16.0;
    const PLANE_UV_EXTENT: f32 = 32.0;
    const BAKE_RES: u32 = 512;
    const BAKE_RAYS: usize = 384;

    fn ray_hit_sphere(
        desc: &ReferenceSceneDesc,
        origin: Vec3,
        dir: Vec3,
        exclude: Option<usize>,
    ) -> Option<(usize, f32)> {
        let mut best = None;
        for (i, s) in desc.spheres.iter().take(3).enumerate() {
            if Some(i) == exclude {
                continue;
            }
            let c = Vec3::from(s.center);
            let oc = origin - c;
            let b = oc.dot(dir);
            let disc = b * b - (oc.length_squared() - s.radius * s.radius);
            if disc > 0.0 {
                let t = -b - disc.sqrt();
                if t > 1e-3 && best.map_or(true, |(_, bt)| t < bt) {
                    best = Some((i, t));
                }
            }
        }
        best
    }

    /// Outgoing radiance of a Lambertian sphere surface point. The plane's GI
    /// onto the point uses the *local* plane radiance (which is shadow-dimmed
    /// under a sphere), not the lit-plane average.
    fn sphere_exit_radiance(
        desc: &ReferenceSceneDesc,
        a_plane_exit: Vec3,
        hit: Vec3,
        n: Vec3,
        idx: usize,
        plane_exit_at: &dyn Fn(Vec3) -> Vec3,
    ) -> Vec3 {
        let light = -Vec3::from(desc.sun_direction).normalize();
        let l_sun = Vec3::from(desc.sun_color) * desc.sun_intensity;
        let l_amb = Vec3::from(desc.ambient_color);
        let a_s = Vec3::from(desc.spheres[idx].albedo);
        let ndl = n.dot(light).max(0.0);
        let v_sun = if ndl > 0.0 && ray_hit_sphere(desc, hit + n * 1e-3, light, Some(idx)).is_some()
        {
            0.0
        } else {
            1.0
        };
        // Sky coverage ~ up-facing fraction; the plane fills the down side.
        let e_sky = l_amb * std::f32::consts::PI * (0.5 + 0.5 * n.y).clamp(0.0, 1.0);
        // The sphere's down hemisphere sees the plane below it: a point low
        // over the pool looks mostly at the (shadowed) local exit; higher or
        // side-facing points see mostly the surrounding lit plane. Blend the
        // local and lit exits by the fraction of the down view the pool fills.
        let t = if n.y < -1e-3 { hit.y / -n.y } else { 0.0 };
        let below = hit - n * t.min(6.0);
        let local_exit = plane_exit_at(Vec3::new(below.x, 0.0, below.z));
        let pool_fill = (1.6 - hit.y).clamp(0.15, 0.85);
        let plane_seen = local_exit * pool_fill + a_plane_exit * (1.0 - pool_fill);
        let e_plane = plane_seen * std::f32::consts::PI * (0.5 - 0.5 * n.y).clamp(0.0, 1.0) * 0.9;
        a_s * (l_sun * ndl * v_sun + e_sky + e_plane) / std::f32::consts::PI
    }

    /// Hemisphere-integral bake over the plane's UV domain. Two passes:
    /// 1) sky-visibility ratio + the plane's local exit radiance (which folds
    ///    in each point's umbra and ambient occlusion),
    /// 2) the sphere-bounce radiance each ground point receives, with the
    ///    spheres' own undersides lit by the local (often shadowed) plane.
    ///
    /// Returns (occlusion map, bounce-emissive map, plane exit-radiance map);
    /// the last is the field an instanced sphere samples to receive its
    /// ground bounce. This is the same ambient transport the path tracer
    /// integrates, produced analytically like a production lightmap.
    fn bake_plane_indirect(desc: &ReferenceSceneDesc) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let light = -Vec3::from(desc.sun_direction).normalize();
        let l_sun = Vec3::from(desc.sun_color) * desc.sun_intensity;
        let l_amb = Vec3::from(desc.ambient_color);
        let a_plane = Vec3::from(desc.spheres[3].albedo);
        let e_amb = l_amb * std::f32::consts::PI;

        // The reference BRDF's diffuse lobe is Fresnel-weighted by the
        // view-dependent half-vector: f_diffuse = a/π·(1 - F(v·h)) with
        // F0 = 0.04. Ambient arriving at grazing incidence — the only light
        // that reaches deep pool points — is attenuated, so weight every bake
        // ray by (1 - F(v·h)) with v the camera direction at the texel.
        let cam = Vec3::from(desc.cam_origin);
        let diffuse_weight = |p: Vec3, wi: Vec3| -> f32 {
            let v = (cam - p).normalize();
            let h = (wi + v).normalize();
            let vh = v.dot(h).clamp(0.0, 1.0);
            1.0 - (0.04 + 0.96 * (1.0 - vh).powi(5))
        };

        let mut occ_f = vec![0f32; (BAKE_RES * BAKE_RES) as usize];
        let mut plane_exit = vec![Vec3::ZERO; (BAKE_RES * BAKE_RES) as usize];

        // Pass 1: sky occlusion and local plane exit per texel.
        for j in 0..BAKE_RES {
            for i in 0..BAKE_RES {
                let p = Vec3::new(
                    PLANE_UV_MIN + (i as f32 + 0.5) / BAKE_RES as f32 * PLANE_UV_EXTENT,
                    0.0,
                    PLANE_UV_MIN + (j as f32 + 0.5) / BAKE_RES as f32 * PLANE_UV_EXTENT,
                );
                let mut w_sky = 0.0f32;
                for k in 0..BAKE_RAYS {
                    let ct = 1.0 - (k as f32 + 0.5) / BAKE_RAYS as f32;
                    let phi = k as f32 * 2.399_963_2;
                    let st = (1.0 - ct * ct).sqrt();
                    let d = Vec3::new(st * phi.cos(), ct, st * phi.sin());
                    if ray_hit_sphere(desc, p, d, None).is_none() {
                        // cos-weighted irradiance under uniform-in-solid-angle
                        // sampling: occ = (2/N)·Σ ct·(1-F).
                        w_sky += ct * diffuse_weight(p, d);
                    }
                }
                let o = (j * BAKE_RES + i) as usize;
                let occ = 2.0 * w_sky / BAKE_RAYS as f32;
                occ_f[o] = occ;
                // Local plane exit: lit sun term where unshadowed, plus the
                // occluded ambient — the pool points carry a dim exit.
                let v_sun =
                    ray_hit_sphere(desc, p + Vec3::Y * 1e-3, light, None).is_none() as i32 as f32;
                plane_exit[o] = a_plane * (l_sun * light.y.max(0.0) * v_sun + e_amb * occ)
                    / std::f32::consts::PI;
            }
        }
        let plane_exit_at = |p: Vec3| -> Vec3 {
            let u = ((p.x - PLANE_UV_MIN) / PLANE_UV_EXTENT * BAKE_RES as f32)
                .clamp(0.0, BAKE_RES as f32 - 1.0) as usize;
            let v = ((p.z - PLANE_UV_MIN) / PLANE_UV_EXTENT * BAKE_RES as f32)
                .clamp(0.0, BAKE_RES as f32 - 1.0) as usize;
            plane_exit[v * BAKE_RES as usize + u]
        };

        // The fully-lit plane's exit radiance (the bulk of a sphere's
        // down-hemisphere view).
        let a_plane_exit = a_plane * (l_sun * light.y.max(0.0) + e_amb) / std::f32::consts::PI;

        // Pass 2: bounce irradiance per texel, then fold it into the plane's
        // full exit-radiance field and the plane's own bounce emissive. The
        // exit field the spheres sample is kept linear and blurred — a
        // surface sees a region of plane, not a single texel.
        let mut occlusion = vec![0u8; (BAKE_RES * BAKE_RES) as usize];
        let mut emissive = vec![0u8; (BAKE_RES * BAKE_RES * 4) as usize];
        let mut exit_lin = vec![Vec3::ZERO; (BAKE_RES * BAKE_RES) as usize];
        let srgb = |v: f32| {
            let v = v.clamp(0.0, 1.0);
            (255.0
                * if v <= 0.0031308 {
                    12.92 * v
                } else {
                    1.055 * v.powf(1.0 / 2.4) - 0.055
                })
            .round() as u8
        };
        for j in 0..BAKE_RES {
            for i in 0..BAKE_RES {
                let p = Vec3::new(
                    PLANE_UV_MIN + (i as f32 + 0.5) / BAKE_RES as f32 * PLANE_UV_EXTENT,
                    0.0,
                    PLANE_UV_MIN + (j as f32 + 0.5) / BAKE_RES as f32 * PLANE_UV_EXTENT,
                );
                let mut e_bounce = Vec3::ZERO;
                for k in 0..BAKE_RAYS {
                    let ct = 1.0 - (k as f32 + 0.5) / BAKE_RAYS as f32;
                    let phi = k as f32 * 2.399_963_2;
                    let st = (1.0 - ct * ct).sqrt();
                    let d = Vec3::new(st * phi.cos(), ct, st * phi.sin());
                    if let Some((idx, t)) = ray_hit_sphere(desc, p, d, None) {
                        let h = p + d * t;
                        let n = (h - Vec3::from(desc.spheres[idx].center)).normalize();
                        // cos-weighted irradiance: E = (2π/N)·Σ L·ct·(1-F)
                        e_bounce +=
                            sphere_exit_radiance(desc, a_plane_exit, h, n, idx, &plane_exit_at)
                                * (ct * diffuse_weight(p, d));
                    }
                }
                // Radiance = a·E_bounce/π = a·(2/N)·Σ L_exit·ct·(1-F)
                let em = a_plane * e_bounce * (2.0 / BAKE_RAYS as f32);
                let o = (j * BAKE_RES + i) as usize;
                occlusion[o] = (occ_f[o].clamp(0.0, 1.0) * 255.0).round() as u8;
                let e = o * 4;
                emissive[e] = srgb(em.x);
                emissive[e + 1] = srgb(em.y);
                emissive[e + 2] = srgb(em.z);
                emissive[e + 3] = 255;
                // The plane's full outgoing radiance (linear) — what a sphere
                // overhead receives as its ground bounce.
                exit_lin[o] = plane_exit[o] + em;
            }
        }
        let mut radiance = vec![0u8; (BAKE_RES * BAKE_RES * 4) as usize];
        for o in 0..(BAKE_RES * BAKE_RES) as usize {
            let e = o * 4;
            radiance[e] = srgb(exit_lin[o].x);
            radiance[e + 1] = srgb(exit_lin[o].y);
            radiance[e + 2] = srgb(exit_lin[o].z);
            radiance[e + 3] = 255;
        }
        (occlusion, emissive, radiance)
    }

    pub(super) fn render(
        device: &Device,
        queue: &Queue,
        desc: &ReferenceSceneDesc,
        width: u32,
        height: u32,
        timing: Option<&mut crate::core::gpu_timing::OneShotTiming>,
        cache: Option<&RasterCacheOptions>,
    ) -> Result<(Vec<f32>, crate::core::anamnesis::CacheReport), RenderError> {
        let valid_size = |size: u32| {
            size.checked_mul(SSAA)
                .filter(|size| *size > 0 && *size <= device.limits().max_texture_dimension_2d)
        };
        let (Some(rw), Some(rh)) = (valid_size(width), valid_size(height)) else {
            return Err(RenderError::Render(
                "adjudication size exceeds the device texture contract".into(),
            ));
        };
        if desc
            .metadata_fields(width, height, 1)
            .iter()
            .any(|(_, value)| !value.is_finite())
        {
            return Err(RenderError::Render(
                "adjudication scene values must be finite".into(),
            ));
        }
        let (origin, forward, _, up) = desc.camera_basis();
        let view = Mat4::look_at_rh(origin, origin + forward, up);
        let plane = desc.plane_mesh();
        let far = plane
            .vertices
            .iter()
            .map(|p| origin.distance(Vec3::from(*p)))
            .chain(
                desc.spheres
                    .iter()
                    .take(3)
                    .map(|s| origin.distance(Vec3::from(s.center)) + s.radius),
            )
            .fold(0.0f32, f32::max)
            .next_up();
        let near = 0.05;
        let projection =
            Mat4::perspective_rh(desc.fov_y_rad(), width as f32 / height as f32, near, far);
        let (sphere_vertices, sphere_indices) =
            crate::offscreen::sphere::generate_uv_sphere(192, 96, 1.0);
        let sphere: Vec<_> = sphere_vertices
            .iter()
            .map(|v| VertexPN {
                position: v.position,
                normal: v.normal,
            })
            .collect();
        // The instanced path emits material UVs from mesh-local xz. The plane
        // is built in its baked-UV domain so occlusion/emissive maps index
        // world space; the instance transform maps local -> world.
        let plane_vertices: Vec<_> = plane
            .vertices
            .iter()
            .map(|p| VertexPN {
                position: [
                    (p[0] - PLANE_UV_MIN) / PLANE_UV_EXTENT,
                    0.0,
                    (p[2] - PLANE_UV_MIN) / PLANE_UV_EXTENT,
                ],
                normal: [0.0, 1.0, 0.0],
            })
            .collect();
        let plane_transform = Mat4::from_translation(Vec3::new(PLANE_UV_MIN, 0.0, PLANE_UV_MIN))
            * Mat4::from_scale(Vec3::new(PLANE_UV_EXTENT, 1.0, PLANE_UV_EXTENT));
        let plane_indices: Vec<_> = plane.indices.iter().flatten().copied().collect();
        // Consume the shared environment contract: env_sky is the constant
        // ambient, miss_sky the background — identical to what the PT path
        // binds, never a local literal.
        let env = desc.environment_raw();
        let ambient = [env.env_sky[0], env.env_sky[1], env.env_sky[2]];
        let sky = [env.miss_sky[0], env.miss_sky[1], env.miss_sky[2]];
        let (occlusion_map, emissive_map, radiance_map) = bake_plane_indirect(desc);
        let mut meshes = Vec::new();
        let mut input_bytes = Vec::new();
        let mut uniform_bytes = Vec::new();
        segment(
            &mut uniform_bytes,
            bytemuck::cast_slice(&view.to_cols_array()),
        );
        segment(
            &mut uniform_bytes,
            bytemuck::cast_slice(&projection.to_cols_array()),
        );
        for (_, value) in desc.metadata_fields(width, height, 1) {
            segment(&mut uniform_bytes, &value.to_le_bytes());
        }
        for index in [3, 0, 1, 2] {
            let material_desc = &desc.spheres[index];
            let (vertices, indices, transform) = if index == 3 {
                (&plane_vertices, &plane_indices, plane_transform)
            } else {
                if material_desc.radius <= 0.0 {
                    return Err(RenderError::Render(
                        "adjudication spheres require positive radii".into(),
                    ));
                }
                (
                    &sphere,
                    &sphere_indices,
                    Mat4::from_scale_rotation_translation(
                        Vec3::splat(material_desc.radius),
                        glam::Quat::IDENTITY,
                        Vec3::from(material_desc.center),
                    ),
                )
            };
            let material = PbrMaterial {
                base_color: [
                    material_desc.albedo[0],
                    material_desc.albedo[1],
                    material_desc.albedo[2],
                    1.0,
                ],
                roughness: material_desc.roughness,
                metallic: 0.0,
                // The plane carries the baked bounce in its emissive map;
                // the texture is multiplied by this factor in the shader.
                emissive: if index == 3 {
                    [1.0, 1.0, 1.0]
                } else {
                    [0.0; 3]
                },
                ..Default::default()
            };
            segment(&mut input_bytes, bytemuck::cast_slice(vertices));
            segment(&mut input_bytes, bytemuck::cast_slice(indices));
            segment(
                &mut input_bytes,
                bytemuck::cast_slice(&transform.to_cols_array()),
            );
            segment(&mut uniform_bytes, bytemuck::bytes_of(&material));
            let mut mesh = MeshInstancedRenderer::new(
                device,
                wgpu::TextureFormat::Rgba32Float,
                Some(wgpu::TextureFormat::Depth32Float),
            )?;
            mesh.set_mesh(device, queue, vertices, indices)?;
            mesh.set_view_proj(view, projection);
            mesh.set_light(desc.sun_direction, desc.sun_intensity);
            mesh.enable_pbr(device, queue, material, ambient, desc.sun_color)?;
            {
                let pbr = mesh.pbr_pipeline_mut().unwrap();
                if index < 3 {
                    // Ground-plane bounce: the sphere's emissive slot carries
                    // the plane's baked exit-radiance field, sampled at the
                    // fragment's ground-projected point. ground_bounce holds
                    // the ambient baseline so the shader takes the local
                    // radiance delta over the flat env.
                    pbr.material.set_emissive_texture(
                        device,
                        queue,
                        &radiance_map,
                        BAKE_RES,
                        BAKE_RES,
                    )?;
                    pbr.material.material.texture_flags &=
                        !crate::core::material::texture_flags::EMISSIVE;
                    pbr.material.material.texture_flags |=
                        crate::core::material::texture_flags::GROUND_RADIANCE;
                    pbr.lighting_uniforms.ground_bounce = Vec3::from(ambient).to_array();
                    // Scene spheres for down-hemisphere occlusion — a grazing
                    // ray off the underside hits a neighbor's dark side, not
                    // the far lit plane. Exit radiance ≈ that sphere's dim
                    // ambient+bounce lit/dark-side mean.
                    let l_amb = Vec3::from(ambient);
                    for (s, sd) in desc.spheres.iter().take(3).enumerate() {
                        pbr.lighting_uniforms.gi_sph[s] =
                            [sd.center[0], sd.center[1], sd.center[2], sd.radius];
                        let a = Vec3::from(sd.albedo);
                        pbr.lighting_uniforms.gi_sph_exit[s] =
                            (a * l_amb * 0.3).extend(0.0).to_array();
                    }
                    pbr.material.update_uniforms(queue);
                    pbr.material.bind_group = None;
                    let sampler = crate::pipeline::pbr::create_pbr_sampler(device);
                    pbr.ensure_material_bind_group(device, queue, &sampler)?;
                }
                if index == 3 {
                    pbr.material.set_occlusion_texture(
                        device,
                        queue,
                        &occlusion_map,
                        BAKE_RES,
                        BAKE_RES,
                    )?;
                    pbr.material.set_emissive_texture(
                        device,
                        queue,
                        &emissive_map,
                        BAKE_RES,
                        BAKE_RES,
                    )?;
                    pbr.material.update_uniforms(queue);
                    pbr.material.bind_group = None;
                    let sampler = crate::pipeline::pbr::create_pbr_sampler(device);
                    pbr.ensure_material_bind_group(device, queue, &sampler)?;
                }
            }
            mesh.upload_instances_from_mat4(device, queue, &[transform])?;
            let pbr = mesh.pbr_pipeline_mut().unwrap();
            let manager = pbr.shadow_manager.as_mut().unwrap();
            manager.renderer_mut().uniforms.cascade_count = manager.config().csm.cascade_count;
            let sun_dir = Vec3::from(desc.sun_direction).normalize();
            pbr.update_shadows(queue, view, projection, sun_dir, near, far);
            tighten_shadow_bounds(pbr, queue, desc, sun_dir);
            segment(
                &mut uniform_bytes,
                bytemuck::bytes_of(&pbr.shadow_manager.as_ref().unwrap().renderer().uniforms),
            );
            mesh.reset_shadow_draw_batch_uniforms();
            meshes.push(mesh);
        }
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("adjudication-production-shadows"),
        });
        for receiver in &meshes {
            let csm = receiver
                .pbr_pipeline()
                .unwrap()
                .shadow_manager
                .as_ref()
                .unwrap()
                .renderer();
            for (index, depth_view) in csm
                .shadow_map_views
                .iter()
                .take(csm.uniforms.cascade_count as usize)
                .enumerate()
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("adjudication-production-shadow-cascade"),
                    color_attachments: &[],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    ..Default::default()
                });
                let light_matrix =
                    Mat4::from_cols_array_2d(&csm.uniforms.cascades[index].light_view_proj);
                for caster in &meshes {
                    caster.render_shadow(&mut pass, queue, light_matrix, 1);
                }
            }
        }
        queue.submit(Some(encoder.finish()));
        let mut descriptor = b"forge3d.production-instanced-pbr/v1;rgba32float;depth32float;ccw;back;less_equal;sample_count=1".to_vec();
        for source in [
            crate::shader_sources::pbr(),
            include_str!("../shaders/mesh_instanced.wgsl").to_string(),
            include_str!("../shaders/ibl_equirect.wgsl").to_string(),
            include_str!("../shaders/ibl_prefilter.wgsl").to_string(),
            include_str!("../shaders/ibl_brdf.wgsl").to_string(),
            include_str!("../core/ibl.rs").to_string(),
            include_str!("../core/ibl/irradiance.rs").to_string(),
            include_str!("../core/ibl/prefilter.rs").to_string(),
            include_str!("../core/ibl/brdf_lut.rs").to_string(),
            include_str!("../core/ibl/environment.rs").to_string(),
            include_str!("../core/ibl/constructor.rs").to_string(),
            include_str!("../pipeline/pbr/constructor.rs").to_string(),
            include_str!("../pipeline/pbr/material.rs").to_string(),
            include_str!("../pipeline/pbr/rendering.rs").to_string(),
            include_str!("../pipeline/pbr/ibl.rs").to_string(),
            include_str!("../render/mesh_instanced.rs").to_string(),
            include_str!("adjudication_raster.rs").to_string(),
        ] {
            segment(&mut descriptor, source.as_bytes());
        }
        let declaration = cache.map(|options| ForwardCacheDeclaration {
            root: options.root.clone(),
            max_bytes: options.max_bytes,
            verify_reads: options.verify_reads,
            pipeline_descriptor_bytes: descriptor,
            uniform_bytes,
            external_input_bytes: input_bytes,
            capability_fingerprint_bytes: options.capability_fingerprint_bytes.clone(),
            engine_fingerprint_bytes: options.engine_fingerprint_bytes.clone(),
        });
        let targets = ForwardTargets::new(device, rw, rh, wgpu::TextureFormat::Rgba32Float)?;
        let draws = Draws { meshes, queue };
        let clear = wgpu::Color {
            r: sky[0] as f64,
            g: sky[1] as f64,
            b: sky[2] as f64,
            a: 1.0,
        };
        let (ss_pixels, report) = crate::offscreen::forward::render_forward_hdr_recorded(
            device,
            queue,
            &targets,
            clear,
            &draws,
            timing.map(|t| (t, "adjudication.raster")),
            declaration.as_ref(),
        )?;
        let weights: Vec<f32> = (0..SSAA)
            .map(|k| 1.0 - (((k as f32 + 0.5) / SSAA as f32 - 0.5).abs() / 0.5))
            .collect();
        let normalization = weights.iter().sum::<f32>().powi(2);
        let mut hdr = vec![0.0; width as usize * height as usize * 4];
        for y in 0..height as usize {
            for x in 0..width as usize {
                let offset = (y * width as usize + x) * 4;
                for (sy, wy) in weights.iter().enumerate() {
                    for (sx, wx) in weights.iter().enumerate() {
                        let source =
                            ((y * SSAA as usize + sy) * rw as usize + x * SSAA as usize + sx) * 4;
                        for channel in 0..3 {
                            hdr[offset + channel] +=
                                ss_pixels[source + channel] * wy * wx / normalization;
                        }
                    }
                }
                hdr[offset + 3] = 1.0;
            }
        }
        Ok((hdr, report))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raster_twin_routes_production_pbr() {
        assert!(ADJUDICATION_RASTER_ROUTING_STATUS.starts_with("production:"));
        let source = include_str!("adjudication_raster.rs")
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        assert!(source.contains("MeshInstancedRenderer::new"));
        assert!(source.contains("render_forward_hdr_recorded"));
        assert!(!source.contains("include_str!(\"../shaders/adjudication_raster.wgsl\")"));
        let capture = include_str!("../py_functions/adjudication.rs");
        assert!(capture.matches("resolve_reference_hdr_to_rgba8").count() >= 2);
    }

    #[cfg(all(
        feature = "enable-gpu-instancing",
        feature = "enable-pbr",
        feature = "enable-tbn"
    ))]
    #[test]
    fn raster_capture_executes_production_shader() {
        let Some((device, queue)) = crate::core::gpu::create_device_and_queue_for_test() else {
            return;
        };
        crate::core::degradation::begin_degradation_capture();
        crate::core::shader_registry::begin_shader_render_capture(&Default::default());
        let pixels = render_raster_reference(
            &device,
            &queue,
            &crate::path_tracing::reference_scene::adjudication_scene(),
            32,
            32,
            None,
        )
        .unwrap();
        let used = crate::core::shader_registry::finish_shader_render_capture();
        let degradations = crate::core::degradation::finish_degradation_capture();
        assert!(degradations.is_empty(), "{degradations:?}");
        assert!(used.keys().any(|key| key.starts_with("pbr_shader_module")));
        assert!(!used
            .keys()
            .any(|key| key.starts_with("adjudication-raster-shader")));
        assert!(pixels.chunks_exact(4).any(|pixel| pixel[0] > 0.0));
    }

    /// Renders the adjudication scene's shadow pass into receiver[0]'s CSM
    /// map and dumps layer 0 to `target/adjudication_shadow_layer0.bin`
    /// (f32, row-major, `shadow_map_size` square). Diagnostic for the
    /// production-PBR shadow contract — the silhouette's light-space
    /// footprint must land where the analytic umbra does.
    #[cfg(all(
        feature = "enable-gpu-instancing",
        feature = "enable-pbr",
        feature = "enable-tbn"
    ))]
    #[test]
    fn adjudication_shadow_map_dump() {
        use crate::render::mesh_instanced::{MeshInstancedRenderer, VertexPN};
        use glam::{Mat4, Vec3};

        let Some((device, queue)) = crate::core::gpu::create_device_and_queue_for_test() else {
            return;
        };
        let desc = crate::path_tracing::reference_scene::adjudication_scene();
        let (origin, forward, _, up) = desc.camera_basis();
        let view = Mat4::look_at_rh(origin, origin + forward, up);
        let plane = desc.plane_mesh();
        let far = plane
            .vertices
            .iter()
            .map(|p| origin.distance(Vec3::from(*p)))
            .chain(
                desc.spheres
                    .iter()
                    .take(3)
                    .map(|s| origin.distance(Vec3::from(s.center)) + s.radius),
            )
            .fold(0.0f32, f32::max)
            .next_up();
        let near = 0.05f32;
        let projection = Mat4::perspective_rh(desc.fov_y_rad(), 1.0, near, far);

        let (sv, si) = crate::offscreen::sphere::generate_uv_sphere(64, 32, 1.0);
        let sphere: Vec<_> = sv
            .iter()
            .map(|v| VertexPN {
                position: v.position,
                normal: v.normal,
            })
            .collect();
        let plane_vertices: Vec<_> = plane
            .vertices
            .iter()
            .map(|p| VertexPN {
                position: *p,
                normal: [0.0, 1.0, 0.0],
            })
            .collect();
        let plane_indices: Vec<_> = plane.indices.iter().flatten().copied().collect();

        let mut meshes = Vec::new();
        for index in [3usize, 0, 1, 2] {
            let (vertices, indices, transform) = if index == 3 {
                (&plane_vertices, &plane_indices, Mat4::IDENTITY)
            } else {
                let s = &desc.spheres[index];
                (
                    &sphere,
                    &si,
                    Mat4::from_scale_rotation_translation(
                        Vec3::splat(s.radius),
                        glam::Quat::IDENTITY,
                        Vec3::from(s.center),
                    ),
                )
            };
            let mut mesh = MeshInstancedRenderer::new(
                &device,
                wgpu::TextureFormat::Rgba32Float,
                Some(wgpu::TextureFormat::Depth32Float),
            )
            .unwrap();
            mesh.set_mesh(&device, &queue, vertices, indices).unwrap();
            mesh.set_view_proj(view, projection);
            let material = crate::core::material::PbrMaterial::default();
            mesh.enable_pbr(
                &device,
                &queue,
                material,
                desc.ambient_color,
                desc.sun_color,
            )
            .unwrap();
            mesh.upload_instances_from_mat4(&device, &queue, &[transform])
                .unwrap();
            let sun_dir = Vec3::from(desc.sun_direction).normalize();
            let pbr = mesh.pbr_pipeline_mut().unwrap();
            pbr.update_shadows(&queue, view, projection, sun_dir, near, far);
            super::production::tighten_shadow_bounds(pbr, &queue, &desc, sun_dir);
            mesh.reset_shadow_draw_batch_uniforms();
            meshes.push(mesh);
        }

        // Depth pass — identical loop to production::render.
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("adjudication-shadow-dump"),
        });
        {
            let receiver = &meshes[0];
            let csm = receiver
                .pbr_pipeline()
                .unwrap()
                .shadow_manager
                .as_ref()
                .unwrap()
                .renderer();
            for (index, depth_view) in csm
                .shadow_map_views
                .iter()
                .take(csm.uniforms.cascade_count as usize)
                .enumerate()
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("adjudication-shadow-dump-pass"),
                    color_attachments: &[],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    ..Default::default()
                });
                let light_matrix =
                    Mat4::from_cols_array_2d(&csm.uniforms.cascades[index].light_view_proj);
                for caster in &meshes {
                    caster.render_shadow(&mut pass, &queue, light_matrix, 1);
                }
            }
        }
        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);

        // Read back layer 0.
        let receiver = &meshes[0];
        let csm = receiver
            .pbr_pipeline()
            .unwrap()
            .shadow_manager
            .as_ref()
            .unwrap()
            .renderer();
        let map_size = csm.uniforms.shadow_map_size as u32;
        let layers = csm.allocation_layers;
        let bpr = crate::core::gpu::align_copy_bpr(map_size * 4);
        let readback = crate::core::resource_tracker::tracked_create_buffer(
            &device,
            &wgpu::BufferDescriptor {
                label: Some("adjudication-shadow-readback"),
                size: (bpr * map_size * layers) as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            },
        )
        .unwrap();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("adjudication-shadow-readback"),
        });
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &csm.shadow_maps,
                mip_level: 0,
                origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &readback,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(bpr),
                    rows_per_image: Some(map_size),
                },
            },
            wgpu::Extent3d {
                width: map_size,
                height: map_size,
                depth_or_array_layers: layers,
            },
        );
        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);
        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range();
        let layer_stride = (map_size * map_size) as usize;
        let mut depths = vec![0f32; layer_stride * layers as usize];
        for layer in 0..layers as usize {
            for y in 0..map_size as usize {
                let off = (layer * map_size as usize + y) * bpr as usize;
                let row = &data[off..off + (map_size * 4) as usize];
                for (x, chunk) in row.chunks_exact(4).enumerate() {
                    depths[layer * layer_stride + y * map_size as usize + x] =
                        f32::from_le_bytes(chunk.try_into().unwrap());
                }
            }
        }
        drop(data);
        readback.unmap();
        std::fs::create_dir_all("target").ok();
        std::fs::write(
            "target/adjudication_shadow_layers.bin",
            bytemuck::cast_slice(&depths),
        )
        .unwrap();
        for layer in 0..layers as usize {
            let slice = &depths[layer * layer_stride..(layer + 1) * layer_stride];
            let occupied = slice.iter().filter(|d| **d < 1.0).count();
            eprintln!(
                "  layer{layer}: occupied={occupied} min={:.4}",
                slice.iter().cloned().fold(1.0f32, f32::min)
            );
        }
        let occupied = depths[..layer_stride].iter().filter(|d| **d < 1.0).count();
        assert!(occupied > 0, "shadow map empty");

        // Debug: print the cascade matrix and a probe point's sampled uv/depth.
        for (i, c) in csm.uniforms.cascades.iter().enumerate() {
            let m = Mat4::from_cols_array_2d(&c.light_view_proj);
            eprintln!("cascade{i} lvp = {:?}", m);
        }
        for probe in [
            Vec3::new(0.85, 0.0, 0.25),
            Vec3::new(0.08, 0.0, -1.52),
            Vec3::new(0.0, 0.0, -1.5),
            Vec3::new(0.1, 0.0, -1.48),
        ] {
            for (i, c) in csm.uniforms.cascades.iter().enumerate() {
                if i >= layers as usize {
                    break;
                }
                let m = Mat4::from_cols_array_2d(&c.light_view_proj);
                let clip = m * probe.extend(1.0);
                let ndc = clip.truncate() / clip.w;
                let uu = ndc.x * 0.5 + 0.5;
                let vv = ndc.y * -0.5 + 0.5;
                let ix = ((uu * 2047.0) as usize).min(2047);
                let iy = ((vv * 2047.0) as usize).min(2047);
                let stored = depths[i * layer_stride + iy * 2048 + ix];
                eprintln!(
                    "  probe {probe:?} cascade{i}: ndc={ndc:?} uv=({uu:.4},{vv:.4}) z={:.4} stored={stored:.4}",
                    clip.z / clip.w
                );
            }
        }
    }
}
