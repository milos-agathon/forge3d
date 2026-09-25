// src/viewer/pbr_scene/reference.rs
// Reference-scene loader: builds a generic `PbrSceneDesc` from the AEQUITAS
// `ReferenceSceneDesc` — sphere/plane geometry, baked plane indirect maps,
// materials, environment, camera and shadow-focus bounds. The analytic bake
// (`bake_plane_indirect`) and all numerics are identical to the former
// offscreen adjudication adapter; only the owner changed (viewer PBR scene).
// RELEVANT FILES: src/path_tracing/reference_scene.rs, src/viewer/pbr_scene/mod.rs

use glam::{DVec3, Vec3};

use crate::core::material::PbrMaterial;
use crate::path_tracing::reference_scene::ReferenceSceneDesc;
use crate::render::mesh_instanced::VertexPN;

use super::{
    BakedMap, GroundRadiance, PbrObjectMaps, PbrSceneCamera, PbrSceneDesc, PbrSceneObject,
};

/// World-space camera consumed by the load command: the viewer camera is set
/// from these values and the scene reports them as its consumed pose.
#[derive(Clone, Copy, Debug)]
pub struct ReferenceCamera {
    pub eye: DVec3,
    pub target: DVec3,
    pub up: Vec3,
    pub fov_deg: f32,
    pub near: f32,
    pub far: f32,
}

/// World-xz domain of the baked plane indirect-transport maps. The plane's
/// instanced mesh-local xz coords are its material UVs (vs_instanced emits
/// position.xz), so these maps index world space through the instance
/// transform.
const PLANE_UV_MIN: f32 = -16.0;
const PLANE_UV_EXTENT: f32 = 32.0;
const BAKE_RES: u32 = 512;
const BAKE_RAYS: usize = 384;
const SSAA: u32 = 4;

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

/// Outgoing radiance of a Lambertian sphere surface point — the same
/// transport model the `FLAG_GROUND_RADIANCE` shader branch integrates
/// for sphere occluders:
///   a/π · ( L_sun·max(n·l,0)·V_sun(h)
///           + π·L_amb·(0.5+0.5·n.y)
///           + E_plane(h,n) )
/// where V_sun tests the sun ray from just off the surface against the
/// OTHER spheres, the sky term is the cosine-weighted up-facing share of
/// the hemisphere about n, and E_plane is the cosine-weighted irradiance
/// the down-facing directions about n receive from the plane exit field —
/// integrated by golden-spiral taps (identical tap set to the shader),
/// not a single projected point: the visible annulus around a sphere is
/// mostly lit plane, and one projection lands inside the sphere's own
/// shadow pool too often.
const SPH_EXIT_TAPS: usize = 16;

fn sphere_exit_radiance(
    desc: &ReferenceSceneDesc,
    hit: Vec3,
    n: Vec3,
    idx: usize,
    plane_exit: &[Vec3],
) -> Vec3 {
    let light = -Vec3::from(desc.sun_direction).normalize();
    let l_sun = Vec3::from(desc.sun_color) * desc.sun_intensity;
    let l_amb = Vec3::from(desc.ambient_color);
    let a_s = Vec3::from(desc.spheres[idx].albedo);
    let ndl = n.dot(light).max(0.0);
    let v_sun = if ndl > 0.0 && ray_hit_sphere(desc, hit + n * 1e-3, light, Some(idx)).is_some() {
        0.0
    } else {
        1.0
    };
    // Sky coverage ~ cosine-weighted up-facing fraction of the hemisphere.
    let e_sky = l_amb * std::f32::consts::PI * (0.5 + 0.5 * n.y);
    // E_plane = (2π/M)·Σ_down P·ct over uniform-solid-angle taps about n.
    let up = if n.y.abs() > 0.95 { Vec3::X } else { Vec3::Y };
    let tangent = up.cross(n).normalize();
    let bitan = n.cross(tangent);
    let mut e_plane = Vec3::ZERO;
    for k in 0..SPH_EXIT_TAPS {
        let ct = 1.0 - (k as f32 + 0.5) / SPH_EXIT_TAPS as f32;
        let phi = k as f32 * 2.399_963_2;
        let st = (1.0 - ct * ct).sqrt();
        let d = tangent * (st * phi.cos()) + bitan * (st * phi.sin()) + n * ct;
        if d.y < -1e-4 {
            let t = hit.y / -d.y;
            e_plane += plane_exit_at(plane_exit, hit + d * t) * ct;
        }
    }
    let e_plane = e_plane * (2.0 * std::f32::consts::PI / SPH_EXIT_TAPS as f32);
    a_s * (l_sun * ndl * v_sun + e_sky + e_plane) / std::f32::consts::PI
}

fn plane_exit_at(plane_exit: &[Vec3], p: Vec3) -> Vec3 {
    let u = ((p.x - PLANE_UV_MIN) / PLANE_UV_EXTENT * BAKE_RES as f32)
        .clamp(0.0, BAKE_RES as f32 - 1.0) as usize;
    let v = ((p.z - PLANE_UV_MIN) / PLANE_UV_EXTENT * BAKE_RES as f32)
        .clamp(0.0, BAKE_RES as f32 - 1.0) as usize;
    plane_exit[v * BAKE_RES as usize + u]
}

// Sphere exit-radiance grids: the hemisphere-integral bounce pass asks for
// a sphere's exit at every surface normal; n_h determines the surface
// point (h = c + r·n_h), so the field over n_h is precomputed once per
// sphere on a lat-long grid and sampled bilinearly. Rebuilt whenever the
// plane exit field it feeds on advances a bounce.
const SPH_GRID_U: usize = 128; // azimuth samples over [0, 2π)
const SPH_GRID_V: usize = 64; // polar samples over [0, π]

fn sphere_exit_grid(desc: &ReferenceSceneDesc, plane_exit: &[Vec3]) -> [Vec<Vec3>; 3] {
    std::array::from_fn(|idx| {
        let s = &desc.spheres[idx];
        let c = Vec3::from(s.center);
        let mut grid = Vec::with_capacity(SPH_GRID_U * SPH_GRID_V);
        for v in 0..SPH_GRID_V {
            let theta = (v as f32 + 0.5) / SPH_GRID_V as f32 * std::f32::consts::PI;
            let (st, ct) = theta.sin_cos();
            for u in 0..SPH_GRID_U {
                let phi = (u as f32 + 0.5) / SPH_GRID_U as f32 * std::f32::consts::TAU;
                let n = Vec3::new(st * phi.cos(), ct, st * phi.sin());
                let h = c + n * s.radius;
                grid.push(sphere_exit_radiance(desc, h, n, idx, plane_exit));
            }
        }
        grid
    })
}

/// Bilinear sample of a sphere exit grid at normal n (periodic azimuth,
/// clamped polar edges).
fn sphere_exit_grid_sample(grid: &[Vec3], n: Vec3) -> Vec3 {
    let phi = n.z.atan2(n.x).rem_euclid(std::f32::consts::TAU);
    let theta = n.y.clamp(-1.0, 1.0).acos();
    let uf = phi / std::f32::consts::TAU * SPH_GRID_U as f32 - 0.5;
    let vf = theta / std::f32::consts::PI * SPH_GRID_V as f32 - 0.5;
    let u0 = uf.floor() as i32;
    let v0 = vf.floor() as i32;
    let fu = uf - u0 as f32;
    let fv = (vf - v0 as f32).clamp(0.0, 1.0);
    let at = |u: i32, v: i32| -> Vec3 {
        let uu = u.rem_euclid(SPH_GRID_U as i32) as usize;
        let vv = v.clamp(0, SPH_GRID_V as i32 - 1) as usize;
        grid[vv * SPH_GRID_U + uu]
    };
    at(u0, v0) * (1.0 - fu) * (1.0 - fv)
        + at(u0 + 1, v0) * fu * (1.0 - fv)
        + at(u0, v0 + 1) * (1.0 - fu) * fv
        + at(u0 + 1, v0 + 1) * fu * fv
}

/// One plane bounce pass: cosine/Fresnel-weighted sphere-exit irradiance
/// each plane texel receives, integrated over the hemisphere above the
/// plane. Returns the plane's bounce radiance (a_plane·E/π) per texel.
fn plane_bounce_pass(
    desc: &ReferenceSceneDesc,
    grids: &[Vec<Vec3>; 3],
    diffuse_weight: &dyn Fn(Vec3, Vec3) -> f32,
) -> Vec<Vec3> {
    let a_plane = Vec3::from(desc.spheres[3].albedo);
    let mut bounce = vec![Vec3::ZERO; (BAKE_RES * BAKE_RES) as usize];
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
                        sphere_exit_grid_sample(&grids[idx], n) * (ct * diffuse_weight(p, d));
                }
            }
            // Radiance = a·E_bounce/π = a·(2/N)·Σ L_exit·ct·(1-F)
            bounce[(j * BAKE_RES + i) as usize] = a_plane * e_bounce * (2.0 / BAKE_RAYS as f32);
        }
    }
    bounce
}

/// Hemisphere-integral bake over the plane's UV domain. Two passes:
/// 1) sky-visibility ratio + the plane's local exit radiance (which folds
///    in each point's umbra and ambient occlusion),
/// 2) the sphere-bounce radiance each ground point receives, iterated
///    plane-exit ↔ sphere-exit twice so the shadow pools carry two-bounce
///    transport (pass-1 plane exit → sphere grids → plane bounce →
///    refreshed sphere grids on the updated exit field → final bounce).
///
/// Returns (occlusion map, bounce-emissive map, plane exit-radiance map);
/// the last is the field an instanced sphere samples to receive its
/// hemisphere GI. This is the same ambient transport the path tracer
/// integrates, produced analytically like a production lightmap.
fn bake_plane_indirect(desc: &ReferenceSceneDesc) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let light = -Vec3::from(desc.sun_direction).normalize();
    let l_sun = Vec3::from(desc.sun_color) * desc.sun_intensity;
    let l_amb = Vec3::from(desc.ambient_color);
    let a_plane = Vec3::from(desc.spheres[3].albedo);
    let e_amb = l_amb * std::f32::consts::PI;

    // The reference BRDF's diffuse lobe is Fresnel-weighted by the
    // view-dependent half-vector: f_diffuse = a/π·(1 - F(v·h)) with
    // F0 = REFERENCE_DIELECTRIC_F0. Ambient arriving at grazing incidence — the only light
    // that reaches deep pool points — is attenuated, so weight every bake
    // ray by (1 - F(v·h)) with v the camera direction at the texel.
    let cam = Vec3::from(desc.cam_origin);
    let diffuse_weight = |p: Vec3, wi: Vec3| -> f32 {
        let v = (cam - p).normalize();
        let h = (wi + v).normalize();
        let vh = v.dot(h).clamp(0.0, 1.0);
        let f0 = crate::path_tracing::reference_scene::REFERENCE_DIELECTRIC_F0;
        1.0 - (f0 + (1.0 - f0) * (1.0 - vh).powi(5))
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
            plane_exit[o] =
                a_plane * (l_sun * light.y.max(0.0) * v_sun + e_amb * occ) / std::f32::consts::PI;
        }
    }
    // Pass 2: bounce irradiance per texel, iterated plane-exit <->
    // sphere-exit twice. The sphere grids consume the plane's current
    // full exit field (direct + ambient + bounce so far); the refreshed
    // field feeds the next grid, so the shadow pools carry two-bounce
    // transport. The exit field the spheres sample stays linear — a
    // surface sees a region of plane, not a single texel.
    let mut exit_lin = plane_exit.clone();
    let mut bounce = vec![Vec3::ZERO; (BAKE_RES * BAKE_RES) as usize];
    for _ in 0..2 {
        let grids = sphere_exit_grid(desc, &exit_lin);
        bounce = plane_bounce_pass(desc, &grids, &diffuse_weight);
        for (o, em) in bounce.iter().enumerate() {
            exit_lin[o] = plane_exit[o] + *em;
        }
    }

    // Pass 2 output: fold the final bounce into the plane's emissive map
    // and its full exit-radiance field (the map the spheres sample).
    let mut occlusion = vec![0u8; (BAKE_RES * BAKE_RES) as usize];
    let mut emissive = vec![0u8; (BAKE_RES * BAKE_RES * 4) as usize];
    let mut radiance = vec![0u8; (BAKE_RES * BAKE_RES * 4) as usize];
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
    for o in 0..(BAKE_RES * BAKE_RES) as usize {
        occlusion[o] = (occ_f[o].clamp(0.0, 1.0) * 255.0).round() as u8;
        let e = o * 4;
        emissive[e] = srgb(bounce[o].x);
        emissive[e + 1] = srgb(bounce[o].y);
        emissive[e + 2] = srgb(bounce[o].z);
        emissive[e + 3] = 255;
        radiance[e] = srgb(exit_lin[o].x);
        radiance[e + 1] = srgb(exit_lin[o].y);
        radiance[e + 2] = srgb(exit_lin[o].z);
        radiance[e + 3] = 255;
    }
    (occlusion, emissive, radiance)
}

/// Build the generic PBR scene + camera from the adjudication
/// `ReferenceSceneDesc`. Object order is [3, 0, 1, 2] (plane first, then the
/// three spheres) exactly as the production draw loop consumed them.
pub fn reference_pbr_scene(desc: &ReferenceSceneDesc) -> (PbrSceneDesc, ReferenceCamera) {
    let (origin, _, _, up) = desc.camera_basis();
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
    // world space; the object transform maps local -> world.
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
    let plane_indices: Vec<_> = plane.indices.iter().flatten().copied().collect();

    // Consume the shared environment contract: env_sky is the constant
    // ambient, miss_sky the background — identical to what the PT path
    // binds, never a local literal.
    let env = desc.environment_raw();
    let ambient = [env.env_sky[0], env.env_sky[1], env.env_sky[2]];
    let sky = [env.miss_sky[0], env.miss_sky[1], env.miss_sky[2]];
    let (occlusion_map, emissive_map, radiance_map) = bake_plane_indirect(desc);

    // Hemisphere-GI occluder spheres + their Lambertian albedo — identical
    // for every sphere object, as the former adapter uploaded per-mesh.
    // The shader derives each occluder's exit radiance in-shader.
    let mut gi_sph = [[0.0f32; 4]; 3];
    let mut gi_sph_albedo = [[0.0f32; 4]; 3];
    for (s, sd) in desc.spheres.iter().take(3).enumerate() {
        gi_sph[s] = [sd.center[0], sd.center[1], sd.center[2], sd.radius];
        gi_sph_albedo[s] = [sd.albedo[0], sd.albedo[1], sd.albedo[2], 0.0];
    }

    // World-space points bounding every caster+receiver in light space:
    // sphere AABB corners, their shadow-pool footprints on the plane (with
    // the AO-fringe pad), and the visible plane patch.
    let sun_dir = Vec3::from(desc.sun_direction).normalize();
    let mut shadow_focus: Vec<DVec3> = Vec::new();
    for s in desc.spheres.iter().take(3) {
        let c = Vec3::from(s.center);
        let r = s.radius;
        for i in 0..8u32 {
            let p = c + Vec3::new(
                if i & 1 != 0 { r } else { -r },
                if i & 2 != 0 { r } else { -r },
                if i & 4 != 0 { r } else { -r },
            );
            shadow_focus.push(DVec3::new(p.x as f64, p.y as f64, p.z as f64));
        }
        // Pool footprint: the sphere's AABB projected onto the plane along
        // the sun direction, padded by the AO fringe.
        if sun_dir.y < -1e-4 {
            let t = -c.y / sun_dir.y;
            let pc = Vec3::new(c.x + t * sun_dir.x, 0.0, c.z + t * sun_dir.z);
            let pr = r + 1.5;
            for i in 0..4u32 {
                let p = Vec3::new(
                    pc.x + if i & 1 != 0 { pr } else { -pr },
                    0.0,
                    pc.z + if i & 2 != 0 { pr } else { -pr },
                );
                shadow_focus.push(DVec3::new(p.x as f64, p.y as f64, p.z as f64));
            }
        }
    }
    for i in 0..4u32 {
        let p = Vec3::new(
            if i & 1 != 0 { 8.0 } else { -8.0 },
            0.0,
            if i & 2 != 0 { 8.0 } else { -8.0 },
        );
        shadow_focus.push(DVec3::new(p.x as f64, p.y as f64, p.z as f64));
    }

    let mut objects = Vec::with_capacity(4);
    for index in [3usize, 0, 1, 2] {
        let material_desc = &desc.spheres[index];
        let object = if index == 3 {
            PbrSceneObject {
                vertices: plane_vertices.clone(),
                indices: plane_indices.clone(),
                translation: DVec3::new(f64::from(PLANE_UV_MIN), 0.0, f64::from(PLANE_UV_MIN)),
                rotation: glam::Quat::IDENTITY,
                scale: Vec3::new(PLANE_UV_EXTENT, 1.0, PLANE_UV_EXTENT),
                material: PbrMaterial {
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
                    emissive: [1.0, 1.0, 1.0],
                    ..Default::default()
                },
                maps: PbrObjectMaps {
                    occlusion: Some(BakedMap {
                        width: BAKE_RES,
                        height: BAKE_RES,
                        data: occlusion_map.clone(),
                    }),
                    emissive: Some(BakedMap {
                        width: BAKE_RES,
                        height: BAKE_RES,
                        data: emissive_map.clone(),
                    }),
                    ground_radiance: None,
                },
            }
        } else {
            PbrSceneObject {
                vertices: sphere.clone(),
                indices: sphere_indices.clone(),
                translation: DVec3::new(
                    f64::from(material_desc.center[0]),
                    f64::from(material_desc.center[1]),
                    f64::from(material_desc.center[2]),
                ),
                rotation: glam::Quat::IDENTITY,
                scale: Vec3::splat(material_desc.radius),
                material: PbrMaterial {
                    base_color: [
                        material_desc.albedo[0],
                        material_desc.albedo[1],
                        material_desc.albedo[2],
                        1.0,
                    ],
                    roughness: material_desc.roughness,
                    metallic: 0.0,
                    emissive: [0.0; 3],
                    ..Default::default()
                },
                maps: PbrObjectMaps {
                    occlusion: None,
                    emissive: None,
                    ground_radiance: Some(GroundRadiance {
                        map: BakedMap {
                            width: BAKE_RES,
                            height: BAKE_RES,
                            data: radiance_map.clone(),
                        },
                        gi_ambient: ambient,
                        gi_sph,
                        gi_sph_albedo,
                    }),
                },
            }
        };
        objects.push(object);
    }

    let camera = PbrSceneCamera {
        eye: DVec3::new(
            f64::from(desc.cam_origin[0]),
            f64::from(desc.cam_origin[1]),
            f64::from(desc.cam_origin[2]),
        ),
        target: DVec3::new(
            f64::from(desc.cam_look_at[0]),
            f64::from(desc.cam_look_at[1]),
            f64::from(desc.cam_look_at[2]),
        ),
        up,
        fov_deg: desc.fov_y_deg,
        near,
        far,
    };
    let scene = PbrSceneDesc {
        objects,
        sun_direction: desc.sun_direction,
        sun_intensity: desc.sun_intensity,
        sun_color: desc.sun_color,
        ambient,
        sky,
        exposure: desc.exposure,
        ssaa: SSAA,
        shadow_focus,
        #[cfg(feature = "extension-module")]
        camera,
    };
    (
        scene,
        ReferenceCamera {
            eye: camera.eye,
            target: camera.target,
            up: camera.up,
            fov_deg: camera.fov_deg,
            near: camera.near,
            far: camera.far,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::path_tracing::reference_scene::adjudication_scene;

    /// Instance/adapter/device/queue triple for headless-viewer tests; the
    /// device+queue request mirrors `create_device_and_queue_for_test`.
    /// Follows `adjudication_test_device`'s hardware-only convention:
    /// software adapters are declined, and with `wavefront` the PT shadow
    /// kernel must compile on this device.
    #[cfg(all(
        feature = "enable-gpu-instancing",
        feature = "enable-pbr",
        feature = "enable-tbn"
    ))]
    fn headless_gpu(
        wavefront: bool,
    ) -> Option<(
        std::sync::Arc<wgpu::Device>,
        std::sync::Arc<wgpu::Queue>,
        std::sync::Arc<wgpu::Adapter>,
    )> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))?;
        let info = adapter.get_info();
        let name = info.name.to_lowercase();
        if info.device_type == wgpu::DeviceType::Cpu
            || [
                "basic render driver",
                "warp",
                "lavapipe",
                "llvmpipe",
                "swiftshader",
            ]
            .iter()
            .any(|token| name.contains(token))
        {
            eprintln!(
                "skipping AEQUITAS GPU test: software adapter '{}'",
                info.name
            );
            return None;
        }
        let mut limits = adapter.limits();
        limits.max_storage_buffers_per_shader_stage =
            limits.max_storage_buffers_per_shader_stage.max(8);
        let capabilities = crate::core::capabilities::CapabilitySet::negotiate(adapter.features());
        let (device, queue) = match pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                required_features: capabilities.granted,
                required_limits: limits.clone(),
                label: Some("forge3d-test-device"),
            },
            None,
        )) {
            Ok(pair) => pair,
            Err(_) => pollster::block_on(adapter.request_device(
                &wgpu::DeviceDescriptor {
                    required_features: wgpu::Features::empty(),
                    required_limits: limits,
                    label: Some("forge3d-test-device"),
                },
                None,
            ))
            .ok()?,
        };
        if wavefront {
            if let Err(error) =
                crate::path_tracing::wavefront::pipeline::WavefrontPipelines::probe_shadow_kernel(
                    &device,
                )
            {
                eprintln!(
                    "skipping AEQUITAS wavefront test: shadow kernel did not compile: {error}"
                );
                return None;
            }
        }
        Some((
            std::sync::Arc::new(device),
            std::sync::Arc::new(queue),
            std::sync::Arc::new(adapter),
        ))
    }

    #[cfg(all(
        feature = "enable-gpu-instancing",
        feature = "enable-pbr",
        feature = "enable-tbn"
    ))]
    #[test]
    fn raster_capture_executes_production_shader() {
        let Some((device, queue, adapter)) = headless_gpu(false) else {
            return;
        };
        crate::core::degradation::begin_degradation_capture();
        crate::core::shader_registry::begin_shader_render_capture(&Default::default());
        let mut viewer = crate::viewer::Viewer::new_headless(
            device,
            queue,
            adapter,
            32,
            32,
            crate::viewer::viewer_config::ViewerConfig::default(),
        )
        .unwrap();
        viewer
            .handle_cmd(crate::viewer::viewer_enums::ViewerCmd::LoadReferenceScene {
                name: "adjudication".into(),
            })
            .unwrap();
        viewer.render_headless_frame(None).unwrap();
        let pixels = viewer.read_pbr_scene_hdr().unwrap();
        let used = crate::core::shader_registry::finish_shader_render_capture();
        let degradations = crate::core::degradation::finish_degradation_capture();
        assert!(degradations.is_empty(), "{degradations:?}");
        assert!(used.keys().any(|key| key.starts_with("pbr_shader_module")));
        assert!(used
            .keys()
            .any(|key| key.starts_with("viewer.pbr_scene.resolve")));
        assert!(!used
            .keys()
            .any(|key| key.starts_with("adjudication-raster-shader")));
        assert!(pixels.chunks_exact(4).any(|pixel| pixel[0] > 0.0));
    }

    /// Raster half of the adjudication pair for an explicit scene desc: a
    /// headless viewer loads `desc` through the same guarded path as
    /// `:load_reference_scene` and returns the resolved linear HDR frame.
    #[cfg(all(
        feature = "enable-gpu-instancing",
        feature = "enable-pbr",
        feature = "enable-tbn"
    ))]
    fn render_viewer_raster(
        gpu: &(
            std::sync::Arc<wgpu::Device>,
            std::sync::Arc<wgpu::Queue>,
            std::sync::Arc<wgpu::Adapter>,
        ),
        desc: &ReferenceSceneDesc,
        size: u32,
    ) -> Vec<f32> {
        let mut viewer = crate::viewer::Viewer::new_headless(
            gpu.0.clone(),
            gpu.1.clone(),
            gpu.2.clone(),
            size,
            size,
            crate::viewer::viewer_config::ViewerConfig {
                width: size,
                height: size,
                ..Default::default()
            },
        )
        .unwrap();
        crate::viewer::cmd::scene_command::load_reference_scene_desc(
            &mut viewer,
            "adjudication-test",
            desc.clone(),
        );
        assert!(
            viewer.pbr_scene.is_some(),
            "reference scene load was rejected: {:?}",
            viewer.command_error
        );
        viewer.render_headless_frame(None).unwrap();
        viewer.read_pbr_scene_hdr().unwrap()
    }

    /// Behavioral half of the constant sky/ambient contract (the source-string
    /// half lives in reference_scene.rs): perturbing `sky_color` and
    /// `ambient_color` on the ReferenceSceneDesc must move BOTH renders. A path
    /// that hardcodes its own sky or ambient literal keeps its old output and
    /// fails here.
    #[cfg(all(
        feature = "enable-gpu-instancing",
        feature = "enable-pbr",
        feature = "enable-tbn"
    ))]
    #[test]
    fn both_paths_consume_scene_sky_and_ambient() {
        use crate::path_tracing::adjudication::render_pt_reference;

        let Some(gpu) = headless_gpu(true) else {
            return;
        };
        const N: u32 = 32;
        // Sun off: the plane is lit by the constant ambient alone, so its
        // radiance is linear in ambient_color on both paths.
        let mut base = adjudication_scene();
        base.sun_intensity = 0.0;
        let mut tinted = base.clone();
        tinted.sky_color = [0.9, 0.1, 0.2];
        tinted.ambient_color = base.ambient_color.map(|c| c * 2.0);

        let pt = |desc: &ReferenceSceneDesc| {
            render_pt_reference(&gpu.0, &gpu.1, desc, N, N, 64, None).unwrap()
        };
        let raster = |desc: &ReferenceSceneDesc| render_viewer_raster(&gpu, desc, N);
        // Top-left pixel is a primary miss (camera pitches ~11 deg down, fov 40).
        let sky_px = |hdr: &[f32]| [hdr[0], hdr[1], hdr[2]];
        // Mean over the bottom quarter of the frame: open plane, no spheres.
        let plane_mean = |hdr: &[f32]| {
            let px = ((N * 3 / 4)..N).flat_map(|y| (0..N).map(move |x| ((y * N + x) * 4) as usize));
            let (sum, count) = px.fold((0.0f64, 0u32), |(s, c), i| {
                (s + f64::from(hdr[i] + hdr[i + 1] + hdr[i + 2]), c + 1)
            });
            sum / f64::from(count)
        };

        for (name, base_hdr, tinted_hdr) in [
            ("pt", pt(&base), pt(&tinted)),
            ("raster", raster(&base), raster(&tinted)),
        ] {
            for (hdr, desc) in [(&base_hdr, &base), (&tinted_hdr, &tinted)] {
                let got = sky_px(hdr);
                for c in 0..3 {
                    assert!(
                        (got[c] - desc.sky_color[c]).abs() < 1e-3,
                        "{name}: sky pixel {got:?} != scene sky_color {:?}",
                        desc.sky_color
                    );
                }
            }
            let ratio = plane_mean(&tinted_hdr) / plane_mean(&base_hdr);
            assert!(
                (1.6..=2.4).contains(&ratio),
                "{name}: doubling ambient_color scaled the plane by {ratio:.3}, expected ~2"
            );
        }
    }

    /// Behavioural lock on the remaining ReferenceSceneDesc inputs (sun,
    /// materials, exposure; sky/ambient are covered above): perturb each on
    /// the committed scene and require the raster path to follow the change
    /// the way the path-traced reference does. A raster path that hardcodes
    /// one of them, or resolves with a different tonemap/exposure, keeps its
    /// old colour while PT moves, so its per-channel bias against PT grows by
    /// the size of the perturbation.
    #[cfg(all(
        feature = "enable-gpu-instancing",
        feature = "enable-pbr",
        feature = "enable-tbn"
    ))]
    #[test]
    fn raster_tracks_sun_material_and_exposure_like_pt() {
        use crate::core::tonemap::resolve_reference_hdr_to_rgba8;
        use crate::path_tracing::adjudication::render_pt_reference;

        let Some(gpu) = headless_gpu(true) else {
            return;
        };
        const N: u32 = 64;
        let mean_rgb = |hdr: &[f32], exposure: f32| -> [f64; 3] {
            let mut sum = [0.0f64; 3];
            for px in resolve_reference_hdr_to_rgba8(hdr, exposure).chunks_exact(4) {
                for c in 0..3 {
                    sum[c] += f64::from(px[c]);
                }
            }
            sum.map(|s| s / f64::from(N * N))
        };
        let render = |desc: &ReferenceSceneDesc| {
            let pt = render_pt_reference(&gpu.0, &gpu.1, desc, N, N, 512, None).unwrap();
            let raster = render_viewer_raster(&gpu, desc, N);
            (
                mean_rgb(&pt, desc.exposure),
                mean_rgb(&raster, desc.exposure),
            )
        };
        let max_abs_diff =
            |a: [f64; 3], b: [f64; 3]| (0..3).map(|c| (a[c] - b[c]).abs()).fold(0.0, f64::max);

        let base = adjudication_scene();
        let (base_pt, base_raster) = render(&base);
        let base_bias = max_abs_diff(base_pt, base_raster);

        let mut sun = base.clone();
        sun.sun_color = [0.80, 0.90, 1.00];
        sun.sun_intensity = 2.4;
        let mut materials = base.clone();
        materials.spheres[0].albedo = [0.25, 0.55, 0.30];
        materials.spheres[1].roughness = 0.30;
        materials.spheres[3].albedo = [0.55, 0.48, 0.36];
        let mut exposure = base.clone();
        exposure.exposure = 1.6;

        for (name, desc) in [
            ("sun", sun),
            ("materials", materials),
            ("exposure", exposure),
        ] {
            let (pt, raster) = render(&desc);
            let shift = max_abs_diff(pt, base_pt);
            let bias = max_abs_diff(pt, raster);
            eprintln!(
                "{name}: PT shift {shift:.3}, PT-raster bias {bias:.3} (base {base_bias:.3})"
            );
            // The perturbation must visibly move the PT reference, or the
            // variant proves nothing.
            assert!(shift > 4.0, "{name}: perturbation too small ({shift:.3})");
            assert!(
                bias <= base_bias + 0.25 * shift,
                "{name}: raster did not follow the ReferenceSceneDesc change \
                 (bias {bias:.3} vs base {base_bias:.3}, PT shift {shift:.3})"
            );
        }
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

        let Some((device, queue)) =
            crate::path_tracing::adjudication::adjudication_test_device(false)
        else {
            return;
        };
        let desc = adjudication_scene();
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
            let (scene_desc, _cam) = reference_pbr_scene(&desc);
            crate::viewer::pbr_scene::ViewerPbrScene::tighten_shadow_bounds(
                pbr,
                &queue,
                &scene_desc.shadow_focus,
                sun_dir,
                &crate::camera::Anchor::new(),
            );
            mesh.reset_shadow_draw_batch_uniforms();
            meshes.push(mesh);
        }

        // Depth pass — identical loop to the scene's shadow stage.
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
        let tex = csm.shadow_maps.clone();
        // wgpu requires depth-texture buffer copies to cover every array
        // layer; copy all cascades and keep layer 0's range.
        let layers = tex.size().depth_or_array_layers;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("adjudication-shadow-dump-copy"),
        });
        let buf = crate::core::resource_tracker::tracked_create_buffer(
            &device,
            &wgpu::BufferDescriptor {
                label: Some("adjudication-shadow-dump-buf"),
                size: (map_size * map_size * 4 * layers) as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            },
        )
        .unwrap();
        let bpp = 4u32;
        let bpr = map_size * bpp;
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &buf,
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
        let slice = buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);
        let data = slice.get_mapped_range()[..(map_size * map_size * 4) as usize].to_vec();
        buf.unmap();
        let path = std::path::Path::new("target/adjudication_shadow_layer0.bin");
        std::fs::write(path, &data).unwrap();
        eprintln!("dumped {} bytes to {}", data.len(), path.display());
    }
}
