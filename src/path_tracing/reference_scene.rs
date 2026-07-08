// src/path_tracing/reference_scene.rs
// AEQUITAS deterministic adjudication reference scene.
// Single source of truth consumed by BOTH the wavefront PT reference
// (src/path_tracing/adjudication.rs) and the raster twin
// (src/offscreen/adjudication_raster.rs). Every numeric constant is a
// hard-coded literal: no RNG, no time, no environment lookups.
// RELEVANT FILES: src/shaders/pt_shade.wgsl, src/shaders/adjudication_raster.wgsl

use crate::accel::cpu_bvh::MeshCPU;
use crate::path_tracing::lighting::{GpuAreaLight, GpuDirectionalLight};
use bytemuck::{Pod, Zeroable};
use glam::Vec3;

/// One analytic sphere (also used as a material slot for the ground plane:
/// a radius-0 sphere can never be intersected — `disc = b^2 - |oc|^2 <= 0` —
/// so slot 3 carries the plane material without adding geometry).
#[derive(Clone, Copy, Debug)]
pub struct SphereDesc {
    pub center: [f32; 3],
    pub radius: f32,
    pub albedo: [f32; 3],
    pub roughness: f32,
}

/// The full adjudication scene description.
#[derive(Clone, Debug)]
pub struct ReferenceSceneDesc {
    pub cam_origin: [f32; 3],
    pub cam_look_at: [f32; 3],
    pub cam_up: [f32; 3],
    pub fov_y_deg: f32,
    pub exposure: f32,
    /// Slots 0..2 are real spheres; slot 3 is the plane material carrier (radius 0).
    pub spheres: [SphereDesc; 4],
    /// Direction the sun light TRAVELS (from light toward scene), normalized lazily.
    pub sun_direction: [f32; 3],
    pub sun_intensity: f32,
    pub sun_color: [f32; 3],
    /// Constant ambient/environment radiance (the env-NEE term in pt_shade and
    /// the raster twin). LITERAL CONSTANT CONTRACT: the environment is flat —
    /// no gradient, no directional variation.
    pub ambient_color: [f32; 3],
    /// Constant primary-miss background (pt_scatter miss + raster sky pass).
    pub sky_color: [f32; 3],
    /// Ground plane at y = 0 spans x/z in [-half_extent, +half_extent].
    pub plane_half_extent: f32,
    pub seed_hi: u32,
    pub seed_lo: u32,
}

/// The committed adjudication scene. Changing any literal invalidates the
/// goldens under tests/golden/adjudication/.
pub fn adjudication_scene() -> ReferenceSceneDesc {
    ReferenceSceneDesc {
        cam_origin: [0.0, 2.2, 6.5],
        cam_look_at: [0.0, 0.9, 0.0],
        cam_up: [0.0, 1.0, 0.0],
        fov_y_deg: 40.0,
        exposure: 1.0,
        spheres: [
            SphereDesc {
                center: [-1.15, 1.0, 0.0],
                radius: 1.0,
                albedo: [0.63, 0.28, 0.22],
                roughness: 0.70,
            },
            SphereDesc {
                center: [1.30, 0.8, 0.55],
                radius: 0.8,
                albedo: [0.24, 0.40, 0.62],
                roughness: 0.55,
            },
            SphereDesc {
                center: [0.25, 0.5, -1.45],
                radius: 0.5,
                albedo: [0.78, 0.68, 0.30],
                roughness: 0.85,
            },
            // Plane material slot (radius 0: material carrier only).
            SphereDesc {
                center: [0.0, -1000.0, 0.0],
                radius: 0.0,
                albedo: [0.42, 0.42, 0.42],
                roughness: 0.90,
            },
        ],
        sun_direction: [-0.45, -0.80, -0.30],
        sun_intensity: 3.2,
        sun_color: [1.0, 0.97, 0.92],
        ambient_color: [0.40, 0.48, 0.62],
        sky_color: [0.35, 0.45, 0.70],
        plane_half_extent: 40.0,
        seed_hi: 0x9E37_79B9,
        seed_lo: 0x85EB_CA6B,
    }
}

/// GPU sphere layout matching WGSL `Sphere` in pt_raygen/pt_intersect/pt_shade
/// (NOT the megakernel layout in compute_types.rs — field order differs):
///   center: vec3 @0, radius @12, albedo: vec3 @16, metallic @28,
///   roughness @32, ior @36, emissive: vec3 @48, ax @60, ay @64; stride 80.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct WavefrontGpuSphere {
    pub center: [f32; 3],
    pub radius: f32,
    pub albedo: [f32; 3],
    pub metallic: f32,
    pub roughness: f32,
    pub ior: f32,
    pub _pad0: [f32; 2],
    pub emissive: [f32; 3],
    pub ax: f32,
    pub ay: f32,
    pub _pad1: [f32; 3],
}

/// GPU environment layout for wavefront binding 21 and the raster uniform tail.
/// Four vec4s keep WGSL uniform alignment boring; `.w` lanes are unused.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct ReferenceEnvironmentRaw {
    pub env_ground: [f32; 4],
    pub env_sky: [f32; 4],
    pub miss_ground: [f32; 4],
    pub miss_sky: [f32; 4],
}

impl ReferenceSceneDesc {
    /// Spheres in the wavefront kernels' GPU layout. All materials are
    /// non-metallic, non-emissive dielectric-off (ior 1.0) Lambert+GGX.
    pub fn wavefront_spheres(&self) -> Vec<WavefrontGpuSphere> {
        self.spheres
            .iter()
            .map(|s| WavefrontGpuSphere {
                center: s.center,
                radius: s.radius,
                albedo: s.albedo,
                metallic: 0.0,
                roughness: s.roughness,
                ior: 1.0,
                _pad0: [0.0; 2],
                emissive: [0.0; 3],
                ax: 0.0,
                ay: 0.0,
                _pad1: [0.0; 3],
            })
            .collect()
    }

    pub fn directional_lights(&self) -> Vec<GpuDirectionalLight> {
        vec![GpuDirectionalLight::new(
            self.sun_direction,
            self.sun_intensity,
            self.sun_color,
            1.0,
        )]
    }

    /// One inert area light: placed far below the plane facing down so
    /// `cos_on_light <= 0` in pt_shade's disc sampler -> pdf 0 -> the shade
    /// kernel never pushes an area-light shadow ray (keeps the per-pixel
    /// shadow-queue traffic at exactly env + sun).
    pub fn area_lights(&self) -> Vec<GpuAreaLight> {
        vec![GpuAreaLight::disc(
            [0.0, -1.0e4, 0.0],
            [0.0, -1.0, 0.0],
            1.0e-6,
            0.0,
            [0.0, 0.0, 0.0],
            0.0,
        )]
    }

    pub fn object_importance(&self) -> [f32; 4] {
        [1.0, 1.0, 1.0, 1.0]
    }

    /// The single environment uniform BOTH renderers consume. The 4-slot
    /// gradient layout is a GPU ABI shared with pt_shade/pt_scatter and
    /// adjudication_raster.wgsl; duplicating each constant into both mix()
    /// endpoints collapses the shader-side gradient to the literal constant
    /// without touching WGSL, and makes it impossible for the two slots of one
    /// term to diverge.
    pub fn environment_raw(&self) -> ReferenceEnvironmentRaw {
        let a = self.ambient_color;
        let s = self.sky_color;
        ReferenceEnvironmentRaw {
            env_ground: [a[0], a[1], a[2], 0.0],
            env_sky: [a[0], a[1], a[2], 0.0],
            miss_ground: [s[0], s[1], s[2], 0.0],
            miss_sky: [s[0], s[1], s[2], 0.0],
        }
    }

    /// Ground plane as a 2-triangle mesh. Winding is chosen so the geometric
    /// normal `normalize(cross(e1, e2))` computed by pt_intersect points +Y.
    pub fn plane_mesh(&self) -> MeshCPU {
        let e = self.plane_half_extent;
        MeshCPU::new(
            vec![[-e, 0.0, -e], [-e, 0.0, e], [e, 0.0, e], [e, 0.0, -e]],
            vec![[0, 1, 2], [0, 2, 3]],
        )
    }

    /// Camera basis exactly as pt_raygen.wgsl consumes it:
    /// rd = normalize(ndc_x*half_w*right + ndc_y*half_h*up + forward).
    pub fn camera_basis(&self) -> (Vec3, Vec3, Vec3, Vec3) {
        let origin = Vec3::from(self.cam_origin);
        let forward = (Vec3::from(self.cam_look_at) - origin).normalize();
        let right = forward.cross(Vec3::from(self.cam_up)).normalize();
        let up = right.cross(forward).normalize();
        (origin, forward, right, up)
    }

    pub fn fov_y_rad(&self) -> f32 {
        self.fov_y_deg.to_radians()
    }

    /// Camera/light metadata reported for BOTH render paths; the gate asserts
    /// the two copies are identical, proving a single scene source of truth.
    pub fn metadata_fields(&self, width: u32, height: u32, spp: u32) -> Vec<(&'static str, f64)> {
        let sun = Vec3::from(self.sun_direction).normalize();
        vec![
            ("cam_origin_x", self.cam_origin[0] as f64),
            ("cam_origin_y", self.cam_origin[1] as f64),
            ("cam_origin_z", self.cam_origin[2] as f64),
            ("cam_look_at_x", self.cam_look_at[0] as f64),
            ("cam_look_at_y", self.cam_look_at[1] as f64),
            ("cam_look_at_z", self.cam_look_at[2] as f64),
            ("fov_y_deg", self.fov_y_deg as f64),
            ("exposure", self.exposure as f64),
            ("sun_dir_x", sun.x as f64),
            ("sun_dir_y", sun.y as f64),
            ("sun_dir_z", sun.z as f64),
            ("sun_intensity", self.sun_intensity as f64),
            ("sun_color_r", self.sun_color[0] as f64),
            ("sun_color_g", self.sun_color[1] as f64),
            ("sun_color_b", self.sun_color[2] as f64),
            ("ambient_r", self.ambient_color[0] as f64),
            ("ambient_g", self.ambient_color[1] as f64),
            ("ambient_b", self.ambient_color[2] as f64),
            ("sky_r", self.sky_color[0] as f64),
            ("sky_g", self.sky_color[1] as f64),
            ("sky_b", self.sky_color[2] as f64),
            ("width", width as f64),
            ("height", height as f64),
            ("spp", spp as f64),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sphere_layout_matches_wavefront_wgsl_stride() {
        assert_eq!(std::mem::size_of::<WavefrontGpuSphere>(), 80);
        assert_eq!(memoffset_of_roughness(), 32);
        assert_eq!(memoffset_of_emissive(), 48);
        assert_eq!(memoffset_of_ay(), 64);
    }

    fn memoffset_of_roughness() -> usize {
        let s = WavefrontGpuSphere::zeroed();
        (&s.roughness as *const f32 as usize) - (&s as *const _ as usize)
    }
    fn memoffset_of_emissive() -> usize {
        let s = WavefrontGpuSphere::zeroed();
        (s.emissive.as_ptr() as usize) - (&s as *const _ as usize)
    }
    fn memoffset_of_ay() -> usize {
        let s = WavefrontGpuSphere::zeroed();
        (&s.ay as *const f32 as usize) - (&s as *const _ as usize)
    }

    #[test]
    fn scene_invariants() {
        let d = adjudication_scene();
        assert_eq!(d.spheres.len(), 4);
        assert_eq!(d.spheres[3].radius, 0.0, "slot 3 is the plane material");
        assert_eq!(std::mem::size_of::<ReferenceEnvironmentRaw>(), 64);
        let (_, f, r, u) = d.camera_basis();
        assert!(f.dot(r).abs() < 1e-6 && f.dot(u).abs() < 1e-6 && r.dot(u).abs() < 1e-6);
        assert!((f.length() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn metadata_includes_constant_ambient_and_sky() {
        let fields = adjudication_scene().metadata_fields(8, 4, 2);
        for key in [
            "ambient_r",
            "ambient_g",
            "ambient_b",
            "sky_r",
            "sky_g",
            "sky_b",
        ] {
            assert!(fields.iter().any(|(name, _)| *name == key), "{key}");
        }
        for key in ["env_ground_r", "env_sky_b", "miss_ground_g", "miss_sky_b"] {
            assert!(
                !fields.iter().any(|(name, _)| *name == key),
                "gradient key {key} must be gone: the scene contract is constant ambient/sky"
            );
        }
    }

    #[test]
    fn environment_raw_is_constant_ambient_and_sky() {
        // Both renderers consume the environment ONLY through environment_raw();
        // this pins the GPU uniform to the two scene constants so neither path
        // can see a gradient or a diverging hardcoded value.
        let d = adjudication_scene();
        let e = d.environment_raw();
        for i in 0..3 {
            assert_eq!(e.env_ground[i], d.ambient_color[i]);
            assert_eq!(e.env_sky[i], d.ambient_color[i]);
            assert_eq!(e.miss_ground[i], d.sky_color[i]);
            assert_eq!(e.miss_sky[i], d.sky_color[i]);
        }
    }

    #[test]
    fn both_render_paths_consume_environment_raw() {
        // Source contract: the PT path and the raster twin must feed their
        // sky/ambient uniforms from ReferenceSceneDesc::environment_raw() —
        // never from local literals.
        let pt = include_str!("adjudication.rs");
        let raster = include_str!("../offscreen/adjudication_raster.rs");
        assert!(pt.contains("set_environment_params(&desc.environment_raw())"));
        assert!(raster.contains("desc.environment_raw()"));
    }

    #[test]
    fn both_render_paths_consume_shared_camera_and_exposure() {
        // Source contract: camera basis, vertical fov, and tonemap exposure
        // must flow from the single ReferenceSceneDesc into BOTH paths —
        // never from divergent hardcoded constants.
        let pt = include_str!("adjudication.rs");
        let raster = include_str!("../offscreen/adjudication_raster.rs");
        let capture = include_str!("../py_functions/adjudication.rs");
        for src in [pt, raster] {
            assert!(src.contains("desc.camera_basis()"));
            assert!(src.contains("desc.fov_y_rad()"));
        }
        assert!(pt.contains("cam_exposure: desc.exposure"));
        // The capture API resolves BOTH HDR buffers with desc.exposure.
        assert!(
            capture.matches("desc.exposure").count() >= 2,
            "both tonemap resolves must use ReferenceSceneDesc::exposure"
        );
    }

    #[test]
    fn plane_winding_points_up() {
        let mesh = adjudication_scene().plane_mesh();
        for tri in &mesh.indices {
            let v0 = Vec3::from(mesh.vertices[tri[0] as usize]);
            let v1 = Vec3::from(mesh.vertices[tri[1] as usize]);
            let v2 = Vec3::from(mesh.vertices[tri[2] as usize]);
            let n = (v1 - v0).cross(v2 - v0);
            assert!(n.y > 0.0, "plane triangle normal must point +Y");
        }
    }
}
