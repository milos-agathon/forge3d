//! PBR Material definitions and core functionality
//!
//! Provides PBR material structs, lighting configuration, and CPU-side BRDF calculations
//! following the metallic-roughness workflow used in glTF and modern game engines.

use glam::{Vec3, Vec4};

/// PBR material properties using metallic-roughness workflow
///
/// Memory layout is binary-compatible with WGSL PbrMaterial struct.
/// See docs/pbr_cpu_gpu_alignment.md for CPU-GPU implementation details.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PbrMaterial {
    /// Base color (albedo) - RGB + alpha
    pub base_color: [f32; 4],

    /// Metallic factor [0, 1] - 0 = dielectric, 1 = metallic
    pub metallic: f32,

    /// Roughness factor [0, 1] - 0 = mirror, 1 = completely rough  
    pub roughness: f32,

    /// Normal map intensity multiplier
    pub normal_scale: f32,

    /// Occlusion strength [0, 1]
    pub occlusion_strength: f32,

    /// Emissive color - RGB
    pub emissive: [f32; 3],

    /// Alpha cutoff for alpha testing
    pub alpha_cutoff: f32,

    /// Texture flags (bitfield for which textures are present)
    pub texture_flags: u32,

    /// Padding for alignment
    pub _padding: [f32; 3],
}

impl Default for PbrMaterial {
    fn default() -> Self {
        Self {
            base_color: [1.0, 1.0, 1.0, 1.0],
            metallic: 0.0,
            roughness: 1.0,
            normal_scale: 1.0,
            occlusion_strength: 1.0,
            emissive: [0.0, 0.0, 0.0],
            alpha_cutoff: 0.5,
            texture_flags: 0,
            _padding: [0.0; 3],
        }
    }
}

/// Texture flags for PBR material
pub mod texture_flags {
    pub const BASE_COLOR: u32 = 1 << 0;
    pub const METALLIC_ROUGHNESS: u32 = 1 << 1;
    pub const NORMAL: u32 = 1 << 2;
    pub const OCCLUSION: u32 = 1 << 3;
    pub const EMISSIVE: u32 = 1 << 4;
}

/// PBR lighting environment
///
/// Memory layout is binary-compatible with WGSL PbrLighting struct.
/// See docs/pbr_cpu_gpu_alignment.md for CPU-GPU implementation details.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PbrLighting {
    /// Directional light direction (world space)
    pub light_direction: [f32; 3],
    pub _padding1: f32,

    /// Directional light color and intensity
    pub light_color: [f32; 3],
    pub light_intensity: f32,

    /// Camera position (world space)
    pub camera_position: [f32; 3],
    pub _padding2: f32,

    /// IBL parameters
    pub ibl_intensity: f32,
    pub ibl_rotation: f32,
    pub exposure: f32,
    pub gamma: f32,
}

impl Default for PbrLighting {
    fn default() -> Self {
        Self {
            light_direction: [0.0, -1.0, 0.3],
            _padding1: 0.0,
            light_color: [1.0, 1.0, 1.0],
            light_intensity: 3.0,
            camera_position: [0.0, 0.0, 5.0],
            _padding2: 0.0,
            ibl_intensity: 1.0,
            ibl_rotation: 0.0,
            exposure: 1.0,
            gamma: 2.2,
        }
    }
}

impl PbrMaterial {
    /// Create a new PBR material with given base color
    pub fn new(base_color: Vec4, metallic: f32, roughness: f32) -> Self {
        Self {
            base_color: base_color.to_array(),
            metallic: metallic.clamp(0.0, 1.0),
            roughness: roughness.clamp(0.04, 1.0), // Minimum roughness to avoid division by zero
            ..Default::default()
        }
    }

    /// Create a metallic material
    pub fn metallic(color: Vec3, roughness: f32) -> Self {
        Self::new(Vec4::new(color.x, color.y, color.z, 1.0), 1.0, roughness)
    }

    /// Create a dielectric (non-metallic) material
    pub fn dielectric(color: Vec3, roughness: f32) -> Self {
        Self::new(Vec4::new(color.x, color.y, color.z, 1.0), 0.0, roughness)
    }

    /// Set the base color
    pub fn with_base_color(mut self, color: Vec4) -> Self {
        self.base_color = color.to_array();
        self
    }

    /// Set the metallic factor
    pub fn with_metallic(mut self, metallic: f32) -> Self {
        self.metallic = metallic.clamp(0.0, 1.0);
        self
    }

    /// Set the roughness factor
    pub fn with_roughness(mut self, roughness: f32) -> Self {
        self.roughness = roughness.clamp(0.04, 1.0);
        self
    }

    /// Set the normal scale
    pub fn with_normal_scale(mut self, scale: f32) -> Self {
        self.normal_scale = scale.max(0.0);
        self
    }

    /// Set the occlusion strength
    pub fn with_occlusion_strength(mut self, strength: f32) -> Self {
        self.occlusion_strength = strength.clamp(0.0, 1.0);
        self
    }

    /// Set the emissive color
    pub fn with_emissive(mut self, emissive: Vec3) -> Self {
        self.emissive = emissive.to_array();
        self
    }

    /// Set the alpha cutoff
    pub fn with_alpha_cutoff(mut self, cutoff: f32) -> Self {
        self.alpha_cutoff = cutoff.clamp(0.0, 1.0);
        self
    }

    /// Check if this is a metallic material
    pub fn is_metallic(&self) -> bool {
        self.metallic >= 0.5
    }

    /// Check if this is a dielectric material
    pub fn is_dielectric(&self) -> bool {
        self.metallic < 0.5
    }

    /// Check if this material has emission
    pub fn is_emissive(&self) -> bool {
        self.emissive[0] > 0.0 || self.emissive[1] > 0.0 || self.emissive[2] > 0.0
    }

    /// Get the base color as Vec4
    pub fn base_color_vec4(&self) -> Vec4 {
        Vec4::from_array(self.base_color)
    }

    /// Get the emissive color as Vec3
    pub fn emissive_vec3(&self) -> Vec3 {
        Vec3::from_array(self.emissive)
    }
}

/// CPU-side BRDF evaluation functions for PBR
pub mod brdf {
    use super::*;
    use std::f32::consts::PI;

    /// GGX distribution function (normal distribution)
    pub fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
        let alpha = roughness * roughness;
        let alpha2 = alpha * alpha;
        let n_dot_h2 = n_dot_h * n_dot_h;

        let num = alpha2;
        let denom = PI * (n_dot_h2 * (alpha2 - 1.0) + 1.0).powi(2);

        num / denom.max(1e-6)
    }

    /// Smith geometry function (masking-shadowing)
    pub fn geometry_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
        let ggx2 = geometry_schlick_ggx(n_dot_v, roughness);
        let ggx1 = geometry_schlick_ggx(n_dot_l, roughness);

        ggx1 * ggx2
    }

    /// Schlick-GGX geometry function
    pub fn geometry_schlick_ggx(n_dot_v: f32, roughness: f32) -> f32 {
        let r = roughness + 1.0;
        let k = (r * r) / 8.0;

        n_dot_v / (n_dot_v * (1.0 - k) + k)
    }

    /// Fresnel-Schlick approximation
    pub fn fresnel_schlick(cos_theta: f32, f0: Vec3) -> Vec3 {
        f0 + (Vec3::ONE - f0) * (1.0 - cos_theta).max(0.0).powi(5)
    }

    /// Lambert diffuse BRDF
    pub fn diffuse_lambert(albedo: Vec3) -> Vec3 {
        albedo / PI
    }

    /// Evaluate Cook-Torrance BRDF
    pub fn evaluate_cook_torrance(
        material: &PbrMaterial,
        light_dir: Vec3, // Direction TO light
        view_dir: Vec3,  // Direction TO viewer
        normal: Vec3,
    ) -> Vec3 {
        let h = (light_dir + view_dir).normalize();

        let n_dot_v = normal.dot(view_dir).max(0.0);
        let n_dot_l = normal.dot(light_dir).max(0.0);
        let n_dot_h = normal.dot(h).max(0.0);
        let v_dot_h = view_dir.dot(h).max(0.0);

        if n_dot_l <= 0.0 || n_dot_v <= 0.0 {
            return Vec3::ZERO;
        }

        let base_color = Vec3::from_array([
            material.base_color[0],
            material.base_color[1],
            material.base_color[2],
        ]);
        let metallic = material.metallic;
        let roughness = material.roughness.max(0.04);

        // Calculate F0 (surface reflection at zero incidence)
        let f0 = Vec3::splat(0.04).lerp(base_color, metallic);

        // Cook-Torrance BRDF components
        let d = distribution_ggx(n_dot_h, roughness);
        let g = geometry_smith(n_dot_v, n_dot_l, roughness);
        let f = fresnel_schlick(v_dot_h, f0);

        // Specular BRDF
        let specular = (d * g * f) / (4.0 * n_dot_v * n_dot_l).max(1e-6);

        // Diffuse BRDF
        let ks = f;
        let kd = (Vec3::ONE - ks) * (1.0 - metallic);
        let diffuse = kd * diffuse_lambert(base_color);

        (diffuse + specular) * n_dot_l
    }
}

/// PBR material presets for common materials
pub mod presets {
    use super::*;

    /// Gold material
    pub fn gold() -> PbrMaterial {
        PbrMaterial::metallic(Vec3::new(1.0, 0.86, 0.57), 0.1)
    }

    /// Silver material
    pub fn silver() -> PbrMaterial {
        PbrMaterial::metallic(Vec3::new(0.95, 0.93, 0.88), 0.05)
    }

    /// Copper material
    pub fn copper() -> PbrMaterial {
        PbrMaterial::metallic(Vec3::new(0.95, 0.64, 0.54), 0.15)
    }

    /// Chrome material
    pub fn chrome() -> PbrMaterial {
        PbrMaterial::metallic(Vec3::new(0.55, 0.56, 0.67), 0.05)
    }

    /// Iron material
    pub fn iron() -> PbrMaterial {
        PbrMaterial::metallic(Vec3::new(0.56, 0.57, 0.58), 0.3)
    }

    /// Plastic material
    pub fn plastic(color: Vec3) -> PbrMaterial {
        PbrMaterial::dielectric(color, 0.7)
    }

    /// Rubber material
    pub fn rubber(color: Vec3) -> PbrMaterial {
        PbrMaterial::dielectric(color, 0.9)
    }

    /// Wood material
    pub fn wood(color: Vec3) -> PbrMaterial {
        PbrMaterial::dielectric(color, 0.8)
    }

    /// Glass material
    pub fn glass(color: Vec3) -> PbrMaterial {
        PbrMaterial::dielectric(color, 0.05)
            .with_base_color(Vec4::new(color.x, color.y, color.z, 0.1))
    }

    /// Ceramic material
    pub fn ceramic(color: Vec3) -> PbrMaterial {
        PbrMaterial::dielectric(color, 0.2)
    }
}
