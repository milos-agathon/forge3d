// src/lighting/types.rs
// P0 lighting type definitions with GPU-aligned layouts
// All types are repr(C) and bytemuck-compatible for GPU upload

use bytemuck::{Pod, Zeroable};

/// Light type enumeration (P0)
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LightType {
    Directional = 0,
    Point = 1,
    Spot = 2,
    Environment = 3,
}

impl LightType {
    pub fn as_u32(self) -> u32 {
        self as u32
    }
}

/// BRDF model enumeration (P0)
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BrdfModel {
    Lambert = 0,
    CookTorranceGgx = 1,
}

impl BrdfModel {
    pub fn as_u32(self) -> u32 {
        self as u32
    }
}

/// Shadow technique enumeration (P0)
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShadowTechnique {
    Hard = 0,
    PCF = 1,
}

impl ShadowTechnique {
    pub fn as_u32(self) -> u32 {
        self as u32
    }
}

/// Global illumination technique enumeration (P0)
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GiTechnique {
    None = 0,
    Ibl = 1,
}

impl GiTechnique {
    pub fn as_u32(self) -> u32 {
        self as u32
    }
}

/// Light configuration (P0)
/// GPU-aligned struct for uniform buffer upload
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Light {
    /// Light type (0=Directional, 1=Point, 2=Spot, 3=Environment)
    pub kind: u32,
    /// Light intensity multiplier
    pub intensity: f32,
    /// Light color (RGB)
    pub color: [f32; 3],

    /// Direction (for directional) or position (for point/spot)
    pub dir_or_pos: [f32; 3],
    /// Range for point/spot lights (unused for directional/environment)
    pub range: f32,

    /// Spot light inner cone angle (degrees)
    pub spot_inner_deg: f32,
    /// Spot light outer cone angle (degrees)
    pub spot_outer_deg: f32,
    /// Padding for 16-byte alignment
    pub _pad: [f32; 2],
}

impl Default for Light {
    fn default() -> Self {
        Self {
            kind: LightType::Directional.as_u32(),
            intensity: 3.0,
            color: [1.0, 1.0, 1.0],
            dir_or_pos: [0.0, 1.0, 0.0],
            range: 1000.0,
            spot_inner_deg: 20.0,
            spot_outer_deg: 35.0,
            _pad: [0.0; 2],
        }
    }
}

impl Light {
    /// Create a directional light from azimuth and elevation angles
    pub fn directional(azimuth_deg: f32, elevation_deg: f32, intensity: f32, color: [f32; 3]) -> Self {
        let az_rad = azimuth_deg.to_radians();
        let el_rad = elevation_deg.to_radians();

        // Convert spherical to Cartesian (pointing down for directional light)
        let x = el_rad.cos() * az_rad.sin();
        let y = -el_rad.sin();
        let z = -el_rad.cos() * az_rad.cos();

        Self {
            kind: LightType::Directional.as_u32(),
            intensity,
            color,
            dir_or_pos: [x, y, z],
            range: 0.0,
            spot_inner_deg: 0.0,
            spot_outer_deg: 0.0,
            _pad: [0.0; 2],
        }
    }

    /// Create a point light at a position
    pub fn point(position: [f32; 3], range: f32, intensity: f32, color: [f32; 3]) -> Self {
        Self {
            kind: LightType::Point.as_u32(),
            intensity,
            color,
            dir_or_pos: position,
            range,
            spot_inner_deg: 0.0,
            spot_outer_deg: 0.0,
            _pad: [0.0; 2],
        }
    }

    /// Create a spot light
    pub fn spot(
        position: [f32; 3],
        _direction: [f32; 3], // TODO: Use direction for spot light orientation in P0-S1
        range: f32,
        inner_deg: f32,
        outer_deg: f32,
        intensity: f32,
        color: [f32; 3],
    ) -> Self {
        Self {
            kind: LightType::Spot.as_u32(),
            intensity,
            color,
            dir_or_pos: position,
            range,
            spot_inner_deg: inner_deg,
            spot_outer_deg: outer_deg,
            _pad: [0.0; 2],
        }
    }

    /// Create an environment/IBL light
    pub fn environment(intensity: f32) -> Self {
        Self {
            kind: LightType::Environment.as_u32(),
            intensity,
            color: [1.0, 1.0, 1.0],
            dir_or_pos: [0.0; 3],
            range: 0.0,
            spot_inner_deg: 0.0,
            spot_outer_deg: 0.0,
            _pad: [0.0; 2],
        }
    }
}

/// Material shading parameters (P0)
/// GPU-aligned struct for uniform buffer upload
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct MaterialShading {
    /// BRDF model (0=Lambert, 1=CookTorranceGgx)
    pub brdf: u32,
    /// Surface roughness [0, 1]
    pub roughness: f32,
    /// Metallic factor [0, 1]
    pub metallic: f32,
    /// Index of refraction (typically ~1.5 for dielectrics)
    pub ior: f32,
}

impl Default for MaterialShading {
    fn default() -> Self {
        Self {
            brdf: BrdfModel::CookTorranceGgx.as_u32(),
            roughness: 0.5,
            metallic: 0.0,
            ior: 1.5,
        }
    }
}

impl MaterialShading {
    /// Validate material parameters
    pub fn validate(&self) -> Result<(), String> {
        if self.roughness < 0.0 || self.roughness > 1.0 {
            return Err(format!("roughness must be in [0,1], got {}", self.roughness));
        }
        if self.metallic < 0.0 || self.metallic > 1.0 {
            return Err(format!("metallic must be in [0,1], got {}", self.metallic));
        }
        if self.ior < 1.0 || self.ior > 3.0 {
            return Err(format!("ior should be in [1,3], got {}", self.ior));
        }
        Ok(())
    }
}

/// Shadow settings (P0)
/// GPU-aligned struct for uniform buffer upload
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ShadowSettings {
    /// Shadow technique (0=Hard, 1=PCF)
    pub tech: u32,
    /// Shadow map resolution (default 2048)
    pub map_res: u32,
    /// Shadow bias to prevent acne
    pub bias: f32,
    /// Normal-based bias
    pub normal_bias: f32,

    /// PCF softness/filter radius
    pub softness: f32,
    /// Padding for 16-byte alignment
    pub _pad: [f32; 3],
}

impl Default for ShadowSettings {
    fn default() -> Self {
        Self {
            tech: ShadowTechnique::PCF.as_u32(),
            map_res: 2048,
            bias: 0.002,
            normal_bias: 0.5,
            softness: 1.25,
            _pad: [0.0; 3],
        }
    }
}

impl ShadowSettings {
    /// Validate shadow settings
    pub fn validate(&self) -> Result<(), String> {
        if self.map_res > 4096 {
            return Err(format!("map_res must be <= 4096, got {}", self.map_res));
        }
        if !self.map_res.is_power_of_two() {
            return Err(format!("map_res must be power of two, got {}", self.map_res));
        }
        Ok(())
    }

    /// Calculate memory budget for shadow atlas (in bytes)
    pub fn memory_budget(&self) -> u64 {
        // D32 format = 4 bytes per pixel
        let pixels = (self.map_res as u64) * (self.map_res as u64);
        pixels * 4
    }
}

/// Global illumination settings (P0)
/// GPU-aligned struct for uniform buffer upload
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GiSettings {
    /// GI technique (0=None, 1=IBL)
    pub tech: u32,
    /// IBL intensity multiplier
    pub ibl_intensity: f32,
    /// IBL environment rotation (degrees)
    pub ibl_rotation_deg: f32,
    /// Padding for 16-byte alignment
    pub _pad: f32,
}

impl Default for GiSettings {
    fn default() -> Self {
        Self {
            tech: GiTechnique::Ibl.as_u32(),
            ibl_intensity: 1.0,
            ibl_rotation_deg: 0.0,
            _pad: 0.0,
        }
    }
}

/// Atmospheric settings (P0)
/// GPU-aligned struct for uniform buffer upload
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Atmosphere {
    /// Fog density
    pub fog_density: f32,
    /// Exposure multiplier
    pub exposure: f32,
    /// Sky model (0=Off, 1=Preetham)
    pub sky_model: u32,
    /// Padding for 16-byte alignment
    pub _pad: f32,
}

impl Default for Atmosphere {
    fn default() -> Self {
        Self {
            fog_density: 0.0,
            exposure: 1.0,
            sky_model: 0,
            _pad: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_light_sizes() {
        // Verify sizes are multiples of 16 for UBO alignment
        assert_eq!(std::mem::size_of::<Light>() % 16, 0);
        assert_eq!(std::mem::size_of::<MaterialShading>() % 16, 0);
        assert_eq!(std::mem::size_of::<ShadowSettings>() % 16, 0);
        assert_eq!(std::mem::size_of::<GiSettings>() % 16, 0);
        assert_eq!(std::mem::size_of::<Atmosphere>() % 16, 0);
    }

    #[test]
    fn test_material_validation() {
        let mut mat = MaterialShading::default();
        assert!(mat.validate().is_ok());

        mat.roughness = 1.5;
        assert!(mat.validate().is_err());

        mat.roughness = 0.5;
        mat.metallic = -0.1;
        assert!(mat.validate().is_err());
    }

    #[test]
    fn test_shadow_validation() {
        let mut shadow = ShadowSettings::default();
        assert!(shadow.validate().is_ok());

        shadow.map_res = 5000;
        assert!(shadow.validate().is_err());

        shadow.map_res = 2047;
        assert!(shadow.validate().is_err());
    }

    #[test]
    fn test_shadow_memory_budget() {
        let shadow = ShadowSettings {
            map_res: 2048,
            ..Default::default()
        };
        // 2048 * 2048 * 4 bytes = 16 MiB
        assert_eq!(shadow.memory_budget(), 16 * 1024 * 1024);
    }
}
