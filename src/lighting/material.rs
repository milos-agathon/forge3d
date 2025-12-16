// src/lighting/material.rs
// Material shading definitions with GPU-aligned layouts
// Split from types.rs for single-responsibility

use bytemuck::{Pod, Zeroable};

/// BRDF model enumeration (P0/P2)
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BrdfModel {
    Lambert = 0,
    Phong = 1,
    BlinnPhong = 2,
    OrenNayar = 3,
    CookTorranceGgx = 4,
    CookTorranceBeckmann = 5,
    DisneyPrincipled = 6,
    AshikhminShirley = 7,
    Ward = 8,
    Toon = 9,
    Minnaert = 10,
    Subsurface = 11,
    Hair = 12,
}

impl BrdfModel {
    pub fn as_u32(self) -> u32 {
        self as u32
    }
}

/// Material shading parameters
/// GPU-aligned struct for uniform buffer upload
/// Size: 32 bytes (2 vec4s)
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct MaterialShading {
    pub brdf: u32,
    pub metallic: f32,
    pub roughness: f32,
    pub sheen: f32,
    pub clearcoat: f32,
    pub subsurface: f32,
    pub anisotropy: f32,
    pub _pad: f32,
}

impl Default for MaterialShading {
    fn default() -> Self {
        Self {
            brdf: BrdfModel::CookTorranceGgx.as_u32(),
            metallic: 0.0,
            roughness: 0.5,
            sheen: 0.0,
            clearcoat: 0.0,
            subsurface: 0.0,
            anisotropy: 0.0,
            _pad: 0.0,
        }
    }
}

/// Type alias for GPU upload
pub type ShadingParamsGPU = MaterialShading;

impl MaterialShading {
    pub fn validate(&self) -> Result<(), String> {
        if self.roughness < 0.0 || self.roughness > 1.0 {
            return Err(format!(
                "roughness must be in [0,1], got {}",
                self.roughness
            ));
        }
        if self.metallic < 0.0 || self.metallic > 1.0 {
            return Err(format!("metallic must be in [0,1], got {}", self.metallic));
        }
        if self.sheen < 0.0 || self.sheen > 1.0 {
            return Err(format!("sheen must be in [0,1], got {}", self.sheen));
        }
        if self.clearcoat < 0.0 || self.clearcoat > 1.0 {
            return Err(format!(
                "clearcoat must be in [0,1], got {}",
                self.clearcoat
            ));
        }
        if self.subsurface < 0.0 || self.subsurface > 1.0 {
            return Err(format!(
                "subsurface must be in [0,1], got {}",
                self.subsurface
            ));
        }
        if self.anisotropy < -1.0 || self.anisotropy > 1.0 {
            return Err(format!(
                "anisotropy must be in [-1,1], got {}",
                self.anisotropy
            ));
        }
        Ok(())
    }

    pub fn lambert(roughness: f32) -> Self {
        Self {
            brdf: BrdfModel::Lambert.as_u32(),
            roughness,
            ..Default::default()
        }
    }

    pub fn phong(roughness: f32, metallic: f32) -> Self {
        Self {
            brdf: BrdfModel::Phong.as_u32(),
            roughness,
            metallic,
            ..Default::default()
        }
    }

    pub fn disney(
        roughness: f32,
        metallic: f32,
        sheen: f32,
        clearcoat: f32,
        subsurface: f32,
    ) -> Self {
        Self {
            brdf: BrdfModel::DisneyPrincipled.as_u32(),
            metallic,
            roughness,
            sheen,
            clearcoat,
            subsurface,
            anisotropy: 0.0,
            _pad: 0.0,
        }
    }

    pub fn anisotropic(brdf_model: BrdfModel, roughness: f32, anisotropy: f32) -> Self {
        Self {
            brdf: brdf_model.as_u32(),
            roughness,
            anisotropy,
            ..Default::default()
        }
    }
}
