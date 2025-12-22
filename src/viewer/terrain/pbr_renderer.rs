// src/viewer/terrain/pbr_renderer.rs
// Bridge to TerrainRenderer for PBR+POM terrain rendering in the interactive viewer

use std::path::PathBuf;
use crate::viewer::viewer_enums::{
    ViewerHeightAoConfig, ViewerSunVisConfig,
    ViewerMaterialLayerConfig, ViewerVectorOverlayConfig, ViewerTonemapConfig,
};

/// Configuration for PBR terrain rendering mode
#[derive(Debug, Clone)]
pub struct ViewerTerrainPbrConfig {
    /// Enable PBR mode (false = legacy simple shader)
    pub enabled: bool,
    /// Path to HDR environment map for IBL
    pub hdr_path: Option<PathBuf>,
    /// IBL intensity multiplier
    pub ibl_intensity: f32,
    /// Shadow technique: "none", "hard", "pcf", "pcss"
    pub shadow_technique: String,
    /// Shadow map resolution
    pub shadow_map_res: u32,
    /// ACES exposure multiplier
    pub exposure: f32,
    /// MSAA samples (1, 4, or 8)
    pub msaa: u32,
    /// Terrain normal strength multiplier
    pub normal_strength: f32,
    /// Heightfield ray-traced AO settings
    pub height_ao: HeightAoConfig,
    /// Heightfield ray-traced sun visibility settings
    pub sun_visibility: SunVisConfig,
    /// M4: Material layer settings (snow/rock/wetness)
    pub materials: MaterialLayerConfig,
    /// M5: Vector overlay settings (depth test, halos)
    pub vector_overlay: VectorOverlayConfig,
    /// M6: Tonemap settings (operator, white balance)
    pub tonemap: TonemapConfig,
    /// M5: Lens effects settings (vignette, distortion, CA)
    pub lens_effects: LensEffectsConfig,
    /// M3: Depth of Field settings
    pub dof: DofConfig,
}

/// Internal heightfield AO configuration
#[derive(Debug, Clone)]
pub struct HeightAoConfig {
    pub enabled: bool,
    pub directions: u32,
    pub steps: u32,
    pub max_distance: f32,
    pub strength: f32,
    pub resolution_scale: f32,
}

impl Default for HeightAoConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            directions: 6,
            steps: 16,
            max_distance: 200.0,
            strength: 1.0,
            resolution_scale: 0.5,
        }
    }
}

/// Internal sun visibility configuration
#[derive(Debug, Clone)]
pub struct SunVisConfig {
    pub enabled: bool,
    pub mode: String,
    pub samples: u32,
    pub steps: u32,
    pub max_distance: f32,
    pub softness: f32,
    pub bias: f32,
    pub resolution_scale: f32,
}

impl Default for SunVisConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            mode: "soft".to_string(),
            samples: 4,
            steps: 24,
            max_distance: 400.0,
            softness: 1.0,
            bias: 0.01,
            resolution_scale: 0.5,
        }
    }
}

/// M4: Internal material layer configuration
#[derive(Debug, Clone)]
pub struct MaterialLayerConfig {
    pub snow_enabled: bool,
    pub snow_altitude_min: f32,
    pub snow_altitude_blend: f32,
    pub snow_slope_max: f32,
    pub rock_enabled: bool,
    pub rock_slope_min: f32,
    pub wetness_enabled: bool,
    pub wetness_strength: f32,
}

impl Default for MaterialLayerConfig {
    fn default() -> Self {
        Self {
            snow_enabled: false,
            snow_altitude_min: 2500.0,
            snow_altitude_blend: 200.0,
            snow_slope_max: 45.0,
            rock_enabled: false,
            rock_slope_min: 45.0,
            wetness_enabled: false,
            wetness_strength: 0.3,
        }
    }
}

/// M5: Internal vector overlay configuration
#[derive(Debug, Clone)]
pub struct VectorOverlayConfig {
    pub depth_test: bool,
    pub depth_bias: f32,
    pub halo_enabled: bool,
    pub halo_width: f32,
    pub halo_color: [f32; 4],
}

impl Default for VectorOverlayConfig {
    fn default() -> Self {
        Self {
            depth_test: false,
            depth_bias: 0.001,
            halo_enabled: false,
            halo_width: 2.0,
            halo_color: [0.0, 0.0, 0.0, 0.5],
        }
    }
}

/// M6: Internal tonemap configuration
#[derive(Debug, Clone)]
pub struct TonemapConfig {
    pub operator: String,
    pub white_point: f32,
    pub white_balance_enabled: bool,
    pub temperature: f32,
    pub tint: f32,
}

/// M5: Internal lens effects configuration
#[derive(Debug, Clone)]
pub struct LensEffectsConfig {
    pub enabled: bool,
    pub vignette_strength: f32,
    pub vignette_radius: f32,
    pub vignette_softness: f32,
    pub distortion: f32,
    pub chromatic_aberration: f32,
}

impl Default for LensEffectsConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            vignette_strength: 0.0,
            vignette_radius: 0.7,
            vignette_softness: 0.3,
            distortion: 0.0,
            chromatic_aberration: 0.0,
        }
    }
}

/// M3: Internal DoF configuration
#[derive(Debug, Clone)]
pub struct DofConfig {
    pub enabled: bool,
    pub focus_distance: f32,
    pub f_stop: f32,
    pub focal_length: f32,
    pub quality: u32,
    pub max_blur_radius: f32,
    pub blur_strength: f32,  // Artistic multiplier for landscape DoF (1.0 = physical, higher = more blur)
}

impl Default for DofConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            focus_distance: 500.0,
            f_stop: 5.6,
            focal_length: 50.0,
            quality: 8,
            max_blur_radius: 16.0,
            blur_strength: 1000.0,  // High default for visible effect at landscape distances
        }
    }
}

impl Default for TonemapConfig {
    fn default() -> Self {
        Self {
            operator: "aces".to_string(),
            white_point: 4.0,
            white_balance_enabled: false,
            temperature: 6500.0,
            tint: 0.0,
        }
    }
}

impl Default for ViewerTerrainPbrConfig {
    fn default() -> Self {
        Self {
            enabled: false, // CRITICAL: Legacy mode is default per AGENTS.md
            hdr_path: None,
            ibl_intensity: 1.0,
            shadow_technique: "pcss".to_string(),
            shadow_map_res: 2048,
            exposure: 1.0,
            msaa: 1,
            normal_strength: 1.0,
            height_ao: HeightAoConfig::default(),
            sun_visibility: SunVisConfig::default(),
            materials: MaterialLayerConfig::default(),
            vector_overlay: VectorOverlayConfig::default(),
            tonemap: TonemapConfig::default(),
            lens_effects: LensEffectsConfig::default(),
            dof: DofConfig::default(),
        }
    }
}

impl ViewerTerrainPbrConfig {
    /// Update config from optional parameters (used by IPC handler)
    #[allow(clippy::too_many_arguments)]
    pub fn apply_updates(
        &mut self,
        enabled: Option<bool>,
        hdr_path: Option<String>,
        ibl_intensity: Option<f32>,
        shadow_technique: Option<String>,
        shadow_map_res: Option<u32>,
        exposure: Option<f32>,
        msaa: Option<u32>,
        normal_strength: Option<f32>,
        height_ao: Option<ViewerHeightAoConfig>,
        sun_visibility: Option<ViewerSunVisConfig>,
        materials: Option<ViewerMaterialLayerConfig>,
        vector_overlay: Option<ViewerVectorOverlayConfig>,
        tonemap: Option<ViewerTonemapConfig>,
    ) {
        if let Some(v) = enabled {
            self.enabled = v;
        }
        if let Some(p) = hdr_path {
            self.hdr_path = Some(PathBuf::from(p));
        }
        if let Some(v) = ibl_intensity {
            self.ibl_intensity = v.max(0.0);
        }
        if let Some(t) = shadow_technique {
            let t_lower = t.to_lowercase();
            if ["none", "hard", "pcf", "pcss"].contains(&t_lower.as_str()) {
                self.shadow_technique = t_lower;
            }
        }
        if let Some(r) = shadow_map_res {
            self.shadow_map_res = r.clamp(512, 8192);
        }
        if let Some(e) = exposure {
            self.exposure = e.max(0.0);
        }
        if let Some(m) = msaa {
            self.msaa = match m {
                1 | 4 | 8 => m,
                _ => 1,
            };
        }
        if let Some(n) = normal_strength {
            self.normal_strength = n.clamp(0.0, 10.0);
        }
        if let Some(ao) = height_ao {
            self.height_ao.enabled = ao.enabled;
            self.height_ao.directions = ao.directions.clamp(4, 16);
            self.height_ao.steps = ao.steps.clamp(8, 64);
            self.height_ao.max_distance = ao.max_distance.max(0.0);
            self.height_ao.strength = ao.strength.clamp(0.0, 2.0);
            self.height_ao.resolution_scale = ao.resolution_scale.clamp(0.1, 1.0);
        }
        if let Some(sv) = sun_visibility {
            self.sun_visibility.enabled = sv.enabled;
            self.sun_visibility.mode = if sv.mode == "hard" { "hard".to_string() } else { "soft".to_string() };
            self.sun_visibility.samples = sv.samples.clamp(1, 16);
            self.sun_visibility.steps = sv.steps.clamp(8, 64);
            self.sun_visibility.max_distance = sv.max_distance.max(0.0);
            self.sun_visibility.softness = sv.softness.clamp(0.0, 4.0);
            self.sun_visibility.bias = sv.bias.clamp(0.0, 0.1);
            self.sun_visibility.resolution_scale = sv.resolution_scale.clamp(0.1, 1.0);
        }
        // M4: Material layer config
        if let Some(mat) = materials {
            self.materials.snow_enabled = mat.snow_enabled;
            self.materials.snow_altitude_min = mat.snow_altitude_min.max(0.0);
            self.materials.snow_altitude_blend = mat.snow_altitude_blend.max(0.0);
            self.materials.snow_slope_max = mat.snow_slope_max.clamp(0.0, 90.0);
            self.materials.rock_enabled = mat.rock_enabled;
            self.materials.rock_slope_min = mat.rock_slope_min.clamp(0.0, 90.0);
            self.materials.wetness_enabled = mat.wetness_enabled;
            self.materials.wetness_strength = mat.wetness_strength.clamp(0.0, 1.0);
        }
        // M5: Vector overlay config
        if let Some(vo) = vector_overlay {
            self.vector_overlay.depth_test = vo.depth_test;
            self.vector_overlay.depth_bias = vo.depth_bias.max(0.0);
            self.vector_overlay.halo_enabled = vo.halo_enabled;
            self.vector_overlay.halo_width = vo.halo_width.max(0.0);
            self.vector_overlay.halo_color = vo.halo_color;
        }
        // M6: Tonemap config
        if let Some(tm) = tonemap {
            let valid_ops = ["reinhard", "reinhard_extended", "aces", "uncharted2", "exposure"];
            if valid_ops.contains(&tm.operator.to_lowercase().as_str()) {
                self.tonemap.operator = tm.operator.to_lowercase();
            }
            self.tonemap.white_point = tm.white_point.max(0.1);
            self.tonemap.white_balance_enabled = tm.white_balance_enabled;
            self.tonemap.temperature = tm.temperature.clamp(2000.0, 12000.0);
            self.tonemap.tint = tm.tint.clamp(-1.0, 1.0);
        }
    }
    
    /// Apply lens effects config from IPC
    pub fn apply_lens_effects(&mut self, enabled: bool, vignette: f32, radius: f32, softness: f32, distortion: f32, ca: f32) {
        self.lens_effects.enabled = enabled;
        self.lens_effects.vignette_strength = vignette.clamp(0.0, 1.0);
        self.lens_effects.vignette_radius = radius.clamp(0.1, 1.0);
        self.lens_effects.vignette_softness = softness.clamp(0.1, 1.0);
        self.lens_effects.distortion = distortion.clamp(-0.5, 0.5);
        self.lens_effects.chromatic_aberration = ca.clamp(0.0, 0.1);
    }
    
    /// Apply DoF config from IPC
    pub fn apply_dof(&mut self, enabled: bool, f_stop: f32, focus_distance: f32, focal_length: f32, quality: &str) {
        self.dof.enabled = enabled;
        self.dof.f_stop = f_stop.clamp(1.4, 22.0);
        self.dof.focus_distance = focus_distance.max(1.0);
        self.dof.focal_length = focal_length.clamp(10.0, 200.0);
        self.dof.quality = match quality.to_lowercase().as_str() {
            "low" => 4,
            "medium" => 8,
            "high" => 16,
            _ => 8,
        };
    }

    /// Format config as display string
    pub fn to_display_string(&self) -> String {
        let mut parts = vec![
            format!("PBR: {}", if self.enabled { "ON" } else { "OFF" }),
            format!("shadow={} res={}", self.shadow_technique, self.shadow_map_res),
            format!("IBL={:.2} exp={:.2}", self.ibl_intensity, self.exposure),
            format!("msaa={} normal={:.2}", self.msaa, self.normal_strength),
        ];
        if self.height_ao.enabled {
            parts.push(format!("height_ao=ON dirs={} steps={}", 
                self.height_ao.directions, self.height_ao.steps));
        }
        if self.sun_visibility.enabled {
            parts.push(format!("sun_vis={} samples={} steps={}", 
                self.sun_visibility.mode, self.sun_visibility.samples, self.sun_visibility.steps));
        }
        // M4-M6 display
        if self.materials.snow_enabled || self.materials.rock_enabled {
            parts.push(format!("materials: snow={} rock={}", 
                self.materials.snow_enabled, self.materials.rock_enabled));
        }
        if self.vector_overlay.depth_test || self.vector_overlay.halo_enabled {
            parts.push(format!("overlay: depth={} halo={}", 
                self.vector_overlay.depth_test, self.vector_overlay.halo_enabled));
        }
        parts.push(format!("tonemap={}", self.tonemap.operator));
        parts.join(" | ")
    }
}
