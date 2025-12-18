// src/viewer/terrain/pbr_renderer.rs
// Bridge to TerrainRenderer for PBR+POM terrain rendering in the interactive viewer

use std::path::PathBuf;
use crate::viewer::viewer_enums::{ViewerHeightAoConfig, ViewerSunVisConfig};

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
        }
    }
}

impl ViewerTerrainPbrConfig {
    /// Update config from optional parameters (used by IPC handler)
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
        parts.join(" | ")
    }
}
