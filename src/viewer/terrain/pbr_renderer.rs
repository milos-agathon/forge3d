// src/viewer/terrain/pbr_renderer.rs
// Bridge to TerrainRenderer for PBR+POM terrain rendering in the interactive viewer

use std::path::PathBuf;

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
    }

    /// Format config as display string
    pub fn to_display_string(&self) -> String {
        format!(
            "PBR: {} | shadow={} res={} | IBL={:.2} exp={:.2} | msaa={} normal={:.2}",
            if self.enabled { "ON" } else { "OFF" },
            self.shadow_technique,
            self.shadow_map_res,
            self.ibl_intensity,
            self.exposure,
            self.msaa,
            self.normal_strength
        )
    }
}
