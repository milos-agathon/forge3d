#[cfg(feature = "extension-module")]
#[derive(Clone)]
pub struct VTLayerFamilyNative {
    // Python accepts "albedo", "normal", and "mask". The current terrain VT
    // runtime pages only the albedo family and ignores the others for now.
    pub family: String,
    pub virtual_size: (u32, u32),
    pub tile_size: u32,
    pub tile_border: u32,
    pub fallback: [f32; 4],
}

#[cfg(feature = "extension-module")]
#[derive(Clone)]
pub struct TerrainVTSettingsNative {
    pub enabled: bool,
    pub layers: Vec<VTLayerFamilyNative>,
    pub atlas_size: u32,
    pub residency_budget_mb: f32,
    pub max_mip_levels: u32,
    pub use_feedback: bool,
}

#[cfg(feature = "extension-module")]
impl Default for TerrainVTSettingsNative {
    fn default() -> Self {
        Self {
            enabled: false,
            layers: vec![],
            atlas_size: 4096,
            residency_budget_mb: 256.0,
            max_mip_levels: 8,
            use_feedback: true,
        }
    }
}
