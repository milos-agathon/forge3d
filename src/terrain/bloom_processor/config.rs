/// Bloom configuration for terrain rendering
#[derive(Debug, Clone, Copy)]
pub struct TerrainBloomConfig {
    /// Enabled flag (false = passthrough, identical output)
    pub enabled: bool,
    /// Brightness threshold for bloom extraction (default 1.5 = HDR only)
    pub threshold: f32,
    /// Softness of threshold transition (0.0 = hard, 1.0 = very soft)
    pub softness: f32,
    /// Bloom intensity when compositing (0.0-1.0+)
    pub intensity: f32,
    /// Blur radius multiplier (affects spread)
    pub radius: f32,
}

impl Default for TerrainBloomConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            threshold: 1.5,
            softness: 0.5,
            intensity: 0.3,
            radius: 1.0,
        }
    }
}
