#[cfg(feature = "extension-module")]
#[derive(Clone)]
pub struct TonemapSettingsNative {
    pub operator_index: u32,
    pub white_point: f32,
    pub lut_enabled: bool,
    pub lut_path: Option<String>,
    pub lut_strength: f32,
    pub white_balance_enabled: bool,
    pub temperature: f32,
    pub tint: f32,
}

#[cfg(feature = "extension-module")]
impl Default for TonemapSettingsNative {
    fn default() -> Self {
        Self {
            operator_index: 2,
            white_point: 4.0,
            lut_enabled: false,
            lut_path: None,
            lut_strength: 1.0,
            white_balance_enabled: false,
            temperature: 6500.0,
            tint: 0.0,
        }
    }
}

#[cfg(feature = "extension-module")]
impl TonemapSettingsNative {
    pub fn operator_from_str(s: &str) -> u32 {
        match s.to_lowercase().as_str() {
            "reinhard" => 0,
            "reinhard_extended" => 1,
            "aces" => 2,
            "uncharted2" => 3,
            "exposure" => 4,
            _ => 2,
        }
    }
}
