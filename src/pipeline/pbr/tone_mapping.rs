/// Tone mapping operators available to the PBR pipeline
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToneMappingMode {
    /// Filmic ACES approximation
    Aces,
    /// Classic Reinhard curve
    Reinhard,
    /// Hable filmic curve (Uncharted 2)
    Hable,
}

impl ToneMappingMode {
    /// Map enum variant to shader index for tone mapping selection
    pub fn as_index(self) -> u32 {
        match self {
            ToneMappingMode::Aces => 0,
            ToneMappingMode::Reinhard => 1,
            ToneMappingMode::Hable => 2,
        }
    }
}

/// Tone mapping configuration shared between CPU previews and GPU passes
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ToneMappingConfig {
    pub mode: ToneMappingMode,
    pub exposure: f32,
}

impl ToneMappingConfig {
    /// Create new tone mapping configuration with explicit exposure
    pub fn new(mode: ToneMappingMode, exposure: f32) -> Self {
        let exposure = exposure.max(1e-6);
        Self { mode, exposure }
    }

    /// Create config from exposure stops (2**stops multiplier)
    pub fn with_stops(mode: ToneMappingMode, stops: f32) -> Self {
        Self::new(mode, exposure_from_stops(stops))
    }

    /// Update exposure using stops value
    pub fn set_exposure_stops(&mut self, stops: f32) {
        self.exposure = exposure_from_stops(stops);
    }

    /// Convert stored exposure back to stops for UI readback
    pub fn exposure_stops(&self) -> f32 {
        self.exposure.max(1e-6).log2()
    }
}

/// Convert exposure stops to scalar multiplier
pub fn exposure_from_stops(stops: f32) -> f32 {
    2.0_f32.powf(stops)
}

const HABLE_WHITE_POINT: f32 = 11.2;

fn tone_curve_reinhard(value: f32) -> f32 {
    value / (1.0 + value)
}

fn tone_curve_aces(value: f32) -> f32 {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    let numerator = value * (a * value + b);
    let denominator = value * (c * value + d) + e;
    if denominator.abs() < f32::EPSILON {
        0.0
    } else {
        (numerator / denominator).clamp(0.0, 1.0)
    }
}

fn tone_curve_hable(value: f32) -> f32 {
    let a = 0.15;
    let b = 0.50;
    let c = 0.10;
    let d = 0.20;
    let e = 0.02;
    let f = 0.30;
    let numerator = value * (value * a + c * b) + d * e;
    let denominator = value * (value * a + b) + d * f;
    if denominator.abs() < f32::EPSILON {
        0.0
    } else {
        let tone = numerator / denominator - e / f;
        let white_scale = {
            let w_num = HABLE_WHITE_POINT * (HABLE_WHITE_POINT * a + c * b) + d * e;
            let w_den = HABLE_WHITE_POINT * (HABLE_WHITE_POINT * a + b) + d * f;
            if w_den.abs() < f32::EPSILON {
                1.0
            } else {
                (w_num / w_den) - e / f
            }
        };
        if white_scale.abs() < f32::EPSILON {
            0.0
        } else {
            (tone / white_scale).clamp(0.0, 1.0)
        }
    }
}

fn tone_map_scalar(value: f32, config: ToneMappingConfig) -> f32 {
    let exposed = (value * config.exposure).max(0.0);
    match config.mode {
        ToneMappingMode::Aces => tone_curve_aces(exposed),
        ToneMappingMode::Reinhard => tone_curve_reinhard(exposed),
        ToneMappingMode::Hable => tone_curve_hable(exposed),
    }
}

/// Apply tone mapping to RGB color using provided configuration
pub fn tone_map_color(color: [f32; 3], config: ToneMappingConfig) -> [f32; 3] {
    [
        tone_map_scalar(color[0], config),
        tone_map_scalar(color[1], config),
        tone_map_scalar(color[2], config),
    ]
}

/// Provide WGSL source for the tone mapping pass used by shaders
pub fn tone_map_shader_source() -> &'static str {
    include_str!("../../shaders/tone_map.wgsl")
}
