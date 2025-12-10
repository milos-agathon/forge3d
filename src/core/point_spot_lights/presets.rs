use super::types::Light;

/// Predefined light configurations
#[derive(Debug, Clone, Copy)]
pub enum LightPreset {
    /// Standard room light
    RoomLight,
    /// Bright desk lamp
    DeskLamp,
    /// Outdoor street light
    StreetLight,
    /// Spotlight for stage/theater
    Spotlight,
    /// Car headlight
    Headlight,
    /// Flashlight
    Flashlight,
    /// Candle flame
    Candle,
    /// Warm living room lamp
    WarmLamp,
}

impl LightPreset {
    pub fn to_light(self, position: [f32; 3]) -> Light {
        match self {
            Self::RoomLight => Light::point(
                position,
                [1.0, 0.95, 0.9], // Warm white
                2.0,              // Medium intensity
                15.0,             // Good room coverage
            ),
            Self::DeskLamp => Light::spot(
                position,
                [0.0, -1.0, 0.2], // Slightly forward
                [1.0, 1.0, 0.95], // Cool white
                3.0,              // Bright
                8.0,              // Focused range
                25.0,             // Inner cone
                40.0,             // Outer cone
                0.8,              // Medium softness
            ),
            Self::StreetLight => Light::point(
                position,
                [1.0, 0.85, 0.6], // Orange-ish
                4.0,              // Very bright
                30.0,             // Large coverage
            ),
            Self::Spotlight => Light::spot(
                position,
                [0.0, -1.0, 0.0], // Straight down
                [1.0, 1.0, 1.0],  // Pure white
                5.0,              // Very bright
                20.0,             // Long range
                15.0,             // Tight inner cone
                25.0,             // Tight outer cone
                0.3,              // Sharp edge
            ),
            Self::Headlight => Light::spot(
                position,
                [0.0, 0.0, -1.0], // Forward
                [1.0, 1.0, 0.95], // Cool white
                4.0,              // Bright
                50.0,             // Long range
                20.0,             // Inner cone
                35.0,             // Outer cone
                0.6,              // Medium softness
            ),
            Self::Flashlight => Light::spot(
                position,
                [0.0, 0.0, -1.0], // Forward
                [1.0, 1.0, 0.9],  // Slightly warm
                2.5,              // Medium-bright
                25.0,             // Good range
                12.0,             // Tight inner
                30.0,             // Wider outer
                1.2,              // Soft edge
            ),
            Self::Candle => Light::point(
                position,
                [1.0, 0.6, 0.2], // Warm orange
                0.8,             // Dim
                3.0,             // Small range
            ),
            Self::WarmLamp => Light::point(
                position,
                [1.0, 0.8, 0.6], // Very warm
                1.5,             // Cozy intensity
                12.0,            // Medium range
            ),
        }
    }
}
