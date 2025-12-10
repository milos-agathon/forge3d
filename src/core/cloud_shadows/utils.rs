use super::types::CloudAnimationParams;
use glam::Vec2;

/// Calculate optimal cloud speed based on wind parameters
pub fn calculate_cloud_speed(wind_direction: f32, wind_strength: f32, base_speed: f32) -> Vec2 {
    Vec2::new(
        wind_direction.cos() * wind_strength * base_speed,
        wind_direction.sin() * wind_strength * base_speed,
    )
}

/// Calculate cloud density based on coverage and base density
pub fn calculate_effective_density(base_density: f32, coverage: f32) -> f32 {
    (base_density * coverage).clamp(0.0, 1.0)
}

/// Calculate shadow intensity based on sun elevation
pub fn calculate_shadow_intensity(base_intensity: f32, sun_elevation: f32) -> f32 {
    // Shadows are stronger when sun is higher
    let elevation_factor = (sun_elevation.to_radians().sin()).max(0.0);
    (base_intensity * elevation_factor).clamp(0.0, 1.0)
}

/// Create cloud animation preset
pub fn create_animation_preset(preset_name: &str) -> CloudAnimationParams {
    match preset_name {
        "calm" => CloudAnimationParams {
            speed: Vec2::new(0.005, 0.003),
            wind_direction: 0.0,
            wind_strength: 0.2,
            turbulence: 0.05,
        },
        "windy" => CloudAnimationParams {
            speed: Vec2::new(0.02, 0.01),
            wind_direction: 45.0_f32.to_radians(),
            wind_strength: 1.5,
            turbulence: 0.2,
        },
        "stormy" => CloudAnimationParams {
            speed: Vec2::new(0.05, 0.03),
            wind_direction: 180.0_f32.to_radians(),
            wind_strength: 3.0,
            turbulence: 0.5,
        },
        _ => CloudAnimationParams::default(),
    }
}
