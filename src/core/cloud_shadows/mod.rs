// src/core/cloud_shadows.rs
// Cloud shadow overlay system for B7 - 2D shadow texture modulation over terrain
// RELEVANT FILES: shaders/cloud_shadows.wgsl, src/scene/mod.rs, examples/cloud_shadows_demo.py

mod renderer;
mod types;
pub mod utils;

pub use renderer::CloudShadowRenderer;
pub use types::{CloudAnimationParams, CloudShadowQuality, CloudShadowUniforms};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cloud_shadow_uniforms_size() {
        // Ensure uniforms match WGSL layout
        assert_eq!(std::mem::size_of::<CloudShadowUniforms>(), 72); // Expected size for alignment
    }

    #[test]
    fn test_quality_settings() {
        assert_eq!(CloudShadowQuality::Low.texture_size(), 256);
        assert_eq!(CloudShadowQuality::Medium.texture_size(), 512);
        assert_eq!(CloudShadowQuality::High.texture_size(), 1024);
        assert_eq!(CloudShadowQuality::Ultra.texture_size(), 2048);

        assert_eq!(CloudShadowQuality::Low.noise_octaves(), 3);
        assert_eq!(CloudShadowQuality::Medium.noise_octaves(), 4);
        assert_eq!(CloudShadowQuality::High.noise_octaves(), 5);
        assert_eq!(CloudShadowQuality::Ultra.noise_octaves(), 6);
    }

    #[test]
    fn test_cloud_speed_calculation() {
        let speed = utils::calculate_cloud_speed(0.0, 1.0, 0.02);
        assert!((speed.x - 0.02).abs() < 0.001);
        assert!(speed.y.abs() < 0.001);

        let speed = utils::calculate_cloud_speed(90.0_f32.to_radians(), 1.0, 0.02);
        assert!(speed.x.abs() < 0.001);
        assert!((speed.y - 0.02).abs() < 0.001);
    }

    #[test]
    fn test_animation_presets() {
        let calm = utils::create_animation_preset("calm");
        assert!(calm.speed.length() < 0.01);
        assert!(calm.wind_strength < 0.5);

        let stormy = utils::create_animation_preset("stormy");
        assert!(stormy.speed.length() > 0.05);
        assert!(stormy.wind_strength > 2.0);
    }
}
