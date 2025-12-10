use bytemuck::{Pod, Zeroable};
use glam::Vec2;

/// Cloud shadow parameters matching WGSL CloudShadowUniforms structure
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct CloudShadowUniforms {
    // Cloud movement parameters
    pub cloud_speed: [f32; 2], // Cloud movement speed (x, y)
    pub time: f32,             // Current time for animation
    pub cloud_scale: f32,      // Scale of cloud patterns

    // Cloud appearance parameters
    pub cloud_density: f32,    // Base cloud density [0, 1]
    pub cloud_coverage: f32,   // Cloud coverage amount [0, 1]
    pub shadow_intensity: f32, // Shadow strength [0, 1]
    pub shadow_softness: f32,  // Shadow edge softness

    // Texture parameters
    pub texture_size: [f32; 2],     // Size of cloud shadow texture
    pub inv_texture_size: [f32; 2], // 1.0 / texture_size

    // Noise parameters
    pub noise_octaves: u32,   // Number of noise octaves
    pub noise_frequency: f32, // Base noise frequency
    pub noise_amplitude: f32, // Noise amplitude
    pub wind_direction: f32,  // Wind direction in radians

    // Debug and visualization
    pub debug_mode: u32,       // Debug visualization mode
    pub show_clouds_only: u32, // Show only cloud patterns
    pub _padding: [f32; 2],    // Padding for alignment
}

impl Default for CloudShadowUniforms {
    fn default() -> Self {
        Self {
            cloud_speed: [0.02, 0.01], // Slow drift
            time: 0.0,
            cloud_scale: 2.0, // Medium scale clouds

            cloud_density: 0.6,    // Moderate density
            cloud_coverage: 0.4,   // 40% coverage
            shadow_intensity: 0.7, // Strong shadows
            shadow_softness: 0.3,  // Soft edges

            texture_size: [512.0, 512.0], // Default texture size
            inv_texture_size: [1.0 / 512.0, 1.0 / 512.0],

            noise_octaves: 4,     // 4 octaves of noise
            noise_frequency: 1.0, // Base frequency
            noise_amplitude: 1.0, // Full amplitude
            wind_direction: 0.0,  // No initial wind

            debug_mode: 0,       // Normal rendering
            show_clouds_only: 0, // Show shadows
            _padding: [0.0; 2],
        }
    }
}

/// Cloud shadow quality settings
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CloudShadowQuality {
    /// Low quality: 256x256 texture, 3 octaves
    Low,
    /// Medium quality: 512x512 texture, 4 octaves
    Medium,
    /// High quality: 1024x1024 texture, 5 octaves
    High,
    /// Ultra quality: 2048x2048 texture, 6 octaves
    Ultra,
}

impl CloudShadowQuality {
    /// Get texture size for this quality setting
    pub fn texture_size(self) -> u32 {
        match self {
            CloudShadowQuality::Low => 256,
            CloudShadowQuality::Medium => 512,
            CloudShadowQuality::High => 1024,
            CloudShadowQuality::Ultra => 2048,
        }
    }

    /// Get noise octaves for this quality
    pub fn noise_octaves(self) -> u32 {
        match self {
            CloudShadowQuality::Low => 3,
            CloudShadowQuality::Medium => 4,
            CloudShadowQuality::High => 5,
            CloudShadowQuality::Ultra => 6,
        }
    }
}

/// Cloud shadow animation parameters
#[derive(Debug, Clone, Copy)]
pub struct CloudAnimationParams {
    pub speed: Vec2,         // Cloud movement speed
    pub wind_direction: f32, // Wind direction in radians
    pub wind_strength: f32,  // Wind strength multiplier
    pub turbulence: f32,     // Amount of turbulence
}

impl Default for CloudAnimationParams {
    fn default() -> Self {
        Self {
            speed: Vec2::new(0.02, 0.01),
            wind_direction: 0.0,
            wind_strength: 1.0,
            turbulence: 0.1,
        }
    }
}
