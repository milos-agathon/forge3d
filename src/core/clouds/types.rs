use glam::{Mat4, Vec2};

/// Cloud rendering quality levels
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CloudQuality {
    Low,    // 32^3 noise, 16 steps, billboard-heavy
    Medium, // 64^3 noise, 32 steps, balanced
    High,   // 128^3 noise, 64 steps, volumetric-heavy
    Ultra,  // 256^3 noise, 128 steps, maximum quality
}

impl CloudQuality {
    pub fn noise_resolution(&self) -> u32 {
        match self {
            CloudQuality::Low => 32,
            CloudQuality::Medium => 64,
            CloudQuality::High => 128,
            CloudQuality::Ultra => 256,
        }
    }

    pub fn max_ray_steps(&self) -> u32 {
        match self {
            CloudQuality::Low => 16,
            CloudQuality::Medium => 32,
            CloudQuality::High => 64,
            CloudQuality::Ultra => 128,
        }
    }

    pub fn billboard_threshold(&self) -> f32 {
        match self {
            CloudQuality::Low => 50.0,     // Use billboard beyond 50 units
            CloudQuality::Medium => 100.0, // Use billboard beyond 100 units
            CloudQuality::High => 200.0,   // Use billboard beyond 200 units
            CloudQuality::Ultra => 500.0,  // Use billboard beyond 500 units
        }
    }
}

/// Cloud rendering mode
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CloudRenderMode {
    Billboard,  // Fast billboard-only rendering
    Volumetric, // High-quality volumetric rendering
    Hybrid,     // Distance-based LOD (billboard far, volumetric near)
}

/// Cloud animation presets
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CloudAnimationPreset {
    Static,   // No animation
    Gentle,   // Slow, calm movement
    Moderate, // Normal wind conditions
    Stormy,   // Fast, chaotic movement
}

impl CloudAnimationPreset {
    pub fn wind_strength(&self) -> f32 {
        match self {
            CloudAnimationPreset::Static => 0.0,
            CloudAnimationPreset::Gentle => 0.2,
            CloudAnimationPreset::Moderate => 0.5,
            CloudAnimationPreset::Stormy => 1.2,
        }
    }

    pub fn animation_speed(&self) -> f32 {
        match self {
            CloudAnimationPreset::Static => 0.0,
            CloudAnimationPreset::Gentle => 0.3,
            CloudAnimationPreset::Moderate => 0.8,
            CloudAnimationPreset::Stormy => 2.0,
        }
    }
}

/// Cloud uniforms structure (must match WGSL exactly)
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CloudUniforms {
    pub view_proj: [[f32; 4]; 4],    // 64 bytes - View-projection matrix
    pub camera_pos: [f32; 4],        // 16 bytes - Camera position (xyz) + cloud_time (w)
    pub sky_params: [f32; 4],        // 16 bytes - Sky color (rgb) + sun_intensity (w)
    pub sun_direction: [f32; 4],     // 16 bytes - Sun direction (xyz) + cloud_density (w)
    pub cloud_params: [f32; 4], // 16 bytes - coverage (x), scale (y), height (z), fade_distance (w)
    pub wind_params: [f32; 4],  // 16 bytes - wind_dir (xy), wind_strength (z), animation_speed (w)
    pub scattering_params: [f32; 4], // 16 bytes - scatter_strength (x), absorption (y), phase_g (z), ambient_strength (w)
    pub render_params: [f32; 4], // 16 bytes - max_steps (x), step_size (y), billboard_size (z), lod_bias (w)
}

impl Default for CloudUniforms {
    fn default() -> Self {
        Self {
            view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            camera_pos: [0.0, 10.0, 0.0, 0.0],
            sky_params: [0.6, 0.8, 1.0, 1.0],    // Light blue sky
            sun_direction: [0.3, 0.7, 0.2, 0.8], // Angled sun + medium density
            cloud_params: [0.6, 200.0, 150.0, 1000.0], // coverage, scale, height, fade_distance
            wind_params: [1.0, 0.0, 0.5, 1.0],   // wind direction + strength + speed
            scattering_params: [1.2, 0.8, 0.3, 0.4], // scatter, absorption, phase_g, ambient
            render_params: [32.0, 5.0, 50.0, 0.0], // max_steps, step_size, billboard_size, lod_bias
        }
    }
}

/// Cloud instance data
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CloudInstance {
    pub world_matrix: [[f32; 4]; 4], // World transformation matrix
    pub cloud_data: [f32; 4],        // size (x), density (y), type (z), blend_factor (w)
    pub animation_data: [f32; 4],    // offset (xy), phase (z), lifetime (w)
}

impl Default for CloudInstance {
    fn default() -> Self {
        Self {
            world_matrix: Mat4::IDENTITY.to_cols_array_2d(),
            cloud_data: [100.0, 0.8, 0.0, 1.0], // size, density, type, blend
            animation_data: [0.0, 0.0, 0.0, 1.0], // offset, phase, lifetime
        }
    }
}

/// Cloud rendering parameters
#[derive(Debug, Clone)]
pub struct CloudParams {
    pub quality: CloudQuality,
    pub render_mode: CloudRenderMode,
    pub animation_preset: CloudAnimationPreset,
    pub density: f32,
    pub coverage: f32,
    pub scale: f32,
    pub height: f32,
    pub fade_distance: f32,
    pub wind_direction: Vec2,
    pub wind_strength: f32,
    pub sun_intensity: f32,
    pub scatter_strength: f32,
    pub absorption: f32,
    pub phase_g: f32,
    pub ambient_strength: f32,
}

impl Default for CloudParams {
    fn default() -> Self {
        Self {
            quality: CloudQuality::Medium,
            render_mode: CloudRenderMode::Hybrid,
            animation_preset: CloudAnimationPreset::Moderate,
            density: 0.8,
            coverage: 0.6,
            scale: 200.0,
            height: 150.0,
            fade_distance: 1000.0,
            wind_direction: Vec2::new(1.0, 0.0),
            wind_strength: 0.5,
            sun_intensity: 1.0,
            scatter_strength: 1.2,
            absorption: 0.8,
            phase_g: 0.3,
            ambient_strength: 0.4,
        }
    }
}
