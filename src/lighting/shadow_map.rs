// src/lighting/shadow_map.rs
// P0-S3: Shadow mapping infrastructure (Hard + PCF)

use wgpu::{Device, TextureView, Texture, TextureFormat};
use crate::lighting::types::{Light, ShadowSettings, ShadowTechnique};

/// Shadow map atlas for rendering depth from light's perspective
pub struct ShadowMap {
    /// Depth texture (D32Float format)
    pub texture: Texture,
    /// View for rendering (write)
    pub depth_view: TextureView,
    /// View for sampling (read)
    pub sampled_view: TextureView,
    /// Sampler (comparison or regular depending on technique)
    pub sampler: wgpu::Sampler,
    /// Resolution (square texture: res×res)
    pub resolution: u32,
    /// Shadow settings
    pub settings: ShadowSettings,
}

impl ShadowMap {
    /// Create a new shadow map with the given settings
    pub fn new(device: &Device, settings: ShadowSettings) -> Self {
        let resolution = settings.map_res;

        // Create depth texture (D32Float format, 4 bytes per pixel)
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Shadow Map Depth Texture"),
            size: wgpu::Extent3d {
                width: resolution,
                height: resolution,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let depth_view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("Shadow Map Depth View"),
            format: Some(TextureFormat::Depth32Float),
            dimension: Some(wgpu::TextureViewDimension::D2),
            aspect: wgpu::TextureAspect::DepthOnly,
            ..Default::default()
        });

        let sampled_view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("Shadow Map Sampled View"),
            format: Some(TextureFormat::Depth32Float),
            dimension: Some(wgpu::TextureViewDimension::D2),
            aspect: wgpu::TextureAspect::DepthOnly,
            ..Default::default()
        });

        // Create sampler based on shadow technique
        let sampler = if settings.tech == ShadowTechnique::Hard as u32 {
            // Comparison sampler for hard shadows (hardware PCF at 1 sample)
            device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("Shadow Map Comparison Sampler"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::FilterMode::Nearest,
                compare: Some(wgpu::CompareFunction::LessEqual),
                ..Default::default()
            })
        } else {
            // Regular sampler for PCF (manual sampling in shader)
            device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("Shadow Map PCF Sampler"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::FilterMode::Nearest,
                compare: None,
                ..Default::default()
            })
        };

        Self {
            texture,
            depth_view,
            sampled_view,
            sampler,
            resolution,
            settings,
        }
    }

    /// Calculate memory usage in bytes
    pub fn memory_bytes(&self) -> u64 {
        // D32Float = 4 bytes per pixel
        (self.resolution as u64) * (self.resolution as u64) * 4
    }

    /// Calculate memory usage in megabytes
    pub fn memory_mb(&self) -> f64 {
        self.memory_bytes() as f64 / (1024.0 * 1024.0)
    }

    /// Validate memory budget (should be ≤32 MiB for P0)
    pub fn validate_budget(&self) -> Result<(), String> {
        let mb = self.memory_mb();
        if mb > 32.0 {
            Err(format!(
                "Shadow map exceeds budget: {:.2} MiB (max 32 MiB)",
                mb
            ))
        } else {
            Ok(())
        }
    }

    /// Recreate shadow map with new settings
    pub fn resize(&mut self, device: &Device, new_settings: ShadowSettings) {
        *self = Self::new(device, new_settings);
    }
}

/// Light-space projection matrix calculation for shadow mapping
pub struct ShadowMatrixCalculator;

impl ShadowMatrixCalculator {
    /// Calculate view-projection matrix for directional light shadow map
    /// Returns a matrix that transforms world space to light's NDC space
    pub fn directional_light_matrix(
        light: &Light,
        scene_bounds: &SceneBounds,
    ) -> [[f32; 4]; 4] {
        // Light looks down its direction vector
        let light_dir = glam::Vec3::from_slice(&light.dir_ws).normalize();
        let light_pos = scene_bounds.center - light_dir * scene_bounds.radius * 2.0;

        // Build view matrix (light looking at scene center)
        let view = glam::Mat4::look_at_rh(
            light_pos,
            scene_bounds.center,
            glam::Vec3::Y, // up vector (will fail if light_dir == Y, TODO: handle)
        );

        // Orthographic projection to cover scene bounds
        let half_size = scene_bounds.radius * 1.5; // 50% margin
        let near = 0.1;
        let far = scene_bounds.radius * 4.0;

        let projection = glam::Mat4::orthographic_rh(
            -half_size,
            half_size,
            -half_size,
            half_size,
            near,
            far,
        );

        // Combine view and projection
        let view_proj = projection * view;

        // Convert to column-major array for GPU upload
        view_proj.to_cols_array_2d()
    }

    /// Calculate view-projection matrix for spot light shadow map
    pub fn spot_light_matrix(light: &Light, aspect: f32) -> [[f32; 4]; 4] {
        // Spot light position
        let light_pos = glam::Vec3::from_slice(&light.pos_ws);

        // Spot direction from light struct
        let spot_dir = glam::Vec3::from_slice(&light.dir_ws).normalize();
        let target = light_pos + spot_dir;

        // Build view matrix
        let view = glam::Mat4::look_at_rh(light_pos, target, glam::Vec3::Z);

        // Perspective projection based on outer cone angle (stored as cosine)
        // Convert back to angle: acos(cone_cos[1]) * 2.0 for full cone
        let outer_angle_rad = light.cone_cos[1].acos();
        let fov_rad = outer_angle_rad * 2.0; // Full cone angle
        let near = 0.1;
        let far = light.range;

        let projection = glam::Mat4::perspective_rh(fov_rad, aspect, near, far);

        // Combine
        let view_proj = projection * view;
        view_proj.to_cols_array_2d()
    }
}

/// Scene bounding volume for shadow frustum calculation
#[derive(Debug, Clone, Copy)]
pub struct SceneBounds {
    pub center: glam::Vec3,
    pub radius: f32,
}

impl SceneBounds {
    /// Create bounds from center and radius
    pub fn new(center: [f32; 3], radius: f32) -> Self {
        Self {
            center: glam::Vec3::from_slice(&center),
            radius,
        }
    }

    /// Create default bounds (origin, 100 unit radius)
    pub fn default() -> Self {
        Self {
            center: glam::Vec3::ZERO,
            radius: 100.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shadow_map_memory() {
        let settings = ShadowSettings {
            tech: ShadowTechnique::Hard.as_u32(),
            map_res: 2048,
            bias: 0.005,
            normal_bias: 0.01,
            softness: 1.0,
            pcss_blocker_radius: 0.03,
            pcss_filter_radius: 0.06,
            light_size: 0.25,
            moment_bias: 0.0005,
            _pad: [0.0; 3],
        };

        // Can't create actual shadow map without Device, but we can calculate memory
        let bytes = (settings.map_res as u64) * (settings.map_res as u64) * 4;
        let mb = bytes as f64 / (1024.0 * 1024.0);

        assert_eq!(bytes, 16_777_216); // 2048*2048*4
        assert!((mb - 16.0).abs() < 0.01); // ~16 MiB
    }

    #[test]
    fn test_directional_light_matrix() {
        // Create downward light: azimuth 0, elevation -90 (straight down)
        let light = Light::directional(
            0.0,    // azimuth_deg
            -90.0,  // elevation_deg (downward)
            1.0,    // intensity
            [1.0, 1.0, 1.0],  // color
        );

        let bounds = SceneBounds::new([0.0, 0.0, 0.0], 50.0);
        let matrix = ShadowMatrixCalculator::directional_light_matrix(&light, &bounds);

        // Matrix should be valid (no NaNs)
        for row in &matrix {
            for &val in row {
                assert!(val.is_finite());
            }
        }
    }
}
