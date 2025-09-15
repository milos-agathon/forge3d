//! Cascade splitting for Cascaded Shadow Maps (CSM)
//!
//! Provides practical split schemes for dividing the view frustum into shadow cascades
//! using a combination of uniform and logarithmic distributions controlled by lambda parameter.

use glam::Vec4Swizzles;
use glam::{Mat4, Vec3, Vec4};

/// Configuration for cascade splitting
#[derive(Debug, Clone)]
pub struct CascadeSplitConfig {
    /// Number of cascades (minimum 3, maximum 8)
    pub cascade_count: u32,

    /// Lambda parameter [0,1] for uniform/logarithmic mixing
    /// 0.0 = pure uniform split, 1.0 = pure logarithmic split
    pub lambda: f32,

    /// Near distance of the camera frustum
    pub near_distance: f32,

    /// Far distance of the camera frustum
    pub far_distance: f32,

    /// Additional padding factor for cascade boundaries
    pub boundary_padding: f32,
}

impl Default for CascadeSplitConfig {
    fn default() -> Self {
        Self {
            cascade_count: 4,
            lambda: 0.5, // Balanced uniform/logarithmic split
            near_distance: 0.1,
            far_distance: 100.0,
            boundary_padding: 1.1, // 10% padding
        }
    }
}

/// Represents a single shadow cascade
#[derive(Debug, Clone)]
pub struct ShadowCascade {
    /// Near distance of this cascade
    pub near_distance: f32,

    /// Far distance of this cascade
    pub far_distance: f32,

    /// Light-space projection matrix for this cascade
    pub light_projection: Mat4,

    /// Texel size in world space for this cascade
    pub texel_size: f32,

    /// Cascade index (0-based)
    pub cascade_index: u32,
}

impl ShadowCascade {
    /// Create a zeroed shadow cascade
    pub fn zeroed() -> Self {
        Self {
            near_distance: 0.0,
            far_distance: 0.0,
            light_projection: Mat4::ZERO,
            texel_size: 0.0,
            cascade_index: 0,
        }
    }
}

/// Calculate cascade split distances using practical logarithmic-uniform mixing
pub fn calculate_cascade_splits(config: &CascadeSplitConfig) -> Vec<f32> {
    let cascade_count = config.cascade_count.clamp(3, 8) as usize;
    let mut splits = Vec::with_capacity(cascade_count + 1);

    // Start with near distance
    splits.push(config.near_distance);

    let near = config.near_distance;
    let far = config.far_distance;
    let lambda = config.lambda.clamp(0.0, 1.0);

    // Calculate intermediate splits
    for i in 1..cascade_count {
        let i_norm = i as f32 / cascade_count as f32;

        // Uniform split
        let uniform_split = near + (far - near) * i_norm;

        // Logarithmic split
        let log_split = near * (far / near).powf(i_norm);

        // Mix uniform and logarithmic
        let mixed_split = lambda * log_split + (1.0 - lambda) * uniform_split;

        splits.push(mixed_split);
    }

    // End with far distance
    splits.push(far);

    // Apply boundary padding to intermediate splits
    if config.boundary_padding > 1.0 {
        for i in 1..splits.len() - 1 {
            let padding_factor = config.boundary_padding;
            let center = (splits[i + 1] + splits[i - 1]) * 0.5;

            // Extend the split boundary
            splits[i] = center + (splits[i] - center) * padding_factor;
        }
    }

    splits
}

/// Generate shadow cascades for a directional light
pub fn generate_cascades(
    config: &CascadeSplitConfig,
    light_direction: Vec3,
    camera_view: Mat4,
    camera_projection: Mat4,
    shadow_map_size: f32,
) -> Vec<ShadowCascade> {
    let splits = calculate_cascade_splits(config);
    let mut cascades = Vec::with_capacity(splits.len() - 1);

    let light_dir = light_direction.normalize();

    for i in 0..splits.len() - 1 {
        let near_dist = splits[i];
        let far_dist = splits[i + 1];

        // Create frustum corners for this cascade
        let frustum_corners =
            extract_frustum_corners(camera_view, camera_projection, near_dist, far_dist);

        // Calculate optimal light space projection for this cascade
        let light_projection =
            calculate_light_projection(&frustum_corners, light_dir, shadow_map_size);

        // Calculate texel size in world space
        let texel_size = calculate_texel_size(&frustum_corners, shadow_map_size);

        cascades.push(ShadowCascade {
            near_distance: near_dist,
            far_distance: far_dist,
            light_projection,
            texel_size,
            cascade_index: i as u32,
        });
    }

    cascades
}

/// Extract frustum corners for a specific near/far range
fn extract_frustum_corners(
    view_matrix: Mat4,
    projection_matrix: Mat4,
    near_dist: f32,
    far_dist: f32,
) -> [Vec3; 8] {
    let inv_view_proj = (projection_matrix * view_matrix).inverse();

    // NDC coordinates for frustum corners
    let ndc_corners = [
        Vec4::new(-1.0, -1.0, -1.0, 1.0), // Near bottom left
        Vec4::new(1.0, -1.0, -1.0, 1.0),  // Near bottom right
        Vec4::new(-1.0, 1.0, -1.0, 1.0),  // Near top left
        Vec4::new(1.0, 1.0, -1.0, 1.0),   // Near top right
        Vec4::new(-1.0, -1.0, 1.0, 1.0),  // Far bottom left
        Vec4::new(1.0, -1.0, 1.0, 1.0),   // Far bottom right
        Vec4::new(-1.0, 1.0, 1.0, 1.0),   // Far top left
        Vec4::new(1.0, 1.0, 1.0, 1.0),    // Far top right
    ];

    let mut world_corners = [Vec3::ZERO; 8];

    for (i, ndc_corner) in ndc_corners.iter().enumerate() {
        let world_pos = inv_view_proj * (*ndc_corner);
        let world_pos = world_pos.xyz() / world_pos.w;
        world_corners[i] = world_pos;
    }

    // Adjust corners for custom near/far distances
    let camera_pos = view_matrix.inverse().col(3).xyz();
    let view_dir = -view_matrix.row(2).xyz();

    // Interpolate based on distance ratios
    let original_near = (world_corners[0] - camera_pos).dot(view_dir);
    let original_far = (world_corners[4] - camera_pos).dot(view_dir);
    let original_range = original_far - original_near;

    if original_range > 0.0 {
        let near_t = (near_dist - original_near) / original_range;
        let far_t = (far_dist - original_near) / original_range;

        // Update near corners
        for i in 0..4 {
            world_corners[i] = world_corners[i].lerp(world_corners[i + 4], near_t);
        }

        // Update far corners
        for i in 4..8 {
            world_corners[i] = world_corners[i - 4].lerp(world_corners[i], far_t);
        }
    }

    world_corners
}

/// Calculate optimal light space projection matrix for frustum corners
fn calculate_light_projection(
    frustum_corners: &[Vec3; 8],
    light_direction: Vec3,
    shadow_map_size: f32,
) -> Mat4 {
    // Calculate light space bounding box
    let light_up = if light_direction.dot(Vec3::Y).abs() > 0.99 {
        Vec3::X // Use X axis if light is nearly vertical
    } else {
        Vec3::Y
    };

    let light_right = light_direction.cross(light_up).normalize();
    let light_up = light_right.cross(light_direction).normalize();

    // Transform frustum corners to light space
    let mut light_space_corners = [Vec3::ZERO; 8];
    for (i, corner) in frustum_corners.iter().enumerate() {
        light_space_corners[i] = Vec3::new(
            corner.dot(light_right),
            corner.dot(light_up),
            corner.dot(light_direction),
        );
    }

    // Find bounding box in light space
    let mut min_bounds = light_space_corners[0];
    let mut max_bounds = light_space_corners[0];

    for corner in light_space_corners.iter().skip(1) {
        min_bounds = min_bounds.min(*corner);
        max_bounds = max_bounds.max(*corner);
    }

    // Extend depth range to include potential occluders
    let depth_extension = (max_bounds.z - min_bounds.z) * 0.5;
    min_bounds.z -= depth_extension;

    // Snap to texel boundaries to reduce shimmering
    let texel_size = (max_bounds.x - min_bounds.x) / shadow_map_size;
    if texel_size > 0.0 {
        min_bounds.x = (min_bounds.x / texel_size).floor() * texel_size;
        min_bounds.y = (min_bounds.y / texel_size).floor() * texel_size;
        max_bounds.x = (max_bounds.x / texel_size).ceil() * texel_size;
        max_bounds.y = (max_bounds.y / texel_size).ceil() * texel_size;
    }

    // Create orthographic projection matrix
    let light_view = Mat4::look_at_rh(Vec3::ZERO, light_direction, light_up);

    let light_projection = Mat4::orthographic_rh(
        min_bounds.x,
        max_bounds.x,
        min_bounds.y,
        max_bounds.y,
        min_bounds.z,
        max_bounds.z,
    );

    light_projection * light_view
}

/// Calculate texel size in world space for a cascade
fn calculate_texel_size(frustum_corners: &[Vec3; 8], shadow_map_size: f32) -> f32 {
    // Find the maximum extent of the frustum
    let mut min_pos = frustum_corners[0];
    let mut max_pos = frustum_corners[0];

    for corner in frustum_corners.iter().skip(1) {
        min_pos = min_pos.min(*corner);
        max_pos = max_pos.max(*corner);
    }

    let extent = (max_pos - min_pos).length();
    extent / shadow_map_size
}

/// Get split distances for debugging/visualization
pub fn get_split_distances(config: &CascadeSplitConfig) -> Vec<f32> {
    calculate_cascade_splits(config)
}

/// Validate cascade configuration
pub fn validate_config(config: &CascadeSplitConfig) -> Result<(), String> {
    if config.cascade_count < 3 {
        return Err("Cascade count must be at least 3".to_string());
    }

    if config.cascade_count > 8 {
        return Err("Cascade count cannot exceed 8".to_string());
    }

    if config.near_distance <= 0.0 {
        return Err("Near distance must be positive".to_string());
    }

    if config.far_distance <= config.near_distance {
        return Err("Far distance must be greater than near distance".to_string());
    }

    if config.lambda < 0.0 || config.lambda > 1.0 {
        return Err("Lambda must be in range [0,1]".to_string());
    }

    if config.boundary_padding < 1.0 {
        return Err("Boundary padding must be >= 1.0".to_string());
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cascade_splits() {
        let config = CascadeSplitConfig {
            cascade_count: 4,
            lambda: 0.5,
            near_distance: 0.1,
            far_distance: 100.0,
            boundary_padding: 1.0,
        };

        let splits = calculate_cascade_splits(&config);

        // Should have cascade_count + 1 splits
        assert_eq!(splits.len(), 5);

        // First should be near, last should be far
        assert_eq!(splits[0], 0.1);
        assert_eq!(splits[4], 100.0);

        // Should be monotonically increasing
        for i in 1..splits.len() {
            assert!(splits[i] > splits[i - 1]);
        }
    }

    #[test]
    fn test_uniform_split() {
        let config = CascadeSplitConfig {
            cascade_count: 3,
            lambda: 0.0, // Pure uniform
            near_distance: 0.0,
            far_distance: 30.0,
            boundary_padding: 1.0,
        };

        let splits = calculate_cascade_splits(&config);

        // Should be uniformly spaced
        assert_eq!(splits, vec![0.0, 10.0, 20.0, 30.0]);
    }

    #[test]
    fn test_config_validation() {
        let mut config = CascadeSplitConfig::default();

        // Valid config should pass
        assert!(validate_config(&config).is_ok());

        // Invalid cascade count
        config.cascade_count = 2;
        assert!(validate_config(&config).is_err());

        config.cascade_count = 10;
        assert!(validate_config(&config).is_err());

        // Invalid distances
        config.cascade_count = 4;
        config.near_distance = -1.0;
        assert!(validate_config(&config).is_err());

        config.near_distance = 10.0;
        config.far_distance = 5.0;
        assert!(validate_config(&config).is_err());
    }
}
