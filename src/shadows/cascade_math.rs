// src/shadows/cascade_math.rs
// Mathematical utilities for cascade shadow map calculations
// RELEVANT FILES: shaders/shadows.wgsl

use glam::{Mat4, Vec3, Vec4};

/// Calculate frustum corners in world space for a given depth range
pub fn calculate_frustum_corners(inv_view_proj: Mat4, near_norm: f32, far_norm: f32) -> Vec<Vec3> {
    let mut corners = Vec::with_capacity(8);

    // NDC corners for near and far planes
    let ndc_corners = [
        // Near plane corners
        [-1.0, -1.0, near_norm],
        [1.0, -1.0, near_norm],
        [1.0, 1.0, near_norm],
        [-1.0, 1.0, near_norm],
        // Far plane corners
        [-1.0, -1.0, far_norm],
        [1.0, -1.0, far_norm],
        [1.0, 1.0, far_norm],
        [-1.0, 1.0, far_norm],
    ];

    // Transform NDC corners to world space
    for ndc in &ndc_corners {
        let world_pos = inv_view_proj * Vec4::new(ndc[0], ndc[1], ndc[2], 1.0);
        corners.push((world_pos / world_pos.w).truncate());
    }

    corners
}

/// Peter-panning detection utility
pub fn detect_peter_panning(
    shadow_factor: f32,
    surface_normal: Vec3,
    light_direction: Vec3,
) -> bool {
    let n_dot_l = surface_normal.dot(-light_direction);

    // Peter-panning occurs when shadows are cast on surfaces facing away from light
    // or when there's insufficient depth bias
    n_dot_l <= 0.01 && shadow_factor < 0.5
}

/// Calculate automatic cascade splits using practical split scheme
pub fn calculate_cascade_splits(
    cascade_count: u32,
    near_plane: f32,
    far_plane: f32,
) -> Vec<f32> {
    let mut splits = Vec::with_capacity(cascade_count as usize + 1);
    splits.push(near_plane);

    let range = far_plane - near_plane;
    let ratio = far_plane / near_plane;

    // Practical Split Scheme (PSS) - blend between logarithmic and uniform
    let lambda = 0.75; // Blend factor (0.0 = uniform, 1.0 = logarithmic)

    for i in 1..cascade_count {
        let i_f = i as f32;
        let count_f = cascade_count as f32;

        // Uniform split
        let uniform_split = near_plane + (i_f / count_f) * range;

        // Logarithmic split
        let log_split = near_plane * ratio.powf(i_f / count_f);

        // Blend the two schemes
        let split = lambda * log_split + (1.0 - lambda) * uniform_split;
        splits.push(split);
    }

    splits.push(far_plane);
    splits
}

/// Calculate optimal cascade splits for unclipped depth
pub fn calculate_unclipped_cascade_splits(
    cascade_count: u32,
    near_plane: f32,
    far_plane: f32,
    depth_clip_factor: f32,
) -> Vec<f32> {
    let effective_far = far_plane * depth_clip_factor;
    let mut splits = Vec::with_capacity(cascade_count as usize + 1);
    splits.push(near_plane);

    let range = effective_far - near_plane;
    let ratio = effective_far / near_plane;

    // More aggressive split scheme for unclipped depth (favors close-up detail)
    let lambda = 0.85; // More logarithmic distribution

    for i in 1..cascade_count {
        let i_f = i as f32;
        let count_f = cascade_count as f32;

        // Uniform split
        let uniform_split = near_plane + (i_f / count_f) * range;

        // More aggressive logarithmic split for unclipped depth
        let log_split = near_plane * ratio.powf(i_f / count_f);

        // Blend with more emphasis on logarithmic
        let split = lambda * log_split + (1.0 - lambda) * uniform_split;
        splits.push(split);
    }

    splits.push(effective_far);
    splits
}

/// Transform frustum corners to light space and calculate AABB bounds
pub fn calculate_light_space_bounds(
    frustum_corners: &[Vec3],
    light_view: Mat4,
) -> (Vec3, Vec3) {
    let mut light_space_corners = Vec::new();
    for corner in frustum_corners {
        let light_space_pos = light_view * corner.extend(1.0);
        light_space_corners.push(light_space_pos.truncate());
    }

    // Calculate AABB in light space
    let mut min_bounds = light_space_corners[0];
    let mut max_bounds = light_space_corners[0];

    for corner in &light_space_corners[1..] {
        min_bounds = min_bounds.min(*corner);
        max_bounds = max_bounds.max(*corner);
    }

    // Expand bounds slightly to prevent edge cases
    let expand = 0.01;
    min_bounds -= Vec3::splat(expand);
    max_bounds += Vec3::splat(expand);

    (min_bounds, max_bounds)
}

/// Snap bounds to texel grid for cascade stabilization (prevents shimmering)
pub fn snap_bounds_to_texel_grid(
    min_bounds: Vec3,
    max_bounds: Vec3,
    shadow_map_size: u32,
) -> (Vec3, Vec3) {
    let world_units_per_texel = (max_bounds.x - min_bounds.x) / shadow_map_size as f32;

    // Round min bounds to nearest texel boundary
    let mut snapped_min = min_bounds;
    snapped_min.x = (min_bounds.x / world_units_per_texel).floor() * world_units_per_texel;
    snapped_min.y = (min_bounds.y / world_units_per_texel).floor() * world_units_per_texel;

    // Recalculate max bounds to maintain exact shadow map size
    let mut snapped_max = max_bounds;
    snapped_max.x = snapped_min.x + world_units_per_texel * shadow_map_size as f32;
    snapped_max.y = snapped_min.y + world_units_per_texel * shadow_map_size as f32;

    (snapped_min, snapped_max)
}
