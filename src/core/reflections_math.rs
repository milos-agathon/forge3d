// src/core/reflections_math.rs
// Mathematical utilities for planar reflections (B5)
// RELEVANT FILES: shaders/planar_reflections.wgsl

use glam::{Mat4, Vec3, Vec4};

/// Create reflection matrix for a plane
pub fn create_reflection_matrix(normal: Vec3, distance: f32) -> Mat4 {
    let n = normal.normalize();
    let d = distance;

    Mat4::from_cols(
        Vec4::new(
            1.0 - 2.0 * n.x * n.x,
            -2.0 * n.x * n.y,
            -2.0 * n.x * n.z,
            -2.0 * n.x * d,
        ),
        Vec4::new(
            -2.0 * n.y * n.x,
            1.0 - 2.0 * n.y * n.y,
            -2.0 * n.y * n.z,
            -2.0 * n.y * d,
        ),
        Vec4::new(
            -2.0 * n.z * n.x,
            -2.0 * n.z * n.y,
            1.0 - 2.0 * n.z * n.z,
            -2.0 * n.z * d,
        ),
        Vec4::new(0.0, 0.0, 0.0, 1.0),
    )
}

/// Reflect a point across a plane
pub fn reflect_point_across_plane(point: Vec3, plane_normal: Vec3, plane_distance: f32) -> Vec3 {
    let n = plane_normal.normalize();
    let distance_to_plane = point.dot(n) + plane_distance;
    point - 2.0 * distance_to_plane * n
}

/// Calculate distance from point to plane
pub fn distance_to_plane(point: Vec3, plane_normal: Vec3, plane_distance: f32) -> f32 {
    point.dot(plane_normal.normalize()) + plane_distance
}

/// Check if point is above plane (in the direction of the normal)
pub fn is_above_plane(point: Vec3, plane_normal: Vec3, plane_distance: f32) -> bool {
    distance_to_plane(point, plane_normal, plane_distance) > 0.001
}

/// Calculate Fresnel reflection factor
pub fn calculate_fresnel(view_dir: Vec3, surface_normal: Vec3, fresnel_power: f32) -> f32 {
    let n_dot_v = surface_normal.dot(view_dir).max(0.0);
    (1.0 - n_dot_v).powf(fresnel_power).clamp(0.0, 1.0)
}

/// Clip frustum against reflection plane for optimized rendering
pub fn clip_frustum_to_plane(
    frustum_corners: &[Vec3; 8],
    plane_normal: Vec3,
    plane_distance: f32,
) -> Vec<Vec3> {
    let mut clipped_corners = Vec::new();

    for &corner in frustum_corners {
        if is_above_plane(corner, plane_normal, plane_distance) {
            clipped_corners.push(corner);
        }
    }

    // Add intersection points where frustum edges cross the plane
    for i in 0..8 {
        let current = frustum_corners[i];
        let next = frustum_corners[(i + 1) % 8];

        let current_above = is_above_plane(current, plane_normal, plane_distance);
        let next_above = is_above_plane(next, plane_normal, plane_distance);

        if current_above != next_above {
            let t = -distance_to_plane(current, plane_normal, plane_distance)
                / (next - current).dot(plane_normal);
            if (0.0..=1.0).contains(&t) {
                let intersection = current + t * (next - current);
                clipped_corners.push(intersection);
            }
        }
    }

    clipped_corners
}
