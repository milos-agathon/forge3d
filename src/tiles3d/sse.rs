//! Screen-Space Error (SSE) computation for 3D Tiles LOD selection

use glam::{Mat4, Vec3};
use super::bounds::BoundingVolume;

/// Parameters for SSE computation
#[derive(Debug, Clone, Copy)]
pub struct SseParams {
    /// Viewport height in pixels
    pub viewport_height: f32,
    /// Vertical field of view in radians
    pub fov_y: f32,
}

impl Default for SseParams {
    fn default() -> Self {
        Self {
            viewport_height: 1080.0,
            fov_y: std::f32::consts::FRAC_PI_4, // 45 degrees
        }
    }
}

impl SseParams {
    /// Create SSE params from viewport dimensions and FOV
    pub fn new(viewport_height: f32, fov_y_radians: f32) -> Self {
        Self {
            viewport_height,
            fov_y: fov_y_radians,
        }
    }

    /// Compute the SSE factor (pixels per meter at distance 1)
    pub fn sse_factor(&self) -> f32 {
        self.viewport_height / (2.0 * (self.fov_y / 2.0).tan())
    }
}

/// Compute Screen-Space Error for a tile
///
/// SSE measures how large the geometric error would appear on screen.
/// Higher SSE means the tile should be refined (load children).
/// Lower SSE means the tile is acceptable at current view.
///
/// # Arguments
/// * `geometric_error` - The tile's geometric error in meters
/// * `bounding_volume` - The tile's bounding volume
/// * `camera_position` - Camera position in world space
/// * `params` - SSE computation parameters
///
/// # Returns
/// Screen-space error in pixels
pub fn compute_sse(
    geometric_error: f32,
    bounding_volume: &BoundingVolume,
    camera_position: Vec3,
    params: &SseParams,
) -> f32 {
    let center = bounding_volume.center();
    let distance = (center - camera_position).length();
    
    if distance < 0.001 {
        return f32::MAX; // Camera inside tile, always refine
    }

    // SSE = (geometric_error / distance) * sse_factor
    // This gives us pixels of error on screen
    (geometric_error / distance) * params.sse_factor()
}

/// Compute SSE using view-projection matrix (for frustum-based computation)
pub fn compute_sse_with_matrix(
    geometric_error: f32,
    bounding_volume: &BoundingVolume,
    view_proj: &Mat4,
    params: &SseParams,
) -> f32 {
    let center = bounding_volume.center();
    let clip = *view_proj * center.extend(1.0);
    
    if clip.w <= 0.0 {
        return f32::MAX; // Behind camera
    }

    let distance = clip.w;
    (geometric_error / distance) * params.sse_factor()
}

/// Determine if a tile should be refined based on SSE threshold
pub fn should_refine(
    geometric_error: f32,
    bounding_volume: &BoundingVolume,
    camera_position: Vec3,
    params: &SseParams,
    threshold: f32,
) -> bool {
    let sse = compute_sse(geometric_error, bounding_volume, camera_position, params);
    sse > threshold
}

/// Compute distance from camera to bounding volume surface (not center)
pub fn distance_to_surface(bounding_volume: &BoundingVolume, camera_position: Vec3) -> f32 {
    let center = bounding_volume.center();
    let radius = bounding_volume.radius();
    let center_dist = (center - camera_position).length();
    (center_dist - radius).max(0.001)
}

/// More accurate SSE using distance to surface
pub fn compute_sse_surface(
    geometric_error: f32,
    bounding_volume: &BoundingVolume,
    camera_position: Vec3,
    params: &SseParams,
) -> f32 {
    let distance = distance_to_surface(bounding_volume, camera_position);
    (geometric_error / distance) * params.sse_factor()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tiles3d::bounds::BoundingSphere;

    #[test]
    fn test_sse_decreases_with_distance() {
        let params = SseParams::default();
        let bounds = BoundingVolume::Sphere(BoundingSphere {
            sphere: [0.0, 0.0, 0.0, 10.0],
        });
        
        let sse_near = compute_sse(10.0, &bounds, Vec3::new(0.0, 0.0, 100.0), &params);
        let sse_far = compute_sse(10.0, &bounds, Vec3::new(0.0, 0.0, 1000.0), &params);
        
        assert!(sse_near > sse_far);
    }

    #[test]
    fn test_sse_increases_with_error() {
        let params = SseParams::default();
        let bounds = BoundingVolume::Sphere(BoundingSphere {
            sphere: [0.0, 0.0, 0.0, 10.0],
        });
        let camera = Vec3::new(0.0, 0.0, 100.0);
        
        let sse_small = compute_sse(1.0, &bounds, camera, &params);
        let sse_large = compute_sse(10.0, &bounds, camera, &params);
        
        assert!(sse_large > sse_small);
    }

    #[test]
    fn test_should_refine() {
        let params = SseParams::default();
        let bounds = BoundingVolume::Sphere(BoundingSphere {
            sphere: [0.0, 0.0, 0.0, 10.0],
        });
        
        // Close camera should trigger refinement
        let near = Vec3::new(0.0, 0.0, 20.0);
        assert!(should_refine(50.0, &bounds, near, &params, 16.0));
        
        // Far camera should not need refinement
        let far = Vec3::new(0.0, 0.0, 10000.0);
        assert!(!should_refine(50.0, &bounds, far, &params, 16.0));
    }
}
