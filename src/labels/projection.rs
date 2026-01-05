//! World-to-screen projection utilities for labels.

use glam::{Mat4, Vec3, Vec4};

/// Projects world coordinates to screen coordinates.
pub struct LabelProjector {
    screen_width: f32,
    screen_height: f32,
}

impl LabelProjector {
    /// Create a new projector for the given screen dimensions.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            screen_width: width as f32,
            screen_height: height as f32,
        }
    }

    /// Update screen dimensions.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.screen_width = width as f32;
        self.screen_height = height as f32;
    }

    /// Project a world position to screen coordinates.
    ///
    /// Returns `Some((screen_pos, depth))` if the point is in front of the camera
    /// and within the view frustum, `None` otherwise.
    ///
    /// # Arguments
    /// * `world_pos` - Position in world space
    /// * `view_proj` - Combined view-projection matrix
    ///
    /// # Returns
    /// * `Some(([x, y], depth))` - Screen position in pixels and normalized depth (0=near, 1=far)
    /// * `None` - Point is behind camera or outside frustum
    pub fn project(&self, world_pos: Vec3, view_proj: Mat4) -> Option<([f32; 2], f32)> {
        // Transform to clip space
        let clip = view_proj * Vec4::new(world_pos.x, world_pos.y, world_pos.z, 1.0);

        // Check if behind camera (w <= 0 means behind or at camera plane)
        if clip.w <= 0.0001 {
            return None;
        }

        // Perspective divide to get NDC
        let ndc = Vec3::new(clip.x / clip.w, clip.y / clip.w, clip.z / clip.w);

        // Check if within NDC bounds [-1, 1] for x,y and [0, 1] for z (wgpu clip space)
        // Allow some margin for labels near edges
        let margin = 0.1;
        if ndc.x < -1.0 - margin
            || ndc.x > 1.0 + margin
            || ndc.y < -1.0 - margin
            || ndc.y > 1.0 + margin
        {
            return None;
        }

        // Check depth (wgpu uses [0, 1] depth range)
        if ndc.z < 0.0 || ndc.z > 1.0 {
            return None;
        }

        // Convert NDC to screen coordinates
        // NDC x: -1 (left) to 1 (right) -> screen x: 0 to width
        // NDC y: -1 (bottom) to 1 (top) -> screen y: height to 0 (flip Y)
        let screen_x = (ndc.x + 1.0) * 0.5 * self.screen_width;
        let screen_y = (1.0 - ndc.y) * 0.5 * self.screen_height;

        Some(([screen_x, screen_y], ndc.z))
    }

    /// Project with depth occlusion check.
    /// If the label's projected depth is greater than the scene depth at that pixel,
    /// the label is considered occluded.
    ///
    /// Note: This requires reading the depth buffer which is not implemented here.
    /// For MVP, we skip depth occlusion and just use the basic projection.
    pub fn project_with_occlusion(
        &self,
        world_pos: Vec3,
        view_proj: Mat4,
        _scene_depth: Option<f32>,
    ) -> Option<([f32; 2], f32)> {
        // For MVP, just use basic projection
        // Depth occlusion would require depth buffer readback
        self.project(world_pos, view_proj)
    }

    /// Get screen dimensions.
    pub fn screen_size(&self) -> (f32, f32) {
        (self.screen_width, self.screen_height)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Mat4;

    #[test]
    fn test_identity_projection() {
        let proj = LabelProjector::new(800, 600);
        // Identity matrix should map (0,0,0) to center of screen
        let result = proj.project(Vec3::ZERO, Mat4::IDENTITY);
        assert!(result.is_some());
        let (pos, _depth) = result.unwrap();
        assert!((pos[0] - 400.0).abs() < 1.0);
        assert!((pos[1] - 300.0).abs() < 1.0);
    }

    #[test]
    fn test_behind_camera() {
        let proj = LabelProjector::new(800, 600);
        // Create a view matrix looking at +Z, point behind would be at -Z
        let view = Mat4::look_at_rh(Vec3::ZERO, Vec3::Z, Vec3::Y);
        let projection = Mat4::perspective_rh(1.0, 800.0 / 600.0, 0.1, 100.0);
        let view_proj = projection * view;

        // Point behind camera should return None
        let result = proj.project(Vec3::new(0.0, 0.0, -10.0), view_proj);
        assert!(result.is_none());
    }
}
