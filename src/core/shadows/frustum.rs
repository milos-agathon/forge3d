//! Camera frustum for shadow cascade calculation
//!
//! Provides frustum extraction from matrices and corner computation
//! for fitting shadow map cascades to the camera view.

use glam::{Mat4, Vec3};

/// Camera frustum representation for cascade calculation
#[derive(Debug, Clone)]
pub struct CameraFrustum {
    /// Camera position
    pub position: Vec3,
    /// Camera forward direction
    pub forward: Vec3,
    /// Camera up direction
    pub up: Vec3,
    /// Camera right direction
    pub right: Vec3,
    /// Vertical field of view (radians)
    pub fov_y: f32,
    /// Aspect ratio (width/height)
    pub aspect: f32,
    /// Near clipping plane
    pub near: f32,
    /// Far clipping plane
    pub far: f32,
}

impl CameraFrustum {
    /// Create camera frustum from view and projection matrices
    pub fn from_matrices(view: &Mat4, projection: &Mat4) -> Self {
        // Extract camera position from inverse view matrix
        let inv_view = view.inverse();
        let position = Vec3::new(inv_view.w_axis.x, inv_view.w_axis.y, inv_view.w_axis.z);

        // Extract camera directions from view matrix
        let forward = -Vec3::new(view.z_axis.x, view.z_axis.y, view.z_axis.z);
        let up = Vec3::new(view.y_axis.x, view.y_axis.y, view.y_axis.z);
        let right = Vec3::new(view.x_axis.x, view.x_axis.y, view.x_axis.z);

        // Extract FOV and aspect from projection matrix
        let fov_y = 2.0 * (1.0 / projection.y_axis.y).atan();
        let aspect = projection.y_axis.y / projection.x_axis.x;

        // Extract near/far planes from projection matrix (assuming reverse Z)
        let near = projection.w_axis.z / (projection.z_axis.z - 1.0);
        let far = projection.w_axis.z / (projection.z_axis.z + 1.0);

        Self {
            position,
            forward: forward.normalize(),
            up: up.normalize(),
            right: right.normalize(),
            fov_y,
            aspect,
            near,
            far,
        }
    }

    /// Get frustum corners at specific depth
    pub fn get_corners_at_depth(&self, depth: f32) -> [Vec3; 8] {
        let h_near = (self.fov_y * 0.5).tan() * self.near;
        let w_near = h_near * self.aspect;
        let h_far = (self.fov_y * 0.5).tan() * depth;
        let w_far = h_far * self.aspect;

        let near_center = self.position + self.forward * self.near;
        let far_center = self.position + self.forward * depth;

        [
            // Near plane corners
            near_center + self.up * h_near - self.right * w_near, // top-left
            near_center + self.up * h_near + self.right * w_near, // top-right
            near_center - self.up * h_near - self.right * w_near, // bottom-left
            near_center - self.up * h_near + self.right * w_near, // bottom-right
            // Far plane corners
            far_center + self.up * h_far - self.right * w_far, // top-left
            far_center + self.up * h_far + self.right * w_far, // top-right
            far_center - self.up * h_far - self.right * w_far, // bottom-left
            far_center - self.up * h_far + self.right * w_far, // bottom-right
        ]
    }
}
