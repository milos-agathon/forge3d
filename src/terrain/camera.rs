//! Orbit camera helpers for the terrain renderer.
//!
//! Computes camera transforms shared between Python and Rust layers.
use glam::{Mat4, Vec3};

/// Calculate an orbit camera position around a target.
///
/// # Arguments
/// - `target`: Center point to orbit around.
/// - `radius`: Distance from target (must be positive and finite).
/// - `phi_deg`: Azimuth angle in degrees (rotation around Y axis).
/// - `theta_deg`: Polar angle in degrees (elevation from vertical).
#[allow(dead_code)]
pub fn orbit_camera(target: Vec3, radius: f32, phi_deg: f32, theta_deg: f32) -> Vec3 {
    if !radius.is_finite() || radius <= 0.0 {
        return target;
    }

    let phi_rad = phi_deg.to_radians();
    let theta_rad = theta_deg.to_radians();

    // Spherical to Cartesian conversion (right-handed, Y up).
    let x = radius * theta_rad.sin() * phi_rad.cos();
    let y = radius * theta_rad.cos();
    let z = radius * theta_rad.sin() * phi_rad.sin();

    target + Vec3::new(x, y, z)
}

/// Build the view-projection matrices for the terrain camera.
///
/// Returns `(view_matrix, projection_matrix)` for right-handed Y-up coordinates.
#[allow(dead_code)]
pub fn build_view_proj(
    eye: Vec3,
    target: Vec3,
    fov_y_deg: f32,
    aspect: f32,
    near: f32,
    far: f32,
) -> (Mat4, Mat4) {
    let up = Vec3::Y;

    let view = Mat4::look_at_rh(eye, target, up);
    let proj = crate::camera::perspective_wgpu(fov_y_deg.to_radians(), aspect, near, far);
    (view, proj)
}
