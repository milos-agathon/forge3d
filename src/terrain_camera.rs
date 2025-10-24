// src/terrain_camera.rs
// Camera helpers for the terrain renderer orbit workflow
// Exists to compute camera transforms shared between Python and Rust layers
// RELEVANT FILES: src/terrain_renderer.rs, python/forge3d/__init__.py, tests/test_terrain_renderer.py, terrain_demo_task_breakdown.md
use glam::{Mat4, Vec3};

/// Calculate an orbit camera position around a target.
#[allow(dead_code)]
pub fn orbit_camera(
    target: Vec3,
    radius: f32,
    phi_deg: f32,
    theta_deg: f32,
    _gamma_deg: f32,
) -> Vec3 {
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
#[allow(dead_code)]
pub fn build_view_proj(
    eye: Vec3,
    target: Vec3,
    fov_y_deg: f32,
    aspect: f32,
    near: f32,
    far: f32,
) -> (Mat4, Mat4) {
    let forward = target - eye;
    let up = if forward.length_squared() < 1e-8 {
        Vec3::Y
    } else {
        Vec3::Y
    };

    let view = Mat4::look_at_rh(eye, target, up);
    let proj = crate::camera::perspective_wgpu(fov_y_deg.to_radians(), aspect, near, far);
    (view, proj)
}
