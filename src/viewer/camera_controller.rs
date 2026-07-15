// src/viewer/camera_controller.rs
// Workstream I1: Camera controllers for interactive viewer
// - Orbit camera: rotate around target with mouse
// - FPS camera: WASD movement with mouse look

use glam::{DVec3, Mat4, Vec3};
use std::f32::consts::PI;

use crate::camera::Anchor;

/// Maximum component-wise distance that a world position may have from the
/// prospective render-frame anchor. Absolute UTM/ECEF magnitudes are valid;
/// only an unsafe residual in the selected local render frame is rejected.
pub const VIEWER_RENDER_FRAME_MAX_COORD: f64 = 1_000_000.0;

/// Which world-coordinate a [`CameraFrameError`] refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoordRole {
    Eye,
    Target,
    Object,
    Content,
}

/// Why a world-coordinate was rejected at the viewer render-frame boundary.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CameraFrameError {
    /// A component was NaN or infinite.
    NonFinite { role: CoordRole },
    /// A component exceeded [`VIEWER_RENDER_FRAME_MAX_COORD`] after subtraction
    /// by the prospective frame anchor.
    OutOfRenderFrame {
        role: CoordRole,
        residual: f64,
        max: f64,
    },
}

impl std::fmt::Display for CameraFrameError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonFinite { role } => write!(f, "non_finite_world_coordinate: role={role:?}"),
            Self::OutOfRenderFrame {
                role,
                residual,
                max,
            } => write!(
                f,
                "out_of_render_frame_residual: role={role:?} residual={residual} bound={max}"
            ),
        }
    }
}

impl std::error::Error for CameraFrameError {}

/// Validate one absolute f64 world point against a copied prospective anchor.
pub fn validate_world_point(
    role: CoordRole,
    v: DVec3,
    anchor: &Anchor,
) -> Result<(), CameraFrameError> {
    for c in [v.x, v.y, v.z] {
        if !c.is_finite() {
            return Err(CameraFrameError::NonFinite { role });
        }
    }
    let residual = (v - anchor.origin()).abs().max_element();
    if residual > VIEWER_RENDER_FRAME_MAX_COORD {
        return Err(CameraFrameError::OutOfRenderFrame {
            role,
            residual,
            max: VIEWER_RENDER_FRAME_MAX_COORD,
        });
    }
    Ok(())
}

/// Copy and prospectively rebase an anchor without mutating the live frame.
pub fn prospective_anchor(current: &Anchor, focus: DVec3) -> Anchor {
    let mut candidate = *current;
    candidate.rebase_if_needed(focus);
    candidate
}

/// Validate a complete requested camera pose transactionally.
pub fn validate_camera_pose(
    current_anchor: &Anchor,
    rebase_focus: DVec3,
    eye: DVec3,
    target: DVec3,
) -> Result<Anchor, CameraFrameError> {
    if !eye.is_finite() {
        return Err(CameraFrameError::NonFinite {
            role: CoordRole::Eye,
        });
    }
    if !target.is_finite() {
        return Err(CameraFrameError::NonFinite {
            role: CoordRole::Target,
        });
    }
    if !rebase_focus.is_finite() {
        return Err(CameraFrameError::NonFinite {
            role: CoordRole::Target,
        });
    }
    let candidate = prospective_anchor(current_anchor, rebase_focus);
    validate_world_point(CoordRole::Eye, eye, &candidate)?;
    validate_world_point(CoordRole::Target, target, &candidate)?;
    Ok(candidate)
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CameraMode {
    Orbit,
    Fps,
}

/// Orbit camera state: rotates around a target point
#[derive(Debug, Clone)]
pub struct OrbitCamera {
    pub target: DVec3,
    pub distance: f32,
    pub yaw: f32,   // Horizontal rotation (radians)
    pub pitch: f32, // Vertical rotation (radians)
    pub up: Vec3,
}

impl OrbitCamera {
    pub fn new(target: DVec3, distance: f32) -> Self {
        Self {
            target,
            distance,
            yaw: 0.0,
            pitch: -0.3, // Slightly above horizon
            up: Vec3::Y,
        }
    }

    pub fn eye(&self) -> DVec3 {
        let x = self.distance * self.pitch.cos() * self.yaw.sin();
        let y = self.distance * self.pitch.sin();
        let z = self.distance * self.pitch.cos() * self.yaw.cos();
        self.target + DVec3::new(f64::from(x), f64::from(y), f64::from(z))
    }

    pub fn rotate(&mut self, delta_yaw: f32, delta_pitch: f32) {
        self.yaw += delta_yaw;
        self.pitch = (self.pitch + delta_pitch).clamp(-PI / 2.0 + 0.01, PI / 2.0 - 0.01);
    }

    pub fn zoom(&mut self, delta: f32) {
        self.distance = (self.distance * (1.0 + delta * 0.1)).clamp(0.1, 1000.0);
    }

    pub fn pan(&mut self, delta_x: f32, delta_y: f32) {
        let forward = (self.target - self.eye()).normalize();
        let right = forward.cross(self.up.as_dvec3()).normalize();
        let up = right.cross(forward).normalize();

        let pan_speed = f64::from(self.distance * 0.001);
        self.target += right * f64::from(delta_x) * pan_speed;
        self.target += up * f64::from(delta_y) * pan_speed;
    }

    pub fn view_matrix(&self, anchor: &Anchor) -> Mat4 {
        anchor.view_look_at(self.eye(), self.target, self.up)
    }
}

/// FPS camera state: free movement with WASD
#[derive(Debug, Clone)]
pub struct FpsCamera {
    pub position: DVec3,
    pub yaw: f32,
    pub pitch: f32,
    pub up: Vec3,
    pub speed: f32,
}

impl FpsCamera {
    pub fn new(position: DVec3) -> Self {
        Self {
            position,
            yaw: 0.0,
            pitch: 0.0,
            up: Vec3::Y,
            speed: 5.0, // Units per second
        }
    }

    pub fn forward(&self) -> DVec3 {
        DVec3::new(
            f64::from(self.pitch.cos() * self.yaw.sin()),
            f64::from(self.pitch.sin()),
            f64::from(self.pitch.cos() * self.yaw.cos()),
        )
    }

    pub fn right(&self) -> DVec3 {
        self.forward().cross(self.up.as_dvec3()).normalize()
    }

    pub fn rotate(&mut self, delta_yaw: f32, delta_pitch: f32) {
        self.yaw += delta_yaw;
        self.pitch = (self.pitch + delta_pitch).clamp(-PI / 2.0 + 0.01, PI / 2.0 - 0.01);
    }

    pub fn move_forward(&mut self, delta: f32) {
        self.position += self.forward() * f64::from(delta * self.speed);
    }

    pub fn move_right(&mut self, delta: f32) {
        self.position += self.right() * f64::from(delta * self.speed);
    }

    pub fn move_up(&mut self, delta: f32) {
        self.position += self.up.as_dvec3() * f64::from(delta * self.speed);
    }

    pub fn view_matrix(&self, anchor: &Anchor) -> Mat4 {
        anchor.view_look_at(self.position, self.position + self.forward(), self.up)
    }
}

/// Combined camera controller with mode switching
pub struct CameraController {
    mode: CameraMode,
    orbit: OrbitCamera,
    fps: FpsCamera,
    mouse_sensitivity: f32,
    pub last_mouse_pos: Option<(f32, f32)>,
    pub mouse_pressed: bool,
}

impl CameraController {
    pub fn new() -> Self {
        Self {
            mode: CameraMode::Orbit,
            orbit: OrbitCamera::new(DVec3::ZERO, 10.0),
            fps: FpsCamera::new(DVec3::new(0.0, 5.0, -10.0)),
            mouse_sensitivity: 0.005,
            last_mouse_pos: None,
            mouse_pressed: false,
        }
    }

    pub fn mode(&self) -> CameraMode {
        self.mode
    }

    pub fn set_mode(&mut self, mode: CameraMode) {
        if self.mode != mode {
            // Sync camera positions on mode switch
            match mode {
                CameraMode::Fps => {
                    self.fps.position = self.orbit.eye();
                    // Compute yaw/pitch from orbit
                    let forward = crate::camera::Anchor::new()
                        .to_render_direction((self.orbit.target - self.orbit.eye()).normalize());
                    self.fps.pitch = forward.y.asin();
                    self.fps.yaw = forward.z.atan2(forward.x);
                }
                CameraMode::Orbit => {
                    self.orbit.target =
                        self.fps.position + self.fps.forward() * f64::from(self.orbit.distance);
                    self.orbit.yaw = self.fps.yaw;
                    self.orbit.pitch = self.fps.pitch;
                }
            }
            self.mode = mode;
        }
    }

    pub fn handle_mouse_move(&mut self, x: f32, y: f32) {
        if let Some((last_x, last_y)) = self.last_mouse_pos {
            let delta_x = x - last_x;
            let delta_y = y - last_y;

            if self.mouse_pressed {
                let delta_yaw = -delta_x * self.mouse_sensitivity;
                let delta_pitch = -delta_y * self.mouse_sensitivity;

                match self.mode {
                    CameraMode::Orbit => self.orbit.rotate(delta_yaw, delta_pitch),
                    CameraMode::Fps => self.fps.rotate(delta_yaw, delta_pitch),
                }
            }
        }
        self.last_mouse_pos = Some((x, y));
    }

    pub fn handle_mouse_scroll(&mut self, delta: f32) {
        if let CameraMode::Orbit = self.mode {
            self.orbit.zoom(delta);
        }
    }

    pub fn handle_pan(&mut self, delta_x: f32, delta_y: f32) {
        if let CameraMode::Orbit = self.mode {
            self.orbit.pan(delta_x, delta_y);
        }
    }

    pub fn update_fps(&mut self, dt: f32, forward: f32, right: f32, up: f32) {
        if let CameraMode::Fps = self.mode {
            self.fps.move_forward(forward * dt);
            self.fps.move_right(right * dt);
            self.fps.move_up(up * dt);
        }
    }

    pub fn view_matrix(&self, anchor: &Anchor) -> Mat4 {
        match self.mode {
            CameraMode::Orbit => self.orbit.view_matrix(anchor),
            CameraMode::Fps => self.fps.view_matrix(anchor),
        }
    }

    pub fn eye(&self) -> DVec3 {
        match self.mode {
            CameraMode::Orbit => self.orbit.eye(),
            CameraMode::Fps => self.fps.position,
        }
    }

    pub fn target(&self) -> DVec3 {
        match self.mode {
            CameraMode::Orbit => self.orbit.target,
            CameraMode::Fps => self.fps.position + self.fps.forward(),
        }
    }

    pub fn up(&self) -> Vec3 {
        match self.mode {
            CameraMode::Orbit => self.orbit.up,
            CameraMode::Fps => self.fps.up,
        }
    }

    /// Force orbit camera to a specific pose (target, distance, yaw, pitch).
    ///
    /// Validates the complete pose against a prospectively rebased frame before
    /// mutating state. Earth-scale absolute coordinates are accepted.
    pub fn set_orbit_pose_target(
        &mut self,
        current_anchor: &Anchor,
        target: DVec3,
        distance: f32,
        yaw: f32,
        pitch: f32,
    ) -> Result<(), CameraFrameError> {
        let candidate = OrbitCamera {
            target,
            distance: distance.max(0.01),
            yaw,
            pitch: pitch.clamp(-PI / 2.0 + 0.01, PI / 2.0 - 0.01),
            up: self.orbit.up,
        };
        let focus = DVec3::new(target.x, 0.0, target.z);
        validate_camera_pose(current_anchor, focus, candidate.eye(), target)?;
        self.mode = CameraMode::Orbit;
        self.orbit.target = target;
        self.orbit.distance = distance.max(0.01);
        // Clamp pitch to avoid gimbal lock
        let p = pitch.clamp(
            -std::f32::consts::FRAC_PI_2 + 0.01,
            std::f32::consts::FRAC_PI_2 - 0.01,
        );
        self.orbit.pitch = p;
        self.orbit.yaw = yaw;
        Ok(())
    }

    /// Set camera from eye/target/up; updates both orbit and FPS states and switches to Orbit mode.
    ///
    /// Validates a prospective f64 world pose before mutating either camera mode.
    pub fn set_look_at(
        &mut self,
        current_anchor: &Anchor,
        eye: DVec3,
        target: DVec3,
        up: Vec3,
    ) -> Result<(), CameraFrameError> {
        validate_camera_pose(current_anchor, eye, eye, target)?;

        let forward = Anchor::new().to_render_direction((target - eye).normalize());
        let pitch = forward.y.asin();
        let yaw = forward.z.atan2(forward.x);
        let distance = (target - eye).length().max(0.01);

        // Update orbit
        self.mode = CameraMode::Orbit;
        self.orbit.target = target;
        self.orbit.distance = Anchor::new()
            .to_render_direction(DVec3::new(distance, 0.0, 0.0))
            .x;
        self.orbit.yaw = yaw;
        self.orbit.pitch = pitch;
        self.orbit.up = if up.length_squared() > 0.0 {
            up.normalize()
        } else {
            Vec3::Y
        };

        // Keep FPS roughly in sync too
        self.fps.position = eye;
        self.fps.yaw = yaw;
        self.fps.pitch = pitch;
        self.fps.up = self.orbit.up;
        Ok(())
    }
}

impl Default for CameraController {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_look_at_accepts_earth_scale_world_coordinates() {
        let mut c = CameraController::new();
        let anchor = Anchor::new();
        let target = DVec3::new(500_000.0, 20.0, 5_500_000.0);
        let eye = DVec3::new(500_000.0, 300.0, 5_500_400.0);
        assert!(c.set_look_at(&anchor, eye, target, Vec3::Y).is_ok());
        assert_eq!(c.orbit.target, target);
        assert_eq!(c.mode, CameraMode::Orbit);
    }

    #[test]
    fn set_look_at_rejects_vacuous_residual_without_mutating() {
        let mut c = CameraController::new();
        let anchor = Anchor::new();
        c.set_look_at(
            &anchor,
            DVec3::new(0.0, 10.0, 20.0),
            DVec3::new(1.0, 2.0, 3.0),
            Vec3::Y,
        )
        .unwrap();
        let before = c.orbit.target;
        let err = c.set_look_at(
            &anchor,
            DVec3::new(0.0, 10.0, 20.0),
            DVec3::new(1.0e300, 0.0, 0.0),
            Vec3::Y,
        );
        assert!(matches!(
            err,
            Err(CameraFrameError::OutOfRenderFrame {
                role: CoordRole::Target,
                ..
            })
        ));
        assert_eq!(
            c.orbit.target, before,
            "a rejected look_at must not move the camera"
        );
    }

    #[test]
    fn set_look_at_rejects_non_finite() {
        let mut c = CameraController::new();
        let anchor = Anchor::new();
        let err = c.set_look_at(
            &anchor,
            DVec3::new(f64::INFINITY, 0.0, 0.0),
            DVec3::ZERO,
            Vec3::Y,
        );
        assert!(matches!(
            err,
            Err(CameraFrameError::NonFinite {
                role: CoordRole::Eye,
            })
        ));
    }

    #[test]
    fn render_frame_bound_is_relative_to_prospective_anchor() {
        let mut c = CameraController::new();
        let anchor = Anchor::new();
        let origin = DVec3::new(4_200_000.0, 0.0, 4_600_000.0);
        let m = VIEWER_RENDER_FRAME_MAX_COORD;
        assert!(c
            .set_look_at(&anchor, origin, origin + DVec3::new(m, 0.0, 0.0), Vec3::Y)
            .is_ok());
        assert!(matches!(
            c.set_look_at(
                &anchor,
                origin,
                origin + DVec3::new(m + 1.0, 0.0, 0.0),
                Vec3::Y
            ),
            Err(CameraFrameError::OutOfRenderFrame { .. })
        ));
    }

    #[test]
    fn set_orbit_pose_target_uses_horizontal_terrain_focus() {
        let mut c = CameraController::new();
        let anchor = Anchor::new();
        assert!(c
            .set_orbit_pose_target(
                &anchor,
                DVec3::new(500_000.0, 250.0, 5_500_000.0),
                400.0,
                0.3,
                -0.4
            )
            .is_ok());
    }
}
