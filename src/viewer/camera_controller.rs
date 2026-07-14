// src/viewer/camera_controller.rs
// Workstream I1: Camera controllers for interactive viewer
// - Orbit camera: rotate around target with mouse
// - FPS camera: WASD movement with mouse look

use glam::{Mat4, Vec3};
use std::f32::consts::PI;

/// Maximum absolute value any camera world-coordinate component may take in the
/// interactive viewer's **terrain-local frame** (MENSURA M-06 contract).
///
/// The viewer renders in a local frame: terrain vertices are pixel-index scaled
/// (grid capped at 2048), the orbit target sits at the terrain centre, and the
/// orbit radius is at most a few thousand units. Even under extreme vertical
/// exaggeration a legitimate coordinate stays well under ~2e5. Absolute
/// *geospatial* coordinates are orders of magnitude larger — UTM northings and
/// Web-Mercator / ECEF magnitudes run ~1.6e5..6.4e6 m, exactly where the
/// single-precision cliff (~0.5 m quantization at Earth radius) corrupts
/// placement. Such coordinates must go through the anchored offscreen `Scene`
/// path (`Anchor::narrow`), never the viewer IPC. This bound turns the former
/// Python-side *convention* into a Rust-enforced contract: values beyond it are
/// rejected rather than silently truncated to `f32`.
pub const VIEWER_LOCAL_FRAME_MAX_COORD: f32 = 1.0e6;

/// Which world-coordinate a [`CameraFrameError`] refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoordRole {
    Eye,
    Target,
}

/// Why a camera world-coordinate was rejected by the viewer's local-frame contract.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CameraFrameError {
    /// A component was NaN or infinite.
    NonFinite { role: CoordRole },
    /// A component exceeded [`VIEWER_LOCAL_FRAME_MAX_COORD`] in magnitude — i.e.
    /// an absolute geospatial coordinate was pushed through the local viewer frame.
    OutOfLocalFrame {
        role: CoordRole,
        value: f32,
        max: f32,
    },
}

/// Reject non-finite or out-of-local-frame world coordinates (M-06 contract).
fn validate_local_frame(role: CoordRole, v: Vec3) -> Result<(), CameraFrameError> {
    for c in [v.x, v.y, v.z] {
        if !c.is_finite() {
            return Err(CameraFrameError::NonFinite { role });
        }
        if c.abs() > VIEWER_LOCAL_FRAME_MAX_COORD {
            return Err(CameraFrameError::OutOfLocalFrame {
                role,
                value: c,
                max: VIEWER_LOCAL_FRAME_MAX_COORD,
            });
        }
    }
    Ok(())
}

/// True when `v` is finite and within the viewer's terrain-local frame
/// ([`VIEWER_LOCAL_FRAME_MAX_COORD`]). Shared with the terrain camera handler so
/// the local-frame contract is enforced on *every* camera world-coordinate entry
/// (the terrain orbit target as well as `set_look_at`), not just one of them.
pub fn coord_within_local_frame(v: Vec3) -> bool {
    validate_local_frame(CoordRole::Target, v).is_ok()
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CameraMode {
    Orbit,
    Fps,
}

/// Orbit camera state: rotates around a target point
#[derive(Debug, Clone)]
pub struct OrbitCamera {
    pub target: Vec3,
    pub distance: f32,
    pub yaw: f32,   // Horizontal rotation (radians)
    pub pitch: f32, // Vertical rotation (radians)
    pub up: Vec3,
}

impl OrbitCamera {
    pub fn new(target: Vec3, distance: f32) -> Self {
        Self {
            target,
            distance,
            yaw: 0.0,
            pitch: -0.3, // Slightly above horizon
            up: Vec3::Y,
        }
    }

    pub fn eye(&self) -> Vec3 {
        let x = self.distance * self.pitch.cos() * self.yaw.sin();
        let y = self.distance * self.pitch.sin();
        let z = self.distance * self.pitch.cos() * self.yaw.cos();
        self.target + Vec3::new(x, y, z)
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
        let right = forward.cross(self.up).normalize();
        let up = right.cross(forward).normalize();

        let pan_speed = self.distance * 0.001;
        self.target += right * delta_x * pan_speed;
        self.target += up * delta_y * pan_speed;
    }

    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.eye(), self.target, self.up)
    }
}

/// FPS camera state: free movement with WASD
#[derive(Debug, Clone)]
pub struct FpsCamera {
    pub position: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub up: Vec3,
    pub speed: f32,
}

impl FpsCamera {
    pub fn new(position: Vec3) -> Self {
        Self {
            position,
            yaw: 0.0,
            pitch: 0.0,
            up: Vec3::Y,
            speed: 5.0, // Units per second
        }
    }

    pub fn forward(&self) -> Vec3 {
        Vec3::new(
            self.pitch.cos() * self.yaw.sin(),
            self.pitch.sin(),
            self.pitch.cos() * self.yaw.cos(),
        )
    }

    pub fn right(&self) -> Vec3 {
        self.forward().cross(self.up).normalize()
    }

    pub fn rotate(&mut self, delta_yaw: f32, delta_pitch: f32) {
        self.yaw += delta_yaw;
        self.pitch = (self.pitch + delta_pitch).clamp(-PI / 2.0 + 0.01, PI / 2.0 - 0.01);
    }

    pub fn move_forward(&mut self, delta: f32) {
        self.position += self.forward() * delta * self.speed;
    }

    pub fn move_right(&mut self, delta: f32) {
        self.position += self.right() * delta * self.speed;
    }

    pub fn move_up(&mut self, delta: f32) {
        self.position += self.up * delta * self.speed;
    }

    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.position, self.position + self.forward(), self.up)
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
            orbit: OrbitCamera::new(Vec3::ZERO, 10.0),
            fps: FpsCamera::new(Vec3::new(0.0, 5.0, -10.0)),
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
                    let forward = (self.orbit.target - self.orbit.eye()).normalize();
                    self.fps.pitch = forward.y.asin();
                    self.fps.yaw = forward.z.atan2(forward.x);
                }
                CameraMode::Orbit => {
                    self.orbit.target =
                        self.fps.position + self.fps.forward() * self.orbit.distance;
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

    pub fn view_matrix(&self) -> Mat4 {
        match self.mode {
            CameraMode::Orbit => self.orbit.view_matrix(),
            CameraMode::Fps => self.fps.view_matrix(),
        }
    }

    pub fn eye(&self) -> Vec3 {
        match self.mode {
            CameraMode::Orbit => self.orbit.eye(),
            CameraMode::Fps => self.fps.position,
        }
    }

    pub fn target(&self) -> Vec3 {
        match self.mode {
            CameraMode::Orbit => self.orbit.target,
            CameraMode::Fps => self.fps.position + self.fps.forward(),
        }
    }

    /// Force orbit camera to a specific pose (target, distance, yaw, pitch).
    ///
    /// Enforces the viewer's terrain-local-frame contract on `target`
    /// ([`VIEWER_LOCAL_FRAME_MAX_COORD`]): an absolute geospatial coordinate is
    /// rejected with [`CameraFrameError`] and no state is mutated.
    pub fn set_orbit_pose_target(
        &mut self,
        target: Vec3,
        distance: f32,
        yaw: f32,
        pitch: f32,
    ) -> Result<(), CameraFrameError> {
        validate_local_frame(CoordRole::Target, target)?;
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
    /// Enforces the viewer's terrain-local-frame contract: `eye` and `target`
    /// must be finite and within [`VIEWER_LOCAL_FRAME_MAX_COORD`]. An absolute
    /// projected/geodetic coordinate is rejected with [`CameraFrameError`] and
    /// **no state is mutated** — the previous valid pose is retained. Such
    /// coordinates belong on the anchored offscreen `Scene` path, not the viewer.
    pub fn set_look_at(
        &mut self,
        eye: Vec3,
        target: Vec3,
        up: Vec3,
    ) -> Result<(), CameraFrameError> {
        validate_local_frame(CoordRole::Eye, eye)?;
        validate_local_frame(CoordRole::Target, target)?;

        let forward = (target - eye).normalize();
        let pitch = forward.y.asin();
        let yaw = forward.z.atan2(forward.x);
        let distance = (target - eye).length().max(0.01);

        // Update orbit
        self.mode = CameraMode::Orbit;
        self.orbit.target = target;
        self.orbit.distance = distance;
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
    fn set_look_at_accepts_local_frame_and_applies_pose() {
        let mut c = CameraController::new();
        let target = Vec3::new(128.0, 20.0, 128.0); // terrain-centre scale
        let eye = Vec3::new(128.0, 300.0, 500.0);
        assert!(c.set_look_at(eye, target, Vec3::Y).is_ok());
        assert_eq!(c.orbit.target, target);
        assert_eq!(c.mode, CameraMode::Orbit);
    }

    #[test]
    fn set_look_at_rejects_ecef_magnitude_target_without_mutating() {
        let mut c = CameraController::new();
        c.set_look_at(
            Vec3::new(0.0, 10.0, 20.0),
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::Y,
        )
        .unwrap();
        let before = c.orbit.target;
        // ECEF magnitude (~6.38e6 m): the exact f32 cliff MENSURA targets.
        let err = c.set_look_at(
            Vec3::new(0.0, 10.0, 20.0),
            Vec3::new(4.2e6, 1.7e6, 4.6e6),
            Vec3::Y,
        );
        assert!(matches!(
            err,
            Err(CameraFrameError::OutOfLocalFrame {
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
    fn set_look_at_rejects_utm_northing_eye() {
        let mut c = CameraController::new();
        // UTM northing ~5.5e6 m in the eye position.
        let err = c.set_look_at(Vec3::new(5.0e5, 5.5e6, 0.0), Vec3::ZERO, Vec3::Y);
        assert!(matches!(
            err,
            Err(CameraFrameError::OutOfLocalFrame {
                role: CoordRole::Eye,
                ..
            })
        ));
    }

    #[test]
    fn set_look_at_rejects_non_finite() {
        let mut c = CameraController::new();
        assert!(matches!(
            c.set_look_at(
                Vec3::new(0.0, 1.0, 2.0),
                Vec3::new(f32::NAN, 0.0, 0.0),
                Vec3::Y
            ),
            Err(CameraFrameError::NonFinite {
                role: CoordRole::Target
            })
        ));
        assert!(matches!(
            c.set_look_at(Vec3::new(f32::INFINITY, 0.0, 0.0), Vec3::ZERO, Vec3::Y),
            Err(CameraFrameError::NonFinite {
                role: CoordRole::Eye
            })
        ));
    }

    #[test]
    fn local_frame_bound_is_inclusive_at_max_and_rejects_just_beyond() {
        let mut c = CameraController::new();
        let m = VIEWER_LOCAL_FRAME_MAX_COORD;
        assert!(c
            .set_look_at(Vec3::ZERO, Vec3::new(m, 0.0, 0.0), Vec3::Y)
            .is_ok());
        // The next representable f32 above the max must be rejected.
        let beyond = f32::from_bits(m.to_bits() + 1);
        assert!(beyond > m);
        assert!(matches!(
            c.set_look_at(Vec3::ZERO, Vec3::new(beyond, 0.0, 0.0), Vec3::Y),
            Err(CameraFrameError::OutOfLocalFrame { .. })
        ));
    }

    #[test]
    fn coord_within_local_frame_separates_local_from_absolute() {
        assert!(coord_within_local_frame(Vec3::new(128.0, 20.0, 128.0)));
        assert!(coord_within_local_frame(Vec3::new(2048.0, 5000.0, -2048.0)));
        assert!(!coord_within_local_frame(Vec3::new(0.0, 0.0, 6.4e6))); // ECEF
        assert!(!coord_within_local_frame(Vec3::new(5.0e5, 5.5e6, 0.0))); // UTM northing
        assert!(!coord_within_local_frame(Vec3::new(f32::NAN, 0.0, 0.0)));
    }

    #[test]
    fn set_orbit_pose_target_enforces_the_same_contract() {
        let mut c = CameraController::new();
        assert!(c
            .set_orbit_pose_target(Vec3::new(100.0, 5.0, 100.0), 400.0, 0.3, -0.4)
            .is_ok());
        assert!(matches!(
            c.set_orbit_pose_target(Vec3::new(0.0, 0.0, 6.4e6), 400.0, 0.0, 0.0),
            Err(CameraFrameError::OutOfLocalFrame {
                role: CoordRole::Target,
                ..
            })
        ));
    }
}
