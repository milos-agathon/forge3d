// src/viewer/camera_controller.rs
// Workstream I1: Camera controllers for interactive viewer
// - Orbit camera: rotate around target with mouse
// - FPS camera: WASD movement with mouse look

use glam::{Mat4, Vec3};
use std::f32::consts::PI;

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
}

impl Default for CameraController {
    fn default() -> Self {
        Self::new()
    }
}
