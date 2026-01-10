//! Camera animation module for keyframe-based camera paths
//!
//! Provides CameraKeyframe storage and CameraAnimation with cubic Hermite interpolation
//! for smooth camera flyovers. Used by the offline render queue for frame export.

pub mod interpolation;
pub mod render_queue;

#[cfg(feature = "extension-module")]
use pyo3::prelude::*;

/// A single camera keyframe with position and timing
#[derive(Debug, Clone, Copy)]
pub struct CameraKeyframe {
    /// Time in seconds from animation start
    pub time: f32,
    /// Azimuth angle in degrees (horizontal rotation)
    pub phi_deg: f32,
    /// Elevation angle in degrees (vertical angle from horizon)
    pub theta_deg: f32,
    /// Distance from target/center
    pub radius: f32,
    /// Field of view in degrees
    pub fov_deg: f32,
}

impl CameraKeyframe {
    pub fn new(time: f32, phi_deg: f32, theta_deg: f32, radius: f32, fov_deg: f32) -> Self {
        Self {
            time,
            phi_deg,
            theta_deg,
            radius,
            fov_deg,
        }
    }
}

/// Interpolated camera state at a given time
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "extension-module", pyclass(module = "forge3d._forge3d", name = "CameraState", get_all))]
pub struct CameraState {
    pub phi_deg: f32,
    pub theta_deg: f32,
    pub radius: f32,
    pub fov_deg: f32,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl CameraState {
    fn __repr__(&self) -> String {
        format!(
            "CameraState(phi={:.2}, theta={:.2}, radius={:.2}, fov={:.2})",
            self.phi_deg, self.theta_deg, self.radius, self.fov_deg
        )
    }
}

/// Camera animation with keyframe storage and interpolation
#[derive(Debug, Clone)]
#[cfg_attr(feature = "extension-module", pyclass(module = "forge3d._forge3d", name = "CameraAnimation"))]
pub struct CameraAnimation {
    keyframes: Vec<CameraKeyframe>,
}

impl Default for CameraAnimation {
    fn default() -> Self {
        Self::new()
    }
}

impl CameraAnimation {
    pub fn new() -> Self {
        Self {
            keyframes: Vec::new(),
        }
    }

    /// Add a keyframe. Keyframes are sorted by time automatically.
    pub fn add_keyframe(&mut self, keyframe: CameraKeyframe) {
        self.keyframes.push(keyframe);
        self.keyframes.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());
    }

    /// Get all keyframes (sorted by time)
    pub fn keyframes(&self) -> &[CameraKeyframe] {
        &self.keyframes
    }

    /// Get animation duration (time of last keyframe)
    pub fn duration(&self) -> f32 {
        self.keyframes.last().map(|k| k.time).unwrap_or(0.0)
    }

    /// Get total frame count for given fps
    pub fn frame_count(&self, fps: u32) -> u32 {
        let duration = self.duration();
        if duration <= 0.0 || fps == 0 {
            return 0;
        }
        (duration * fps as f32).ceil() as u32 + 1
    }

    /// Evaluate camera state at a given time using cubic Hermite interpolation
    pub fn evaluate(&self, time: f32) -> Option<CameraState> {
        if self.keyframes.is_empty() {
            return None;
        }

        // Clamp time to valid range [first_keyframe_time, last_keyframe_time]
        let first_time = self.keyframes.first().map(|k| k.time).unwrap_or(0.0);
        let last_time = self.keyframes.last().map(|k| k.time).unwrap_or(0.0);
        let time = time.clamp(first_time, last_time);

        // Find surrounding keyframes
        let (k0, k1, k2, k3, t) = self.find_keyframes_for_time(time);

        Some(CameraState {
            phi_deg: interpolation::cubic_hermite(k0.phi_deg, k1.phi_deg, k2.phi_deg, k3.phi_deg, t),
            theta_deg: interpolation::cubic_hermite(k0.theta_deg, k1.theta_deg, k2.theta_deg, k3.theta_deg, t),
            radius: interpolation::cubic_hermite(k0.radius, k1.radius, k2.radius, k3.radius, t),
            fov_deg: interpolation::cubic_hermite(k0.fov_deg, k1.fov_deg, k2.fov_deg, k3.fov_deg, t),
        })
    }

    /// Find the 4 keyframes surrounding a given time for Catmull-Rom interpolation
    fn find_keyframes_for_time(&self, time: f32) -> (CameraKeyframe, CameraKeyframe, CameraKeyframe, CameraKeyframe, f32) {
        let n = self.keyframes.len();
        
        if n == 1 {
            let k = self.keyframes[0];
            return (k, k, k, k, 0.0);
        }

        // Find the segment containing time
        let mut idx = 0;
        for (i, kf) in self.keyframes.iter().enumerate() {
            if kf.time > time {
                idx = i.saturating_sub(1);
                break;
            }
            idx = i;
        }

        // Clamp to valid segment
        if idx >= n - 1 {
            idx = n - 2;
        }

        let k1 = self.keyframes[idx];
        let k2 = self.keyframes[idx + 1];

        // Get surrounding keyframes for Catmull-Rom (with clamping at boundaries)
        let k0 = if idx > 0 { self.keyframes[idx - 1] } else { k1 };
        let k3 = if idx + 2 < n { self.keyframes[idx + 2] } else { k2 };

        // Calculate normalized t within segment
        let segment_duration = k2.time - k1.time;
        let t = if segment_duration > 0.0 {
            (time - k1.time) / segment_duration
        } else {
            0.0
        };

        (k0, k1, k2, k3, t)
    }
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl CameraAnimation {
    #[new]
    pub fn py_new() -> Self {
        Self::new()
    }

    /// Add a keyframe to the animation
    #[pyo3(name = "add_keyframe", signature = (time, phi, theta, radius, fov))]
    pub fn add_keyframe_py(&mut self, time: f64, phi: f64, theta: f64, radius: f64, fov: f64) {
        self.add_keyframe(CameraKeyframe::new(
            time as f32,
            phi as f32,
            theta as f32,
            radius as f32,
            fov as f32,
        ));
    }

    /// Get animation duration in seconds
    #[getter]
    pub fn get_duration(&self) -> f64 {
        self.keyframes.last().map(|k| k.time).unwrap_or(0.0) as f64
    }

    /// Get number of keyframes
    #[getter]
    pub fn keyframe_count(&self) -> usize {
        self.keyframes.len()
    }

    /// Get total frame count for given fps
    pub fn get_frame_count(&self, fps: u32) -> u32 {
        self.frame_count(fps)
    }

    /// Evaluate camera state at given time
    #[pyo3(name = "evaluate")]
    pub fn evaluate_py(&self, time: f64) -> Option<CameraState> {
        self.evaluate(time as f32)
    }

    fn __repr__(&self) -> String {
        format!(
            "CameraAnimation(keyframes={}, duration={:.2}s)",
            self.keyframes.len(),
            self.duration()
        )
    }
}
