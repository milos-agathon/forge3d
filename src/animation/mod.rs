//! Camera animation module for keyframe-based camera paths.
//!
//! Provides `CameraKeyframe` storage and `CameraAnimation` with cubic Hermite
//! interpolation for smooth camera flyovers. Used by the offline render queue
//! for frame export.

pub mod interpolation;
pub mod render_queue;

#[cfg(feature = "extension-module")]
use pyo3::{prelude::*, types::PyAny};

/// A single camera keyframe with position and timing.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(
    feature = "extension-module",
    pyclass(module = "forge3d._forge3d", name = "CameraKeyframe")
)]
pub struct CameraKeyframe {
    /// Time in seconds from animation start.
    pub time: f32,
    /// Azimuth angle in degrees (horizontal rotation).
    pub phi_deg: f32,
    /// Polar angle in degrees measured down from the vertical axis.
    pub theta_deg: f32,
    /// Distance from target/center.
    pub radius: f32,
    /// Field of view in degrees.
    pub fov_deg: f32,
    /// Optional explicit terrain target in world space.
    pub target: Option<[f32; 3]>,
}

impl CameraKeyframe {
    pub fn new(
        time: f32,
        phi_deg: f32,
        theta_deg: f32,
        radius: f32,
        fov_deg: f32,
        target: Option<[f32; 3]>,
    ) -> Self {
        Self {
            time,
            phi_deg,
            theta_deg,
            radius,
            fov_deg,
            target,
        }
    }
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl CameraKeyframe {
    #[new]
    #[pyo3(signature = (time, phi, theta, radius, fov, target=None))]
    fn py_new(
        time: f64,
        phi: f64,
        theta: f64,
        radius: f64,
        fov: f64,
        target: Option<(f64, f64, f64)>,
    ) -> Self {
        Self::new(
            time as f32,
            phi as f32,
            theta as f32,
            radius as f32,
            fov as f32,
            target.map(|value| [value.0 as f32, value.1 as f32, value.2 as f32]),
        )
    }

    #[getter]
    fn time(&self) -> f64 {
        self.time as f64
    }

    #[getter]
    fn phi_deg(&self) -> f64 {
        self.phi_deg as f64
    }

    #[getter]
    fn theta_deg(&self) -> f64 {
        self.theta_deg as f64
    }

    #[getter]
    fn radius(&self) -> f64 {
        self.radius as f64
    }

    #[getter]
    fn fov_deg(&self) -> f64 {
        self.fov_deg as f64
    }

    #[getter]
    fn target(&self) -> Option<(f64, f64, f64)> {
        self.target
            .map(|value| (value[0] as f64, value[1] as f64, value[2] as f64))
    }

    fn __repr__(&self) -> String {
        match self.target {
            Some(target) => format!(
                "CameraKeyframe(time={:.2}, phi={:.2}, theta={:.2}, radius={:.2}, fov={:.2}, target=({:.2}, {:.2}, {:.2}))",
                self.time,
                self.phi_deg,
                self.theta_deg,
                self.radius,
                self.fov_deg,
                target[0],
                target[1],
                target[2]
            ),
            None => format!(
                "CameraKeyframe(time={:.2}, phi={:.2}, theta={:.2}, radius={:.2}, fov={:.2}, target=None)",
                self.time, self.phi_deg, self.theta_deg, self.radius, self.fov_deg
            ),
        }
    }
}

/// Interpolated camera state at a given time.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(
    feature = "extension-module",
    pyclass(module = "forge3d._forge3d", name = "CameraState")
)]
pub struct CameraState {
    pub phi_deg: f32,
    pub theta_deg: f32,
    pub radius: f32,
    pub fov_deg: f32,
    pub target: Option<[f32; 3]>,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl CameraState {
    #[getter]
    fn phi_deg(&self) -> f64 {
        self.phi_deg as f64
    }

    #[getter]
    fn theta_deg(&self) -> f64 {
        self.theta_deg as f64
    }

    #[getter]
    fn radius(&self) -> f64 {
        self.radius as f64
    }

    #[getter]
    fn fov_deg(&self) -> f64 {
        self.fov_deg as f64
    }

    #[getter]
    fn target(&self) -> Option<(f64, f64, f64)> {
        self.target
            .map(|value| (value[0] as f64, value[1] as f64, value[2] as f64))
    }

    fn __repr__(&self) -> String {
        match self.target {
            Some(target) => format!(
                "CameraState(phi={:.2}, theta={:.2}, radius={:.2}, fov={:.2}, target=({:.2}, {:.2}, {:.2}))",
                self.phi_deg,
                self.theta_deg,
                self.radius,
                self.fov_deg,
                target[0],
                target[1],
                target[2]
            ),
            None => format!(
                "CameraState(phi={:.2}, theta={:.2}, radius={:.2}, fov={:.2}, target=None)",
                self.phi_deg, self.theta_deg, self.radius, self.fov_deg
            ),
        }
    }
}

/// Camera animation with keyframe storage and interpolation.
#[derive(Debug, Clone)]
#[cfg_attr(
    feature = "extension-module",
    pyclass(module = "forge3d._forge3d", name = "CameraAnimation")
)]
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

    fn sort_keyframes(keyframes: &mut [CameraKeyframe]) {
        keyframes.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());
    }

    /// Add a keyframe. Keyframes are sorted by time automatically.
    pub fn add_keyframe(&mut self, keyframe: CameraKeyframe) {
        self.keyframes.push(keyframe);
        Self::sort_keyframes(&mut self.keyframes);
    }

    /// Replace all keyframes. The new list is sorted by time automatically.
    pub fn replace_keyframes(&mut self, mut keyframes: Vec<CameraKeyframe>) {
        Self::sort_keyframes(&mut keyframes);
        self.keyframes = keyframes;
    }

    /// Remove all keyframes.
    pub fn clear_keyframes(&mut self) {
        self.keyframes.clear();
    }

    /// Get all keyframes (sorted by time).
    pub fn keyframes(&self) -> &[CameraKeyframe] {
        &self.keyframes
    }

    /// Get animation duration (time of last keyframe).
    pub fn duration(&self) -> f32 {
        self.keyframes.last().map(|k| k.time).unwrap_or(0.0)
    }

    /// Get total frame count for given fps.
    pub fn frame_count(&self, fps: u32) -> u32 {
        let duration = self.duration();
        if duration <= 0.0 || fps == 0 {
            return 0;
        }
        (duration * fps as f32).ceil() as u32 + 1
    }

    fn interpolate_target(
        k0: CameraKeyframe,
        k1: CameraKeyframe,
        k2: CameraKeyframe,
        k3: CameraKeyframe,
        t: f32,
    ) -> Option<[f32; 3]> {
        let p1 = k1.target?;
        let p2 = k2.target?;
        let p0 = k0.target.unwrap_or(p1);
        let p3 = k3.target.unwrap_or(p2);

        Some([
            interpolation::cubic_hermite(p0[0], p1[0], p2[0], p3[0], t),
            interpolation::cubic_hermite(p0[1], p1[1], p2[1], p3[1], t),
            interpolation::cubic_hermite(p0[2], p1[2], p2[2], p3[2], t),
        ])
    }

    /// Evaluate camera state at a given time using cubic Hermite interpolation.
    pub fn evaluate(&self, time: f32) -> Option<CameraState> {
        if self.keyframes.is_empty() {
            return None;
        }

        let first_time = self.keyframes.first().map(|k| k.time).unwrap_or(0.0);
        let last_time = self.keyframes.last().map(|k| k.time).unwrap_or(0.0);
        let time = time.clamp(first_time, last_time);

        let (k0, k1, k2, k3, t) = self.find_keyframes_for_time(time);

        Some(CameraState {
            phi_deg: interpolation::cubic_hermite(
                k0.phi_deg, k1.phi_deg, k2.phi_deg, k3.phi_deg, t,
            ),
            theta_deg: interpolation::cubic_hermite(
                k0.theta_deg,
                k1.theta_deg,
                k2.theta_deg,
                k3.theta_deg,
                t,
            ),
            radius: interpolation::cubic_hermite(k0.radius, k1.radius, k2.radius, k3.radius, t),
            fov_deg: interpolation::cubic_hermite(
                k0.fov_deg, k1.fov_deg, k2.fov_deg, k3.fov_deg, t,
            ),
            target: Self::interpolate_target(k0, k1, k2, k3, t),
        })
    }

    /// Find the 4 keyframes surrounding a given time for Catmull-Rom interpolation.
    fn find_keyframes_for_time(
        &self,
        time: f32,
    ) -> (
        CameraKeyframe,
        CameraKeyframe,
        CameraKeyframe,
        CameraKeyframe,
        f32,
    ) {
        let n = self.keyframes.len();

        if n == 1 {
            let k = self.keyframes[0];
            return (k, k, k, k, 0.0);
        }

        let mut idx = 0;
        for (i, kf) in self.keyframes.iter().enumerate() {
            if kf.time > time {
                idx = i.saturating_sub(1);
                break;
            }
            idx = i;
        }

        if idx >= n - 1 {
            idx = n - 2;
        }

        let k1 = self.keyframes[idx];
        let k2 = self.keyframes[idx + 1];
        let k0 = if idx > 0 { self.keyframes[idx - 1] } else { k1 };
        let k3 = if idx + 2 < n {
            self.keyframes[idx + 2]
        } else {
            k2
        };

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

    /// Add a keyframe to the animation.
    #[pyo3(name = "add_keyframe", signature = (time, phi, theta, radius, fov, target=None))]
    pub fn add_keyframe_py(
        &mut self,
        time: f64,
        phi: f64,
        theta: f64,
        radius: f64,
        fov: f64,
        target: Option<(f64, f64, f64)>,
    ) {
        self.add_keyframe(CameraKeyframe::new(
            time as f32,
            phi as f32,
            theta as f32,
            radius as f32,
            fov as f32,
            target.map(|value| [value.0 as f32, value.1 as f32, value.2 as f32]),
        ));
    }

    /// Return a copy of the keyframes in sorted time order.
    #[pyo3(name = "get_keyframes")]
    pub fn get_keyframes_py(&self, py: Python<'_>) -> PyResult<Vec<Py<CameraKeyframe>>> {
        self.keyframes
            .iter()
            .copied()
            .map(|keyframe| Py::new(py, keyframe))
            .collect()
    }

    /// Replace all keyframes from a Python iterable of CameraKeyframe objects.
    #[pyo3(name = "replace_keyframes")]
    pub fn replace_keyframes_py(&mut self, keyframes: &Bound<'_, PyAny>) -> PyResult<()> {
        let mut updated = Vec::new();
        for item in keyframes.iter()? {
            let item = item?;
            let keyframe = item.extract::<PyRef<'_, CameraKeyframe>>()?;
            updated.push(*keyframe);
        }
        self.replace_keyframes(updated);
        Ok(())
    }

    /// Remove all keyframes.
    #[pyo3(name = "clear_keyframes")]
    pub fn clear_keyframes_py(&mut self) {
        self.clear_keyframes();
    }

    /// Get animation duration in seconds.
    #[getter]
    pub fn get_duration(&self) -> f64 {
        self.keyframes.last().map(|k| k.time).unwrap_or(0.0) as f64
    }

    /// Get number of keyframes.
    #[getter]
    pub fn keyframe_count(&self) -> usize {
        self.keyframes.len()
    }

    /// Get total frame count for given fps.
    pub fn get_frame_count(&self, fps: u32) -> u32 {
        self.frame_count(fps)
    }

    /// Evaluate camera state at given time.
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
