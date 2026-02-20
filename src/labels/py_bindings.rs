// src/labels/py_bindings.rs
// PyO3 bindings for label style types (P2.3)
// Exposes LabelStyle and LabelFlags to Python

use pyo3::prelude::*;

use super::types::{LabelFlags, LabelStyle};

/// Python wrapper for LabelFlags.
#[cfg(feature = "extension-module")]
#[pyclass(module = "forge3d._forge3d", name = "LabelFlags")]
#[derive(Clone)]
pub struct PyLabelFlags {
    #[pyo3(get, set)]
    pub underline: bool,
    #[pyo3(get, set)]
    pub small_caps: bool,
    #[pyo3(get, set)]
    pub leader: bool,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl PyLabelFlags {
    #[new]
    #[pyo3(signature = (underline=false, small_caps=false, leader=false))]
    fn new(underline: bool, small_caps: bool, leader: bool) -> Self {
        Self {
            underline,
            small_caps,
            leader,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "LabelFlags(underline={}, small_caps={}, leader={})",
            self.underline, self.small_caps, self.leader
        )
    }
}

#[cfg(feature = "extension-module")]
impl From<LabelFlags> for PyLabelFlags {
    fn from(f: LabelFlags) -> Self {
        Self {
            underline: f.underline,
            small_caps: f.small_caps,
            leader: f.leader,
        }
    }
}

#[cfg(feature = "extension-module")]
impl From<&PyLabelFlags> for LabelFlags {
    fn from(f: &PyLabelFlags) -> Self {
        Self {
            underline: f.underline,
            small_caps: f.small_caps,
            leader: f.leader,
        }
    }
}

/// Python wrapper for LabelStyle.
#[cfg(feature = "extension-module")]
#[pyclass(module = "forge3d._forge3d", name = "LabelStyle")]
#[derive(Clone)]
pub struct PyLabelStyle {
    #[pyo3(get, set)]
    pub size: f32,
    #[pyo3(get, set)]
    pub color: (f32, f32, f32, f32),
    #[pyo3(get, set)]
    pub halo_color: (f32, f32, f32, f32),
    #[pyo3(get, set)]
    pub halo_width: f32,
    #[pyo3(get, set)]
    pub priority: i32,
    #[pyo3(get, set)]
    pub min_depth: f32,
    #[pyo3(get, set)]
    pub max_depth: f32,
    #[pyo3(get, set)]
    pub depth_fade: f32,
    #[pyo3(get, set)]
    pub min_zoom: f32,
    #[pyo3(get, set)]
    pub max_zoom: f32,
    #[pyo3(get, set)]
    pub rotation: f32,
    #[pyo3(get, set)]
    pub offset: (f32, f32),
    #[pyo3(get, set)]
    pub flags: PyLabelFlags,
    #[pyo3(get, set)]
    pub horizon_fade_angle: f32,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl PyLabelStyle {
    #[new]
    #[pyo3(signature = (
        size = 14.0,
        color = (0.1, 0.1, 0.1, 1.0),
        halo_color = (1.0, 1.0, 1.0, 0.8),
        halo_width = 1.5,
        priority = 0,
        min_depth = 0.0,
        max_depth = 1.0,
        depth_fade = 0.0,
        min_zoom = 0.0,
        max_zoom = 3.4028235e38,
        rotation = 0.0,
        offset = (0.0, 0.0),
        flags = None,
        horizon_fade_angle = 5.0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        size: f32,
        color: (f32, f32, f32, f32),
        halo_color: (f32, f32, f32, f32),
        halo_width: f32,
        priority: i32,
        min_depth: f32,
        max_depth: f32,
        depth_fade: f32,
        min_zoom: f32,
        max_zoom: f32,
        rotation: f32,
        offset: (f32, f32),
        flags: Option<PyLabelFlags>,
        horizon_fade_angle: f32,
    ) -> Self {
        Self {
            size,
            color,
            halo_color,
            halo_width,
            priority,
            min_depth,
            max_depth,
            depth_fade,
            min_zoom,
            max_zoom,
            rotation,
            offset,
            flags: flags.unwrap_or_else(|| PyLabelFlags::new(false, false, false)),
            horizon_fade_angle,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "LabelStyle(size={}, color={:?}, priority={}, halo_width={})",
            self.size, self.color, self.priority, self.halo_width
        )
    }
}

#[cfg(feature = "extension-module")]
impl From<LabelStyle> for PyLabelStyle {
    fn from(s: LabelStyle) -> Self {
        Self {
            size: s.size,
            color: (s.color[0], s.color[1], s.color[2], s.color[3]),
            halo_color: (
                s.halo_color[0],
                s.halo_color[1],
                s.halo_color[2],
                s.halo_color[3],
            ),
            halo_width: s.halo_width,
            priority: s.priority,
            min_depth: s.min_depth,
            max_depth: s.max_depth,
            depth_fade: s.depth_fade,
            min_zoom: s.min_zoom,
            max_zoom: s.max_zoom,
            rotation: s.rotation,
            offset: (s.offset[0], s.offset[1]),
            flags: PyLabelFlags::from(s.flags),
            horizon_fade_angle: s.horizon_fade_angle,
        }
    }
}

#[cfg(feature = "extension-module")]
impl From<&PyLabelStyle> for LabelStyle {
    fn from(s: &PyLabelStyle) -> Self {
        Self {
            size: s.size,
            color: [s.color.0, s.color.1, s.color.2, s.color.3],
            halo_color: [
                s.halo_color.0,
                s.halo_color.1,
                s.halo_color.2,
                s.halo_color.3,
            ],
            halo_width: s.halo_width,
            priority: s.priority,
            min_depth: s.min_depth,
            max_depth: s.max_depth,
            depth_fade: s.depth_fade,
            min_zoom: s.min_zoom,
            max_zoom: s.max_zoom,
            rotation: s.rotation,
            offset: [s.offset.0, s.offset.1],
            flags: LabelFlags::from(&s.flags),
            horizon_fade_angle: s.horizon_fade_angle,
        }
    }
}
