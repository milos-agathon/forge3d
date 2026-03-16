use super::super::*;

// Plan 2: Selection style Python bindings
#[cfg(feature = "extension-module")]
#[pyclass(module = "forge3d._forge3d", name = "SelectionStyle")]
#[derive(Clone)]
pub struct PySelectionStyle {
    #[pyo3(get, set)]
    pub color: (f32, f32, f32, f32),
    #[pyo3(get, set)]
    pub outline: bool,
    #[pyo3(get, set)]
    pub outline_width: f32,
    #[pyo3(get, set)]
    pub glow: bool,
    #[pyo3(get, set)]
    pub glow_intensity: f32,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl PySelectionStyle {
    #[new]
    #[pyo3(signature = (color=(1.0, 0.8, 0.0, 0.5), outline=false, outline_width=2.0, glow=false, glow_intensity=0.5))]
    fn new(
        color: (f32, f32, f32, f32),
        outline: bool,
        outline_width: f32,
        glow: bool,
        glow_intensity: f32,
    ) -> Self {
        Self {
            color,
            outline,
            outline_width,
            glow,
            glow_intensity,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "SelectionStyle(color={:?}, outline={}, glow={})",
            self.color, self.outline, self.glow
        )
    }
}

#[cfg(feature = "extension-module")]
impl From<PySelectionStyle> for crate::picking::SelectionStyle {
    fn from(s: PySelectionStyle) -> Self {
        Self {
            color: [s.color.0, s.color.1, s.color.2, s.color.3],
            outline: s.outline,
            outline_width: s.outline_width,
            glow: s.glow,
            glow_intensity: s.glow_intensity,
        }
    }
}

#[cfg(feature = "extension-module")]
impl From<&crate::picking::SelectionStyle> for PySelectionStyle {
    fn from(s: &crate::picking::SelectionStyle) -> Self {
        Self {
            color: (s.color[0], s.color[1], s.color[2], s.color[3]),
            outline: s.outline,
            outline_width: s.outline_width,
            glow: s.glow,
            glow_intensity: s.glow_intensity,
        }
    }
}

// Plan 3: Highlight style Python bindings
#[cfg(feature = "extension-module")]
#[pyclass(module = "forge3d._forge3d", name = "HighlightStyle")]
#[derive(Clone)]
pub struct PyHighlightStyle {
    #[pyo3(get, set)]
    pub color: (f32, f32, f32, f32),
    #[pyo3(get, set)]
    pub effect: String,
    #[pyo3(get, set)]
    pub outline_width: f32,
    #[pyo3(get, set)]
    pub glow_intensity: f32,
    #[pyo3(get, set)]
    pub glow_radius: f32,
    #[pyo3(get, set)]
    pub pulse_speed: f32,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl PyHighlightStyle {
    #[new]
    #[pyo3(signature = (
        color=(1.0, 0.8, 0.0, 0.5),
        effect="color_tint",
        outline_width=2.0,
        glow_intensity=0.5,
        glow_radius=8.0,
        pulse_speed=0.0
    ))]
    fn new(
        color: (f32, f32, f32, f32),
        effect: &str,
        outline_width: f32,
        glow_intensity: f32,
        glow_radius: f32,
        pulse_speed: f32,
    ) -> Self {
        Self {
            color,
            effect: effect.to_string(),
            outline_width,
            glow_intensity,
            glow_radius,
            pulse_speed,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "HighlightStyle(effect='{}', color={:?})",
            self.effect, self.color
        )
    }

    /// Create outline style
    #[staticmethod]
    fn outline(color: (f32, f32, f32, f32), width: f32) -> Self {
        Self {
            color,
            effect: "outline".to_string(),
            outline_width: width,
            glow_intensity: 0.5,
            glow_radius: 8.0,
            pulse_speed: 0.0,
        }
    }

    /// Create glow style
    #[staticmethod]
    fn glow(color: (f32, f32, f32, f32), intensity: f32, radius: f32) -> Self {
        Self {
            color,
            effect: "glow".to_string(),
            outline_width: 2.0,
            glow_intensity: intensity,
            glow_radius: radius,
            pulse_speed: 0.0,
        }
    }
}

#[cfg(feature = "extension-module")]
impl From<PyHighlightStyle> for crate::picking::HighlightStyle {
    fn from(s: PyHighlightStyle) -> Self {
        use crate::picking::HighlightEffect;

        let effect = match s.effect.as_str() {
            "none" => HighlightEffect::None,
            "color_tint" => HighlightEffect::ColorTint,
            "outline" => HighlightEffect::Outline,
            "glow" => HighlightEffect::Glow,
            "outline_glow" => HighlightEffect::OutlineGlow,
            _ => HighlightEffect::ColorTint,
        };

        Self {
            color: [s.color.0, s.color.1, s.color.2, s.color.3],
            secondary_color: [1.0, 1.0, 1.0, 0.3],
            effect,
            outline_width: s.outline_width,
            glow_intensity: s.glow_intensity,
            glow_radius: s.glow_radius,
            pulse_speed: s.pulse_speed,
            depth_bias: 0.001,
        }
    }
}
