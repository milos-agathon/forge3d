// src/overlay_layer.rs
// PyO3 overlay layer wrapper describing colormap-driven terrain overlays
// Exists to ferry overlay configuration from Python into the renderer core
// RELEVANT FILES: src/colormap1d.rs, src/terrain_render_params.rs, src/terrain_renderer.rs, tests/test_overlay_layer.py
#[cfg(feature = "extension-module")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "extension-module")]
use pyo3::prelude::*;

#[cfg(feature = "extension-module")]
#[derive(Clone)]
pub(crate) enum OverlaySource {
    Colormap {
        colormap: crate::colormap1d::Colormap1D,
    },
}

#[cfg(feature = "extension-module")]
#[derive(Copy, Clone)]
pub(crate) enum OverlayBlendMode {
    Alpha,
    Add,
    Multiply,
}

#[cfg(feature = "extension-module")]
impl OverlayBlendMode {
    fn as_str(self) -> &'static str {
        match self {
            OverlayBlendMode::Alpha => "Alpha",
            OverlayBlendMode::Add => "Add",
            OverlayBlendMode::Multiply => "Multiply",
        }
    }
}

#[cfg(feature = "extension-module")]
fn parse_blend_mode(value: &str) -> PyResult<OverlayBlendMode> {
    match value.to_lowercase().as_str() {
        "alpha" => Ok(OverlayBlendMode::Alpha),
        "add" | "additive" => Ok(OverlayBlendMode::Add),
        "multiply" | "mul" => Ok(OverlayBlendMode::Multiply),
        other => Err(PyValueError::new_err(format!(
            "Unsupported blend_mode '{}'. Expected 'Alpha', 'Add', or 'Multiply'",
            other
        ))),
    }
}

/// Overlay layer configuration used by the terrain renderer.
#[cfg(feature = "extension-module")]
#[pyclass(module = "forge3d._forge3d", name = "OverlayLayer")]
#[derive(Clone)]
pub struct OverlayLayer {
    source: OverlaySource,
    strength: f32,
    offset: f32,
    domain: (f32, f32),
    blend_mode: OverlayBlendMode,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl OverlayLayer {
    /// Create an overlay sourced from a 1D colormap.
    #[staticmethod]
    #[pyo3(signature = (colormap, strength=1.0, offset=0.0, blend_mode="Alpha", domain=(0.0, 1.0)))]
    pub fn from_colormap1d(
        colormap: &crate::colormap1d::Colormap1D,
        strength: f32,
        offset: f32,
        blend_mode: &str,
        domain: (f32, f32),
    ) -> PyResult<Self> {
        if !strength.is_finite() || strength < 0.0 {
            return Err(PyValueError::new_err("strength must be finite and >= 0"));
        }
        if !offset.is_finite() {
            return Err(PyValueError::new_err("offset must be a finite value"));
        }
        if !domain.0.is_finite() || !domain.1.is_finite() {
            return Err(PyValueError::new_err("domain values must be finite floats"));
        }
        if domain.0 >= domain.1 {
            return Err(PyValueError::new_err("domain min must be < max"));
        }

        let blend = parse_blend_mode(blend_mode)?;

        Ok(Self {
            source: OverlaySource::Colormap {
                colormap: colormap.clone(),
            },
            strength,
            offset,
            domain,
            blend_mode: blend,
        })
    }

    /// Overlay strength multiplier.
    #[getter]
    pub fn strength(&self) -> f32 {
        self.strength
    }

    /// Overlay value offset applied before sampling.
    #[getter]
    pub fn offset(&self) -> f32 {
        self.offset
    }

    /// Overlay blend mode.
    #[getter]
    pub fn blend_mode(&self) -> &'static str {
        self.blend_mode.as_str()
    }

    /// Domain used when sampling the source colormap.
    #[getter]
    pub fn domain(&self) -> (f32, f32) {
        self.domain
    }

    /// Kind of overlay source (currently only `Colormap1D`).
    #[getter]
    pub fn kind(&self) -> &'static str {
        match self.source {
            OverlaySource::Colormap { .. } => "Colormap1D",
        }
    }

    /// Return the backing colormap if available.
    #[getter]
    pub fn colormap(&self) -> Option<crate::colormap1d::Colormap1D> {
        match &self.source {
            OverlaySource::Colormap { colormap } => Some(colormap.clone()),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "OverlayLayer(kind='{}', strength={:.3}, offset={:.3}, blend_mode='{}')",
            self.kind(),
            self.strength,
            self.offset,
            self.blend_mode.as_str()
        )
    }
}

#[cfg(feature = "extension-module")]
impl OverlayLayer {
    pub fn strength_value(&self) -> f32 {
        self.strength
    }

    pub fn domain_tuple(&self) -> (f32, f32) {
        self.domain
    }

    pub fn colormap_clone(&self) -> Option<crate::colormap1d::Colormap1D> {
        match &self.source {
            OverlaySource::Colormap { colormap } => Some(colormap.clone()),
        }
    }
}
