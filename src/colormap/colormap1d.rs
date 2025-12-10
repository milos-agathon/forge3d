// src/colormap1d.rs
// PyO3 colormap wrapper for height-based 1D lookup tables
// Exists to expose dynamic terrain colormaps to the Python API
// RELEVANT FILES: src/terrain/mod.rs, python/forge3d/__init__.py, tests/test_colormap1d.py, src/session.rs
#[cfg(feature = "extension-module")]
use pyo3::exceptions::{PyRuntimeError, PyValueError};
#[cfg(feature = "extension-module")]
use pyo3::prelude::*;
#[cfg(feature = "extension-module")]
use std::cmp::Ordering;
#[cfg(feature = "extension-module")]
use std::sync::Arc;

#[cfg(feature = "extension-module")]
#[pyclass(module = "forge3d._forge3d", name = "Colormap1D")]
#[derive(Clone)]
pub struct Colormap1D {
    #[allow(dead_code)]
    pub(crate) lut: Arc<crate::terrain::ColormapLUT>,
    pub(crate) domain: (f32, f32),
    pub(crate) stops: Vec<(f32, String)>,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl Colormap1D {
    /// Create a colormap from color stops.
    ///
    /// Args:
    ///     stops: List of (value, color) tuples, color as "#RRGGBB"
    ///     domain: (min, max) value range
    #[staticmethod]
    #[pyo3(signature = (stops, domain))]
    pub fn from_stops(
        _py: Python<'_>,
        stops: Vec<(f32, String)>,
        domain: (f32, f32),
    ) -> PyResult<Self> {
        if stops.is_empty() {
            return Err(PyValueError::new_err("stops must not be empty"));
        }
        if stops.len() < 2 {
            return Err(PyValueError::new_err(
                "stops must contain at least two entries",
            ));
        }
        if stops.iter().any(|(value, _)| !value.is_finite()) {
            return Err(PyValueError::new_err("stop values must be finite numbers"));
        }
        if !domain.0.is_finite() || !domain.1.is_finite() {
            return Err(PyValueError::new_err("domain values must be finite"));
        }
        if domain.0 >= domain.1 {
            return Err(PyValueError::new_err("domain min must be < max"));
        }

        let mut stops_sorted = stops.clone();
        stops_sorted.sort_by(|a, b| match a.0.partial_cmp(&b.0) {
            Some(ordering) => ordering,
            None => Ordering::Equal,
        });

        let colors: Vec<[u8; 4]> = stops_sorted
            .iter()
            .map(|(_, hex)| parse_html_color(hex))
            .collect::<PyResult<_>>()?;

        let resolution = 256usize;
        let lut_data = interpolate_colormap(&stops_sorted, &colors, domain, resolution)?;

        let ctx = crate::core::gpu::ctx();
        let lut =
            crate::terrain::ColormapLUT::new_single_palette(&ctx.device, &ctx.queue, &lut_data)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        Ok(Self {
            lut: Arc::new(lut),
            domain,
            stops: stops_sorted,
        })
    }

    /// Get the domain (min, max).
    #[getter]
    pub fn domain(&self) -> (f32, f32) {
        self.domain
    }

    /// Python repr for debugging.
    fn __repr__(&self) -> String {
        let lut_refs = Arc::strong_count(&self.lut);
        format!(
            "Colormap1D(stops={}, domain=({:.3}, {:.3}), lut_refs={})",
            self.stops.len(),
            self.domain.0,
            self.domain.1,
            lut_refs
        )
    }
}

#[cfg(feature = "extension-module")]
fn parse_html_color(hex: &str) -> PyResult<[u8; 4]> {
    let trimmed = hex.trim().trim_start_matches('#');
    if trimmed.len() != 6 {
        return Err(PyValueError::new_err(format!(
            "HTML color must be #RRGGBB, got: #{}",
            trimmed
        )));
    }
    let r = u8::from_str_radix(&trimmed[0..2], 16)
        .map_err(|e| PyValueError::new_err(format!("Invalid hex color: {}", e)))?;
    let g = u8::from_str_radix(&trimmed[2..4], 16)
        .map_err(|e| PyValueError::new_err(format!("Invalid hex color: {}", e)))?;
    let b = u8::from_str_radix(&trimmed[4..6], 16)
        .map_err(|e| PyValueError::new_err(format!("Invalid hex color: {}", e)))?;
    Ok([r, g, b, 255])
}

#[cfg(feature = "extension-module")]
fn interpolate_colormap(
    stops: &[(f32, String)],
    colors: &[[u8; 4]],
    domain: (f32, f32),
    resolution: usize,
) -> PyResult<Vec<u8>> {
    if colors.len() != stops.len() {
        return Err(PyRuntimeError::new_err(
            "stops and colors must have the same length",
        ));
    }

    let mut data = Vec::with_capacity(resolution * 4);
    for i in 0..resolution {
        let t = i as f32 / (resolution - 1) as f32;
        let value = domain.0 + t * (domain.1 - domain.0);
        let color = interpolate_color_at_value(value, stops, colors);
        data.extend_from_slice(&color);
    }
    Ok(data)
}

#[cfg(feature = "extension-module")]
fn interpolate_color_at_value(value: f32, stops: &[(f32, String)], colors: &[[u8; 4]]) -> [u8; 4] {
    if value <= stops[0].0 {
        return colors[0];
    }
    if value >= stops.last().map(|s| s.0).unwrap_or(value) {
        return *colors.last().unwrap();
    }

    for i in 0..stops.len() - 1 {
        let v0 = stops[i].0;
        let v1 = stops[i + 1].0;
        if value >= v0 && value <= v1 {
            let t = if v1 > v0 {
                (value - v0) / (v1 - v0)
            } else {
                0.0
            };
            let c0 = colors[i];
            let c1 = colors[i + 1];
            return [
                lerp_u8(c0[0], c1[0], t),
                lerp_u8(c0[1], c1[1], t),
                lerp_u8(c0[2], c1[2], t),
                lerp_u8(c0[3], c1[3], t),
            ];
        }
    }

    *colors.last().unwrap()
}

#[cfg(feature = "extension-module")]
fn lerp_u8(a: u8, b: u8, t: f32) -> u8 {
    let a_f = a as f32;
    let b_f = b as f32;
    (a_f + (b_f - a_f) * t.clamp(0.0, 1.0))
        .round()
        .clamp(0.0, 255.0) as u8
}
