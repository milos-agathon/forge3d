//! Renderer module utilities exposed to Python bindings.
//!
//! Hosts shared rendering helpers including terrain metadata and readback operations.

pub mod readback;

use crate::terrain::stats as terrain_stats;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Minimum height separation to prevent division by zero in normalization.
const MIN_HEIGHT_EPSILON: f32 = 1e-5;

/// Terrain metadata holding the height normalization range.
pub struct TerrainMeta {
    pub h_min: f32,
    pub h_max: f32,
}

impl Default for TerrainMeta {
    fn default() -> Self {
        Self {
            h_min: 0.0,
            h_max: 1.0,
        }
    }
}

impl TerrainMeta {
    /// Compute and store the height range from heightmap data.
    ///
    /// Uses percentile clamping (1st–99th) for robustness against outliers.
    pub fn compute_and_store_h_range(&mut self, heights: &[f32]) {
        let (h_min, h_max) = terrain_stats::min_max(heights, true);
        self.h_min = h_min;
        self.h_max = h_max.max(h_min + MIN_HEIGHT_EPSILON);
    }

    /// Override the height normalization range used for color & lighting.
    /// Raises `ValueError` if `min >= max`.
    pub fn set_height_range(&mut self, min: f32, max: f32) -> PyResult<()> {
        if !min.is_finite() || !max.is_finite() {
            return Err(PyValueError::new_err("min/max must be finite floats"));
        }
        if min >= max {
            return Err(PyValueError::new_err("min must be < max"));
        }
        self.h_min = min;
        self.h_max = max;
        Ok(())
    }
}
