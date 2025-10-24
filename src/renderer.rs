// src/renderer.rs
// Renderer module utilities exposed to Python bindings
// Exists to host shared rendering helpers available to extension callers
// RELEVANT FILES: src/renderer/readback.rs, src/terrain_renderer.rs, src/lib.rs, python/forge3d/__init__.py
// T02-BEGIN:dem-stats
pub mod readback;

use crate::terrain_stats;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// High-level rendering utilities and compositors
pub mod terrain_drape;

// Terrain metadata structure to hold height range
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

// This would be included in the main Renderer struct
impl TerrainMeta {
    /// Called from `add_terrain` after validation.
    pub fn compute_and_store_h_range(&mut self, heights: &[f32]) {
        let (h_min, h_max) = terrain_stats::min_max(heights, true);
        self.h_min = h_min;
        self.h_max = h_max.max(h_min + 1e-5); // avoid div/0
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
// T02-END:dem-stats
