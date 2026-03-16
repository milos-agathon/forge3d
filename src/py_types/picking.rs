use super::super::*;

// Feature B: Picking system Python bindings
#[cfg(feature = "extension-module")]
#[pyclass(module = "forge3d._forge3d", name = "PickResult")]
pub struct PyPickResult {
    #[pyo3(get)]
    pub feature_id: u32,
    #[pyo3(get)]
    pub screen_x: u32,
    #[pyo3(get)]
    pub screen_y: u32,
    #[pyo3(get)]
    pub world_pos: Option<(f32, f32, f32)>,
    #[pyo3(get)]
    pub layer_name: Option<String>,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl PyPickResult {
    #[new]
    fn new(feature_id: u32, screen_x: u32, screen_y: u32) -> Self {
        Self {
            feature_id,
            screen_x,
            screen_y,
            world_pos: None,
            layer_name: None,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "PickResult(feature_id={}, screen=({}, {}), layer={:?})",
            self.feature_id, self.screen_x, self.screen_y, self.layer_name
        )
    }
}

#[cfg(feature = "extension-module")]
impl From<crate::picking::PickResult> for PyPickResult {
    fn from(r: crate::picking::PickResult) -> Self {
        Self {
            feature_id: r.feature_id,
            screen_x: r.screen_pos.0,
            screen_y: r.screen_pos.1,
            world_pos: r.world_pos.map(|p| (p[0], p[1], p[2])),
            layer_name: r.layer_name,
        }
    }
}

// Plan 2: Terrain query result Python bindings
#[cfg(feature = "extension-module")]
#[pyclass(module = "forge3d._forge3d", name = "TerrainQueryResult")]
pub struct PyTerrainQueryResult {
    #[pyo3(get)]
    pub elevation: f32,
    #[pyo3(get)]
    pub slope: f32,
    #[pyo3(get)]
    pub aspect: f32,
    #[pyo3(get)]
    pub world_pos: (f32, f32, f32),
    #[pyo3(get)]
    pub normal: (f32, f32, f32),
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl PyTerrainQueryResult {
    #[new]
    fn new(elevation: f32, slope: f32, aspect: f32) -> Self {
        Self {
            elevation,
            slope,
            aspect,
            world_pos: (0.0, 0.0, 0.0),
            normal: (0.0, 1.0, 0.0),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "TerrainQueryResult(elevation={:.2}, slope={:.1}°, aspect={:.1}°)",
            self.elevation, self.slope, self.aspect
        )
    }
}

#[cfg(feature = "extension-module")]
impl From<crate::picking::TerrainQueryResult> for PyTerrainQueryResult {
    fn from(r: crate::picking::TerrainQueryResult) -> Self {
        Self {
            elevation: r.elevation,
            slope: r.slope,
            aspect: r.aspect,
            world_pos: (r.world_pos[0], r.world_pos[1], r.world_pos[2]),
            normal: (r.normal[0], r.normal[1], r.normal[2]),
        }
    }
}

// Plan 3: Rich pick result with full attributes
#[cfg(feature = "extension-module")]
#[pyclass(module = "forge3d._forge3d", name = "RichPickResult")]
#[derive(Clone)]
pub struct PyRichPickResult {
    #[pyo3(get)]
    pub feature_id: u32,
    #[pyo3(get)]
    pub layer_name: String,
    #[pyo3(get)]
    pub world_pos: (f32, f32, f32),
    #[pyo3(get)]
    pub attributes: std::collections::HashMap<String, String>,
    #[pyo3(get)]
    pub hit_distance: f32,
    #[pyo3(get)]
    pub terrain_elevation: Option<f32>,
    #[pyo3(get)]
    pub terrain_slope: Option<f32>,
    #[pyo3(get)]
    pub terrain_aspect: Option<f32>,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl PyRichPickResult {
    #[new]
    fn new(feature_id: u32, layer_name: String) -> Self {
        Self {
            feature_id,
            layer_name,
            world_pos: (0.0, 0.0, 0.0),
            attributes: std::collections::HashMap::new(),
            hit_distance: 0.0,
            terrain_elevation: None,
            terrain_slope: None,
            terrain_aspect: None,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "RichPickResult(feature_id={}, layer='{}', pos={:?})",
            self.feature_id, self.layer_name, self.world_pos
        )
    }

    /// Get attribute by key
    fn get_attribute(&self, key: &str) -> Option<String> {
        self.attributes.get(key).cloned()
    }
}

#[cfg(feature = "extension-module")]
impl From<crate::picking::RichPickResult> for PyRichPickResult {
    fn from(r: crate::picking::RichPickResult) -> Self {
        let (terrain_elevation, terrain_slope, terrain_aspect) = if let Some(info) = r.terrain_info
        {
            (Some(info.elevation), Some(info.slope), Some(info.aspect))
        } else {
            (None, None, None)
        };

        Self {
            feature_id: r.feature_id,
            layer_name: r.layer_name,
            world_pos: (r.world_pos[0], r.world_pos[1], r.world_pos[2]),
            attributes: r.attributes,
            hit_distance: r.hit_distance,
            terrain_elevation,
            terrain_slope,
            terrain_aspect,
        }
    }
}

// Plan 3: Lasso state class (string-based for simplicity)
#[cfg(feature = "extension-module")]
#[pyclass(module = "forge3d._forge3d", name = "LassoState")]
#[derive(Clone)]
pub struct PyLassoState {
    #[pyo3(get)]
    pub state: String,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl PyLassoState {
    #[new]
    #[pyo3(signature = (state="inactive"))]
    fn new(state: &str) -> Self {
        Self {
            state: state.to_string(),
        }
    }

    fn __repr__(&self) -> String {
        format!("LassoState('{}')", self.state)
    }

    /// Check if inactive
    fn is_inactive(&self) -> bool {
        self.state == "inactive"
    }

    /// Check if drawing
    fn is_drawing(&self) -> bool {
        self.state == "drawing"
    }

    /// Check if complete
    fn is_complete(&self) -> bool {
        self.state == "complete"
    }

    #[staticmethod]
    fn inactive() -> Self {
        Self {
            state: "inactive".to_string(),
        }
    }

    #[staticmethod]
    fn drawing() -> Self {
        Self {
            state: "drawing".to_string(),
        }
    }

    #[staticmethod]
    fn complete() -> Self {
        Self {
            state: "complete".to_string(),
        }
    }
}

// Plan 3: Heightfield hit result
#[cfg(feature = "extension-module")]
#[pyclass(module = "forge3d._forge3d", name = "HeightfieldHit")]
#[derive(Clone)]
pub struct PyHeightfieldHit {
    #[pyo3(get)]
    pub position: (f32, f32, f32),
    #[pyo3(get)]
    pub t: f32,
    #[pyo3(get)]
    pub uv: (f32, f32),
    #[pyo3(get)]
    pub elevation: f32,
    #[pyo3(get)]
    pub normal: (f32, f32, f32),
    #[pyo3(get)]
    pub slope: f32,
    #[pyo3(get)]
    pub aspect: f32,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl PyHeightfieldHit {
    fn __repr__(&self) -> String {
        format!(
            "HeightfieldHit(elevation={:.2}, slope={:.1}°, aspect={:.1}°)",
            self.elevation, self.slope, self.aspect
        )
    }
}

#[cfg(feature = "extension-module")]
impl From<crate::picking::HeightfieldHit> for PyHeightfieldHit {
    fn from(h: crate::picking::HeightfieldHit) -> Self {
        Self {
            position: (h.position[0], h.position[1], h.position[2]),
            t: h.t,
            uv: (h.uv[0], h.uv[1]),
            elevation: h.elevation,
            normal: (h.normal[0], h.normal[1], h.normal[2]),
            slope: h.slope,
            aspect: h.aspect,
        }
    }
}
