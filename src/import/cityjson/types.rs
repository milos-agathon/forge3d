use serde_json::Value as JsonValue;
use std::collections::HashMap;

use crate::import::building_materials::BuildingMaterial;
use crate::import::osm_buildings::RoofType;

/// Error type for CityJSON parsing
#[derive(Debug, Clone)]
pub struct CityJsonError {
    pub message: String,
}

impl std::fmt::Display for CityJsonError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CityJSON error: {}", self.message)
    }
}

impl std::error::Error for CityJsonError {}

impl CityJsonError {
    pub fn new(msg: impl Into<String>) -> Self {
        Self {
            message: msg.into(),
        }
    }
}

pub type CityJsonResult<T> = Result<T, CityJsonError>;

/// A parsed CityJSON building with geometry and attributes
#[derive(Debug, Clone)]
pub struct BuildingGeom {
    /// Unique identifier
    pub id: String,
    /// Vertex positions as flat [x, y, z, x, y, z, ...] in transformed coordinates
    pub positions: Vec<f32>,
    /// Triangle indices into positions (each group of 3 = one triangle)
    pub indices: Vec<u32>,
    /// Normal vectors per vertex (optional)
    pub normals: Option<Vec<f32>>,
    /// Level of detail (1-3, 0 = unknown)
    pub lod: u8,
    /// Building height in meters (if available)
    pub height: Option<f32>,
    /// Ground elevation in meters (if available)
    pub ground_height: Option<f32>,
    /// Roof type inferred from attributes
    pub roof_type: RoofType,
    /// Material properties
    pub material: BuildingMaterial,
    /// Original attributes from CityJSON
    pub attributes: HashMap<String, JsonValue>,
}

impl BuildingGeom {
    /// Create a new empty building geometry
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            positions: Vec::new(),
            indices: Vec::new(),
            normals: None,
            lod: 0,
            height: None,
            ground_height: None,
            roof_type: RoofType::Flat,
            material: BuildingMaterial::default(),
            attributes: HashMap::new(),
        }
    }

    /// Get vertex count
    pub fn vertex_count(&self) -> usize {
        self.positions.len() / 3
    }

    /// Get triangle count
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// Check if geometry is empty
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty() || self.indices.is_empty()
    }
}

/// CityJSON file metadata
#[derive(Debug, Clone)]
pub struct CityJsonMeta {
    /// CityJSON version (e.g., "1.1")
    pub version: String,
    /// Coordinate reference system EPSG code
    pub crs_epsg: Option<u32>,
    /// Transform scale factors
    pub scale: [f64; 3],
    /// Transform translation offsets
    pub translate: [f64; 3],
    /// Geographic extent [minx, miny, minz, maxx, maxy, maxz]
    pub extent: Option<[f64; 6]>,
}

impl Default for CityJsonMeta {
    fn default() -> Self {
        Self {
            version: "1.1".to_string(),
            crs_epsg: None,
            scale: [1.0, 1.0, 1.0],
            translate: [0.0, 0.0, 0.0],
            extent: None,
        }
    }
}
