// src/import/mod.rs
// Import helpers for 3D data formats (P4: 3D Buildings Pipeline)

pub mod osm_buildings;
pub mod building_materials;
pub mod cityjson;

// Re-export key types for convenience
pub use building_materials::{BuildingMaterial, material_from_tags, material_from_name};
pub use cityjson::{BuildingGeom, CityJsonMeta, parse_cityjson};
pub use osm_buildings::{RoofType, infer_roof_type};
