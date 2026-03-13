// src/import/mod.rs
// Import helpers for 3D data formats (P4: 3D Buildings Pipeline)

pub mod building_materials;
pub mod cityjson;
pub mod osm_buildings;

// Re-export key types for convenience
pub use building_materials::{material_from_name, material_from_tags, BuildingMaterial};
pub use cityjson::{parse_cityjson, BuildingGeom, CityJsonMeta};
pub use osm_buildings::{infer_roof_type, RoofType};
