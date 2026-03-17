mod geometry;
mod parser;
mod types;

pub use parser::parse_cityjson;
pub use types::{BuildingGeom, CityJsonError, CityJsonMeta, CityJsonResult};

#[cfg(feature = "extension-module")]
mod bindings;
#[cfg(feature = "extension-module")]
pub use bindings::parse_cityjson_py;

#[cfg(test)]
mod tests;
