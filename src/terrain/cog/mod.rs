//! P3: Cloud Optimized GeoTIFF (COG) streaming module.
//!
//! Provides HTTP range-based tile streaming from COG files without pre-tiling.
//! Implements the `HeightReader` trait for integration with existing terrain pipeline.

mod range_reader;
mod ifd_parser;
mod cog_reader;
mod cache;
mod error;

#[cfg(feature = "extension-module")]
pub mod py_bindings;

pub use range_reader::RangeReader;
pub use ifd_parser::{IfdEntry, parse_cog_header, CogHeader};
pub use cog_reader::CogHeightReader;
pub use cache::{CogTileCache, CogCacheStats};
pub use error::CogError;
