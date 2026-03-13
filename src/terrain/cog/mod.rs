//! P3: Cloud Optimized GeoTIFF (COG) streaming module.
//!
//! Provides HTTP range-based tile streaming from COG files without pre-tiling.
//! Implements the `HeightReader` trait for integration with existing terrain pipeline.

mod cache;
mod cog_reader;
mod error;
mod ifd_parser;
mod range_reader;

#[cfg(feature = "extension-module")]
pub mod py_bindings;

pub use cache::{CogCacheStats, CogTileCache};
pub use cog_reader::CogHeightReader;
pub use error::CogError;
pub use ifd_parser::{parse_cog_header, CogHeader, IfdEntry};
pub use range_reader::RangeReader;
