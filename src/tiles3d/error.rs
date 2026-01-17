//! Error types for 3D Tiles module

use std::fmt;

/// Result type for 3D Tiles operations
pub type Tiles3dResult<T> = Result<T, Tiles3dError>;

/// Errors that can occur during 3D Tiles operations
#[derive(Debug)]
pub enum Tiles3dError {
    /// IO error reading tileset or tile files
    Io(std::io::Error),
    /// JSON parsing error
    Json(serde_json::Error),
    /// Invalid tileset structure
    InvalidTileset(String),
    /// Invalid b3dm payload
    InvalidB3dm(String),
    /// Invalid pnts payload
    InvalidPnts(String),
    /// Invalid glTF data
    InvalidGltf(String),
    /// Unsupported feature
    Unsupported(String),
    /// HTTP error for remote tilesets
    Http(String),
}

impl fmt::Display for Tiles3dError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "IO error: {}", e),
            Self::Json(e) => write!(f, "JSON error: {}", e),
            Self::InvalidTileset(msg) => write!(f, "Invalid tileset: {}", msg),
            Self::InvalidB3dm(msg) => write!(f, "Invalid b3dm: {}", msg),
            Self::InvalidPnts(msg) => write!(f, "Invalid pnts: {}", msg),
            Self::InvalidGltf(msg) => write!(f, "Invalid glTF: {}", msg),
            Self::Unsupported(msg) => write!(f, "Unsupported: {}", msg),
            Self::Http(msg) => write!(f, "HTTP error: {}", msg),
        }
    }
}

impl std::error::Error for Tiles3dError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Json(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Tiles3dError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<serde_json::Error> for Tiles3dError {
    fn from(e: serde_json::Error) -> Self {
        Self::Json(e)
    }
}
