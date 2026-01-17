//! Error types for Point Cloud module

use std::fmt;

pub type PointCloudResult<T> = Result<T, PointCloudError>;

#[derive(Debug)]
pub enum PointCloudError {
    Io(std::io::Error),
    Json(serde_json::Error),
    InvalidCopc(String),
    InvalidEpt(String),
    InvalidLaz(String),
    Unsupported(String),
    Http(String),
}

impl fmt::Display for PointCloudError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "IO error: {}", e),
            Self::Json(e) => write!(f, "JSON error: {}", e),
            Self::InvalidCopc(msg) => write!(f, "Invalid COPC: {}", msg),
            Self::InvalidEpt(msg) => write!(f, "Invalid EPT: {}", msg),
            Self::InvalidLaz(msg) => write!(f, "Invalid LAZ: {}", msg),
            Self::Unsupported(msg) => write!(f, "Unsupported: {}", msg),
            Self::Http(msg) => write!(f, "HTTP error: {}", msg),
        }
    }
}

impl std::error::Error for PointCloudError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Json(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for PointCloudError {
    fn from(e: std::io::Error) -> Self { Self::Io(e) }
}

impl From<serde_json::Error> for PointCloudError {
    fn from(e: serde_json::Error) -> Self { Self::Json(e) }
}
