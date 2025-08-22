//! Central error handling for forge3d renderer
//! 
//! Provides a unified RenderError enum with consistent categorization
//! and conversion to Python exceptions via PyO3.

use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;

/// Centralized error type for all renderer operations
#[derive(thiserror::Error, Debug)]
pub enum RenderError {
    #[error("Device error: {0}")]
    Device(String),
    
    #[error("Upload error: {0}")]
    Upload(String),
    
    #[error("Render error: {0}")]
    Render(String),
    
    #[error("Readback error: {0}")]
    Readback(String),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

impl RenderError {
    /// Convert RenderError to PyErr with categorized prefixes
    pub fn to_py_err(self) -> PyErr {
        let category = match &self {
            RenderError::Device(_) => "Device",
            RenderError::Upload(_) => "Upload", 
            RenderError::Render(_) => "Render",
            RenderError::Readback(_) => "Readback",
            RenderError::Io(_) => "IO",
        };
        
        PyRuntimeError::new_err(format!("[{}] {}", category, self))
    }
    
    /// Convenience constructors for common error types
    pub fn device<T: ToString>(msg: T) -> Self {
        RenderError::Device(msg.to_string())
    }
    
    pub fn upload<T: ToString>(msg: T) -> Self {
        RenderError::Upload(msg.to_string())
    }
    
    pub fn render<T: ToString>(msg: T) -> Self {
        RenderError::Render(msg.to_string())
    }
    
    pub fn readback<T: ToString>(msg: T) -> Self {
        RenderError::Readback(msg.to_string())
    }
    
    pub fn io<T: ToString>(msg: T) -> Self {
        RenderError::Io(std::io::Error::new(std::io::ErrorKind::Other, msg.to_string()))
    }
}

impl From<RenderError> for PyErr {
    fn from(err: RenderError) -> Self {
        err.to_py_err()
    }
}

/// Result type alias for renderer operations
pub type RenderResult<T> = Result<T, RenderError>;