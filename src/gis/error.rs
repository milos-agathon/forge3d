use std::fmt;
use std::io;
use std::path::PathBuf;

#[derive(Debug)]
pub enum GisError {
    NotFound(PathBuf),
    Io(String),
    UnsupportedDriver(String),
    InvalidRaster(String),
    InvalidCrs(String),
    AlreadyExists(PathBuf),
    InvalidShape(String),
    UnsupportedDType(String),
    InvalidTransform(String),
    MissingTransform(String),
    InvalidNodata(String),
    InvalidArgument(String),
    InvalidGeometry(String),
    InvalidBounds(String),
    MissingCrs(String),
    CrsAlreadyExists(String),
    CrsMismatch(String),
    ResamplingRequired(String),
    UnsupportedResamplingMethod(String),
    TransformFailed(String),
    /// A raster reprojection failed to transform one or more pixels. Carries a
    /// structured, stable payload (count, first pixel, source/destination CRS,
    /// policy) so Python callers read attributes instead of parsing text.
    ReprojectTransformFailed {
        count: usize,
        first_pixel: (usize, usize),
        src_crs: String,
        dst_crs: String,
        policy: String,
    },
    BackendUnavailable(String),
    ShapeMismatch(String),
    UnsupportedCreationOption(String),
    WriteFailed(String),
    PostWriteValidationFailed(String),
}

impl GisError {
    pub fn code(&self) -> &'static str {
        match self {
            GisError::NotFound(_) => "NotFound",
            GisError::Io(_) => "Io",
            GisError::UnsupportedDriver(_) => "UnsupportedDriver",
            GisError::InvalidRaster(_) => "InvalidRaster",
            GisError::InvalidCrs(_) => "InvalidCrs",
            GisError::AlreadyExists(_) => "AlreadyExists",
            GisError::InvalidShape(_) => "InvalidShape",
            GisError::UnsupportedDType(_) => "UnsupportedDType",
            GisError::InvalidTransform(_) => "InvalidTransform",
            GisError::MissingTransform(_) => "MissingTransform",
            GisError::InvalidNodata(_) => "InvalidNodata",
            GisError::InvalidArgument(_) => "InvalidArgument",
            GisError::InvalidGeometry(_) => "InvalidGeometry",
            GisError::InvalidBounds(_) => "InvalidBounds",
            GisError::MissingCrs(_) => "MissingCrs",
            GisError::CrsAlreadyExists(_) => "CrsAlreadyExists",
            GisError::CrsMismatch(_) => "CrsMismatch",
            GisError::ResamplingRequired(_) => "resampling_required",
            GisError::UnsupportedResamplingMethod(_) => "unsupported_resampling_method",
            GisError::TransformFailed(_) => "TransformFailed",
            GisError::ReprojectTransformFailed { .. } => "TransformFailed",
            GisError::BackendUnavailable(_) => "BackendUnavailable",
            GisError::ShapeMismatch(_) => "ShapeMismatch",
            GisError::UnsupportedCreationOption(_) => "UnsupportedCreationOption",
            GisError::WriteFailed(_) => "WriteFailed",
            GisError::PostWriteValidationFailed(_) => "PostWriteValidationFailed",
        }
    }

    pub fn message(&self) -> String {
        match self {
            GisError::NotFound(path) => format!("path not found: {}", path.display()),
            GisError::Io(message)
            | GisError::UnsupportedDriver(message)
            | GisError::InvalidRaster(message)
            | GisError::InvalidCrs(message)
            | GisError::InvalidShape(message)
            | GisError::UnsupportedDType(message)
            | GisError::InvalidTransform(message)
            | GisError::MissingTransform(message)
            | GisError::InvalidNodata(message)
            | GisError::InvalidArgument(message)
            | GisError::InvalidGeometry(message)
            | GisError::InvalidBounds(message)
            | GisError::MissingCrs(message)
            | GisError::CrsAlreadyExists(message)
            | GisError::CrsMismatch(message)
            | GisError::ResamplingRequired(message)
            | GisError::UnsupportedResamplingMethod(message)
            | GisError::TransformFailed(message)
            | GisError::BackendUnavailable(message)
            | GisError::ShapeMismatch(message)
            | GisError::UnsupportedCreationOption(message)
            | GisError::WriteFailed(message)
            | GisError::PostWriteValidationFailed(message) => message.clone(),
            GisError::AlreadyExists(path) => {
                format!("output path already exists: {}", path.display())
            }
            GisError::ReprojectTransformFailed {
                count,
                first_pixel: (row, col),
                src_crs,
                dst_crs,
                policy: _,
            } => format!(
                "transform_failed: {count} pixel(s) failed to transform from {src_crs} to \
                 {dst_crs} (first at row {row}, col {col}); pass on_transform_error='nodata' \
                 to fill them with nodata instead"
            ),
        }
    }
}

impl fmt::Display for GisError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.code(), self.message())
    }
}

impl std::error::Error for GisError {}

impl From<io::Error> for GisError {
    fn from(value: io::Error) -> Self {
        GisError::Io(value.to_string())
    }
}

impl From<tiff::TiffError> for GisError {
    fn from(value: tiff::TiffError) -> Self {
        GisError::InvalidRaster(value.to_string())
    }
}

#[cfg(feature = "extension-module")]
pyo3::create_exception!(
    _forge3d,
    TransformFailed,
    pyo3::exceptions::PyRuntimeError,
    "Raised when raster reprojection cannot transform one or more coordinates. \
     Exposes stable .count (int), .first_pixel ((row, col) tuple), .src_crs (str), \
     .dst_crs (str), and .policy (str) attributes so callers never parse text."
);

#[cfg(feature = "extension-module")]
impl From<GisError> for pyo3::PyErr {
    fn from(value: GisError) -> Self {
        use pyo3::exceptions::{
            PyFileExistsError, PyFileNotFoundError, PyOSError, PyRuntimeError, PyTypeError,
            PyValueError,
        };

        let message = value.to_string();
        // Structured reprojection failure: raise the dedicated TransformFailed
        // exception with attributes attached, not a bare RuntimeError.
        if let GisError::ReprojectTransformFailed {
            count,
            first_pixel,
            src_crs,
            dst_crs,
            policy,
        } = &value
        {
            let (count, first_pixel, src_crs, dst_crs, policy) = (
                *count,
                *first_pixel,
                src_crs.clone(),
                dst_crs.clone(),
                policy.clone(),
            );
            return pyo3::Python::with_gil(|py| {
                use pyo3::types::PyAnyMethods;
                let err = TransformFailed::new_err(message);
                let obj = err.value_bound(py);
                let _ = obj.setattr("count", count);
                let _ = obj.setattr("first_pixel", first_pixel);
                let _ = obj.setattr("src_crs", src_crs);
                let _ = obj.setattr("dst_crs", dst_crs);
                let _ = obj.setattr("policy", policy);
                err
            });
        }
        match value {
            GisError::NotFound(_) => PyFileNotFoundError::new_err(message),
            GisError::AlreadyExists(_) => PyFileExistsError::new_err(message),
            GisError::Io(_) => PyOSError::new_err(message),
            GisError::UnsupportedDType(_) => PyTypeError::new_err(message),
            // ReprojectTransformFailed is handled by the early return above;
            // this arm only satisfies exhaustiveness.
            GisError::ReprojectTransformFailed { .. }
            | GisError::InvalidRaster(_)
            | GisError::WriteFailed(_)
            | GisError::PostWriteValidationFailed(_)
            | GisError::BackendUnavailable(_)
            | GisError::TransformFailed(_) => PyRuntimeError::new_err(message),
            GisError::UnsupportedDriver(_)
            | GisError::InvalidCrs(_)
            | GisError::InvalidShape(_)
            | GisError::InvalidTransform(_)
            | GisError::MissingTransform(_)
            | GisError::InvalidNodata(_)
            | GisError::InvalidArgument(_)
            | GisError::InvalidGeometry(_)
            | GisError::InvalidBounds(_)
            | GisError::MissingCrs(_)
            | GisError::CrsAlreadyExists(_)
            | GisError::CrsMismatch(_)
            | GisError::ResamplingRequired(_)
            | GisError::UnsupportedResamplingMethod(_)
            | GisError::ShapeMismatch(_)
            | GisError::UnsupportedCreationOption(_) => PyValueError::new_err(message),
        }
    }
}

pub type GisResult<T> = Result<T, GisError>;
