use std::collections::HashMap;
use std::path::PathBuf;

use crate::gis::error::{GisError, GisResult};

#[cfg(feature = "extension-module")]
use pyo3::types::PyDictMethods;

pub const WARNING_MISSING_CRS: &str = "missing_crs";
pub const WARNING_MISSING_TRANSFORM: &str = "missing_transform";
pub const WARNING_NOT_GEOREFERENCED: &str = "not_georeferenced";
pub const WARNING_ROTATED_OR_SHEARED: &str = "rotated_or_sheared_transform";
pub const WARNING_PER_BAND_NODATA_MISMATCH: &str = "per_band_nodata_mismatch";
pub const WARNING_METADATA_UNAVAILABLE: &str = "metadata_unavailable";
pub const WARNING_ASSIGNMENT_NOT_REPROJECTION: &str = "assignment_not_reprojection";

#[derive(Debug, Clone, PartialEq)]
pub struct RasterWarning {
    pub code: String,
    pub message: String,
    pub field: Option<String>,
}

impl RasterWarning {
    pub fn new(
        code: &'static str,
        message: impl Into<String>,
        field: Option<&'static str>,
    ) -> Self {
        Self {
            code: code.to_string(),
            message: message.into(),
            field: field.map(str::to_string),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RasterDType {
    UInt8,
    Int16,
    UInt16,
    Int32,
    UInt32,
    Float32,
    Float64,
}

impl RasterDType {
    pub fn name(self) -> &'static str {
        match self {
            RasterDType::UInt8 => "uint8",
            RasterDType::Int16 => "int16",
            RasterDType::UInt16 => "uint16",
            RasterDType::Int32 => "int32",
            RasterDType::UInt32 => "uint32",
            RasterDType::Float32 => "float32",
            RasterDType::Float64 => "float64",
        }
    }

    pub fn is_float(self) -> bool {
        matches!(self, RasterDType::Float32 | RasterDType::Float64)
    }

    pub fn nodata_fits(self, value: f64) -> bool {
        if value.is_nan() {
            return self.is_float();
        }
        if !value.is_finite() {
            return false;
        }
        match self {
            RasterDType::UInt8 => value.fract() == 0.0 && (0.0..=u8::MAX as f64).contains(&value),
            RasterDType::Int16 => {
                value.fract() == 0.0 && (i16::MIN as f64..=i16::MAX as f64).contains(&value)
            }
            RasterDType::UInt16 => value.fract() == 0.0 && (0.0..=u16::MAX as f64).contains(&value),
            RasterDType::Int32 => {
                value.fract() == 0.0 && (i32::MIN as f64..=i32::MAX as f64).contains(&value)
            }
            RasterDType::UInt32 => value.fract() == 0.0 && (0.0..=u32::MAX as f64).contains(&value),
            RasterDType::Float32 => (-(f32::MAX as f64)..=f32::MAX as f64).contains(&value),
            RasterDType::Float64 => true,
        }
    }
}

#[cfg_attr(
    feature = "extension-module",
    pyo3::pyclass(module = "forge3d._forge3d", name = "AffineTransform")
)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AffineTransform {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub d: f64,
    pub e: f64,
    pub f: f64,
}

impl AffineTransform {
    pub fn new(coefficients: [f64; 6]) -> GisResult<Self> {
        if coefficients.iter().any(|value| !value.is_finite()) {
            return Err(GisError::InvalidTransform(
                "transform coefficients must be finite".to_string(),
            ));
        }
        let transform = Self {
            a: coefficients[0],
            b: coefficients[1],
            c: coefficients[2],
            d: coefficients[3],
            e: coefficients[4],
            f: coefficients[5],
        };
        let resolution = transform.resolution();
        if resolution.0 <= 0.0 || resolution.1 <= 0.0 {
            return Err(GisError::InvalidTransform(
                "transform must have positive pixel resolution".to_string(),
            ));
        }
        Ok(transform)
    }

    pub fn tuple(self) -> (f64, f64, f64, f64, f64, f64) {
        (self.a, self.b, self.c, self.d, self.e, self.f)
    }

    pub fn resolution(self) -> (f64, f64) {
        (
            (self.a.mul_add(self.a, self.d * self.d)).sqrt(),
            (self.b.mul_add(self.b, self.e * self.e)).sqrt(),
        )
    }

    pub fn bounds(self, width: u32, height: u32) -> RasterBounds {
        let width = width as f64;
        let height = height as f64;
        let corners = [
            self.apply(0.0, 0.0),
            self.apply(width, 0.0),
            self.apply(0.0, height),
            self.apply(width, height),
        ];
        let left = corners
            .iter()
            .map(|corner| corner.0)
            .fold(f64::INFINITY, f64::min);
        let right = corners
            .iter()
            .map(|corner| corner.0)
            .fold(f64::NEG_INFINITY, f64::max);
        let bottom = corners
            .iter()
            .map(|corner| corner.1)
            .fold(f64::INFINITY, f64::min);
        let top = corners
            .iter()
            .map(|corner| corner.1)
            .fold(f64::NEG_INFINITY, f64::max);
        RasterBounds {
            left,
            bottom,
            right,
            top,
        }
    }

    pub fn is_rotated_or_sheared(self) -> bool {
        self.b != 0.0 || self.d != 0.0
    }

    pub(crate) fn apply(self, col: f64, row: f64) -> (f64, f64) {
        (
            self.a.mul_add(col, self.b.mul_add(row, self.c)),
            self.d.mul_add(col, self.e.mul_add(row, self.f)),
        )
    }
}

#[cfg(feature = "extension-module")]
#[pyo3::pymethods]
impl AffineTransform {
    #[new]
    fn py_new(a: f64, b: f64, c: f64, d: f64, e: f64, f: f64) -> pyo3::PyResult<Self> {
        Self::new([a, b, c, d, e, f]).map_err(Into::into)
    }

    #[getter]
    fn coefficients(&self) -> (f64, f64, f64, f64, f64, f64) {
        self.tuple()
    }

    #[getter(resolution)]
    fn resolution_py(&self) -> (f64, f64) {
        self.resolution()
    }

    #[getter]
    fn rotated_or_sheared(&self) -> bool {
        self.is_rotated_or_sheared()
    }

    fn __repr__(&self) -> String {
        format!("AffineTransform{:?}", self.tuple())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RasterBounds {
    pub left: f64,
    pub bottom: f64,
    pub right: f64,
    pub top: f64,
}

impl RasterBounds {
    pub fn tuple(self) -> (f64, f64, f64, f64) {
        (self.left, self.bottom, self.right, self.top)
    }
}

#[cfg_attr(
    feature = "extension-module",
    pyo3::pyclass(module = "forge3d._forge3d", name = "RasterInfo")
)]
#[derive(Debug, Clone)]
pub struct RasterInfo {
    pub path: String,
    pub driver: String,
    pub width: u32,
    pub height: u32,
    pub band_count: u16,
    pub dtype_per_band: Vec<String>,
    pub crs_wkt: Option<String>,
    pub crs_authority: Option<HashMap<String, String>>,
    pub transform: Option<(f64, f64, f64, f64, f64, f64)>,
    pub bounds: Option<(f64, f64, f64, f64)>,
    pub resolution: Option<(f64, f64)>,
    pub nodata_per_band: Vec<Option<f64>>,
    pub block_size: Option<Vec<(u32, u32)>>,
    pub tiling: Option<String>,
    pub compression: Option<String>,
    pub is_georeferenced: bool,
    pub warnings: Vec<RasterWarning>,
}

impl RasterInfo {
    pub fn new(path: PathBuf, width: u32, height: u32, band_count: u16) -> Self {
        Self {
            path: path.to_string_lossy().to_string(),
            driver: "GTiff".to_string(),
            width,
            height,
            band_count,
            dtype_per_band: Vec::new(),
            crs_wkt: None,
            crs_authority: None,
            transform: None,
            bounds: None,
            resolution: None,
            nodata_per_band: vec![None; band_count as usize],
            block_size: None,
            tiling: None,
            compression: None,
            is_georeferenced: false,
            warnings: Vec::new(),
        }
    }
}

#[cfg(feature = "extension-module")]
#[pyo3::pymethods]
impl RasterInfo {
    #[getter]
    fn path(&self) -> String {
        self.path.clone()
    }

    #[getter]
    fn driver(&self) -> String {
        self.driver.clone()
    }

    #[getter]
    fn width(&self) -> u32 {
        self.width
    }

    #[getter]
    fn height(&self) -> u32 {
        self.height
    }

    #[getter]
    fn band_count(&self) -> u16 {
        self.band_count
    }

    #[getter]
    fn dtype_per_band(&self) -> Vec<String> {
        self.dtype_per_band.clone()
    }

    #[getter]
    fn crs_wkt(&self) -> Option<String> {
        self.crs_wkt.clone()
    }

    #[getter]
    fn crs_authority(&self) -> Option<HashMap<String, String>> {
        self.crs_authority.clone()
    }

    #[getter]
    fn transform(&self) -> Option<(f64, f64, f64, f64, f64, f64)> {
        self.transform
    }

    #[getter]
    fn bounds(&self) -> Option<(f64, f64, f64, f64)> {
        self.bounds
    }

    #[getter]
    fn resolution(&self) -> Option<(f64, f64)> {
        self.resolution
    }

    #[getter]
    fn nodata_per_band(&self) -> Vec<Option<f64>> {
        self.nodata_per_band.clone()
    }

    #[getter]
    fn block_size(&self) -> Option<Vec<(u32, u32)>> {
        self.block_size.clone()
    }

    #[getter]
    fn tiling(&self) -> Option<String> {
        self.tiling.clone()
    }

    #[getter]
    fn compression(&self) -> Option<String> {
        self.compression.clone()
    }

    #[getter]
    fn is_georeferenced(&self) -> bool {
        self.is_georeferenced
    }

    #[getter]
    fn warnings<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::PyObject> {
        warnings_to_py(py, &self.warnings)
    }

    fn as_dict<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::PyObject> {
        use pyo3::types::PyDict;
        use pyo3::IntoPy;

        let dict = PyDict::new_bound(py);
        dict.set_item("path", self.path.clone())?;
        dict.set_item("driver", self.driver.clone())?;
        dict.set_item("width", self.width)?;
        dict.set_item("height", self.height)?;
        dict.set_item("band_count", self.band_count)?;
        dict.set_item("dtype_per_band", self.dtype_per_band.clone())?;
        dict.set_item("crs_wkt", self.crs_wkt.clone())?;
        dict.set_item("crs_authority", self.crs_authority.clone())?;
        dict.set_item("transform", self.transform)?;
        dict.set_item("bounds", self.bounds)?;
        dict.set_item("resolution", self.resolution)?;
        dict.set_item("nodata_per_band", self.nodata_per_band.clone())?;
        dict.set_item("block_size", self.block_size.clone())?;
        dict.set_item("tiling", self.tiling.clone())?;
        dict.set_item("compression", self.compression.clone())?;
        dict.set_item("is_georeferenced", self.is_georeferenced)?;
        dict.set_item("warnings", warnings_to_py(py, &self.warnings)?)?;
        Ok(dict.into_py(py))
    }

    fn __repr__(&self) -> String {
        format!(
            "RasterInfo(path={:?}, driver={:?}, width={}, height={}, band_count={})",
            self.path, self.driver, self.width, self.height, self.band_count
        )
    }
}

#[cfg(feature = "extension-module")]
fn warnings_to_py(
    py: pyo3::Python<'_>,
    warnings: &[RasterWarning],
) -> pyo3::PyResult<pyo3::PyObject> {
    use pyo3::types::{PyDict, PyList};
    use pyo3::IntoPy;

    let mut items = Vec::with_capacity(warnings.len());
    for warning in warnings {
        let dict = PyDict::new_bound(py);
        dict.set_item("code", warning.code.clone())?;
        dict.set_item("message", warning.message.clone())?;
        dict.set_item("field", warning.field.clone())?;
        items.push(dict.into_py(py));
    }
    Ok(PyList::new_bound(py, items).into_py(py))
}
