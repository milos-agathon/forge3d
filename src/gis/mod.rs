pub mod affine;
pub mod crs;
pub mod error;
pub mod geometry;
#[cfg(feature = "extension-module")]
pub(crate) mod py_json;
pub mod raster_info;
pub mod raster_write;
pub mod types;
pub mod vector;
pub mod warp;

pub use affine::{window_from_bounds, PixelWindow, RasterWindow};
pub use crs::{web_mercator_bounds, CrsInspection, CrsTransform};
pub use error::{GisError, GisResult};
pub use geometry::{
    geometry_centroid, geometry_measure, interpolate_line, repair_geometry, representative_point,
    validate_geometry,
};
#[cfg(feature = "extension-module")]
pub use geometry::{
    geometry_centroid_py, geometry_measure_py, interpolate_line_py, repair_geometry_py,
    representative_point_py, validate_geometry_py,
};
pub use raster_info::{read_raster, read_raster_info};
pub use raster_write::{
    write_raster, CreationOptions, CrsSpec, RasterArray, RasterData, WriteRasterOptions,
};
pub use types::{AffineTransform, RasterBounds, RasterDType, RasterInfo, RasterWarning};
#[cfg(feature = "extension-module")]
pub use vector::{
    feature_count_py, geometry_type_py, read_vector_py, reproject_vector_py, vector_bounds_py,
    vector_crs_py, vector_schema_py,
};
pub use vector::{
    read_vector, read_vector_info, reproject_vector, VectorInfo, VectorReadOptions,
    VectorReadResult, VectorReprojectInput, VectorReprojectResult,
};
pub use warp::{align_raster_to, assign_crs, reproject_raster, resample_array, resample_raster};

#[cfg(feature = "extension-module")]
use std::collections::HashMap;
#[cfg(feature = "extension-module")]
use std::path::Path;

#[cfg(feature = "extension-module")]
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArrayMethods};
#[cfg(feature = "extension-module")]
use pyo3::prelude::*;
#[cfg(feature = "extension-module")]
use pyo3::types::{PyAny, PyDict, PyDictMethods, PyFloat, PyList, PyTuple};

#[cfg(feature = "extension-module")]
#[pyfunction(name = "read_raster_info")]
pub fn read_raster_info_py(path: String) -> PyResult<RasterInfo> {
    read_raster_info(path).map_err(Into::into)
}

#[cfg(feature = "extension-module")]
#[pyfunction(name = "read_raster", signature = (path, bands = None, window = None, masked = false))]
pub fn read_raster_py(
    py: Python<'_>,
    path: String,
    bands: Option<&Bound<'_, PyAny>>,
    window: Option<&Bound<'_, PyAny>>,
    masked: bool,
) -> PyResult<PyObject> {
    let bands = parse_read_bands_py(bands)?;
    let window = parse_read_window_py(window)?;
    let result = read_raster(path, bands, window, masked)?;
    read_result_to_py(py, &result)
}

#[cfg(feature = "extension-module")]
#[pyfunction(
    name = "write_raster",
    signature = (
        path,
        array,
        *,
        crs = None,
        transform = None,
        nodata = None,
        driver = "GTiff",
        overwrite = false,
        creation_options = None,
        like_path = None,
        like_info = None
    )
)]
#[allow(clippy::too_many_arguments)]
pub fn write_raster_py(
    path: String,
    array: &Bound<'_, PyAny>,
    crs: Option<&Bound<'_, PyAny>>,
    transform: Option<(f64, f64, f64, f64, f64, f64)>,
    nodata: Option<&Bound<'_, PyAny>>,
    driver: &str,
    overwrite: bool,
    creation_options: Option<&Bound<'_, PyAny>>,
    like_path: Option<String>,
    like_info: Option<&Bound<'_, PyAny>>,
) -> PyResult<RasterInfo> {
    if like_path.is_some() && like_info.is_some() {
        return Err(GisError::InvalidArgument(
            "exactly one of like_path or like_info may be supplied".to_string(),
        )
        .into());
    }

    let raster_array = extract_raster_array(array)?;
    let creation_options_explicit = creation_options.is_some();
    let creation_options = CreationOptions::from_map(&extract_creation_options(creation_options)?)?;
    let crs = extract_crs(crs)?;
    let transform = transform
        .map(|values| {
            AffineTransform::new([values.0, values.1, values.2, values.3, values.4, values.5])
        })
        .transpose()?;
    let nodata = extract_nodata(nodata, raster_array.bands)?;
    let like_info = if let Some(path) = like_path {
        Some(read_raster_info(path)?)
    } else if let Some(info) = like_info {
        Some(info.extract::<PyRef<'_, RasterInfo>>()?.clone())
    } else {
        None
    };

    let options = WriteRasterOptions {
        crs,
        transform,
        nodata,
        driver: driver.to_string(),
        overwrite,
        creation_options,
        creation_options_explicit,
        like_info,
    };
    write_raster(path, raster_array, options).map_err(Into::into)
}

#[cfg(feature = "extension-module")]
#[pyfunction(name = "inspect_crs")]
pub fn inspect_crs_py(source: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    if let Ok(info) = source.extract::<PyRef<'_, RasterInfo>>() {
        return crs_inspection_to_py(
            source.py(),
            &crate::gis::crs::inspect_info_crs(&info, "raster")?,
        );
    }
    if let Ok(value) = source.extract::<String>() {
        if Path::new(&value).exists() {
            let info = read_raster_info(&value)?;
            return crs_inspection_to_py(
                source.py(),
                &crate::gis::crs::inspect_info_crs(&info, "path")?,
            );
        }
        if looks_like_dataset_path(&value) {
            return Err(GisError::NotFound(value.into()).into());
        }
        let spec = crate::gis::crs::parse_crs_string(value)?;
        return crs_inspection_to_py(
            source.py(),
            &crate::gis::crs::inspect_crs_spec(&spec, "crs"),
        );
    }
    let spec = extract_required_crs(Some(source))?;
    crs_inspection_to_py(
        source.py(),
        &crate::gis::crs::inspect_crs_spec(&spec, "crs"),
    )
}

#[cfg(feature = "extension-module")]
#[pyfunction(name = "parse_crs")]
pub fn parse_crs_py(value: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    let spec = extract_required_crs(Some(value))?;
    crs_inspection_to_py(value.py(), &crate::gis::crs::inspect_crs_spec(&spec, "crs"))
}

#[cfg(feature = "extension-module")]
#[pyfunction(name = "raster_crs")]
pub fn raster_crs_py(source: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    let (info, source_kind) = extract_raster_info_source(source)?;
    crs_inspection_to_py(
        source.py(),
        &crate::gis::crs::inspect_info_crs(&info, source_kind)?,
    )
}

#[cfg(feature = "extension-module")]
#[pyfunction(name = "create_crs_transformer", signature = (src_crs, dst_crs, *, always_xy = true))]
pub fn create_crs_transformer_py(
    src_crs: &Bound<'_, PyAny>,
    dst_crs: &Bound<'_, PyAny>,
    always_xy: bool,
) -> PyResult<CrsTransform> {
    CrsTransform::new(
        extract_required_crs(Some(src_crs))?,
        extract_required_crs(Some(dst_crs))?,
        always_xy,
    )
    .map_err(Into::into)
}

#[cfg(feature = "extension-module")]
#[pyfunction(name = "transform_bounds", signature = (src_crs, dst_crs, bounds, *, densify = None))]
pub fn transform_bounds_py(
    src_crs: &Bound<'_, PyAny>,
    dst_crs: &Bound<'_, PyAny>,
    bounds: (f64, f64, f64, f64),
    densify: Option<u32>,
) -> PyResult<(f64, f64, f64, f64)> {
    if densify.unwrap_or(0) > 0 {
        return Err(GisError::InvalidArgument(
            "unsupported_option: transform_bounds densify is not supported by the built-in backend"
                .to_string(),
        )
        .into());
    }
    let bounds = crate::gis::affine::validate_bounds_tuple(bounds, false)?;
    let src_crs = extract_required_crs(Some(src_crs))?;
    let dst_crs = extract_required_crs(Some(dst_crs))?;
    crate::gis::crs::transform_bounds(bounds, &src_crs, &dst_crs)
        .map(|bounds| bounds.tuple())
        .map_err(Into::into)
}

#[cfg(feature = "extension-module")]
#[pyfunction(name = "bounds")]
pub fn bounds_py(source: &Bound<'_, PyAny>) -> PyResult<(f64, f64, f64, f64)> {
    let (info, _) = extract_raster_info_source(source)?;
    crate::gis::affine::raster_bounds(&info)
        .map(|bounds| bounds.tuple())
        .map_err(Into::into)
}

#[cfg(feature = "extension-module")]
#[pyfunction(name = "raster_bounds")]
pub fn raster_bounds_py(source: &Bound<'_, PyAny>) -> PyResult<(f64, f64, f64, f64)> {
    bounds_py(source)
}

#[cfg(feature = "extension-module")]
#[pyfunction(name = "raster_transform")]
pub fn raster_transform_py(source: &Bound<'_, PyAny>) -> PyResult<(f64, f64, f64, f64, f64, f64)> {
    let (info, _) = extract_raster_info_source(source)?;
    crate::gis::affine::raster_transform(&info)
        .map(|transform| transform.tuple())
        .map_err(Into::into)
}

#[cfg(feature = "extension-module")]
#[pyfunction(name = "raster_resolution")]
pub fn raster_resolution_py(source: &Bound<'_, PyAny>) -> PyResult<(f64, f64)> {
    let (info, _) = extract_raster_info_source(source)?;
    crate::gis::affine::raster_resolution(&info).map_err(Into::into)
}

#[cfg(feature = "extension-module")]
#[pyfunction(name = "transform_from_origin")]
pub fn transform_from_origin_py(
    west: f64,
    north: f64,
    xsize: f64,
    ysize: f64,
) -> PyResult<(f64, f64, f64, f64, f64, f64)> {
    crate::gis::affine::transform_from_origin(west, north, xsize, ysize)
        .map(|transform| transform.tuple())
        .map_err(Into::into)
}

#[cfg(feature = "extension-module")]
#[pyfunction(name = "transform_from_bounds")]
pub fn transform_from_bounds_py(
    bounds: (f64, f64, f64, f64),
    width: u32,
    height: u32,
) -> PyResult<(f64, f64, f64, f64, f64, f64)> {
    let bounds = crate::gis::affine::validate_bounds_tuple(bounds, false)?;
    crate::gis::affine::transform_from_bounds(bounds, width, height)
        .map(|transform| transform.tuple())
        .map_err(Into::into)
}

#[cfg(feature = "extension-module")]
#[pyfunction(name = "array_bounds")]
pub fn array_bounds_py(
    height: u32,
    width: u32,
    transform: &Bound<'_, PyAny>,
) -> PyResult<(f64, f64, f64, f64)> {
    crate::gis::affine::array_bounds(height, width, extract_affine_transform(transform)?)
        .map(|bounds| bounds.tuple())
        .map_err(Into::into)
}

#[cfg(feature = "extension-module")]
#[pyfunction(name = "validate_transform", signature = (transform, *, require_north_up = false))]
pub fn validate_transform_py(
    py: Python<'_>,
    transform: &Bound<'_, PyAny>,
    require_north_up: bool,
) -> PyResult<PyObject> {
    use pyo3::IntoPy;

    let transform = extract_affine_transform(transform)?;
    if require_north_up && transform.is_rotated_or_sheared() {
        return Err(GisError::InvalidTransform(
            "rotated_or_sheared_transform: north-up transform required".to_string(),
        )
        .into());
    }
    let dict = PyDict::new_bound(py);
    dict.set_item("valid", true)?;
    dict.set_item("rotated_or_sheared", transform.is_rotated_or_sheared())?;
    dict.set_item("resolution", transform.resolution())?;
    dict.set_item(
        "warnings",
        warnings_to_py(py, &transform_warnings(transform))?,
    )?;
    Ok(dict.into_py(py))
}

#[cfg(feature = "extension-module")]
#[pyfunction(name = "pixel_convention")]
pub fn pixel_convention_py(py: Python<'_>, transform: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    use pyo3::IntoPy;

    let _ = extract_affine_transform(transform)?;
    let dict = PyDict::new_bound(py);
    dict.set_item("default_offset", "center")?;
    dict.set_item("corner_offset", "ul")?;
    dict.set_item(
        "warnings",
        warnings_to_py(
            py,
            &[RasterWarning::new(
                "pixel_convention_explicit",
                "xy uses pixel centers by default; affine coefficients map pixel corners",
                Some("transform"),
            )],
        )?,
    )?;
    Ok(dict.into_py(py))
}

#[cfg(feature = "extension-module")]
#[pyfunction(name = "xy")]
pub fn xy_py(transform: &Bound<'_, PyAny>, row: i64, col: i64) -> PyResult<(f64, f64)> {
    crate::gis::affine::xy(extract_affine_transform(transform)?, row, col).map_err(Into::into)
}

#[cfg(feature = "extension-module")]
#[pyfunction(name = "rowcol")]
pub fn rowcol_py(transform: &Bound<'_, PyAny>, x: f64, y: f64) -> PyResult<(i64, i64)> {
    crate::gis::affine::rowcol(extract_affine_transform(transform)?, x, y).map_err(Into::into)
}

#[cfg(feature = "extension-module")]
#[pyfunction(name = "index")]
pub fn index_py(transform: &Bound<'_, PyAny>, x: f64, y: f64) -> PyResult<(i64, i64)> {
    rowcol_py(transform, x, y)
}

#[cfg(feature = "extension-module")]
#[pyfunction(name = "assign_crs", signature = (source_or_path, crs, *, overwrite = false))]
pub fn assign_crs_py(
    source_or_path: &Bound<'_, PyAny>,
    crs: &Bound<'_, PyAny>,
    overwrite: bool,
) -> PyResult<RasterInfo> {
    if let Ok(info) = source_or_path.extract::<PyRef<'_, RasterInfo>>() {
        return assign_crs_to_info(&info, extract_required_crs(Some(crs))?, overwrite);
    }
    let path = extract_path_or_info_path(source_or_path)?;
    let crs = extract_required_crs(Some(crs))?;
    assign_crs(path, crs, overwrite).map_err(Into::into)
}

#[cfg(feature = "extension-module")]
#[pyfunction(name = "window_from_bounds", signature = (info_or_path, bounds, *, boundless = false))]
pub fn window_from_bounds_py(
    info_or_path: &Bound<'_, PyAny>,
    bounds: &Bound<'_, PyAny>,
    boundless: bool,
) -> PyResult<PyObject> {
    let py = bounds.py();
    let (info, _) = extract_raster_info_source(info_or_path)?;
    let (bounds, bounds_crs) = extract_bounds_arg(bounds)?;
    if let Some(bounds_crs) = bounds_crs {
        let raster_crs = crate::gis::crs::require_crs(&info)?;
        if !crate::gis::crs::crs_equal(&raster_crs, &bounds_crs) {
            return Err(GisError::CrsMismatch(
                "crs_mismatch: bounds CRS does not match raster CRS".to_string(),
            )
            .into());
        }
    }
    let window = crate::gis::affine::window_from_bounds(&info, bounds, boundless)?;
    raster_window_to_py(py, &window)
}

#[cfg(feature = "extension-module")]
#[pyfunction(name = "window_transform")]
pub fn window_transform_py(
    info_or_path: &Bound<'_, PyAny>,
    window: (i64, i64, u32, u32),
) -> PyResult<(f64, f64, f64, f64, f64, f64)> {
    let (info, _) = extract_raster_info_source(info_or_path)?;
    let window = PixelWindow {
        col_off: window.0,
        row_off: window.1,
        width: window.2,
        height: window.3,
    };
    crate::gis::affine::window_transform(&info, window)
        .map(|transform| transform.tuple())
        .map_err(Into::into)
}

#[cfg(feature = "extension-module")]
#[pyfunction(name = "apply_nodata", signature = (array, nodata, *, mask = None))]
pub fn apply_nodata_py(
    py: Python<'_>,
    array: &Bound<'_, PyAny>,
    nodata: &Bound<'_, PyAny>,
    mask: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyObject> {
    use pyo3::IntoPy;

    let raster_array = extract_raster_array(array)?;
    let nodata = extract_nodata(Some(nodata), raster_array.bands)?;
    let explicit_mask = mask.map(|mask| extract_bool_mask(mask)).transpose()?;
    if let Some(mask) = &explicit_mask {
        validate_mask_len(mask.len(), &raster_array)?;
    }
    let mask = raster_info::valid_mask(&raster_array, &nodata, explicit_mask.as_deref());
    let valid_count = mask.iter().filter(|&&valid| valid).count();
    let warnings = if valid_count == 0 {
        vec![RasterWarning::new(
            "empty_raster",
            "nodata and masks leave no valid pixels",
            None,
        )]
    } else {
        Vec::new()
    };
    let dict = PyDict::new_bound(py);
    dict.set_item("array", raster_array_to_py(py, &raster_array)?)?;
    dict.set_item("mask", bool_array_to_py(py, mask, &raster_array)?)?;
    dict.set_item("mask_polarity", "true_valid")?;
    dict.set_item("valid_count", valid_count)?;
    dict.set_item("nodata_per_band", nodata)?;
    dict.set_item("warnings", warnings_to_py(py, &warnings)?)?;
    Ok(dict.into_py(py))
}

#[cfg(feature = "extension-module")]
#[pyfunction(name = "read_raster_mask", signature = (path, band = None))]
pub fn read_raster_mask_py(
    py: Python<'_>,
    path: String,
    band: Option<usize>,
) -> PyResult<PyObject> {
    use pyo3::IntoPy;

    let loaded = raster_info::read_raster_data(path)?;
    let mut mask = raster_info::valid_mask(&loaded.array, &loaded.info.nodata_per_band, None);
    if let Some(band) = band {
        if band == 0 || band > loaded.array.bands {
            return Err(GisError::InvalidArgument(
                "band is 1-based and must be within raster band count".to_string(),
            )
            .into());
        }
        let band_index = band - 1;
        let pixels = loaded.array.width * loaded.array.height;
        mask = mask[band_index * pixels..(band_index + 1) * pixels].to_vec();
    }
    let shape = if band.is_some() {
        (1, loaded.array.height, loaded.array.width)
    } else {
        (loaded.array.bands, loaded.array.height, loaded.array.width)
    };
    let dict = PyDict::new_bound(py);
    dict.set_item("mask", bool_array_to_py_shape(py, mask, shape)?)?;
    dict.set_item("mask_polarity", "true_valid")?;
    dict.set_item(
        "mask_flags",
        if loaded.info.nodata_per_band.iter().any(Option::is_some) {
            vec!["nodata"]
        } else {
            vec!["all_valid"]
        },
    )?;
    dict.set_item("nodata_per_band", loaded.info.nodata_per_band)?;
    dict.set_item(
        "warnings",
        warnings_to_py(
            py,
            &[RasterWarning::new(
                "mask_polarity_explicit",
                "mask values are true for valid pixels",
                Some("mask"),
            )],
        )?,
    )?;
    Ok(dict.into_py(py))
}

#[cfg(feature = "extension-module")]
#[pyfunction(name = "resample_raster", signature = (source, shape_or_resolution, *, method = None))]
pub fn resample_raster_py(
    source: &Bound<'_, PyAny>,
    shape_or_resolution: &Bound<'_, PyAny>,
    method: Option<&str>,
) -> PyResult<PyObject> {
    let target = extract_resample_target(shape_or_resolution)?;
    let method = warp::ResamplingMethod::parse(method)?;
    let result = if let Ok(array) = extract_raster_array(source) {
        resample_array(array, target, method)?
    } else {
        let source = extract_raster_source_arg(source)?;
        resample_raster(source.path, target, method)?
    };
    raster_operation_to_py(source.py(), &result)
}

#[cfg(feature = "extension-module")]
#[pyfunction(name = "assert_grid_compatible", signature = (left, right, *, compare_nodata = true))]
pub fn assert_grid_compatible_py(
    py: Python<'_>,
    left: &Bound<'_, PyAny>,
    right: &Bound<'_, PyAny>,
    compare_nodata: bool,
) -> PyResult<PyObject> {
    use pyo3::IntoPy;

    let (left, _) = extract_raster_info_source(left)?;
    let (right, _) = extract_raster_info_source(right)?;
    let diagnostics = warp::assert_grid_compatible(&left, &right, compare_nodata);
    let dict = PyDict::new_bound(py);
    dict.set_item("compatible", diagnostics.is_empty())?;
    dict.set_item("diagnostics", warnings_to_py(py, &diagnostics)?)?;
    Ok(dict.into_py(py))
}

#[cfg(feature = "extension-module")]
#[pyfunction(name = "align_raster_grid", signature = (source, target_info, *, resampling = None))]
pub fn align_raster_grid_py(
    source: &Bound<'_, PyAny>,
    target_info: &Bound<'_, PyAny>,
    resampling: Option<&str>,
) -> PyResult<PyObject> {
    align_raster_to_py(source, target_info, resampling)
}

#[cfg(feature = "extension-module")]
#[pyfunction(name = "align_raster_to", signature = (source, target_info, *, resampling = None))]
pub fn align_raster_to_py(
    source: &Bound<'_, PyAny>,
    target_info: &Bound<'_, PyAny>,
    resampling: Option<&str>,
) -> PyResult<PyObject> {
    let source_arg = extract_raster_source_arg(source)?;
    let (target, _) = extract_raster_info_source(target_info)?;
    let method = warp::ResamplingMethod::parse(resampling)?;
    if source_arg.categorical && method != warp::ResamplingMethod::Nearest {
        return Err(GisError::InvalidArgument(
            "categorical_resampling_requires_nearest: categorical rasters require nearest resampling"
                .to_string(),
        )
        .into());
    }
    let result = align_raster_to(source_arg.path, &target, method)?;
    raster_operation_to_py(source.py(), &result)
}

#[cfg(feature = "extension-module")]
#[pyfunction(name = "reproject_raster", signature = (source, dst_crs, *, resampling = None))]
pub fn reproject_raster_py(
    source: &Bound<'_, PyAny>,
    dst_crs: &Bound<'_, PyAny>,
    resampling: Option<&str>,
) -> PyResult<PyObject> {
    let path = extract_path_or_info_path(source)?;
    let dst_crs = extract_required_crs(Some(dst_crs))?;
    let method = warp::ResamplingMethod::parse(resampling)?;
    let result = reproject_raster(path, dst_crs, method)?;
    raster_operation_to_py(source.py(), &result)
}

#[cfg(feature = "extension-module")]
#[pyfunction(name = "web_mercator_bounds")]
pub fn web_mercator_bounds_py(
    bounds: (f64, f64, f64, f64),
    src_crs: &Bound<'_, PyAny>,
) -> PyResult<(f64, f64, f64, f64)> {
    let bounds = crate::gis::affine::validate_bounds_tuple(bounds, true)?;
    let src_crs = extract_required_crs(Some(src_crs))?;
    crate::gis::crs::web_mercator_bounds(bounds, &src_crs)
        .map(|bounds| bounds.tuple())
        .map_err(Into::into)
}

#[cfg(feature = "extension-module")]
#[pyfunction(name = "calculate_default_transform", signature = (src_info, dst_crs, *, resolution = None))]
pub fn calculate_default_transform_py(
    py: Python<'_>,
    src_info: &Bound<'_, PyAny>,
    dst_crs: &Bound<'_, PyAny>,
    resolution: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyObject> {
    use pyo3::IntoPy;

    let (src_info, _) = extract_raster_info_source(src_info)?;
    let src_crs = crate::gis::crs::require_crs(&src_info)?;
    let dst_crs = extract_required_crs(Some(dst_crs))?;
    let source_bounds = crate::gis::affine::raster_bounds(&src_info)?;
    let dst_bounds = crate::gis::crs::transform_bounds(source_bounds, &src_crs, &dst_crs)?;
    let (width, height) = if let Some(resolution) = resolution {
        let target = extract_resample_target(resolution)?;
        match target {
            warp::ResampleTarget::Shape { height, width } => (width, height),
            warp::ResampleTarget::Resolution { x, y } => (
                ((dst_bounds.right - dst_bounds.left) / x).ceil().max(1.0) as u32,
                ((dst_bounds.top - dst_bounds.bottom) / y).ceil().max(1.0) as u32,
            ),
        }
    } else {
        (src_info.width, src_info.height)
    };
    let transform = crate::gis::affine::transform_from_bounds(dst_bounds, width, height)?;
    let dict = PyDict::new_bound(py);
    dict.set_item("transform", transform.tuple())?;
    dict.set_item("width", width)?;
    dict.set_item("height", height)?;
    dict.set_item("bounds", dst_bounds.tuple())?;
    dict.set_item(
        "src_crs",
        crs_inspection_to_py(py, &crate::gis::crs::inspect_crs_spec(&src_crs, "crs"))?,
    )?;
    dict.set_item(
        "dst_crs",
        crs_inspection_to_py(py, &crate::gis::crs::inspect_crs_spec(&dst_crs, "crs"))?,
    )?;
    Ok(dict.into_py(py))
}

#[cfg(feature = "extension-module")]
#[pyfunction(name = "read_raster_window", signature = (path, bounds_or_window, *, boundless = false, masked = false))]
pub fn read_raster_window_py(
    py: Python<'_>,
    path: String,
    bounds_or_window: &Bound<'_, PyAny>,
    boundless: bool,
    masked: bool,
) -> PyResult<PyObject> {
    read_raster_window_impl(py, path, bounds_or_window, boundless, masked)
}

#[cfg(feature = "extension-module")]
fn parse_read_bands_py(bands: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Vec<u16>>> {
    let Some(bands) = bands else {
        return Ok(None);
    };
    if bands.is_none() {
        return Ok(None);
    }
    if let Ok(band) = bands.extract::<i64>() {
        return Ok(Some(vec![band_to_u16(band)?]));
    }
    let values = bands.extract::<Vec<i64>>().map_err(|_| {
        GisError::InvalidArgument("bands must be an int or sequence of ints".to_string())
    })?;
    values
        .into_iter()
        .map(band_to_u16)
        .collect::<PyResult<Vec<_>>>()
        .map(Some)
}

#[cfg(feature = "extension-module")]
fn band_to_u16(value: i64) -> PyResult<u16> {
    u16::try_from(value).map_err(|_| {
        GisError::InvalidArgument("bands are 1-based positive integers".to_string()).into()
    })
}

#[cfg(feature = "extension-module")]
fn parse_read_window_py(window: Option<&Bound<'_, PyAny>>) -> PyResult<Option<PixelWindow>> {
    let Some(window) = window else {
        return Ok(None);
    };
    if window.is_none() {
        return Ok(None);
    }
    if let Ok(values) = window.extract::<(i64, i64, i64, i64)>() {
        return Ok(Some(pixel_window_from_i64(
            values.0, values.1, values.2, values.3,
        )?));
    }
    let dict = window.downcast::<PyDict>().map_err(|_| {
        GisError::InvalidArgument(
            "window must be (col_off, row_off, width, height) or a dict".to_string(),
        )
    })?;
    let col_off = read_window_dict_i64(dict, "col_off")?;
    let row_off = read_window_dict_i64(dict, "row_off")?;
    let width = read_window_dict_i64(dict, "width")?;
    let height = read_window_dict_i64(dict, "height")?;
    Ok(Some(pixel_window_from_i64(
        col_off, row_off, width, height,
    )?))
}

#[cfg(feature = "extension-module")]
fn read_window_dict_i64(dict: &Bound<'_, PyDict>, key: &'static str) -> PyResult<i64> {
    let Some(value) = dict.get_item(key)? else {
        return Err(pyo3::PyErr::from(GisError::InvalidArgument(format!(
            "window dict must include '{key}'"
        ))));
    };
    value.extract::<i64>().map_err(|_| {
        pyo3::PyErr::from(GisError::InvalidArgument(format!(
            "window field '{key}' must be an integer"
        )))
    })
}

#[cfg(feature = "extension-module")]
fn pixel_window_from_i64(
    col_off: i64,
    row_off: i64,
    width: i64,
    height: i64,
) -> PyResult<PixelWindow> {
    let width = u32::try_from(width).map_err(|_| {
        GisError::InvalidArgument("window width must be a positive integer".to_string())
    })?;
    let height = u32::try_from(height).map_err(|_| {
        GisError::InvalidArgument("window height must be a positive integer".to_string())
    })?;
    Ok(PixelWindow {
        col_off,
        row_off,
        width,
        height,
    })
}

#[cfg(feature = "extension-module")]
fn looks_like_dataset_path(value: &str) -> bool {
    let lower = value.to_ascii_lowercase();
    value.contains('\\')
        || value.contains('/')
        || lower.ends_with(".tif")
        || lower.ends_with(".tiff")
        || lower.ends_with(".geotiff")
        || lower.ends_with(".gpkg")
        || lower.ends_with(".geojson")
        || lower.ends_with(".shp")
        || value
            .as_bytes()
            .get(1..3)
            .is_some_and(|slice| slice[0] == b':' && matches!(slice[1], b'\\' | b'/'))
}

#[cfg(feature = "extension-module")]
pub(crate) fn extract_crs(crs: Option<&Bound<'_, PyAny>>) -> PyResult<Option<CrsSpec>> {
    let Some(crs) = crs else {
        return Ok(None);
    };
    if crs.is_none() {
        return Ok(None);
    }
    if let Ok(value) = crs.extract::<u32>() {
        return CrsSpec::from_string(format!("EPSG:{value}"))
            .map(Some)
            .map_err(Into::into);
    }
    if let Ok(value) = crs.extract::<String>() {
        return CrsSpec::from_string(value).map(Some).map_err(Into::into);
    }
    let dict = crs
        .downcast::<PyDict>()
        .map_err(|_| GisError::InvalidCrs("crs must be a string, dict, or None".to_string()))?;
    for (key, _) in dict.iter() {
        let key = key
            .extract::<String>()
            .map_err(|_| GisError::InvalidCrs("CRS dict keys must be strings".to_string()))?;
        if !matches!(key.as_str(), "name" | "code") {
            return Err(GisError::InvalidCrs(format!(
                "unsupported CRS dict key {key:?}; use name/code"
            ))
            .into());
        }
    }
    let authority = dict
        .get_item("name")?
        .map(|value| py_value_to_string(&value))
        .transpose()?;
    let code = dict
        .get_item("code")?
        .map(|value| py_value_to_string(&value))
        .transpose()?;
    CrsSpec::from_parts(authority, code, None)
        .map(Some)
        .map_err(Into::into)
}

#[cfg(feature = "extension-module")]
pub(crate) fn extract_required_crs(crs: Option<&Bound<'_, PyAny>>) -> PyResult<CrsSpec> {
    extract_crs(crs)?.ok_or_else(|| {
        GisError::MissingCrs("missing_crs: CRS argument is required".to_string()).into()
    })
}

#[cfg(feature = "extension-module")]
fn extract_raster_array(array: &Bound<'_, PyAny>) -> PyResult<RasterArray> {
    let dtype_name = array
        .getattr("dtype")
        .and_then(|dtype| dtype.getattr("name"))
        .and_then(|name| name.extract::<String>())
        .map_err(|_| GisError::UnsupportedDType("expected a NumPy ndarray".to_string()))?;

    match dtype_name.as_str() {
        "uint8" => extract_typed_array::<u8>(array)
            .map(|(data, shape)| RasterArray::new(RasterData::U8(data), &shape))?,
        "int16" => extract_typed_array::<i16>(array)
            .map(|(data, shape)| RasterArray::new(RasterData::I16(data), &shape))?,
        "uint16" => extract_typed_array::<u16>(array)
            .map(|(data, shape)| RasterArray::new(RasterData::U16(data), &shape))?,
        "int32" => extract_typed_array::<i32>(array)
            .map(|(data, shape)| RasterArray::new(RasterData::I32(data), &shape))?,
        "uint32" => extract_typed_array::<u32>(array)
            .map(|(data, shape)| RasterArray::new(RasterData::U32(data), &shape))?,
        "float32" => extract_typed_array::<f32>(array)
            .map(|(data, shape)| RasterArray::new(RasterData::F32(data), &shape))?,
        "float64" => extract_typed_array::<f64>(array)
            .map(|(data, shape)| RasterArray::new(RasterData::F64(data), &shape))?,
        other => {
            Err(GisError::UnsupportedDType(format!("unsupported NumPy dtype {other:?}")).into())
        }
    }
    .map_err(Into::into)
}

#[cfg(feature = "extension-module")]
fn extract_typed_array<T>(array: &Bound<'_, PyAny>) -> PyResult<(Vec<T>, Vec<usize>)>
where
    T: numpy::Element + Copy,
{
    let array: PyReadonlyArrayDyn<'_, T> = array.extract()?;
    let shape = array.shape().to_vec();
    let data = array.as_array().iter().copied().collect::<Vec<_>>();
    Ok((data, shape))
}

#[cfg(feature = "extension-module")]
fn extract_creation_options(
    creation_options: Option<&Bound<'_, PyAny>>,
) -> PyResult<HashMap<String, String>> {
    let mut values = HashMap::new();
    let Some(options) = creation_options else {
        return Ok(values);
    };
    let dict = options
        .downcast::<PyDict>()
        .map_err(|_| GisError::InvalidArgument("creation_options must be a dict".to_string()))?;
    for (key, value) in dict.iter() {
        let key = key.extract::<String>().map_err(|_| {
            GisError::InvalidArgument("creation option keys must be strings".to_string())
        })?;
        let value = if value.is_none() {
            String::new()
        } else if let Ok(text) = value.extract::<String>() {
            text
        } else {
            value.str()?.to_str()?.to_string()
        };
        values.insert(key.to_ascii_lowercase(), value);
    }
    Ok(values)
}

#[cfg(feature = "extension-module")]
fn py_value_to_string(value: &Bound<'_, PyAny>) -> PyResult<String> {
    if let Ok(text) = value.extract::<String>() {
        Ok(text)
    } else {
        Ok(value.str()?.to_str()?.to_string())
    }
}

#[cfg(feature = "extension-module")]
fn extract_affine_transform(value: &Bound<'_, PyAny>) -> PyResult<AffineTransform> {
    if let Ok(transform) = value.extract::<PyRef<'_, AffineTransform>>() {
        return Ok(*transform);
    }
    let values = value
        .extract::<(f64, f64, f64, f64, f64, f64)>()
        .map_err(|_| {
            GisError::InvalidTransform("transform must be AffineTransform or 6-tuple".to_string())
        })?;
    AffineTransform::new([values.0, values.1, values.2, values.3, values.4, values.5])
        .map_err(Into::into)
}

#[cfg(feature = "extension-module")]
fn assign_crs_to_info(info: &RasterInfo, crs: CrsSpec, overwrite: bool) -> PyResult<RasterInfo> {
    if (info.crs_wkt.is_some() || info.crs_authority.is_some()) && !overwrite {
        return Err(GisError::CrsAlreadyExists(
            "crs_exists: raster metadata already has CRS; pass overwrite=True to replace it"
                .to_string(),
        )
        .into());
    }
    let mut out = info.clone();
    out.crs_wkt = crs.wkt.clone();
    out.crs_authority = crate::gis::crs::authority_map(&crs);
    out.is_georeferenced = out.transform.is_some();
    out.warnings
        .retain(|warning| warning.code != types::WARNING_MISSING_CRS);
    out.warnings.push(RasterWarning::new(
        types::WARNING_ASSIGNMENT_NOT_REPROJECTION,
        "CRS assignment updates metadata only; coordinates and pixel values were not reprojected",
        Some("crs"),
    ));
    Ok(out)
}

#[cfg(feature = "extension-module")]
fn transform_warnings(transform: AffineTransform) -> Vec<RasterWarning> {
    if transform.is_rotated_or_sheared() {
        vec![RasterWarning::new(
            types::WARNING_ROTATED_OR_SHEARED,
            "transform contains rotation or shear terms",
            Some("transform"),
        )]
    } else {
        Vec::new()
    }
}

#[cfg(feature = "extension-module")]
fn extract_bool_mask(mask: &Bound<'_, PyAny>) -> PyResult<Vec<bool>> {
    if mask.is_none() {
        return Ok(Vec::new());
    }
    extract_typed_array::<bool>(mask).map(|(values, _)| values)
}

#[cfg(feature = "extension-module")]
fn validate_mask_len(len: usize, array: &RasterArray) -> PyResult<()> {
    if len != array.bands * array.height * array.width {
        return Err(GisError::InvalidShape(format!(
            "mask has {len} values but raster array has {}",
            array.bands * array.height * array.width
        ))
        .into());
    }
    Ok(())
}

#[cfg(feature = "extension-module")]
fn extract_nodata(
    nodata: Option<&Bound<'_, PyAny>>,
    band_count: usize,
) -> PyResult<Vec<Option<f64>>> {
    let Some(nodata) = nodata else {
        return Ok(vec![None; band_count]);
    };
    if nodata.is_none() {
        return Ok(vec![None; band_count]);
    }
    if let Ok(value) = nodata.extract::<f64>() {
        return Ok(vec![Some(value); band_count]);
    }
    if let Ok(values) = nodata.extract::<Vec<Option<f64>>>() {
        if values.len() != band_count {
            return Err(GisError::InvalidNodata(format!(
                "nodata length {} does not match band count {band_count}",
                values.len()
            ))
            .into());
        }
        return Ok(values);
    }
    let values = nodata.extract::<Vec<f64>>().map_err(|_| {
        GisError::InvalidNodata("nodata must be a scalar or per-band list".to_string())
    })?;
    if values.len() != band_count {
        return Err(GisError::InvalidNodata(format!(
            "nodata length {} does not match band count {band_count}",
            values.len()
        ))
        .into());
    }
    Ok(values.into_iter().map(Some).collect())
}

#[cfg(feature = "extension-module")]
fn extract_path_or_info_path(value: &Bound<'_, PyAny>) -> PyResult<String> {
    if let Ok(info) = value.extract::<PyRef<'_, RasterInfo>>() {
        return Ok(info.path.clone());
    }
    value.extract::<String>().map_err(|_| {
        GisError::InvalidArgument("source must be a path or RasterInfo".to_string()).into()
    })
}

#[cfg(feature = "extension-module")]
struct RasterSourceArg {
    path: String,
    categorical: bool,
}

#[cfg(feature = "extension-module")]
fn extract_raster_source_arg(value: &Bound<'_, PyAny>) -> PyResult<RasterSourceArg> {
    if let Ok(info) = value.extract::<PyRef<'_, RasterInfo>>() {
        return Ok(RasterSourceArg {
            path: info.path.clone(),
            categorical: false,
        });
    }
    if let Ok(path) = value.extract::<String>() {
        return Ok(RasterSourceArg {
            path,
            categorical: false,
        });
    }
    let dict = value.downcast::<PyDict>().map_err(|_| {
        GisError::InvalidArgument("source must be a path, RasterInfo, or source dict".to_string())
    })?;
    let path_value = dict
        .get_item("path")?
        .ok_or_else(|| GisError::InvalidArgument("source dict must include 'path'".to_string()))?;
    let path = py_value_to_string(&path_value)?;
    let categorical = dict
        .get_item("categorical")?
        .map(|value| value.extract::<bool>())
        .transpose()?
        .unwrap_or(false);
    Ok(RasterSourceArg { path, categorical })
}

#[cfg(feature = "extension-module")]
fn extract_raster_info_source(value: &Bound<'_, PyAny>) -> PyResult<(RasterInfo, &'static str)> {
    if let Ok(info) = value.extract::<PyRef<'_, RasterInfo>>() {
        return Ok((info.clone(), "raster"));
    }
    let path = value.extract::<String>().map_err(|_| {
        GisError::InvalidArgument("source must be a path or RasterInfo".to_string())
    })?;
    Ok((read_raster_info(path)?, "path"))
}

#[cfg(feature = "extension-module")]
fn extract_bounds_arg(value: &Bound<'_, PyAny>) -> PyResult<(RasterBounds, Option<CrsSpec>)> {
    if let Ok(values) = value.extract::<(f64, f64, f64, f64)>() {
        return Ok((
            crate::gis::affine::validate_bounds_tuple(values, false)?,
            None,
        ));
    }
    let dict = value
        .downcast::<PyDict>()
        .map_err(|_| GisError::InvalidBounds("bounds must be a 4-tuple or dict".to_string()))?;
    let bounds_value = dict
        .get_item("bounds")?
        .ok_or_else(|| GisError::InvalidBounds("bounds dict must include 'bounds'".to_string()))?;
    let bounds = bounds_value.extract::<(f64, f64, f64, f64)>()?;
    let crs = dict
        .get_item("crs")?
        .map(|value| extract_required_crs(Some(&value)))
        .transpose()?;
    Ok((
        crate::gis::affine::validate_bounds_tuple(bounds, false)?,
        crs,
    ))
}

#[cfg(feature = "extension-module")]
fn extract_resample_target(value: &Bound<'_, PyAny>) -> PyResult<warp::ResampleTarget> {
    if let Ok(dict) = value.downcast::<PyDict>() {
        let shape = dict.get_item("shape")?;
        let resolution = dict.get_item("resolution")?;
        if shape.is_some() && resolution.is_some() {
            return Err(GisError::InvalidArgument(
                "ambiguous_target: target dict must not include both 'shape' and 'resolution'"
                    .to_string(),
            )
            .into());
        }
        if let Some(shape) = shape {
            let shape = shape.extract::<(u32, u32)>()?;
            return Ok(warp::ResampleTarget::Shape {
                height: shape.0,
                width: shape.1,
            });
        }
        if let Some(resolution) = resolution {
            if let Ok(value) = resolution.extract::<f64>() {
                return Ok(warp::ResampleTarget::Resolution { x: value, y: value });
            }
            let resolution = resolution.extract::<(f64, f64)>()?;
            return Ok(warp::ResampleTarget::Resolution {
                x: resolution.0,
                y: resolution.1,
            });
        }
        return Err(GisError::InvalidArgument(
            "target dict must include 'shape' or 'resolution'".to_string(),
        )
        .into());
    }
    if let Ok(tuple) = value.downcast::<PyTuple>() {
        if tuple.len() != 2 {
            return Err(GisError::InvalidArgument(
                "shape_or_resolution tuple must contain exactly two values".to_string(),
            )
            .into());
        }
        let first = tuple.get_item(0)?;
        let second = tuple.get_item(1)?;
        if first.is_instance_of::<PyFloat>() || second.is_instance_of::<PyFloat>() {
            return Ok(warp::ResampleTarget::Resolution {
                x: first.extract::<f64>()?,
                y: second.extract::<f64>()?,
            });
        }
        let shape = value.extract::<(u32, u32)>()?;
        return Ok(warp::ResampleTarget::Shape {
            height: shape.0,
            width: shape.1,
        });
    }
    if let Ok(shape) = value.extract::<(u32, u32)>() {
        return Ok(warp::ResampleTarget::Shape {
            height: shape.0,
            width: shape.1,
        });
    }
    if let Ok(resolution) = value.extract::<f64>() {
        return Ok(warp::ResampleTarget::Resolution {
            x: resolution,
            y: resolution,
        });
    }
    let resolution = value.extract::<(f64, f64)>().map_err(|_| {
        GisError::InvalidArgument(
            "shape_or_resolution must be (height, width), a resolution scalar, or a dict"
                .to_string(),
        )
    })?;
    Ok(warp::ResampleTarget::Resolution {
        x: resolution.0,
        y: resolution.1,
    })
}

#[cfg(feature = "extension-module")]
fn crs_inspection_to_py(py: Python<'_>, inspection: &crs::CrsInspection) -> PyResult<PyObject> {
    use pyo3::IntoPy;

    let dict = PyDict::new_bound(py);
    dict.set_item("source_kind", inspection.source_kind.clone())?;
    dict.set_item("missing", inspection.missing)?;
    dict.set_item("wkt", inspection.wkt.clone())?;
    dict.set_item("authority", inspection.authority.clone())?;
    dict.set_item("axis_order", inspection.axis_order.clone())?;
    dict.set_item("axis_order_policy", inspection.axis_order_policy.clone())?;
    dict.set_item("warnings", warnings_to_py(py, &inspection.warnings)?)?;
    Ok(dict.into_py(py))
}

#[cfg(feature = "extension-module")]
fn raster_window_to_py(py: Python<'_>, window: &RasterWindow) -> PyResult<PyObject> {
    use pyo3::IntoPy;

    let dict = PyDict::new_bound(py);
    dict.set_item(
        "window",
        (
            window.window.col_off,
            window.window.row_off,
            window.window.width,
            window.window.height,
        ),
    )?;
    dict.set_item("clipped_bounds", window.clipped_bounds.tuple())?;
    dict.set_item("output_transform", window.output_transform.tuple())?;
    dict.set_item("output_shape", window.output_shape)?;
    Ok(dict.into_py(py))
}

#[cfg(feature = "extension-module")]
fn raster_operation_to_py(
    py: Python<'_>,
    result: &warp::RasterOperationResult,
) -> PyResult<PyObject> {
    use pyo3::IntoPy;

    let dict = PyDict::new_bound(py);
    dict.set_item("array", raster_array_to_py(py, &result.array)?)?;
    dict.set_item("info", raster_info_to_py_dict(py, &result.info)?)?;
    dict.set_item("resampling", result.resampling.clone())?;
    dict.set_item("diagnostics", warnings_to_py(py, &result.diagnostics)?)?;
    Ok(dict.into_py(py))
}

#[cfg(feature = "extension-module")]
fn read_result_to_py(py: Python<'_>, result: &raster_info::RasterReadResult) -> PyResult<PyObject> {
    use pyo3::{IntoPy, ToPyObject};

    let dict = PyDict::new_bound(py);
    dict.set_item("array", raster_array_to_py(py, &result.array)?)?;
    dict.set_item("info", raster_info_to_py_dict(py, &result.info)?)?;
    dict.set_item(
        "bands",
        PyTuple::new_bound(py, result.bands.iter().map(|band| band.to_object(py))),
    )?;
    dict.set_item(
        "window",
        result
            .window
            .map(|window| (window.col_off, window.row_off, window.width, window.height)),
    )?;
    dict.set_item(
        "window_transform",
        result.window_transform.map(AffineTransform::tuple),
    )?;
    match &result.mask {
        Some(mask) => {
            dict.set_item(
                "mask",
                bool_array_to_py_shape(
                    py,
                    mask.clone(),
                    (result.array.bands, result.array.height, result.array.width),
                )?,
            )?;
            dict.set_item("mask_polarity", "true_valid")?;
        }
        None => {
            dict.set_item("mask", py.None())?;
            dict.set_item("mask_polarity", py.None())?;
        }
    }
    dict.set_item("nodata_per_band", result.nodata_per_band.clone())?;
    dict.set_item("warnings", warnings_to_py(py, &result.warnings)?)?;
    Ok(dict.into_py(py))
}

#[cfg(feature = "extension-module")]
fn raster_array_to_py(py: Python<'_>, array: &RasterArray) -> PyResult<PyObject> {
    use pyo3::IntoPy;

    let shape = [array.bands, array.height, array.width];
    let object = match &array.data {
        RasterData::U8(data) => PyArray1::from_vec_bound(py, data.clone())
            .reshape(shape)?
            .into_py(py),
        RasterData::I16(data) => PyArray1::from_vec_bound(py, data.clone())
            .reshape(shape)?
            .into_py(py),
        RasterData::U16(data) => PyArray1::from_vec_bound(py, data.clone())
            .reshape(shape)?
            .into_py(py),
        RasterData::I32(data) => PyArray1::from_vec_bound(py, data.clone())
            .reshape(shape)?
            .into_py(py),
        RasterData::U32(data) => PyArray1::from_vec_bound(py, data.clone())
            .reshape(shape)?
            .into_py(py),
        RasterData::F32(data) => PyArray1::from_vec_bound(py, data.clone())
            .reshape(shape)?
            .into_py(py),
        RasterData::F64(data) => PyArray1::from_vec_bound(py, data.clone())
            .reshape(shape)?
            .into_py(py),
    };
    Ok(object)
}

#[cfg(feature = "extension-module")]
fn bool_array_to_py(py: Python<'_>, mask: Vec<bool>, array: &RasterArray) -> PyResult<PyObject> {
    bool_array_to_py_shape(py, mask, (array.bands, array.height, array.width))
}

#[cfg(feature = "extension-module")]
fn bool_array_to_py_shape(
    py: Python<'_>,
    mask: Vec<bool>,
    shape: (usize, usize, usize),
) -> PyResult<PyObject> {
    use pyo3::IntoPy;

    Ok(PyArray1::from_vec_bound(py, mask)
        .reshape([shape.0, shape.1, shape.2])?
        .into_py(py))
}

#[cfg(feature = "extension-module")]
fn read_raster_window_impl(
    py: Python<'_>,
    path: String,
    bounds_or_window: &Bound<'_, PyAny>,
    boundless: bool,
    masked: bool,
) -> PyResult<PyObject> {
    use pyo3::IntoPy;

    let loaded = raster_info::read_raster_data(&path)?;
    let window = if let Ok(values) = bounds_or_window.extract::<(i64, i64, u32, u32)>() {
        PixelWindow {
            col_off: values.0,
            row_off: values.1,
            width: values.2,
            height: values.3,
        }
    } else {
        let (bounds, bounds_crs) = extract_bounds_arg(bounds_or_window)?;
        if let Some(bounds_crs) = bounds_crs {
            let raster_crs = crate::gis::crs::require_crs(&loaded.info)?;
            if !crate::gis::crs::crs_equal(&raster_crs, &bounds_crs) {
                return Err(GisError::CrsMismatch(
                    "crs_mismatch: bounds CRS does not match raster CRS".to_string(),
                )
                .into());
            }
        }
        crate::gis::affine::window_from_bounds(&loaded.info, bounds, boundless)?.window
    };
    if window.width == 0 || window.height == 0 {
        return Err(GisError::InvalidBounds(
            "window width and height must be positive".to_string(),
        )
        .into());
    }
    if !boundless
        && (window.col_off < 0
            || window.row_off < 0
            || window.col_off + window.width as i64 > loaded.info.width as i64
            || window.row_off + window.height as i64 > loaded.info.height as i64)
    {
        return Err(GisError::InvalidBounds(
            "window is outside raster extent; pass boundless=True to allow padding".to_string(),
        )
        .into());
    }
    let output_transform = crate::gis::affine::window_transform(&loaded.info, window)?;
    let array = raster_info::copy_window(&loaded.array, &loaded.info.nodata_per_band, window)?;
    let info = warp::operation_info(
        &loaded.info,
        window.width,
        window.height,
        Some(output_transform),
        None,
        &array,
    )?;
    let dict = PyDict::new_bound(py);
    dict.set_item("array", raster_array_to_py(py, &array)?)?;
    dict.set_item(
        "window",
        (window.col_off, window.row_off, window.width, window.height),
    )?;
    dict.set_item("window_transform", output_transform.tuple())?;
    dict.set_item("info", raster_info_to_py_dict(py, &info)?)?;
    if masked {
        dict.set_item(
            "mask",
            bool_array_to_py(
                py,
                raster_info::valid_mask(&array, &info.nodata_per_band, None),
                &array,
            )?,
        )?;
        dict.set_item("mask_polarity", "true_valid")?;
    } else {
        dict.set_item("mask", py.None())?;
        dict.set_item("mask_polarity", py.None())?;
    }
    Ok(dict.into_py(py))
}

#[cfg(feature = "extension-module")]
fn raster_info_to_py_dict(py: Python<'_>, info: &RasterInfo) -> PyResult<PyObject> {
    use pyo3::IntoPy;

    let dict = PyDict::new_bound(py);
    dict.set_item("path", info.path.clone())?;
    dict.set_item("driver", info.driver.clone())?;
    dict.set_item("width", info.width)?;
    dict.set_item("height", info.height)?;
    dict.set_item("band_count", info.band_count)?;
    dict.set_item("dtype_per_band", info.dtype_per_band.clone())?;
    dict.set_item("crs_wkt", info.crs_wkt.clone())?;
    dict.set_item("crs_authority", info.crs_authority.clone())?;
    dict.set_item("transform", info.transform)?;
    dict.set_item("bounds", info.bounds)?;
    dict.set_item("resolution", info.resolution)?;
    dict.set_item("nodata_per_band", info.nodata_per_band.clone())?;
    dict.set_item("block_size", info.block_size.clone())?;
    dict.set_item("tiling", info.tiling.clone())?;
    dict.set_item("compression", info.compression.clone())?;
    dict.set_item("is_georeferenced", info.is_georeferenced)?;
    dict.set_item("warnings", warnings_to_py(py, &info.warnings)?)?;
    Ok(dict.into_py(py))
}

#[cfg(feature = "extension-module")]
fn warnings_to_py(py: Python<'_>, warnings: &[RasterWarning]) -> PyResult<PyObject> {
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
