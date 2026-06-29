pub mod error;
pub mod raster_info;
pub mod raster_write;
pub mod types;

pub use error::{GisError, GisResult};
pub use raster_info::read_raster_info;
pub use raster_write::{
    write_raster, CreationOptions, CrsSpec, RasterArray, RasterData, WriteRasterOptions,
};
pub use types::{AffineTransform, RasterBounds, RasterDType, RasterInfo, RasterWarning};

#[cfg(feature = "extension-module")]
use std::collections::HashMap;

#[cfg(feature = "extension-module")]
use numpy::{PyReadonlyArrayDyn, PyUntypedArrayMethods};
#[cfg(feature = "extension-module")]
use pyo3::prelude::*;
#[cfg(feature = "extension-module")]
use pyo3::types::{PyAny, PyDict, PyDictMethods};

#[cfg(feature = "extension-module")]
#[pyfunction(name = "read_raster_info")]
pub fn read_raster_info_py(path: String) -> PyResult<RasterInfo> {
    read_raster_info(path).map_err(Into::into)
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
fn extract_crs(crs: Option<&Bound<'_, PyAny>>) -> PyResult<Option<CrsSpec>> {
    let Some(crs) = crs else {
        return Ok(None);
    };
    if crs.is_none() {
        return Ok(None);
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
