#[cfg(feature = "extension-module")]
pub use py::{classify_raster_py, normalize_raster_py};

#[cfg(feature = "extension-module")]
mod py {
    use pyo3::prelude::*;
    use pyo3::types::{PyAny, PyDict, PyDictMethods, PyList, PyTuple};
    use pyo3::IntoPy;

    use crate::gis::error::GisError;
    use crate::gis::raster_info;
    use crate::gis::raster_write::{RasterArray, RasterData};
    use crate::gis::types::{RasterDType, RasterInfo, RasterWarning};

    const EMPTY_RASTER: &str = "empty_raster";
    const INVALID_ARGUMENT: &str = "invalid_argument";
    const INVALID_NODATA: &str = "invalid_nodata";
    const SHAPE_MISMATCH: &str = "shape_mismatch";
    const UNSUPPORTED_DTYPE: &str = "unsupported_dtype";
    const UNSUPPORTED_OPTION: &str = "unsupported_option";

    struct RasterSource {
        array: RasterArray,
        info: RasterInfo,
    }

    struct Stats {
        valid: Vec<bool>,
        values: Vec<f64>,
        valid_count: usize,
        nodata_count: usize,
        min: f64,
        max: f64,
        mean: f64,
        std: f64,
    }

    #[pyfunction(
        name = "normalize_raster",
        signature = (source, *, method = "minmax", valid_mask = None, nodata = None, clip = None)
    )]
    pub fn normalize_raster_py(
        py: Python<'_>,
        source: &Bound<'_, PyAny>,
        method: &str,
        valid_mask: Option<&Bound<'_, PyAny>>,
        nodata: Option<&Bound<'_, PyAny>>,
        clip: Option<(f64, f64)>,
    ) -> PyResult<PyObject> {
        if method != "minmax" {
            return Err(GisError::InvalidArgument(format!(
                "{UNSUPPORTED_OPTION}: normalize_raster supports only method='minmax'"
            ))
            .into());
        }
        let clip = validate_clip(clip)?;
        let source = raster_source_from_py(source)?;
        let nodata = nodata_or_source(nodata, &source)?;
        let mask = valid_mask_from_py(valid_mask, &source.array)?;
        let stats = stats(&source.array, &nodata, mask.as_deref(), clip)?;

        let mut out = Vec::with_capacity(stats.values.len());
        let span = stats.max - stats.min;
        for (index, value) in stats.values.iter().enumerate() {
            if stats.valid[index] {
                out.push(if span == 0.0 {
                    0.0
                } else {
                    ((value - stats.min) / span) as f32
                });
            } else {
                out.push(f32::NAN);
            }
        }
        let array = RasterArray::new(
            RasterData::F32(out),
            &[source.array.bands, source.array.height, source.array.width],
        )?;
        let mut info = output_info(&source.info, &array, vec![Some(f64::NAN); array.bands]);
        info.dtype_per_band = vec![RasterDType::Float32.name().to_string(); array.bands];
        result_to_py(py, &array, &info, "minmax", &stats, None)
    }

    #[pyfunction(
        name = "classify_raster",
        signature = (source, *, bins = None, labels = None, right = false, valid_mask = None, nodata = None, dtype = "uint16")
    )]
    pub fn classify_raster_py(
        py: Python<'_>,
        source: &Bound<'_, PyAny>,
        bins: Option<&Bound<'_, PyAny>>,
        labels: Option<&Bound<'_, PyAny>>,
        right: bool,
        valid_mask: Option<&Bound<'_, PyAny>>,
        nodata: Option<&Bound<'_, PyAny>>,
        dtype: &str,
    ) -> PyResult<PyObject> {
        let bins = parse_bins(bins)?;
        let labels = parse_labels(labels, bins.len() + 1)?;
        let dtype = parse_class_dtype(dtype, bins.len() + 1)?;
        let source = raster_source_from_py(source)?;
        let nodata = nodata_or_source(nodata, &source)?;
        let mask = valid_mask_from_py(valid_mask, &source.array)?;
        let stats = stats(&source.array, &nodata, mask.as_deref(), None)?;

        let mut counts = vec![0usize; bins.len() + 2];
        let ids = stats
            .values
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let id = if stats.valid[index] {
                    digitize(*value, &bins, right) + 1
                } else {
                    0
                };
                counts[id] += 1;
                id as u32
            })
            .collect::<Vec<_>>();
        let array = class_array(
            dtype,
            ids,
            source.array.bands,
            source.array.height,
            source.array.width,
        )?;
        let info = output_info(&source.info, &array, vec![Some(0.0); array.bands]);
        let table = class_table_py(py, &bins, &labels, right, &counts)?;
        result_to_py(py, &array, &info, "explicit_bins", &stats, Some(table))
    }

    fn validate_clip(clip: Option<(f64, f64)>) -> PyResult<Option<(f64, f64)>> {
        match clip {
            Some((lo, hi)) if lo.is_finite() && hi.is_finite() && lo < hi => Ok(Some((lo, hi))),
            Some(_) => Err(GisError::InvalidArgument(format!(
                "{INVALID_ARGUMENT}: clip must be finite and ordered as (min, max)"
            ))
            .into()),
            None => Ok(None),
        }
    }

    fn raster_source_from_py(value: &Bound<'_, PyAny>) -> PyResult<RasterSource> {
        if let Ok(path) = value.extract::<String>() {
            let result = raster_info::read_raster(path, None, None, false)?;
            return Ok(RasterSource {
                array: result.array,
                info: result.info,
            });
        }
        if let Ok(info) = value.extract::<PyRef<'_, RasterInfo>>() {
            if info.path.is_empty() {
                return Err(GisError::InvalidArgument(format!(
                    "{INVALID_ARGUMENT}: RasterInfo source must include a path"
                ))
                .into());
            }
            let result = raster_info::read_raster(&info.path, None, None, false)?;
            return Ok(RasterSource {
                array: result.array,
                info: result.info,
            });
        }
        if let Ok(dict) = value.downcast::<PyDict>() {
            let Some(array_value) = dict.get_item("array")? else {
                return Err(GisError::InvalidArgument(format!(
                    "{INVALID_ARGUMENT}: source dict must include array"
                ))
                .into());
            };
            let array = extract_source_array(&array_value)?;
            let info = if let Some(info_value) = dict.get_item("info")? {
                raster_info_from_py(&info_value)?
            } else {
                synthetic_info(&array)
            };
            validate_info_shape(&info, &array)?;
            return Ok(RasterSource { array, info });
        }
        let array = extract_source_array(value)?;
        let info = synthetic_info(&array);
        Ok(RasterSource { array, info })
    }

    fn extract_source_array(value: &Bound<'_, PyAny>) -> PyResult<RasterArray> {
        let dtype_name = value
            .getattr("dtype")
            .and_then(|dtype| dtype.getattr("name"))
            .and_then(|name| name.extract::<String>())
            .map_err(|_| {
                GisError::UnsupportedDType(format!(
                    "{UNSUPPORTED_DTYPE}: source must be a supported NumPy ndarray"
                ))
            })?;
        match dtype_name.as_str() {
            "uint8" | "int16" | "uint16" | "int32" | "uint32" | "float32" | "float64" => {
                super::super::extract_raster_array(value)
            }
            other => Err(GisError::UnsupportedDType(format!(
                "{UNSUPPORTED_DTYPE}: unsupported source dtype {other:?}"
            ))
            .into()),
        }
    }

    fn raster_info_from_py(value: &Bound<'_, PyAny>) -> PyResult<RasterInfo> {
        if let Ok(info) = value.extract::<PyRef<'_, RasterInfo>>() {
            return Ok(info.clone());
        }
        let dict = value.downcast::<PyDict>().map_err(|_| {
            GisError::InvalidArgument(format!(
                "{INVALID_ARGUMENT}: raster info must be RasterInfo or dict"
            ))
        })?;
        let width = required_u32(dict, "width")?;
        let height = required_u32(dict, "height")?;
        let band_count = optional_u16(dict.get_item("band_count")?)?.unwrap_or(1);
        let mut info = RasterInfo::new(
            dict.get_item("path")?
                .and_then(|value| value.extract::<String>().ok())
                .unwrap_or_default()
                .into(),
            width,
            height,
            band_count,
        );
        info.driver = dict
            .get_item("driver")?
            .and_then(|value| value.extract::<String>().ok())
            .unwrap_or_else(|| "memory".to_string());
        info.dtype_per_band = dict
            .get_item("dtype_per_band")?
            .map(|value| value.extract::<Vec<String>>())
            .transpose()?
            .unwrap_or_else(|| vec!["uint8".to_string(); band_count as usize]);
        info.nodata_per_band = dict
            .get_item("nodata_per_band")?
            .map(|value| value.extract::<Vec<Option<f64>>>())
            .transpose()?
            .unwrap_or_else(|| vec![None; band_count as usize]);
        info.crs_wkt = optional_string(dict.get_item("crs_wkt")?)?;
        info.crs_authority = dict
            .get_item("crs_authority")?
            .map(|value| {
                if value.is_none() {
                    Ok(None)
                } else {
                    value.extract().map(Some)
                }
            })
            .transpose()?
            .flatten();
        info.transform = optional_tuple6(dict.get_item("transform")?)?;
        info.bounds = optional_tuple4(dict.get_item("bounds")?)?;
        info.resolution = optional_tuple2(dict.get_item("resolution")?)?;
        info.warnings = warnings_from_py(dict.get_item("warnings")?)?;
        info.is_georeferenced =
            info.transform.is_some() && (info.crs_wkt.is_some() || info.crs_authority.is_some());
        Ok(info)
    }

    fn synthetic_info(array: &RasterArray) -> RasterInfo {
        let mut info = RasterInfo::new(
            "".into(),
            array.width as u32,
            array.height as u32,
            array.bands as u16,
        );
        info.driver = "memory".to_string();
        info.dtype_per_band = vec![array.dtype().name().to_string(); array.bands];
        info.nodata_per_band = vec![None; array.bands];
        info
    }

    fn validate_info_shape(info: &RasterInfo, array: &RasterArray) -> PyResult<()> {
        if info.width as usize != array.width
            || info.height as usize != array.height
            || info.band_count as usize != array.bands
        {
            return Err(GisError::ShapeMismatch(format!(
                "{SHAPE_MISMATCH}: source info shape ({}, {}, {}) does not match array shape ({}, {}, {})",
                info.band_count, info.height, info.width, array.bands, array.height, array.width
            ))
            .into());
        }
        Ok(())
    }

    fn nodata_or_source(
        value: Option<&Bound<'_, PyAny>>,
        source: &RasterSource,
    ) -> PyResult<Vec<Option<f64>>> {
        let nodata = match value {
            Some(value) if !value.is_none() => parse_nodata(value, source.array.bands)?,
            _ => normalize_nodata_len(source.info.nodata_per_band.clone(), source.array.bands)?,
        };
        for item in nodata.iter().flatten() {
            validate_nodata(source.array.dtype(), *item)?;
        }
        Ok(nodata)
    }

    fn parse_nodata(value: &Bound<'_, PyAny>, bands: usize) -> PyResult<Vec<Option<f64>>> {
        if let Ok(number) = value.extract::<f64>() {
            return Ok(vec![Some(number); bands]);
        }
        if let Ok(list) = value.downcast::<PyList>() {
            return nodata_from_iter(list.iter(), bands);
        }
        if let Ok(tuple) = value.downcast::<PyTuple>() {
            return nodata_from_iter(tuple.iter(), bands);
        }
        Err(GisError::InvalidNodata(format!(
            "{INVALID_NODATA}: nodata must be a scalar or per-band list"
        ))
        .into())
    }

    fn nodata_from_iter<'py>(
        items: impl Iterator<Item = Bound<'py, PyAny>>,
        bands: usize,
    ) -> PyResult<Vec<Option<f64>>> {
        let values = items
            .map(|item| {
                if item.is_none() {
                    Ok(None)
                } else {
                    item.extract::<f64>().map(Some).map_err(|_| {
                        GisError::InvalidNodata(format!(
                            "{INVALID_NODATA}: nodata list values must be numeric or None"
                        ))
                        .into()
                    })
                }
            })
            .collect::<PyResult<Vec<_>>>()?;
        normalize_nodata_len(values, bands).map_err(Into::into)
    }

    fn normalize_nodata_len(
        mut values: Vec<Option<f64>>,
        bands: usize,
    ) -> Result<Vec<Option<f64>>, GisError> {
        if values.is_empty() {
            values = vec![None; bands];
        }
        if values.len() != bands {
            return Err(GisError::InvalidNodata(format!(
                "{INVALID_NODATA}: nodata length {} does not match band count {bands}",
                values.len()
            )));
        }
        Ok(values)
    }

    fn validate_nodata(dtype: RasterDType, value: f64) -> PyResult<()> {
        if dtype.nodata_fits(value) {
            Ok(())
        } else {
            Err(GisError::InvalidNodata(format!(
                "{INVALID_NODATA}: nodata value {value} does not fit {}",
                dtype.name()
            ))
            .into())
        }
    }

    fn valid_mask_from_py(
        value: Option<&Bound<'_, PyAny>>,
        array: &RasterArray,
    ) -> PyResult<Option<Vec<bool>>> {
        let Some(value) = value else {
            return Ok(None);
        };
        if value.is_none() {
            return Ok(None);
        }
        let (mask, shape) = super::super::extract_typed_array::<bool>(value)?;
        expand_mask(mask, &shape, array)
            .map(Some)
            .map_err(Into::into)
    }

    fn expand_mask(
        mask: Vec<bool>,
        shape: &[usize],
        array: &RasterArray,
    ) -> Result<Vec<bool>, GisError> {
        let pixels = array.height * array.width;
        match shape {
            [height, width] if *height == array.height && *width == array.width => {
                let mut out = Vec::with_capacity(array.bands * pixels);
                for _ in 0..array.bands {
                    out.extend(mask.iter().copied());
                }
                Ok(out)
            }
            [bands, height, width]
                if *bands == array.bands && *height == array.height && *width == array.width =>
            {
                Ok(mask)
            }
            _ => Err(GisError::ShapeMismatch(format!(
                "{SHAPE_MISMATCH}: valid_mask shape {:?} is not compatible with raster shape ({}, {}, {})",
                shape, array.bands, array.height, array.width
            ))),
        }
    }

    fn stats(
        array: &RasterArray,
        nodata: &[Option<f64>],
        explicit_mask: Option<&[bool]>,
        clip: Option<(f64, f64)>,
    ) -> PyResult<Stats> {
        let mut values = raster_info::raster_to_f64(array);
        let pixels = array.height * array.width;
        let mut valid = vec![false; values.len()];
        let mut observed = Vec::new();
        for band in 0..array.bands {
            let nodata = nodata.get(band).copied().flatten();
            for pixel in 0..pixels {
                let index = band * pixels + pixel;
                let mask_valid = explicit_mask
                    .and_then(|mask| mask.get(index))
                    .copied()
                    .unwrap_or(true);
                let value = values[index];
                let is_valid = mask_valid && value.is_finite() && !nodata_matches(value, nodata);
                if is_valid {
                    if let Some((lo, hi)) = clip {
                        values[index] = values[index].clamp(lo, hi);
                    }
                    valid[index] = true;
                    observed.push(values[index]);
                }
            }
        }
        if observed.is_empty() {
            return Err(GisError::InvalidArgument(format!(
                "{EMPTY_RASTER}: no valid raster cells remain"
            ))
            .into());
        }
        let valid_count = observed.len();
        let nodata_count = values.len() - valid_count;
        let min = observed.iter().copied().fold(f64::INFINITY, f64::min);
        let max = observed.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mean = observed.iter().sum::<f64>() / valid_count as f64;
        let variance = observed
            .iter()
            .map(|value| {
                let delta = value - mean;
                delta * delta
            })
            .sum::<f64>()
            / valid_count as f64;
        Ok(Stats {
            valid,
            values,
            valid_count,
            nodata_count,
            min,
            max,
            mean,
            std: variance.sqrt(),
        })
    }

    fn nodata_matches(value: f64, nodata: Option<f64>) -> bool {
        nodata.is_some_and(|nodata| value == nodata || (value.is_nan() && nodata.is_nan()))
    }

    fn parse_bins(value: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<f64>> {
        let Some(value) = value else {
            return Err(invalid_bins());
        };
        if value.is_none() {
            return Err(invalid_bins());
        }
        let bins = if let Ok(values) = value.extract::<Vec<f64>>() {
            values
        } else if let Ok((values, shape)) = super::super::extract_typed_array::<f64>(value) {
            if shape.len() != 1 {
                return Err(invalid_bins());
            }
            values
        } else {
            return Err(invalid_bins());
        };
        if bins.is_empty()
            || bins.iter().any(|value| !value.is_finite())
            || bins.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(invalid_bins());
        }
        Ok(bins)
    }

    fn invalid_bins() -> PyErr {
        GisError::InvalidArgument(format!(
            "{INVALID_ARGUMENT}: bins must be a finite strictly increasing sequence"
        ))
        .into()
    }

    fn parse_labels(value: Option<&Bound<'_, PyAny>>, expected: usize) -> PyResult<Vec<String>> {
        let labels = match value {
            Some(value) if !value.is_none() => value.extract::<Vec<String>>().map_err(|_| {
                GisError::InvalidArgument(format!(
                    "{INVALID_ARGUMENT}: labels must be a sequence of strings"
                ))
            })?,
            _ => (1..=expected)
                .map(|class_id| format!("class_{class_id}"))
                .collect(),
        };
        if labels.len() != expected {
            return Err(GisError::InvalidArgument(format!(
                "{INVALID_ARGUMENT}: labels length {} does not match class count {expected}",
                labels.len()
            ))
            .into());
        }
        Ok(labels)
    }

    fn parse_class_dtype(value: &str, max_class_id: usize) -> PyResult<RasterDType> {
        let dtype = match value.to_ascii_lowercase().as_str() {
            "uint8" => RasterDType::UInt8,
            "uint16" => RasterDType::UInt16,
            "int16" => RasterDType::Int16,
            "uint32" => RasterDType::UInt32,
            "int32" => RasterDType::Int32,
            _ => {
                return Err(GisError::UnsupportedDType(format!(
                    "{UNSUPPORTED_DTYPE}: classify_raster dtype must be uint8, uint16, int16, uint32, or int32"
                ))
                .into())
            }
        };
        let max_value = match dtype {
            RasterDType::UInt8 => u8::MAX as usize,
            RasterDType::UInt16 => u16::MAX as usize,
            RasterDType::Int16 => i16::MAX as usize,
            RasterDType::UInt32 => u32::MAX as usize,
            RasterDType::Int32 => i32::MAX as usize,
            RasterDType::Float32 | RasterDType::Float64 => unreachable!(),
        };
        if max_class_id > max_value {
            return Err(GisError::UnsupportedDType(format!(
                "{UNSUPPORTED_DTYPE}: dtype {} cannot store class id {max_class_id}",
                dtype.name()
            ))
            .into());
        }
        Ok(dtype)
    }

    fn digitize(value: f64, bins: &[f64], right: bool) -> usize {
        if right {
            bins.iter().take_while(|&&bin| value > bin).count()
        } else {
            bins.iter().take_while(|&&bin| value >= bin).count()
        }
    }

    fn class_array(
        dtype: RasterDType,
        ids: Vec<u32>,
        bands: usize,
        height: usize,
        width: usize,
    ) -> PyResult<RasterArray> {
        let shape = [bands, height, width];
        let data = match dtype {
            RasterDType::UInt8 => RasterData::U8(ids.into_iter().map(|id| id as u8).collect()),
            RasterDType::UInt16 => RasterData::U16(ids.into_iter().map(|id| id as u16).collect()),
            RasterDType::Int16 => RasterData::I16(ids.into_iter().map(|id| id as i16).collect()),
            RasterDType::UInt32 => RasterData::U32(ids),
            RasterDType::Int32 => RasterData::I32(ids.into_iter().map(|id| id as i32).collect()),
            RasterDType::Float32 | RasterDType::Float64 => unreachable!(),
        };
        RasterArray::new(data, &shape).map_err(Into::into)
    }

    fn output_info(
        source: &RasterInfo,
        array: &RasterArray,
        nodata: Vec<Option<f64>>,
    ) -> RasterInfo {
        let mut info = source.clone();
        info.width = array.width as u32;
        info.height = array.height as u32;
        info.band_count = array.bands as u16;
        info.dtype_per_band = vec![array.dtype().name().to_string(); array.bands];
        info.nodata_per_band = nodata;
        info
    }

    fn result_to_py(
        py: Python<'_>,
        array: &RasterArray,
        info: &RasterInfo,
        method: &str,
        stats: &Stats,
        class_table: Option<PyObject>,
    ) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        dict.set_item("array", super::super::raster_array_to_py(py, array)?)?;
        dict.set_item("info", super::super::raster_info_to_py_dict(py, info)?)?;
        dict.set_item("method", method)?;
        dict.set_item("valid_count", stats.valid_count)?;
        dict.set_item("nodata_count", stats.nodata_count)?;
        dict.set_item("min", stats.min)?;
        dict.set_item("max", stats.max)?;
        dict.set_item("mean", stats.mean)?;
        dict.set_item("std", stats.std)?;
        match class_table {
            Some(table) => dict.set_item("class_table", table)?,
            None => dict.set_item("class_table", py.None())?,
        }
        dict.set_item(
            "warnings",
            super::super::warnings_to_py(py, &info.warnings)?,
        )?;
        Ok(dict.into_py(py))
    }

    fn class_table_py(
        py: Python<'_>,
        bins: &[f64],
        labels: &[String],
        right: bool,
        counts: &[usize],
    ) -> PyResult<PyObject> {
        let mut rows = Vec::with_capacity(labels.len() + 1);
        rows.push(class_row_py(
            py, 0, "nodata", None, None, false, counts[0], true,
        )?);
        for class_id in 1..=labels.len() {
            let left = if class_id == 1 {
                None
            } else {
                Some(bins[class_id - 2])
            };
            let right_bound = bins.get(class_id - 1).copied();
            rows.push(class_row_py(
                py,
                class_id,
                &labels[class_id - 1],
                left,
                right_bound,
                right,
                counts[class_id],
                false,
            )?);
        }
        Ok(PyList::new_bound(py, rows).into_py(py))
    }

    fn class_row_py(
        py: Python<'_>,
        class_id: usize,
        label: &str,
        left: Option<f64>,
        right: Option<f64>,
        right_inclusive: bool,
        count: usize,
        nodata: bool,
    ) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        dict.set_item("class_id", class_id)?;
        dict.set_item("label", label)?;
        dict.set_item("left", left)?;
        dict.set_item("right", right)?;
        dict.set_item("right_inclusive", right_inclusive)?;
        dict.set_item("count", count)?;
        dict.set_item("nodata", nodata)?;
        Ok(dict.into_py(py))
    }

    fn required_u32(dict: &Bound<'_, PyDict>, key: &'static str) -> PyResult<u32> {
        dict.get_item(key)?
            .ok_or_else(|| {
                GisError::InvalidArgument(format!("{INVALID_ARGUMENT}: raster info missing {key}"))
            })?
            .extract()
            .map_err(Into::into)
    }

    fn optional_u16(value: Option<Bound<'_, PyAny>>) -> PyResult<Option<u16>> {
        optional_extract(value)
    }

    fn optional_string(value: Option<Bound<'_, PyAny>>) -> PyResult<Option<String>> {
        optional_extract(value)
    }

    fn optional_tuple2(value: Option<Bound<'_, PyAny>>) -> PyResult<Option<(f64, f64)>> {
        optional_extract(value)
    }

    fn optional_tuple4(value: Option<Bound<'_, PyAny>>) -> PyResult<Option<(f64, f64, f64, f64)>> {
        optional_extract(value)
    }

    fn optional_tuple6(
        value: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Option<(f64, f64, f64, f64, f64, f64)>> {
        optional_extract(value)
    }

    fn optional_extract<T>(value: Option<Bound<'_, PyAny>>) -> PyResult<Option<T>>
    where
        T: for<'py> FromPyObject<'py>,
    {
        value
            .map(|value| {
                if value.is_none() {
                    Ok(None)
                } else {
                    value.extract::<T>().map(Some)
                }
            })
            .transpose()
            .map(Option::flatten)
    }

    fn warnings_from_py(value: Option<Bound<'_, PyAny>>) -> PyResult<Vec<RasterWarning>> {
        let Some(value) = value else {
            return Ok(Vec::new());
        };
        if value.is_none() {
            return Ok(Vec::new());
        }
        let Ok(list) = value.downcast::<PyList>() else {
            return Ok(Vec::new());
        };
        let mut warnings = Vec::with_capacity(list.len());
        for item in list.iter() {
            let Ok(dict) = item.downcast::<PyDict>() else {
                continue;
            };
            let Some(code) = optional_string(dict.get_item("code")?)? else {
                continue;
            };
            warnings.push(RasterWarning {
                code,
                message: optional_string(dict.get_item("message")?)?.unwrap_or_default(),
                field: optional_string(dict.get_item("field")?)?,
            });
        }
        Ok(warnings)
    }
}
