//! Python boundary for the unconditional F3DZ core.

use super::super::*;
use numpy::{PyArray2, PyReadonlyArray2};

fn py_codec_error(error: crate::codec::f3dz::F3dzError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

#[pyfunction]
#[pyo3(signature = (dem, eps, progressive=true))]
pub(crate) fn compress_dem<'py>(
    py: Python<'py>,
    dem: PyReadonlyArray2<'py, f32>,
    eps: f32,
    progressive: bool,
) -> PyResult<Bound<'py, PyBytes>> {
    let array = dem.as_array();
    let height = u32::try_from(array.nrows())
        .map_err(|_| PyValueError::new_err("DEM height exceeds u32"))?;
    let width =
        u32::try_from(array.ncols()).map_err(|_| PyValueError::new_err("DEM width exceeds u32"))?;
    let values: Vec<f32> = array.iter().copied().collect();
    let mut options = crate::codec::f3dz::EncodeOptions::new(eps);
    options.progressive = progressive;
    let encoded =
        crate::codec::f3dz::encode_dem(&values, width, height, &options).map_err(py_codec_error)?;
    Ok(PyBytes::new_bound(py, &encoded))
}

#[pyfunction]
#[pyo3(signature = (data))]
pub(crate) fn decompress_dem<'py>(
    py: Python<'py>,
    data: &[u8],
) -> PyResult<(Bound<'py, PyArray2<f32>>, Py<PyDict>)> {
    let decoded = crate::codec::f3dz::decode_dem(data, None).map_err(py_codec_error)?;
    let (_, entries) = crate::codec::f3dz::format::parse_prefix(data).map_err(py_codec_error)?;
    let array = ndarray::Array2::from_shape_vec(
        (decoded.height as usize, decoded.width as usize),
        decoded.values,
    )
    .map_err(|error| PyRuntimeError::new_err(format!("F3DZ array reshape failed: {error}")))?;
    let info = PyDict::new_bound(py);
    info.set_item("codec", "f3dz/1")?;
    info.set_item("width", decoded.width)?;
    info.set_item("height", decoded.height)?;
    info.set_item("eps", decoded.epsilon)?;
    info.set_item(
        "progressive",
        entries.first().is_some_and(|entry| entry.progressive()),
    )?;
    info.set_item("base_quality", decoded.base_quality)?;
    info.set_item("page_count", entries.len())?;
    info.set_item("height_datum", decoded.height_datum)?;
    Ok((PyArray2::from_owned_array_bound(py, array), info.into()))
}

#[pyfunction]
#[pyo3(signature = (data, source=None))]
pub(crate) fn verify_dem<'py>(
    py: Python<'py>,
    data: &[u8],
    source: Option<PyReadonlyArray2<'py, f32>>,
) -> PyResult<Py<PyDict>> {
    // decode_dem is the authoritative fail-closed structural/CRC verifier.
    let decoded = crate::codec::f3dz::decode_dem(data, None).map_err(py_codec_error)?;
    let (header, entries) =
        crate::codec::f3dz::format::parse_prefix(data).map_err(py_codec_error)?;
    let source = source.map(|array| array.as_array().to_owned());
    if let Some(source) = &source {
        if source.shape() != [decoded.height as usize, decoded.width as usize] {
            return Err(PyValueError::new_err(format!(
                "source shape {:?} does not match F3DZ grid ({}, {})",
                source.shape(),
                decoded.height,
                decoded.width
            )));
        }
    }

    let page_reports = pyo3::types::PyList::empty_bound(py);
    let mut all_within = true;
    let mut stored_match = true;
    let mut global_max = 0.0f32;
    for (page, entry) in entries.iter().enumerate() {
        let start_x = entry.page_x as usize * header.tile_size as usize;
        let start_y = entry.page_y as usize * header.tile_size as usize;
        let mut max_error = 0.0f32;
        let mut page_within = true;
        if let Some(source) = &source {
            for y in 0..entry.height as usize {
                for x in 0..entry.width as usize {
                    let index = (start_y + y) * decoded.width as usize + start_x + x;
                    let original = source[(start_y + y, start_x + x)];
                    let reconstructed = decoded.values[index];
                    if original.is_nan() || reconstructed.is_nan() {
                        if !(original.is_nan() && reconstructed.is_nan()) {
                            page_within = false;
                            max_error = f32::INFINITY;
                        }
                    } else {
                        let error = (reconstructed - original).abs();
                        max_error = max_error.max(error);
                        page_within &= error <= decoded.epsilon;
                    }
                }
            }
            all_within &= page_within;
            stored_match &= max_error.to_bits() == entry.max_abs_err.to_bits();
            global_max = global_max.max(max_error);
        }
        let report = PyDict::new_bound(py);
        report.set_item("page", page)?;
        report.set_item("page_x", entry.page_x)?;
        report.set_item("page_y", entry.page_y)?;
        report.set_item("crc32", entry.crc32)?;
        report.set_item("stored_max_abs_err", entry.max_abs_err)?;
        report.set_item("measured_max_abs_err", source.as_ref().map(|_| max_error))?;
        report.set_item("within_epsilon", source.as_ref().map(|_| page_within))?;
        report.set_item(
            "stored_error_matches",
            source
                .as_ref()
                .map(|_| max_error.to_bits() == entry.max_abs_err.to_bits()),
        )?;
        page_reports.append(report)?;
    }

    let result = PyDict::new_bound(py);
    result.set_item("valid", true)?;
    result.set_item("codec", "f3dz/1")?;
    result.set_item("crc_ok", true)?;
    result.set_item("header_consistent", true)?;
    result.set_item("width", decoded.width)?;
    result.set_item("height", decoded.height)?;
    result.set_item("eps", decoded.epsilon)?;
    result.set_item("base_quality", decoded.base_quality)?;
    result.set_item("source_checked", source.is_some())?;
    result.set_item("max_abs_err", source.as_ref().map(|_| global_max))?;
    result.set_item("all_within_epsilon", source.as_ref().map(|_| all_within))?;
    result.set_item("stored_errors_match", source.as_ref().map(|_| stored_match))?;
    result.set_item("pages", page_reports)?;
    Ok(result.into())
}
