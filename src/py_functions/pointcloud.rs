use super::super::*;

/// Read a LAZ/LAS file and return (point_count, first_3_coords, has_rgb).
///
/// `first_3_coords` contains up to 3 XYZ triples from the first points.
/// Uses the `las` crate's built-in LAZ decompression.
#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(name = "read_laz_points_info")]
pub(crate) fn read_laz_points_info_py(path: &str) -> PyResult<(u64, Vec<f64>, bool)> {
    use las::{Read as LasRead, Reader};

    let mut reader = Reader::from_path(path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{}", e)))?;

    let point_count = reader.header().number_of_points();
    let mut coords: Vec<f64> = Vec::with_capacity(9);
    let mut has_rgb = false;

    for result in reader.points().take(3) {
        let pt = result.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{}", e)))?;
        coords.push(pt.x);
        coords.push(pt.y);
        coords.push(pt.z);
        if pt.color.is_some() {
            has_rgb = true;
        }
    }

    Ok((point_count, coords, has_rgb))
}

/// Read a LAZ/LAS file and return sample point attributes.
///
/// This keeps the legacy `read_laz_points_info` tuple stable while exposing
/// classification/intensity samples for point cloud pipelines that need them.
#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(name = "read_laz_point_attributes")]
#[pyo3(signature = (path, sample_count = 3))]
pub(crate) fn read_laz_point_attributes_py(
    py: Python<'_>,
    path: &str,
    sample_count: usize,
) -> PyResult<Py<PyDict>> {
    use las::{Read as LasRead, Reader};

    let mut reader = Reader::from_path(path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{}", e)))?;

    let point_count = reader.header().number_of_points();
    let mut coords: Vec<f64> = Vec::with_capacity(sample_count.saturating_mul(3));
    let mut intensities: Vec<u16> = Vec::with_capacity(sample_count);
    let mut classifications: Vec<u8> = Vec::with_capacity(sample_count);
    let mut has_rgb = false;

    for result in reader.points().take(sample_count) {
        let pt = result.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{}", e)))?;
        coords.push(pt.x);
        coords.push(pt.y);
        coords.push(pt.z);
        intensities.push(pt.intensity);
        classifications.push(u8::from(pt.classification));
        if pt.color.is_some() {
            has_rgb = true;
        }
    }

    let d = PyDict::new_bound(py);
    d.set_item("point_count", point_count)?;
    d.set_item("coords", coords)?;
    d.set_item("has_rgb", has_rgb)?;
    d.set_item("intensities", intensities)?;
    d.set_item("classifications", classifications)?;
    Ok(d.into())
}

/// Read points for one COPC octree node using the native Rust decoder.
#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(name = "copc_read_node_points")]
#[pyo3(signature = (path, depth = 0, x = 0, y = 0, z = 0, budget = None))]
pub(crate) fn copc_read_node_points_py(
    py: Python<'_>,
    path: &str,
    depth: u32,
    x: u32,
    y: u32,
    z: u32,
    budget: Option<usize>,
) -> PyResult<Py<PyDict>> {
    let dataset = crate::pointcloud::CopcDataset::open(path)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
    let key = crate::pointcloud::OctreeKey::new(depth, x, y, z);
    let mut data = dataset
        .read_points(&key)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;

    let point_count = data.positions.len() / 3;
    if let Some(limit) = budget {
        if limit < point_count {
            data.positions.truncate(limit * 3);
            if let Some(colors) = data.colors.as_mut() {
                colors.truncate(limit * 3);
            }
            if let Some(intensities) = data.intensities.as_mut() {
                intensities.truncate(limit);
            }
            if let Some(classifications) = data.classifications.as_mut() {
                classifications.truncate(limit);
            }
        }
    }

    let d = PyDict::new_bound(py);
    d.set_item("positions", PyArray1::from_vec_bound(py, data.positions))?;
    d.set_item(
        "colors",
        data.colors
            .map(|values| PyArray1::from_vec_bound(py, values).into_py(py))
            .unwrap_or_else(|| py.None()),
    )?;
    d.set_item(
        "intensities",
        data.intensities
            .map(|values| PyArray1::from_vec_bound(py, values).into_py(py))
            .unwrap_or_else(|| py.None()),
    )?;
    d.set_item(
        "classifications",
        data.classifications
            .map(|values| PyArray1::from_vec_bound(py, values).into_py(py))
            .unwrap_or_else(|| py.None()),
    )?;
    Ok(d.into())
}

/// Return whether the `copc_laz` Cargo feature is enabled.
#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(name = "copc_laz_enabled")]
pub(crate) fn copc_laz_enabled_py() -> bool {
    cfg!(feature = "copc_laz")
}
