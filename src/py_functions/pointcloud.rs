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

/// Return whether the `copc_laz` Cargo feature is enabled.
#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(name = "copc_laz_enabled")]
pub(crate) fn copc_laz_enabled_py() -> bool {
    cfg!(feature = "copc_laz")
}
