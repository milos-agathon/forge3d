use super::super::*;

#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(name = "decode_b3dm_py")]
pub(crate) fn decode_b3dm_py(py: Python<'_>, data: &[u8]) -> PyResult<PyObject> {
    let payload = crate::tiles3d::decode_b3dm(data)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let d = PyDict::new_bound(py);

    d.set_item("version", payload.header.version)?;
    d.set_item("byte_length", payload.header.byte_length)?;
    d.set_item(
        "feature_table",
        serde_json::to_string(&payload.feature_table).unwrap_or_default(),
    )?;
    d.set_item(
        "batch_table",
        payload
            .batch_table
            .as_ref()
            .map(serde_json::to_string)
            .transpose()
            .unwrap_or_default(),
    )?;
    d.set_item("vertex_count", payload.vertex_count())?;
    d.set_item("triangle_count", payload.triangle_count())?;
    d.set_item("positions", PyArray1::from_vec_bound(py, payload.positions))?;
    d.set_item(
        "normals",
        payload
            .normals
            .map(|values| PyArray1::from_vec_bound(py, values).into_py(py))
            .unwrap_or_else(|| py.None()),
    )?;
    d.set_item(
        "colors",
        payload
            .colors
            .map(|values| PyArray1::from_vec_bound(py, values).into_py(py))
            .unwrap_or_else(|| py.None()),
    )?;
    d.set_item("indices", PyArray1::from_vec_bound(py, payload.indices))?;
    d.set_item(
        "batch_ids",
        payload
            .batch_ids
            .map(|values| PyArray1::from_vec_bound(py, values).into_py(py))
            .unwrap_or_else(|| py.None()),
    )?;

    Ok(d.into())
}

#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(name = "tiles3d_traverse_py")]
#[pyo3(signature = (path, camera_position, sse_threshold = 16.0, max_depth = 32))]
pub(crate) fn tiles3d_traverse_py(
    py: Python<'_>,
    path: &str,
    camera_position: (f64, f64, f64),
    sse_threshold: f32,
    max_depth: usize,
) -> PyResult<Vec<PyObject>> {
    let tileset = crate::tiles3d::Tileset::load(path)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let traverser = crate::tiles3d::TilesetTraverser::new(sse_threshold).with_max_depth(max_depth);
    let camera = glam::DVec3::new(camera_position.0, camera_position.1, camera_position.2);

    let mut out = Vec::new();
    for tile in traverser.visible_tiles(&tileset, camera, None) {
        let Some(uri) = tile.tile.content_uri() else {
            continue;
        };
        let d = PyDict::new_bound(py);
        d.set_item("uri", uri)?;
        d.set_item(
            "resolved_path",
            tileset.resolve_uri(uri).to_string_lossy().to_string(),
        )?;
        d.set_item("sse", tile.sse)?;
        d.set_item("depth", tile.depth)?;
        out.push(d.into());
    }
    Ok(out)
}

#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(name = "decode_pnts_py")]
pub(crate) fn decode_pnts_py(py: Python<'_>, data: &[u8]) -> PyResult<PyObject> {
    let payload = crate::tiles3d::decode_pnts(data)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let d = PyDict::new_bound(py);

    d.set_item("version", payload.header.version)?;
    d.set_item("byte_length", payload.header.byte_length)?;
    d.set_item("point_count", payload.point_count())?;
    d.set_item(
        "feature_table",
        serde_json::to_string(&payload.feature_table).unwrap_or_default(),
    )?;
    d.set_item(
        "batch_table",
        payload
            .batch_table
            .as_ref()
            .map(serde_json::to_string)
            .transpose()
            .unwrap_or_default(),
    )?;
    d.set_item("positions", PyArray1::from_vec_bound(py, payload.positions))?;
    d.set_item(
        "colors",
        payload
            .colors
            .map(|values| PyArray1::from_vec_bound(py, values).into_py(py))
            .unwrap_or_else(|| py.None()),
    )?;
    d.set_item(
        "colors_rgba",
        payload
            .colors_rgba
            .map(|values| PyArray1::from_vec_bound(py, values).into_py(py))
            .unwrap_or_else(|| py.None()),
    )?;
    d.set_item(
        "normals",
        payload
            .normals
            .map(|values| PyArray1::from_vec_bound(py, values).into_py(py))
            .unwrap_or_else(|| py.None()),
    )?;
    d.set_item(
        "batch_ids",
        payload
            .batch_ids
            .map(|values| PyArray1::from_vec_bound(py, values).into_py(py))
            .unwrap_or_else(|| py.None()),
    )?;
    Ok(d.into())
}
