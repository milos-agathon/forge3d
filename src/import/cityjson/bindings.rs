use pyo3::{exceptions::PyValueError, prelude::*, types::PyBytes};

use super::parse_cityjson;

/// P4.3: Python binding for CityJSON parsing
#[pyfunction]
pub fn parse_cityjson_py(data: &Bound<'_, PyBytes>) -> PyResult<PyObject> {
    let bytes = data.as_bytes();
    let (buildings, meta) =
        parse_cityjson(bytes).map_err(|e| PyValueError::new_err(e.to_string()))?;

    Python::with_gil(|py| {
        let result = pyo3::types::PyDict::new_bound(py);
        let meta_dict = pyo3::types::PyDict::new_bound(py);
        meta_dict.set_item("version", &meta.version)?;
        meta_dict.set_item("crs_epsg", meta.crs_epsg)?;
        meta_dict.set_item("scale", (meta.scale[0], meta.scale[1], meta.scale[2]))?;
        meta_dict.set_item(
            "translate",
            (meta.translate[0], meta.translate[1], meta.translate[2]),
        )?;
        if let Some(ext) = &meta.extent {
            meta_dict.set_item("extent", (ext[0], ext[1], ext[2], ext[3], ext[4], ext[5]))?;
        }
        result.set_item("metadata", meta_dict)?;

        let buildings_list = pyo3::types::PyList::empty_bound(py);
        for building in &buildings {
            let bdict = pyo3::types::PyDict::new_bound(py);
            bdict.set_item("id", &building.id)?;
            bdict.set_item("vertex_count", building.vertex_count())?;
            bdict.set_item("triangle_count", building.triangle_count())?;
            bdict.set_item("lod", building.lod)?;
            bdict.set_item("height", building.height)?;
            bdict.set_item("ground_height", building.ground_height)?;
            bdict.set_item(
                "roof_type",
                format!("{:?}", building.roof_type).to_lowercase(),
            )?;

            let mat_dict = pyo3::types::PyDict::new_bound(py);
            mat_dict.set_item(
                "albedo",
                (
                    building.material.albedo[0],
                    building.material.albedo[1],
                    building.material.albedo[2],
                ),
            )?;
            mat_dict.set_item("roughness", building.material.roughness)?;
            mat_dict.set_item("metallic", building.material.metallic)?;
            bdict.set_item("material", mat_dict)?;

            bdict.set_item("positions", building.positions.clone())?;
            bdict.set_item("indices", building.indices.clone())?;
            if let Some(ref normals) = building.normals {
                bdict.set_item("normals", normals.clone())?;
            }

            buildings_list.append(bdict)?;
        }
        result.set_item("buildings", buildings_list)?;
        result.set_item("building_count", buildings.len())?;

        Ok(result.into())
    })
}
