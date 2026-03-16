use super::*;

pub(super) fn register_io_import_py_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(crate::io::obj_read::io_import_obj_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::io::obj_write::io_export_obj_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::io::stl_write::io_export_stl_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        crate::io::gltf_read::io_import_gltf_py,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(
        crate::import::osm_buildings::import_osm_buildings_extrude_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::import::osm_buildings::import_osm_buildings_from_geojson_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::import::osm_buildings::infer_roof_type_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::import::building_materials::material_from_tags_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::import::building_materials::material_from_name_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::import::cityjson::parse_cityjson_py,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(crate::uv::unwrap::uv_planar_unwrap_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        crate::uv::unwrap::uv_spherical_unwrap_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::converters::multipolygonz_to_obj::converters_multipolygonz_to_obj_py,
        m
    )?)?;
    Ok(())
}
