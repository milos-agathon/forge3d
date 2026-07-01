use super::*;

#[cfg(feature = "extension-module")]
pub(crate) fn register_gis_py_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(crate::gis::read_raster_info_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::gis::read_raster_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::gis::write_raster_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::gis::read_vector_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::gis::reproject_vector_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::gis::geometry_type_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::gis::vector_schema_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::gis::feature_count_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::gis::vector_crs_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::gis::vector_bounds_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::gis::validate_geometry_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::gis::repair_geometry_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::gis::geometry_measure_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::gis::geometry_centroid_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::gis::representative_point_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::gis::interpolate_line_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::gis::parse_crs_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::gis::inspect_crs_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::gis::raster_crs_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::gis::create_crs_transformer_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::gis::transform_bounds_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::gis::raster_transform_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::gis::transform_from_origin_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::gis::transform_from_bounds_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::gis::array_bounds_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::gis::raster_bounds_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::gis::raster_resolution_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::gis::validate_transform_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::gis::pixel_convention_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::gis::rowcol_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::gis::xy_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::gis::index_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::gis::bounds_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::gis::assign_crs_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::gis::apply_nodata_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::gis::read_raster_mask_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::gis::window_from_bounds_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::gis::read_raster_window_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::gis::window_transform_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::gis::assert_grid_compatible_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::gis::resample_raster_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::gis::align_raster_grid_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::gis::align_raster_to_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::gis::reproject_raster_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        crate::gis::calculate_default_transform_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(crate::gis::web_mercator_bounds_py, m)?)?;
    Ok(())
}
