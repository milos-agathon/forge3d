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
    m.add_function(wrap_pyfunction!(crate::gis::is_valid_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::gis::repair_geometry_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::gis::geometry_measure_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::gis::measure_geometries_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::gis::geometry_centroid_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::gis::representative_point_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::gis::interpolate_line_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::gis::union_geometries_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::gis::union_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::gis::intersection_geometries_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::gis::difference_geometries_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        crate::gis::symmetric_difference_geometries_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(crate::gis::dissolve_vector_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::gis::buffer_geometry_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::gis::clip_vector_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::gis::intersect_vectors_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::gis::simplify_geometry_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::gis::load_boundary_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::gis::rasterize_vectors_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::gis::geometry_mask_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::gis::mask_raster_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::gis::normalize_raster_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::gis::classify_raster_py, m)?)?;
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
    m.add_function(wrap_pyfunction!(crate::gis::warped_vrt_info_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::gis::web_mercator_bounds_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::gis::fetch_remote_geodata_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::gis::cache_geodata_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::gis::fetch_vector_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::gis::read_cog_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::gis::slippy_tile_index_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::gis::query_osm_features_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::gis::parse_osm_features_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::gis::load_context_vectors_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::gis::prepare_osm_scene_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::gis::prepare_dem_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        crate::gis::prepare_terrain_derivatives_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(crate::gis::read_gridded_dataset_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::gis::subset_grid_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::gis::decode_terrarium_dem_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::gis::build_terrarium_dem_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        crate::gis::prepare_landcover_raster_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::gis::prepare_population_raster_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::gis::load_building_footprints_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::gis::extract_building_heights_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(crate::gis::estimate_local_utm_py, m)?)?;
    Ok(())
}
