use super::*;

pub(super) fn register_interactive_py_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(open_viewer, m)?)?;
    m.add_function(wrap_pyfunction!(open_terrain_viewer, m)?)?;
    m.add_function(wrap_pyfunction!(run_interactive_viewer_cli, m)?)?;
    m.add_function(wrap_pyfunction!(set_point_shape_mode, m)?)?;
    m.add_function(wrap_pyfunction!(set_point_lod_threshold, m)?)?;
    m.add_function(wrap_pyfunction!(is_weighted_oit_available, m)?)?;
    m.add_function(wrap_pyfunction!(vector_oit_and_pick_demo, m)?)?;
    m.add_function(wrap_pyfunction!(vector_render_oit_py, m)?)?;
    m.add_function(wrap_pyfunction!(vector_render_pick_map_py, m)?)?;
    m.add_function(wrap_pyfunction!(vector_render_oit_and_pick_py, m)?)?;
    m.add_function(wrap_pyfunction!(vector_render_polygons_fill_py, m)?)?;
    m.add_function(wrap_pyfunction!(configure_csm, m)?)?;
    m.add_function(wrap_pyfunction!(set_csm_enabled, m)?)?;
    m.add_function(wrap_pyfunction!(set_csm_light_direction, m)?)?;
    m.add_function(wrap_pyfunction!(set_csm_pcf_kernel, m)?)?;
    m.add_function(wrap_pyfunction!(set_csm_bias_params, m)?)?;
    m.add_function(wrap_pyfunction!(set_csm_debug_mode, m)?)?;
    m.add_function(wrap_pyfunction!(get_csm_cascade_info, m)?)?;
    m.add_function(wrap_pyfunction!(validate_csm_peter_panning, m)?)?;
    m.add_function(wrap_pyfunction!(_pt_render_gpu_mesh, m)?)?;
    m.add_function(wrap_pyfunction!(hybrid_render, m)?)?;
    Ok(())
}
