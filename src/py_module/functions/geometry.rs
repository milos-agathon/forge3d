use super::*;

pub(super) fn register_geometry_py_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(crate::vector::extrude_polygon_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::vector::add_polygons_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::vector::add_lines_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::vector::add_points_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::vector::add_graph_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::vector::clear_vectors_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::vector::get_vector_counts_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        crate::vector::api::extrude_polygon_gpu_py,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(
        crate::geometry::geometry_extrude_polygon_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::geometry::geometry_generate_primitive_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::geometry::geometry_validate_mesh_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(crate::geometry::geometry_weld_mesh_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        crate::geometry::geometry_transform_center_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::geometry::geometry_transform_scale_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::geometry::geometry_transform_flip_axis_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::geometry::geometry_transform_swap_axes_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::geometry::geometry_transform_bounds_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(crate::geometry::geometry_subdivide_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        crate::geometry::geometry_displace_heightmap_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::geometry::geometry_displace_procedural_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::geometry::geometry_generate_ribbon_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::geometry::geometry_generate_tube_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::geometry::geometry_generate_thick_polyline_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::geometry::geometry_generate_tangents_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::geometry::geometry_attach_tangents_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::geometry::geometry_subdivide_adaptive_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::geometry::geometry_simplify_mesh_py,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(
        crate::render::instancing::geometry_instance_mesh_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::render::instancing::gpu_instancing_available_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::render::instancing::geometry_instance_mesh_gpu_stub_py,
        m
    )?)?;
    #[cfg(all(feature = "enable-gpu-instancing"))]
    {
        m.add_function(wrap_pyfunction!(
            crate::render::instancing::geometry_instance_mesh_gpu_py,
            m
        )?)?;
        m.add_function(wrap_pyfunction!(
            crate::render::instancing::geometry_instance_mesh_gpu_render_py,
            m
        )?)?;
    }

    m.add_function(wrap_pyfunction!(crate::geometry::grid::grid_generate, m)?)?;
    Ok(())
}
