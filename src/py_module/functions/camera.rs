use super::*;

pub(super) fn register_camera_py_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(crate::camera::camera_look_at, m)?)?;
    m.add_function(wrap_pyfunction!(crate::camera::camera_perspective, m)?)?;
    m.add_function(wrap_pyfunction!(crate::camera::camera_orthographic, m)?)?;
    m.add_function(wrap_pyfunction!(crate::camera::camera_view_proj, m)?)?;
    m.add_function(wrap_pyfunction!(crate::camera::camera_dof_params, m)?)?;
    m.add_function(wrap_pyfunction!(
        crate::camera::camera_f_stop_to_aperture,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::camera::camera_aperture_to_f_stop,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::camera::camera_hyperfocal_distance,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::camera::camera_depth_of_field_range,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::camera::camera_circle_of_confusion,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(crate::geometry::transforms::translate, m)?)?;
    m.add_function(wrap_pyfunction!(crate::geometry::transforms::rotate_x, m)?)?;
    m.add_function(wrap_pyfunction!(crate::geometry::transforms::rotate_y, m)?)?;
    m.add_function(wrap_pyfunction!(crate::geometry::transforms::rotate_z, m)?)?;
    m.add_function(wrap_pyfunction!(crate::geometry::transforms::scale, m)?)?;
    m.add_function(wrap_pyfunction!(
        crate::geometry::transforms::scale_uniform,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::geometry::transforms::compose_trs,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::geometry::transforms::look_at_transform,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::geometry::transforms::multiply_matrices,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::geometry::transforms::invert_matrix,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::geometry::transforms::compute_normal_matrix,
        m
    )?)?;
    Ok(())
}
