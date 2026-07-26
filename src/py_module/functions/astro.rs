use super::*;

pub(super) fn register_astro_py_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(astro_body_position, m)?)?;
    m.add_function(wrap_pyfunction!(astro_moon_phase, m)?)?;
    m.add_function(wrap_pyfunction!(sky_set_observation, m)?)?;
    m.add_function(wrap_pyfunction!(astro_validation_metrics, m)?)?;
    Ok(())
}
