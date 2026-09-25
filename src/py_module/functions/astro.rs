use super::*;

pub(super) fn register_astro_py_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(
        crate::py_functions::astro::astro_body_position,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::py_functions::astro::astro_body_position_refracted,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::py_functions::astro::astro_moon_phase,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::py_functions::astro::astro_moon_phase_at,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::py_functions::astro::astro_validation_metrics,
        m
    )?)?;
    Ok(())
}
