use super::*;

pub(super) fn register_provenance_py_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(seal_provenance, m)?)?;
    m.add_function(wrap_pyfunction!(verify_provenance, m)?)?;
    Ok(())
}
