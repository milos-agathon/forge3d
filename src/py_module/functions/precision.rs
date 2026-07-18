use super::*;

#[cfg(feature = "extension-module")]
pub(crate) fn register_precision_py_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(crate::py_functions::dd_selftest, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_functions::dd_harness, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_functions::dd_jitter_demo, m)?)?;
    Ok(())
}
