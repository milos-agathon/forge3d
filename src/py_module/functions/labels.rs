use super::*;

pub(super) fn register_labels_py_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(declutter_optimal_py, m)?)?;
    m.add_function(wrap_pyfunction!(shape_text_py, m)?)?;
    Ok(())
}
