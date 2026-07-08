use super::*;

pub(super) fn register_tiles3d_py_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(decode_b3dm_py, m)?)?;
    m.add_function(wrap_pyfunction!(tiles3d_traverse_py, m)?)?;
    m.add_function(wrap_pyfunction!(decode_pnts_py, m)?)?;
    Ok(())
}
