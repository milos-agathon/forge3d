use super::*;

#[cfg(feature = "extension-module")]
pub(crate) fn register_gis_py_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(crate::gis::read_raster_info_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::gis::write_raster_py, m)?)?;
    Ok(())
}
