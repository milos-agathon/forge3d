use super::*;

#[pyfunction]
pub(crate) fn set_point_shape_mode(mode: u32) -> PyResult<()> {
    crate::vector::point::set_global_shape_mode(mode);
    Ok(())
}

#[cfg(feature = "extension-module")]
#[pyfunction]
pub(crate) fn set_point_lod_threshold(threshold: f32) -> PyResult<()> {
    crate::vector::point::set_global_lod_threshold(threshold);
    Ok(())
}

#[cfg(feature = "extension-module")]
#[pyfunction]
pub(crate) fn is_weighted_oit_available() -> PyResult<bool> {
    Ok(crate::vector::oit::is_weighted_oit_enabled())
}
