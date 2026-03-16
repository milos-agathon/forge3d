use super::super::*;

#[pyfunction]
pub(crate) fn configure_csm(
    cascade_count: u32,
    shadow_map_size: u32,
    max_shadow_distance: f32,
    pcf_kernel_size: u32,
    depth_bias: f32,
    slope_bias: f32,
    peter_panning_offset: f32,
    enable_evsm: bool,
    debug_mode: u32,
) -> PyResult<()> {
    let config = CpuCsmConfig::new(
        cascade_count,
        shadow_map_size,
        max_shadow_distance,
        pcf_kernel_size,
        depth_bias,
        slope_bias,
        peter_panning_offset,
        enable_evsm,
        debug_mode,
    )
    .map_err(PyValueError::new_err)?;

    let mut state = GLOBAL_CSM_STATE.lock().expect("csm state poisoned");
    state.apply_config(config).map_err(PyValueError::new_err)?;
    Ok(())
}

// -------------------------
// C1: Engine info (context)
// -------------------------
#[cfg(feature = "extension-module")]
#[pyfunction]
pub(crate) fn set_csm_enabled(enabled: bool) -> PyResult<()> {
    let mut state = GLOBAL_CSM_STATE.lock().expect("csm state poisoned");
    state.set_enabled(enabled);
    Ok(())
}

#[cfg(feature = "extension-module")]
#[pyfunction]
pub(crate) fn set_csm_light_direction(direction: (f32, f32, f32)) -> PyResult<()> {
    let mut state = GLOBAL_CSM_STATE.lock().expect("csm state poisoned");
    state.set_light_direction([direction.0, direction.1, direction.2]);
    Ok(())
}

#[cfg(feature = "extension-module")]
#[pyfunction]
pub(crate) fn set_csm_pcf_kernel(kernel_size: u32) -> PyResult<()> {
    let mut state = GLOBAL_CSM_STATE.lock().expect("csm state poisoned");
    state
        .set_pcf_kernel(kernel_size)
        .map_err(PyValueError::new_err)?;
    Ok(())
}

#[cfg(feature = "extension-module")]
#[pyfunction]
pub(crate) fn set_csm_bias_params(
    depth_bias: f32,
    slope_bias: f32,
    peter_panning_offset: f32,
) -> PyResult<()> {
    let mut state = GLOBAL_CSM_STATE.lock().expect("csm state poisoned");
    state
        .set_bias_params(depth_bias, slope_bias, peter_panning_offset)
        .map_err(PyValueError::new_err)?;
    Ok(())
}

#[cfg(feature = "extension-module")]
#[pyfunction]
pub(crate) fn set_csm_debug_mode(mode: u32) -> PyResult<()> {
    let mut state = GLOBAL_CSM_STATE.lock().expect("csm state poisoned");
    state.set_debug_mode(mode).map_err(PyValueError::new_err)?;
    Ok(())
}

#[cfg(feature = "extension-module")]
#[pyfunction]
pub(crate) fn get_csm_cascade_info() -> PyResult<Vec<(f32, f32, f32)>> {
    let state = GLOBAL_CSM_STATE.lock().expect("csm state poisoned");
    Ok(state.cascade_info())
}

#[cfg(feature = "extension-module")]
#[pyfunction]
pub(crate) fn validate_csm_peter_panning() -> PyResult<bool> {
    let state = GLOBAL_CSM_STATE.lock().expect("csm state poisoned");
    Ok(state.validate_peter_panning())
}
