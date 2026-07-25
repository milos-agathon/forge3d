use super::super::*;

#[cfg(feature = "extension-module")]
#[pyfunction]
/// Run the selected backend's double-float exactness canary.
pub(crate) fn dd_selftest(py: Python<'_>) -> PyResult<PyObject> {
    let report = py
        .allow_threads(crate::core::dd::selftest)
        .map_err(PyErr::from)?;
    let dict = PyDict::new_bound(py);
    dict.set_item("passed", report.passed)?;
    dict.set_item("backend", report.backend)?;
    dict.set_item("adapter", report.adapter)?;
    dict.set_item("two_prod_variant", report.two_prod_variant.as_str())?;
    dict.set_item("shader_label", report.shader_label)?;
    dict.set_item("shader_hash", report.shader_hash)?;
    dict.set_item("canary_count", report.canary_count)?;
    dict.set_item("mismatch_count", report.mismatch_count)?;
    dict.set_item(
        "rejected_variants",
        report
            .rejected_variants
            .iter()
            .map(|variant| variant.as_str())
            .collect::<Vec<_>>(),
    )?;
    dict.set_item("failure_details", report.failure_details)?;
    Ok(dict.into_py(py))
}

#[cfg(feature = "extension-module")]
#[pyfunction(signature = (operation, n=100_000_000))]
/// Prove one double-float operation over generated and adversarial vectors.
pub(crate) fn dd_harness(py: Python<'_>, operation: &str, n: u64) -> PyResult<PyObject> {
    let report = py
        .allow_threads(|| crate::core::dd::harness(operation, n))
        .map_err(PyErr::from)?;
    let dict = PyDict::new_bound(py);
    dict.set_item("operation", report.operation.as_str())?;
    dict.set_item("backend", report.backend)?;
    dict.set_item("adapter", report.adapter)?;
    dict.set_item("two_prod_variant", report.two_prod_variant.as_str())?;
    dict.set_item("shader_label", report.shader_label)?;
    dict.set_item("shader_hash", report.shader_hash)?;
    dict.set_item("generated_count", report.generated_count)?;
    dict.set_item("adversarial_count", report.adversarial_count)?;
    dict.set_item("mismatch_count", report.mismatch_count)?;
    dict.set_item("max_err_u2", report.max_err_u2)?;
    dict.set_item("cited_bound_u2", report.cited_bound_u2)?;
    dict.set_item("certificate_json", report.certificate_json)?;
    Ok(dict.into_py(py))
}

#[cfg(feature = "extension-module")]
#[pyfunction(signature = (frames=1_000))]
/// Run the Everest ECEF absolute-coordinate jitter demonstration.
pub(crate) fn dd_jitter_demo(py: Python<'_>, frames: u32) -> PyResult<PyObject> {
    let report = py
        .allow_threads(|| crate::core::dd::jitter_demo(frames))
        .map_err(PyErr::from)?;
    let dict = PyDict::new_bound(py);
    dict.set_item("dd_errors_px", report.dd_errors_px)?;
    dict.set_item("f32_errors_px", report.f32_errors_px)?;
    dict.set_item("dd_max_error_px", report.dd_max_error_px)?;
    dict.set_item("f32_max_error_px", report.f32_max_error_px)?;
    dict.set_item("raw_over_one_px", report.raw_over_one_px)?;
    dict.set_item("dd_hash_a", report.dd_hash_a)?;
    dict.set_item("dd_hash_b", report.dd_hash_b)?;
    dict.set_item("backend", report.backend)?;
    dict.set_item("shader_label", report.shader_label)?;
    dict.set_item("certificate_json", report.certificate_json)?;
    Ok(dict.into_py(py))
}
