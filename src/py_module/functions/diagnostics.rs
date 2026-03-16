use super::*;

pub(super) fn register_diagnostics_py_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(enumerate_adapters, m)?)?;
    m.add_function(wrap_pyfunction!(device_probe, m)?)?;
    m.add_function(wrap_pyfunction!(global_memory_metrics, m)?)?;
    m.add_function(wrap_pyfunction!(render_debug_pattern_frame, m)?)?;
    m.add_function(wrap_pyfunction!(numpy_to_exr, m)?)?;

    m.add_function(wrap_pyfunction!(engine_info, m)?)?;
    m.add_function(wrap_pyfunction!(report_device, m)?)?;
    m.add_function(wrap_pyfunction!(c5_build_framegraph_report, m)?)?;
    m.add_function(wrap_pyfunction!(c6_mt_record_demo, m)?)?;
    m.add_function(wrap_pyfunction!(c7_async_compute_demo, m)?)?;
    Ok(())
}
