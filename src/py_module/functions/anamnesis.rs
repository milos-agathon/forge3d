use super::*;

#[pyfunction]
pub(crate) fn anamnesis_leaf_key(content: Vec<u8>) -> String {
    crate::core::anamnesis::leaf_key(&content).to_hex()
}

#[pyfunction]
pub(crate) fn anamnesis_pass_key(
    label: &str,
    pipeline_descriptor: Vec<u8>,
    uniform_bytes: Vec<u8>,
    input_keys: Vec<String>,
    capability_fingerprint: Vec<u8>,
    engine_fingerprint: Vec<u8>,
) -> PyResult<String> {
    let inputs = input_keys
        .iter()
        .map(|value| {
            crate::core::anamnesis::PassKey::from_hex(value).map_err(PyValueError::new_err)
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok(crate::core::anamnesis::pass_key(
        label,
        &pipeline_descriptor,
        &uniform_bytes,
        &inputs,
        &capability_fingerprint,
        &engine_fingerprint,
    )
    .0
    .to_hex())
}

#[pyfunction]
pub(crate) fn anamnesis_engine_fingerprint() -> PyResult<String> {
    serde_json::to_string(&crate::core::anamnesis::EngineFingerprint::current())
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))
}

pub(super) fn register_anamnesis_py_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(anamnesis_leaf_key, m)?)?;
    m.add_function(wrap_pyfunction!(anamnesis_pass_key, m)?)?;
    m.add_function(wrap_pyfunction!(anamnesis_engine_fingerprint, m)?)?;
    Ok(())
}
