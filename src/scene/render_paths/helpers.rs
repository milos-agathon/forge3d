fn reflection_err(error: String) -> pyo3::PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(format!("Reflection rendering failed: {}", error))
}

fn cloud_shadow_err(error: String) -> pyo3::PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(format!("Cloud shadow generation failed: {}", error))
}

fn cloud_render_err(error: String) -> pyo3::PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(format!("Cloud rendering failed: {}", error))
}

fn dof_err(error: String) -> pyo3::PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(format!(
        "Depth-of-field rendering failed: {}",
        error
    ))
}
