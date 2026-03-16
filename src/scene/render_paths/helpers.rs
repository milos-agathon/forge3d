fn render_targets(
    scene: &Scene,
) -> (
    &wgpu::TextureView,
    Option<&wgpu::TextureView>,
    &wgpu::TextureView,
    Option<&wgpu::TextureView>,
) {
    let (target_view, resolve_target) = if scene.sample_count > 1 {
        (
            scene
                .msaa_view
                .as_ref()
                .expect("MSAA view missing when sample_count > 1"),
            Some(&scene.color_view),
        )
    } else {
        (&scene.color_view, None)
    };
    let (normal_target, normal_resolve) = if scene.sample_count > 1 {
        (
            scene
                .msaa_normal_view
                .as_ref()
                .expect("MSAA normal view missing when sample_count > 1"),
            Some(&scene.normal_view),
        )
    } else {
        (&scene.normal_view, None)
    };
    (target_view, resolve_target, normal_target, normal_resolve)
}

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
