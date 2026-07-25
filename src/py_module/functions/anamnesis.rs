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
    input_keys: Vec<(String, String)>,
    capability_fingerprint: Vec<u8>,
    engine_fingerprint: Vec<u8>,
) -> PyResult<String> {
    let inputs = input_keys
        .iter()
        .map(|(binding, value)| {
            crate::core::anamnesis::PassKey::from_hex(value)
                .map(|key| crate::core::anamnesis::InputKey::new(binding, key))
                .map_err(PyValueError::new_err)
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

#[pyfunction]
pub(crate) fn anamnesis_store_verify(
    py: Python<'_>,
    root: std::path::PathBuf,
    max_bytes: u64,
) -> PyResult<PyObject> {
    let store = crate::core::anamnesis::ContentStore::new(root, max_bytes.max(1), true)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    let report = store
        .verify()
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    let result = PyDict::new_bound(py);
    result.set_item("valid", report.valid)?;
    result.set_item("quarantined", report.quarantined)?;
    result.set_item("bytes_checked", report.bytes_checked)?;
    Ok(result.into())
}

#[pyfunction]
pub(crate) fn anamnesis_store_gc(root: std::path::PathBuf, max_bytes: u64) -> PyResult<u64> {
    let store = crate::core::anamnesis::ContentStore::new(&root, u64::MAX, false)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    store
        .gc(max_bytes)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))
}

#[pyfunction]
pub(crate) fn anamnesis_store_put_leaf(
    root: std::path::PathBuf,
    blob: Vec<u8>,
    label: &str,
    max_bytes: u64,
) -> PyResult<String> {
    let store = crate::core::anamnesis::ContentStore::new(root, max_bytes.max(1), true)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    let key = crate::core::anamnesis::leaf_key(&blob);
    store
        .put_leaf(key, &blob, label)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    Ok(key.to_hex())
}

#[pyfunction]
pub(crate) fn anamnesis_store_get(
    py: Python<'_>,
    root: std::path::PathBuf,
    key: &str,
    max_bytes: u64,
) -> PyResult<Option<PyObject>> {
    let key = crate::core::anamnesis::PassKey::from_hex(key).map_err(PyValueError::new_err)?;
    let store = crate::core::anamnesis::ContentStore::new(root, max_bytes.max(1), true)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    Ok(store
        .get(key)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?
        .map(|(blob, _)| PyBytes::new_bound(py, &blob).into()))
}

#[pyfunction]
pub(crate) fn anamnesis_restore_rgba8(
    py: Python<'_>,
    blob: Vec<u8>,
    width: u32,
    height: u32,
) -> PyResult<PyObject> {
    if width == 0 || height == 0 || blob.len() != width as usize * height as usize * 4 {
        return Err(PyValueError::new_err(
            "ANAMNESIS RGBA8 restore requires width*height*4 tightly packed bytes",
        ));
    }
    let ctx = crate::core::gpu::try_ctx()?;
    let texture = crate::core::resource_tracker::tracked_create_texture(
        ctx.device.as_ref(),
        &wgpu::TextureDescriptor {
            label: Some("anamnesis.portable.restore.rgba8"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        },
    )?;
    ctx.queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &blob,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(width * 4),
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    let restored = crate::read_texture_tight(
        ctx.device.as_ref(),
        ctx.queue.as_ref(),
        &texture,
        (width, height),
        wgpu::TextureFormat::Rgba8Unorm,
    )
    .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    Ok(PyBytes::new_bound(py, &restored).into())
}

pub(super) fn register_anamnesis_py_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(anamnesis_leaf_key, m)?)?;
    m.add_function(wrap_pyfunction!(anamnesis_pass_key, m)?)?;
    m.add_function(wrap_pyfunction!(anamnesis_engine_fingerprint, m)?)?;
    m.add_function(wrap_pyfunction!(anamnesis_store_verify, m)?)?;
    m.add_function(wrap_pyfunction!(anamnesis_store_gc, m)?)?;
    m.add_function(wrap_pyfunction!(anamnesis_store_put_leaf, m)?)?;
    m.add_function(wrap_pyfunction!(anamnesis_store_get, m)?)?;
    m.add_function(wrap_pyfunction!(anamnesis_restore_rgba8, m)?)?;
    Ok(())
}
