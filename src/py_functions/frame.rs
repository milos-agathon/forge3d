use super::super::*;
use crate::py_types::frame::Frame;
#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(signature = (width, height, certificate=None))]
pub(crate) fn render_debug_pattern_frame(
    py: Python<'_>,
    width: u32,
    height: u32,
    certificate: Option<Bound<'_, PyAny>>,
) -> PyResult<Py<Frame>> {
    let certificate_capture =
        crate::core::certificate::begin_render_capture("render_debug_pattern_frame");
    let ctx = crate::core::gpu::try_ctx()?;
    let texture = crate::util::debug_pattern::render_debug_pattern(
        ctx.device.as_ref(),
        ctx.queue.as_ref(),
        width,
        height,
    )
    .map_err(|err| PyRuntimeError::new_err(format!("failed to render debug pattern: {err:#}")))?;

    let frame = Frame::new(
        ctx.device.clone(),
        ctx.queue.clone(),
        texture,
        width,
        height,
        wgpu::TextureFormat::Rgba8UnormSrgb,
    );

    let frame = Py::new(py, frame)?;
    // The `debug_pattern` pass (live gpu_ms when timestamps are granted) is
    // recorded inside `util::debug_pattern::render_debug_pattern`.
    certificate_capture.finish();
    crate::core::certificate::emit_certificate_for_kwarg(py, certificate.as_ref())?;
    Ok(frame)
}

#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(signature = (path, array, channel_prefix=None))]
pub(crate) fn numpy_to_exr(
    path: &str,
    array: PyReadonlyArrayDyn<f32>,
    channel_prefix: Option<&str>,
) -> PyResult<()> {
    let path_obj = Path::new(path);
    if let Some(ext) = path_obj.extension().and_then(|ext| ext.to_str()) {
        if !ext.eq_ignore_ascii_case("exr") {
            return Err(PyValueError::new_err(format!(
                "expected .exr extension, got .{}",
                ext
            )));
        }
    }

    let prefix = channel_prefix.unwrap_or("beauty").trim();
    if prefix.is_empty() {
        return Err(PyValueError::new_err(
            "EXR channel prefix must be non-empty",
        ));
    }

    let view = array.as_array();
    let shape = view.shape();
    let (height, width, channels) = match shape.len() {
        2 => (shape[0], shape[1], 1usize),
        3 => (shape[0], shape[1], shape[2]),
        _ => {
            return Err(PyValueError::new_err(format!(
                "expected array shape (H,W), (H,W,3), or (H,W,4); got {:?}",
                shape
            )))
        }
    };

    if height == 0 || width == 0 {
        return Err(PyValueError::new_err("array dimensions must be positive"));
    }

    if channels != 1 && channels != 3 && channels != 4 {
        return Err(PyValueError::new_err(format!(
            "unsupported channel count {}; expected 1, 3, or 4",
            channels
        )));
    }

    #[cfg(feature = "images")]
    {
        let height_u32 =
            u32::try_from(height).map_err(|_| PyValueError::new_err("array height exceeds u32"))?;
        let width_u32 =
            u32::try_from(width).map_err(|_| PyValueError::new_err("array width exceeds u32"))?;
        let data = view.to_owned().into_raw_vec();
        let write_result = match channels {
            1 => exr_write::write_exr_scalar_f32(path_obj, width_u32, height_u32, &data, prefix),
            3 => exr_write::write_exr_rgb_f32(path_obj, width_u32, height_u32, &data, prefix),
            4 => exr_write::write_exr_rgba_f32(path_obj, width_u32, height_u32, &data, prefix),
            _ => unreachable!("channel validation guards this"),
        };
        write_result
            .map_err(|err| PyRuntimeError::new_err(format!("failed to write EXR: {err:#}")))?;
        Ok(())
    }

    #[cfg(not(feature = "images"))]
    {
        Err(PyRuntimeError::new_err(
            "writing EXR requires the 'images' feature",
        ))
    }
}

// Core modules
