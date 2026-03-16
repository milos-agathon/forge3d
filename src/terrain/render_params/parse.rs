use super::*;

pub(crate) fn tuple_to_u32_pair(value: &PyAny, name: &str) -> PyResult<(u32, u32)> {
    let pair: (i64, i64) = value.extract().map_err(|_| {
        PyValueError::new_err(format!(
            "{} must be a tuple of two integers, got {:?}",
            name, value
        ))
    })?;
    if pair.0 < 0 || pair.1 < 0 {
        return Err(PyValueError::new_err(format!(
            "{} components must be non-negative",
            name
        )));
    }
    Ok((pair.0 as u32, pair.1 as u32))
}

#[cfg(feature = "extension-module")]
pub(crate) fn tuple_to_f32_pair(value: &PyAny, name: &str) -> PyResult<(f32, f32)> {
    let pair: (f32, f32) = value.extract().map_err(|_| {
        PyValueError::new_err(format!(
            "{} must be a tuple of two floats, got {:?}",
            name, value
        ))
    })?;
    if !pair.0.is_finite() || !pair.1.is_finite() {
        return Err(PyValueError::new_err(format!(
            "{} values must be finite floats",
            name
        )));
    }
    Ok(pair)
}

#[cfg(feature = "extension-module")]
pub(crate) fn list_to_vec3(value: &PyAny, name: &str) -> PyResult<[f32; 3]> {
    let seq: Vec<f32> = value.extract().map_err(|_| {
        PyValueError::new_err(format!(
            "{} must be a list of three floats, got {:?}",
            name, value
        ))
    })?;
    if seq.len() != 3 {
        return Err(PyValueError::new_err(format!(
            "{} must contain exactly three floats",
            name
        )));
    }
    if seq.iter().any(|v| !v.is_finite()) {
        return Err(PyValueError::new_err(format!(
            "{} entries must be finite floats",
            name
        )));
    }
    Ok([seq[0], seq[1], seq[2]])
}

#[cfg(feature = "extension-module")]
pub(crate) fn to_finite_f32(value: &PyAny, name: &str) -> PyResult<f32> {
    let v: f32 = value
        .extract()
        .map_err(|_| PyValueError::new_err(format!("{} must be a float value", name)))?;
    if !v.is_finite() {
        return Err(PyValueError::new_err(format!(
            "{} must be a finite float",
            name
        )));
    }
    Ok(v)
}

#[cfg(feature = "extension-module")]
pub(crate) fn extract_overlays(
    obj: &PyAny,
) -> PyResult<Vec<Py<crate::core::overlay_layer::OverlayLayer>>> {
    if obj.is_none() {
        return Ok(Vec::new());
    }
    obj.extract().map_err(|_| {
        PyValueError::new_err("overlays must be a sequence of forge3d.OverlayLayer objects")
    })
}

#[cfg(feature = "extension-module")]
pub(crate) fn normalize_direction(x: f32, y: f32, z: f32) -> [f32; 3] {
    let len = (x * x + y * y + z * z).sqrt();
    if len <= 1e-6 {
        [0.0, 1.0, 0.0]
    } else {
        [x / len, y / len, z / len]
    }
}

#[cfg(feature = "extension-module")]
pub(crate) fn parse_filter_mode(value: &str, field: &str) -> PyResult<FilterModeNative> {
    match value {
        "Linear" | "linear" => Ok(FilterModeNative::Linear),
        "Nearest" | "nearest" => Ok(FilterModeNative::Nearest),
        other => Err(PyValueError::new_err(format!(
            "{} must be 'Linear' or 'Nearest', got {}",
            field, other
        ))),
    }
}

#[cfg(feature = "extension-module")]
pub(crate) fn parse_address_mode(value: &str, field: &str) -> PyResult<AddressModeNative> {
    match value {
        "Repeat" | "repeat" => Ok(AddressModeNative::Repeat),
        "ClampToEdge" | "clamp_to_edge" | "Clamp" | "clamp" => Ok(AddressModeNative::ClampToEdge),
        "MirrorRepeat" | "mirror_repeat" => Ok(AddressModeNative::MirrorRepeat),
        other => Err(PyValueError::new_err(format!(
            "{} must be 'Repeat', 'ClampToEdge', or 'MirrorRepeat', got {}",
            field, other
        ))),
    }
}
