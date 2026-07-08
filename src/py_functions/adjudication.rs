// src/py_functions/adjudication.rs
// AEQUITAS ground-truth capture API: renders the committed adjudication
// reference scene through BOTH the wavefront path tracer and the raster twin
// on the same device/queue, resolves both through the single shared tonemap
// operator, and returns (pt_rgba, raster_rgba, metadata).
// RELEVANT FILES: src/path_tracing/adjudication.rs, src/offscreen/adjudication_raster.rs

use super::super::*;

/// Render the adjudication pair: high-spp path-traced ground truth and the
/// raster render of the same ReferenceSceneDesc.
///
/// Returns `(pt_rgba, raster_rgba, meta)` where both arrays are
/// `(height, width, 4) uint8` and `meta` is `{"pt": {...}, "raster": {...}}`
/// with identical camera/light fields for both paths (single scene source).
#[cfg(feature = "extension-module")]
#[pyfunction]
pub(crate) fn render_adjudication_pair(
    py: Python<'_>,
    width: u32,
    height: u32,
    spp: u32,
) -> PyResult<Py<PyAny>> {
    use numpy::PyArray1;

    if width == 0 || height == 0 || spp == 0 {
        return Err(PyValueError::new_err(
            "render_adjudication_pair requires width > 0, height > 0, spp > 0",
        ));
    }

    let g = crate::core::gpu::try_ctx()?;
    let desc = crate::path_tracing::reference_scene::adjudication_scene();

    let pt_hdr = crate::path_tracing::adjudication::render_pt_reference(
        &g.device, &g.queue, &desc, width, height, spp,
    )?;
    let raster_hdr = crate::offscreen::adjudication_raster::render_raster_reference(
        g.device.as_ref(),
        g.queue.as_ref(),
        &desc,
        width,
        height,
    )?;

    // Tonemap parity by construction: one shared operator (Reinhard, see
    // core::tonemap::resolve_reference_hdr_to_rgba8), same exposure, both paths.
    let pt_rgba = crate::core::tonemap::resolve_reference_hdr_to_rgba8(&pt_hdr, desc.exposure);
    let raster_rgba =
        crate::core::tonemap::resolve_reference_hdr_to_rgba8(&raster_hdr, desc.exposure);

    let to_array = |v: Vec<u8>| -> PyResult<Py<PyAny>> {
        let arr1 = PyArray1::<u8>::from_vec_bound(py, v);
        let arr3 = arr1.reshape([height as usize, width as usize, 4])?;
        Ok(arr3.into_py(py))
    };
    let pt_arr = to_array(pt_rgba)?;
    let raster_arr = to_array(raster_rgba)?;

    // Both metadata dicts come from the same metadata_fields() call so the
    // gate can assert byte-identical camera/light descriptions per path.
    let fields = desc.metadata_fields(width, height, spp);
    let meta = pyo3::types::PyDict::new_bound(py);
    for key in ["pt", "raster"] {
        let sub = pyo3::types::PyDict::new_bound(py);
        for (name, value) in &fields {
            sub.set_item(name, value)?;
        }
        meta.set_item(key, sub)?;
    }

    Ok((pt_arr, raster_arr, meta).into_py(py))
}
