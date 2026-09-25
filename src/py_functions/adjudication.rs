// src/py_functions/adjudication.rs
// AEQUITAS ground-truth capture API: renders the committed adjudication
// reference scene through BOTH the wavefront path tracer and the raster twin
// on the same device/queue, resolves both through the single shared tonemap
// operator, and returns (pt_rgba, raster_rgba, metadata). The raster side is
// a headless interactive viewer executing its PBR scene through the shared
// Viewer::render frame pipeline.
// RELEVANT FILES: src/path_tracing/adjudication.rs,
// src/viewer/pbr_scene/reference.rs, src/core/anamnesis/hdr_graph.rs

use super::super::*;

/// Render the adjudication pair: high-spp path-traced ground truth and the
/// raster render of the same ReferenceSceneDesc.
///
/// Returns `(pt_rgba, raster_rgba, meta)` where both arrays are
/// `(height, width, 4) uint8` and `meta` is `{"pt": {...}, "raster": {...}}`
/// with identical camera/light fields for both paths (single scene source).
#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(signature = (width, height, spp, certificate = None, cache = None))]
pub(crate) fn render_adjudication_pair(
    py: Python<'_>,
    width: u32,
    height: u32,
    spp: u32,
    certificate: Option<Bound<'_, PyAny>>,
    cache: Option<Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    use numpy::PyArray1;

    let certificate_capture =
        crate::core::certificate::begin_render_capture("render_adjudication_pair");
    if width == 0 || height == 0 || spp == 0 {
        return Err(PyValueError::new_err(
            "render_adjudication_pair requires width > 0, height > 0, spp > 0",
        ));
    }

    let g = crate::core::gpu::try_ctx()?;
    let desc = crate::path_tracing::reference_scene::adjudication_scene();

    // CENSOR F-04: one shared timing instance so the certificate records the
    // PT scope first and the raster scope second (same order as the fallback);
    // falls back to 0.0 pass records when TIMESTAMP_QUERY is not granted.
    let mut timing = crate::core::gpu_timing::OneShotTiming::for_current_device();
    let pt_hdr = crate::path_tracing::adjudication::render_pt_reference(
        &g.device,
        &g.queue,
        &desc,
        width,
        height,
        spp,
        Some(&mut timing),
    )?;
    let certificate_enabled = certificate
        .as_ref()
        .is_some_and(|value| !value.is_none() && !matches!(value.extract::<bool>(), Ok(false)));
    let cache_options = if certificate_enabled {
        None
    } else {
        cache
            .as_ref()
            .map(|value| -> PyResult<_> {
                let path = value.extract::<std::path::PathBuf>().or_else(|_| {
                    value
                        .call_method0("__fspath__")?
                        .extract::<std::path::PathBuf>()
                })?;
                let limits = g.device.limits();
                let capability = crate::core::anamnesis::CapabilityFingerprint {
                    granted_features: g
                        .capabilities
                        .granted_names()
                        .into_iter()
                        .map(str::to_string)
                        .collect(),
                    limits: std::collections::BTreeMap::from([
                        (
                            "max_texture_dimension_2d".into(),
                            limits.max_texture_dimension_2d as u64,
                        ),
                        ("max_buffer_size".into(), limits.max_buffer_size),
                        ("max_bind_groups".into(), limits.max_bind_groups as u64),
                        (
                            "max_storage_buffers_per_shader_stage".into(),
                            limits.max_storage_buffers_per_shader_stage as u64,
                        ),
                        (
                            "min_uniform_buffer_offset_alignment".into(),
                            limits.min_uniform_buffer_offset_alignment as u64,
                        ),
                        (
                            "min_storage_buffer_offset_alignment".into(),
                            limits.min_storage_buffer_offset_alignment as u64,
                        ),
                    ]),
                    backend: format!("{:?}", g.adapter.get_info().backend).to_lowercase(),
                    dx12_compiler: g.dx12_compiler.into(),
                    naga_capabilities: vec![format!(
                        "wgpu-validation-default@naga-{}",
                        env!("FORGE3D_NAGA_VERSION")
                    )],
                };
                Ok((
                    path,
                    capability.canonical_bytes(),
                    crate::core::anamnesis::EngineFingerprint::current().canonical_bytes(),
                ))
            })
            .transpose()?
    };

    // Raster twin: a headless interactive viewer loads the shared reference
    // scene and renders it through the same Viewer::render frame pipeline
    // the windowed viewer uses.
    let mut viewer = crate::viewer::Viewer::new_headless(
        g.device.clone(),
        g.queue.clone(),
        g.adapter.clone(),
        width,
        height,
        crate::viewer::viewer_config::ViewerConfig {
            width,
            height,
            ..Default::default()
        },
    )
    .map_err(|error| PyRuntimeError::new_err(format!("headless viewer init: {error}")))?;
    viewer
        .handle_cmd(crate::viewer::viewer_enums::ViewerCmd::LoadReferenceScene {
            name: "adjudication".into(),
        })
        .map_err(PyRuntimeError::new_err)?;

    let frame_camera = viewer.current_frame_camera();
    let cache_declaration = cache_options
        .map(|(root, capability_bytes, engine_bytes)| {
            let (descriptor, uniforms, inputs) = viewer
                .pbr_scene_cache_key_parts(frame_camera)
                .ok_or_else(|| PyRuntimeError::new_err("pbr scene missing after load"))?;
            Ok::<_, PyErr>(crate::core::anamnesis::ForwardCacheDeclaration {
                root,
                max_bytes: 512 * 1024 * 1024,
                verify_reads: true,
                pipeline_descriptor_bytes: descriptor,
                uniform_bytes: uniforms,
                external_input_bytes: inputs,
                capability_fingerprint_bytes: capability_bytes,
                engine_fingerprint_bytes: engine_bytes,
            })
        })
        .transpose()?;
    let hdr_target = viewer
        .pbr_scene_hdr_target()
        .ok_or_else(|| PyRuntimeError::new_err("pbr scene HDR target missing"))?;
    let target = crate::core::anamnesis::HdrGraphTarget {
        texture: &hdr_target,
        width,
        height,
        format: wgpu::TextureFormat::Rgba32Float,
    };
    let labels = crate::core::anamnesis::HdrGraphLabels {
        graphics: "viewer.pbr_scene",
        readback: "viewer.readback",
    };
    let mut render = || -> crate::core::error::RenderResult<()> {
        viewer.render_headless_frame(Some((&mut timing, "adjudication.raster")))
    };
    let (raster_hdr, cache_report) = crate::core::anamnesis::render_hdr_graph(
        &g.device,
        &g.queue,
        &target,
        &labels,
        cache_declaration.as_ref(),
        &mut render,
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

    // meta["raster"] carries only the values the viewer PBR scene actually
    // consumed this frame; the gate compares the shared keys against
    // meta["pt"] (which additionally carries "spp").
    let meta = pyo3::types::PyDict::new_bound(py);
    let pt_sub = pyo3::types::PyDict::new_bound(py);
    for (name, value) in desc.metadata_fields(width, height, spp) {
        pt_sub.set_item(name, value)?;
    }
    meta.set_item("pt", pt_sub)?;
    let mut raster_consumed = viewer.pbr_scene_consumed_metadata();
    let metadata_source = if raster_consumed.is_empty() {
        // ANAMNESIS hit: no frame rendered. The restored image's cache key
        // describes exactly the desc + this frame camera, so record that
        // keyed metadata as what the image consumed.
        viewer.pbr_scene_record_keyed_metadata(frame_camera);
        raster_consumed = viewer.pbr_scene_consumed_metadata();
        "cache_key"
    } else {
        "rendered_frame"
    };
    let raster_sub = pyo3::types::PyDict::new_bound(py);
    for (name, value) in raster_consumed {
        raster_sub.set_item(name, value)?;
    }
    meta.set_item("raster", raster_sub)?;
    meta.set_item("raster_metadata_source", metadata_source)?;
    meta.set_item("raster_route", crate::viewer::pbr_scene::PBR_SCENE_ROUTE)?;
    let cache_meta = pyo3::types::PyDict::new_bound(py);
    cache_meta.set_item("hits", cache_report.hits)?;
    cache_meta.set_item("misses", cache_report.misses)?;
    cache_meta.set_item("bytes_read", cache_report.bytes_read)?;
    cache_meta.set_item("bytes_written", cache_report.bytes_written)?;
    cache_meta.set_item("wall_ms_saved", cache_report.wall_ms_saved)?;
    meta.set_item("cache", cache_meta)?;

    if !timing.record_into_certificate() {
        crate::core::certificate::record_pass("adjudication.path_trace", 0.0, spp);
        crate::core::certificate::record_pass("adjudication.raster", 0.0, 5);
    }
    certificate_capture.finish();
    crate::core::certificate::emit_certificate_for_kwarg(py, certificate.as_ref())?;

    Ok((pt_arr, raster_arr, meta).into_py(py))
}
