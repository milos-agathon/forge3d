use super::*;

#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(signature = (width, height, points_xy=None, point_rgba=None, point_size=None, polylines=None, polyline_rgba=None, stroke_width=None, certificate=None))]
pub(crate) fn vector_render_oit_py(
    py: Python<'_>,
    width: u32,
    height: u32,
    points_xy: Option<&Bound<'_, PyAny>>,
    point_rgba: Option<&Bound<'_, PyAny>>,
    point_size: Option<&Bound<'_, PyAny>>,
    polylines: Option<&Bound<'_, PyAny>>,
    polyline_rgba: Option<&Bound<'_, PyAny>>,
    stroke_width: Option<&Bound<'_, PyAny>>,
    certificate: Option<Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let _certificate_capture =
        crate::core::certificate::begin_render_capture("vector_render_oit_py");
    #[cfg(not(feature = "weighted-oit"))]
    {
        let _ = (
            py,
            width,
            height,
            points_xy,
            point_rgba,
            point_size,
            polylines,
            polyline_rgba,
            stroke_width,
            certificate,
        );
        Err(weighted_oit_not_enabled_err())
    }
    #[cfg(feature = "weighted-oit")]
    {
        let points = extract_xy_list(points_xy)?;
        let point_colors = extract_rgba_list(point_rgba)?;
        let point_sizes = extract_f32_list(point_size)?;
        let lines = extract_polylines(polylines)?;
        let line_colors = extract_rgba_list(polyline_rgba)?;
        let line_widths = extract_f32_list(stroke_width)?;
        let point_defs = build_point_defs(&points, &point_colors, &point_sizes);
        let poly_defs = build_poly_defs(&lines, &line_colors, &line_widths);
        let mut scene = upload_vector_scene(&point_defs, &poly_defs)?;

        let oit = crate::vector::oit::WeightedOIT::new(
            &scene.device,
            width,
            height,
            wgpu::TextureFormat::Rgba8UnormSrgb,
        )
        .map_err(vector_runtime_err)?;
        let (final_tex, final_view) =
            create_rgba_target(&scene.device, "vf.Vector.RenderOIT.Final", width, height)?;
        let mut encoder = scene
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("vf.Vector.RenderOIT.Encoder"),
            });

        {
            let mut pass = oit.begin_accumulation(&mut encoder);
            render_oit_scene(&mut scene, &mut pass, width, height)?;
        }
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("vf.Vector.RenderOIT.Compose"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &final_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            oit.compose(&mut pass);
        }

        scene.queue.submit(Some(encoder.finish()));
        scene.device.poll(wgpu::Maintain::Wait);
        let result = read_rgba_texture_to_py(
            py,
            &scene.device,
            &scene.queue,
            &final_tex,
            width,
            height,
            "vf.Vector.RenderOIT.Copy",
            "vf.Vector.RenderOIT.Read",
            "map_async cancelled",
        )?;
        crate::core::certificate::record_pass("vector.oit", 0.0, 1);
        crate::core::certificate::record_pass("vector.oit.compose", 0.0, 1);
        _certificate_capture.finish();
        crate::core::certificate::emit_certificate_for_kwarg(py, certificate.as_ref())?;
        Ok(result)
    }
}

#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(signature = (width, height, points_xy=None, point_rgba=None, point_size=None, polylines=None, polyline_rgba=None, stroke_width=None, edl_strength=1.5, edl_radius_px=1.0, certificate=None))]
pub(crate) fn vector_render_oit_edl_py(
    py: Python<'_>,
    width: u32,
    height: u32,
    points_xy: Option<&Bound<'_, PyAny>>,
    point_rgba: Option<&Bound<'_, PyAny>>,
    point_size: Option<&Bound<'_, PyAny>>,
    polylines: Option<&Bound<'_, PyAny>>,
    polyline_rgba: Option<&Bound<'_, PyAny>>,
    stroke_width: Option<&Bound<'_, PyAny>>,
    edl_strength: f32,
    edl_radius_px: f32,
    certificate: Option<Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let _certificate_capture =
        crate::core::certificate::begin_render_capture("vector_render_oit_edl_py");
    #[cfg(not(feature = "weighted-oit"))]
    {
        let _ = (
            py,
            width,
            height,
            points_xy,
            point_rgba,
            point_size,
            polylines,
            polyline_rgba,
            stroke_width,
            edl_strength,
            edl_radius_px,
            certificate,
        );
        Err(weighted_oit_not_enabled_err())
    }
    #[cfg(feature = "weighted-oit")]
    {
        let points = extract_xy_list(points_xy)?;
        let point_colors = extract_rgba_list(point_rgba)?;
        let point_sizes = extract_f32_list(point_size)?;
        let lines = extract_polylines(polylines)?;
        let line_colors = extract_rgba_list(polyline_rgba)?;
        let line_widths = extract_f32_list(stroke_width)?;
        let point_defs = build_point_defs(&points, &point_colors, &point_sizes);
        let poly_defs = build_poly_defs(&lines, &line_colors, &line_widths);
        let mut scene = upload_vector_scene(&point_defs, &poly_defs)?;

        let oit = crate::vector::oit::WeightedOIT::new(
            &scene.device,
            width,
            height,
            wgpu::TextureFormat::Rgba8UnormSrgb,
        )
        .map_err(vector_runtime_err)?;
        let (final_tex, final_view) =
            create_rgba_target(&scene.device, "vf.Vector.RenderOITEDL.Final", width, height)?;
        let (edl_tex, edl_view) = create_rgba_target(
            &scene.device,
            "vf.Vector.RenderOITEDL.Output",
            width,
            height,
        )?;
        let mut encoder = scene
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("vf.Vector.RenderOITEDL.Encoder"),
            });

        {
            let mut pass = oit.begin_accumulation(&mut encoder);
            render_oit_scene(&mut scene, &mut pass, width, height)?;
        }
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("vf.Vector.RenderOITEDL.Compose"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &final_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            oit.compose(&mut pass);
        }

        let (edl_pipeline, edl_bind_group) = oit
            .create_edl_pipeline(
                &scene.device,
                &final_view,
                edl_strength.max(0.0),
                edl_radius_px.max(1.0),
            )
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("vf.Vector.RenderOITEDL.Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &edl_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            crate::core::shader_registry::record_shader_use("vf.Vector.PointEDL");
            pass.set_pipeline(&edl_pipeline);
            pass.set_bind_group(0, &edl_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        scene.queue.submit(Some(encoder.finish()));
        scene.device.poll(wgpu::Maintain::Wait);
        drop(final_tex);
        let result = read_rgba_texture_to_py(
            py,
            &scene.device,
            &scene.queue,
            &edl_tex,
            width,
            height,
            "vf.Vector.RenderOITEDL.Copy",
            "vf.Vector.RenderOITEDL.Read",
            "map_async cancelled",
        )?;
        crate::core::certificate::record_pass("vector.oit", 0.0, 1);
        crate::core::certificate::record_pass("vector.oit.compose", 0.0, 1);
        crate::core::certificate::record_pass("vector.edl", 0.0, 1);
        _certificate_capture.finish();
        crate::core::certificate::emit_certificate_for_kwarg(py, certificate.as_ref())?;
        Ok(result)
    }
}
