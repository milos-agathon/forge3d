use super::*;

#[pyfunction]
#[pyo3(signature = (width, height, points_xy=None, polylines=None, base_pick_id=None, certificate=None))]
pub(crate) fn vector_render_pick_map_py(
    py: Python<'_>,
    width: u32,
    height: u32,
    points_xy: Option<&Bound<'_, PyAny>>,
    polylines: Option<&Bound<'_, PyAny>>,
    base_pick_id: Option<u32>,
    certificate: Option<Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let _certificate_capture =
        crate::core::certificate::begin_render_capture("vector_render_pick_map_py");
    let points = extract_xy_list(points_xy)?;
    let lines = extract_polylines(polylines)?;
    let point_defs = build_point_defs(&points, &[], &[]);
    let poly_defs = build_poly_defs(&lines, &[], &[]);
    let mut scene = upload_vector_scene(&point_defs, &poly_defs)?;
    let (pick_tex, pick_view) =
        create_pick_target(&scene.device, "vf.Vector.RenderPick.Pick", width, height)?;
    let mut encoder = scene
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("vf.Vector.RenderPick.Encoder"),
        });

    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("vf.Vector.RenderPick.Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &pick_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.0,
                        g: 0.0,
                        b: 0.0,
                        a: 0.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });
        render_pick_scene(
            &mut scene,
            &mut pass,
            width,
            height,
            base_pick_id.unwrap_or(1),
        )?;
    }

    scene.queue.submit(Some(encoder.finish()));
    scene.device.poll(wgpu::Maintain::Wait);
    let result = read_u32_texture_to_py(
        py,
        &scene.device,
        &scene.queue,
        &pick_tex,
        width,
        height,
        "vf.Vector.RenderPick.Copy",
        "vf.Vector.RenderPick.Read",
        "map_async cancelled",
    )?;
    crate::core::certificate::record_pass("vector.pick", 0.0, 1);
    _certificate_capture.finish();
    crate::core::certificate::emit_certificate_for_kwarg(py, certificate.as_ref())?;
    Ok(result)
}

#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(signature = (width, height, points_xy=None, point_rgba=None, point_size=None, polylines=None, polyline_rgba=None, stroke_width=None, base_pick_id=None, certificate=None))]
pub(crate) fn vector_render_oit_and_pick_py(
    py: Python<'_>,
    width: u32,
    height: u32,
    points_xy: Option<&Bound<'_, PyAny>>,
    point_rgba: Option<&Bound<'_, PyAny>>,
    point_size: Option<&Bound<'_, PyAny>>,
    polylines: Option<&Bound<'_, PyAny>>,
    polyline_rgba: Option<&Bound<'_, PyAny>>,
    stroke_width: Option<&Bound<'_, PyAny>>,
    base_pick_id: Option<u32>,
    certificate: Option<Bound<'_, PyAny>>,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let _certificate_capture =
        crate::core::certificate::begin_render_capture("vector_render_oit_and_pick_py");
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
            base_pick_id,
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
            create_rgba_target(&scene.device, "vf.Vector.Combine.Final", width, height)?;
        let (pick_tex, pick_view) =
            create_pick_target(&scene.device, "vf.Vector.Combine.Pick", width, height)?;
        let mut encoder = scene
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("vf.Vector.Combine.Encoder"),
            });

        {
            let mut pass = oit.begin_accumulation(&mut encoder);
            render_oit_scene(&mut scene, &mut pass, width, height)?;
        }
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("vf.Vector.Combine.Compose"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &final_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            oit.compose(&mut pass);
        }
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("vf.Vector.Combine.PickPass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &pick_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            render_pick_scene(
                &mut scene,
                &mut pass,
                width,
                height,
                base_pick_id.unwrap_or(1),
            )?;
        }

        scene.queue.submit(Some(encoder.finish()));
        scene.device.poll(wgpu::Maintain::Wait);

        let rgba = read_rgba_texture_to_py(
            py,
            &scene.device,
            &scene.queue,
            &final_tex,
            width,
            height,
            "vf.Vector.Combine.CopyFinal",
            "vf.Vector.Combine.FinalRead",
            "final map cancelled",
        )?;
        let ids = read_u32_texture_to_py(
            py,
            &scene.device,
            &scene.queue,
            &pick_tex,
            width,
            height,
            "vf.Vector.Combine.CopyPick",
            "vf.Vector.Combine.PickRead",
            "pick map cancelled",
        )?;
        crate::core::certificate::record_pass("vector.oit", 0.0, 1);
        crate::core::certificate::record_pass("vector.oit.compose", 0.0, 1);
        crate::core::certificate::record_pass("vector.pick", 0.0, 1);
        _certificate_capture.finish();
        crate::core::certificate::emit_certificate_for_kwarg(py, certificate.as_ref())?;
        Ok((rgba, ids))
    }
}
