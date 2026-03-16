use super::*;

#[cfg(feature = "extension-module")]
#[pyfunction]
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
) -> PyResult<Py<PyAny>> {
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
            create_rgba_target(&scene.device, "vf.Vector.RenderOIT.Final", width, height);
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

        scene.queue.submit(Some(encoder.finish()));
        scene.device.poll(wgpu::Maintain::Wait);
        read_rgba_texture_to_py(
            py,
            &scene.device,
            &scene.queue,
            &final_tex,
            width,
            height,
            "vf.Vector.RenderOIT.Copy",
            "vf.Vector.RenderOIT.Read",
            "map_async cancelled",
        )
    }
}
