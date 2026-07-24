use super::coverage::{decode_coverage_scene, CoveragePolygonInput, CoveragePolylineInput};
use super::*;
use crate::vector::api::{PolygonDef, PolylineDef, VectorStyle};

fn point_to_ndc(point: [f32; 2], width: u32, height: u32) -> glam::Vec2 {
    glam::Vec2::new(
        2.0 * point[0] / width as f32 - 1.0,
        1.0 - 2.0 * point[1] / height as f32,
    )
}

fn ablation_polygon(input: CoveragePolygonInput, width: u32, height: u32) -> PolygonDef {
    PolygonDef {
        exterior: input
            .exterior
            .into_iter()
            .map(|point| point_to_ndc(point, width, height))
            .collect(),
        holes: input
            .holes
            .into_iter()
            .map(|ring| {
                ring.into_iter()
                    .map(|point| point_to_ndc(point, width, height))
                    .collect()
            })
            .collect(),
        style: VectorStyle {
            fill_color: [1.0; 4],
            ..VectorStyle::default()
        },
    }
}

fn ablation_polyline(input: CoveragePolylineInput, width: u32, height: u32) -> PolylineDef {
    PolylineDef {
        path: input
            .path
            .into_iter()
            .map(|point| point_to_ndc(point, width, height))
            .collect(),
        style: VectorStyle {
            stroke_color: [1.0; 4],
            stroke_width: input.width,
            ..VectorStyle::default()
        },
    }
}

fn create_ablation_multisample_target(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    sample_count: u32,
) -> Result<crate::core::resource_tracker::TrackedTexture, RenderError> {
    crate::core::resource_tracker::tracked_create_texture(
        device,
        &wgpu::TextureDescriptor {
            label: Some("vf.Vector.LimesAblation.Msaa"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        },
    )
}

/// Render one torture-sheet layer through the pre-LIMES raster pipelines.
///
/// This deliberately uses the real lyon fill and smoothstep-feathered line
/// renderers. `sample_count=4` changes only raster sample count and resolves
/// into the same single-sample target, which makes the MSAA comparison a true
/// ablation rather than a NumPy simulation.
#[cfg(feature = "extension-module")]
#[pyfunction(signature = (scene_json, sample_count, certificate=None))]
pub(crate) fn _vector_render_coverage_ablation_py(
    py: Python<'_>,
    scene_json: &str,
    sample_count: u32,
    certificate: Option<Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    if !matches!(sample_count, 1 | 4) {
        return Err(PyValueError::new_err(
            "sample_count must be 1 (current) or 4 (MSAA ablation)",
        ));
    }
    let mut input = decode_coverage_scene(scene_json)?;
    if input.layers.len() != 1 {
        return Err(PyValueError::new_err(
            "LIMES ablation accepts exactly one torture-sheet layer",
        ));
    }
    let layer = input.layers.pop().expect("length checked");
    if layer.polygon_grid.is_some() || (layer.polygons.is_empty() == layer.polylines.is_empty()) {
        return Err(PyValueError::new_err(
            "LIMES ablation layer must contain polygons XOR polylines and no compact grid",
        ));
    }

    let capture =
        crate::core::certificate::begin_render_capture("_vector_render_coverage_ablation_py");
    let (device, queue) = gpu_device_queue()?;
    let (final_texture, final_view) = create_rgba_target(
        &device,
        "vf.Vector.LimesAblation.Final",
        input.width,
        input.height,
    )?;
    let multisample_texture = if sample_count == 4 {
        Some(create_ablation_multisample_target(
            &device,
            input.width,
            input.height,
            sample_count,
        )?)
    } else {
        None
    };
    let multisample_view = multisample_texture
        .as_ref()
        .map(|texture| texture.create_view(&wgpu::TextureViewDescriptor::default()));
    let render_view = multisample_view.as_ref().unwrap_or(&final_view);
    let resolve_target = (sample_count == 4).then_some(&final_view);

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("vf.Vector.LimesAblation.Encoder"),
    });
    let mut timing = crate::core::gpu_timing::OneShotTiming::for_current_device();
    let timing_scope = timing.begin(&mut encoder, "vector.coverage.ablation");
    let draw_count;

    if !layer.polygons.is_empty() {
        let polygons = layer
            .polygons
            .into_iter()
            .map(|polygon| ablation_polygon(polygon, input.width, input.height))
            .collect::<Vec<_>>();
        let mut renderer = crate::vector::PolygonRenderer::new_with_sample_count(
            &device,
            wgpu::TextureFormat::Rgba8Unorm,
            sample_count,
        )
        .map_err(vector_runtime_err)?;
        let packed = polygons
            .iter()
            .map(|polygon| renderer.tessellate_polygon(polygon))
            .collect::<Result<Vec<_>, _>>()
            .map_err(vector_runtime_err)?;
        renderer
            .upload_polygons(&device, &queue, &packed)
            .map_err(vector_runtime_err)?;
        draw_count = packed.len() as u32;
        let total_indices = packed
            .iter()
            .map(|polygon| polygon.indices.len() as u32)
            .sum();
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("vf.Vector.LimesAblation.Polygon"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: render_view,
                    resolve_target,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            renderer
                .render(
                    &mut pass,
                    &queue,
                    &glam::Mat4::IDENTITY.to_cols_array_2d(),
                    [1.0; 4],
                    [0.0; 4],
                    0.0,
                    total_indices,
                )
                .map_err(vector_runtime_err)?;
        }
    } else {
        let polylines = layer
            .polylines
            .into_iter()
            .map(|polyline| ablation_polyline(polyline, input.width, input.height))
            .collect::<Vec<_>>();
        let mut renderer = crate::vector::LineRenderer::new_with_sample_count(
            &device,
            wgpu::TextureFormat::Rgba8Unorm,
            sample_count,
        )
        .map_err(vector_runtime_err)?;
        let instances = renderer
            .pack_polylines(&polylines)
            .map_err(vector_runtime_err)?;
        draw_count = instances.len() as u32;
        renderer
            .upload_lines(&device, &queue, &instances)
            .map_err(vector_runtime_err)?;
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("vf.Vector.LimesAblation.Line"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: render_view,
                    resolve_target,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            renderer
                .render(
                    &mut pass,
                    &queue,
                    &glam::Mat4::IDENTITY.to_cols_array_2d(),
                    viewport_dims(input.width, input.height),
                    draw_count,
                    crate::vector::line::LineCap::Round,
                    crate::vector::line::LineJoin::Round,
                    4.0,
                )
                .map_err(vector_runtime_err)?;
        }
    }
    timing.end(&mut encoder, timing_scope, draw_count);
    timing.resolve(&mut encoder);
    queue.submit(Some(encoder.finish()));
    device.poll(wgpu::Maintain::Wait);

    let rgba = read_rgba_texture_to_py(
        py,
        &device,
        &queue,
        &final_texture,
        input.width,
        input.height,
        "vf.Vector.LimesAblation.Copy",
        "vf.Vector.LimesAblation.Read",
        "LIMES ablation readback cancelled",
    )?;
    if !timing.record_into_certificate() {
        crate::core::certificate::record_pass("vector.coverage.ablation", 0.0, draw_count);
    }
    capture.finish();
    crate::core::certificate::emit_certificate_for_kwarg(py, certificate.as_ref())?;
    Ok(rgba)
}
