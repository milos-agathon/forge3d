use super::*;
use crate::vector::api::PolygonDef;

fn parse_and_normalize_polygons(
    exteriors: Vec<numpy::PyReadonlyArray2<'_, f64>>,
    holes: Option<Vec<Vec<numpy::PyReadonlyArray2<'_, f64>>>>,
) -> PyResult<Vec<PolygonDef>> {
    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    let mut polys = Vec::with_capacity(exteriors.len());

    for (index, exterior) in exteriors.into_iter().enumerate() {
        let exterior =
            crate::vector::api::parse_polygon_from_numpy(exterior).map_err(vector_runtime_err)?;
        for vertex in &exterior {
            min_x = min_x.min(vertex.x);
            min_y = min_y.min(vertex.y);
            max_x = max_x.max(vertex.x);
            max_y = max_y.max(vertex.y);
        }

        let mut hole_rings = Vec::new();
        if let Some(all_holes) = &holes {
            if let Some(hole_set) = all_holes.get(index) {
                for hole in hole_set {
                    let hole_vertices = crate::vector::api::parse_polygon_from_numpy(hole.clone())
                        .map_err(vector_runtime_err)?;
                    for vertex in &hole_vertices {
                        min_x = min_x.min(vertex.x);
                        min_y = min_y.min(vertex.y);
                        max_x = max_x.max(vertex.x);
                        max_y = max_y.max(vertex.y);
                    }
                    hole_rings.push(hole_vertices);
                }
            }
        }

        polys.push(PolygonDef {
            exterior,
            holes: hole_rings,
            style: crate::vector::api::VectorStyle::default(),
        });
    }

    if !min_x.is_finite() || !min_y.is_finite() || !max_x.is_finite() || !max_y.is_finite() {
        min_x = -1.0;
        min_y = -1.0;
        max_x = 1.0;
        max_y = 1.0;
    }

    let center_x = 0.5 * (min_x + max_x);
    let center_y = 0.5 * (min_y + max_y);
    let dx = (max_x - min_x).max(1e-6);
    let dy = (max_y - min_y).max(1e-6);
    let norm_scale = 100.0 / dx.max(dy);

    for poly in &mut polys {
        for vertex in &mut poly.exterior {
            vertex.x = (vertex.x - center_x) * norm_scale;
            vertex.y = (vertex.y - center_y) * norm_scale;
        }
        for hole in &mut poly.holes {
            for vertex in hole {
                vertex.x = (vertex.x - center_x) * norm_scale;
                vertex.y = (vertex.y - center_y) * norm_scale;
            }
        }
    }

    Ok(polys)
}

fn compute_polygon_view_proj(polys: &[PolygonDef], width: u32, height: u32) -> [[f32; 4]; 4] {
    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;

    for poly in polys {
        for vertex in &poly.exterior {
            min_x = min_x.min(vertex.x);
            min_y = min_y.min(vertex.y);
            max_x = max_x.max(vertex.x);
            max_y = max_y.max(vertex.y);
        }
        for hole in &poly.holes {
            for vertex in hole {
                min_x = min_x.min(vertex.x);
                min_y = min_y.min(vertex.y);
                max_x = max_x.max(vertex.x);
                max_y = max_y.max(vertex.y);
            }
        }
    }

    if !min_x.is_finite() || !min_y.is_finite() || !max_x.is_finite() || !max_y.is_finite() {
        min_x = 0.0;
        min_y = 0.0;
        max_x = 100.0;
        max_y = 100.0;
    }

    let center_x = 0.5 * (min_x + max_x);
    let center_y = 0.5 * (min_y + max_y);
    let dx = (max_x - min_x).max(1e-6);
    let dy = (max_y - min_y).max(1e-6);
    let viewport_aspect = width as f32 / height as f32;
    let data_aspect = dx / dy;
    let scale = if data_aspect > viewport_aspect {
        2.0 / dx
    } else {
        2.0 / dy
    };

    [
        [scale, 0.0, 0.0, -scale * center_x],
        [0.0, -scale, 0.0, scale * center_y],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

#[cfg(feature = "extension-module")]
#[pyfunction]
pub(crate) fn vector_render_polygons_fill_py(
    py: Python<'_>,
    width: u32,
    height: u32,
    exteriors: Vec<numpy::PyReadonlyArray2<'_, f64>>,
    holes: Option<Vec<Vec<numpy::PyReadonlyArray2<'_, f64>>>>,
    fill_rgba: Option<(f32, f32, f32, f32)>,
    stroke_rgba: Option<(f32, f32, f32, f32)>,
    stroke_width: Option<f32>,
) -> PyResult<Py<PyAny>> {
    let polys = parse_and_normalize_polygons(exteriors, holes)?;
    let (device, queue) = gpu_device_queue();

    let mut renderer =
        crate::vector::PolygonRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb)
            .map_err(vector_runtime_err)?;
    let mut packed = Vec::with_capacity(polys.len());
    for poly in &polys {
        packed.push(
            renderer
                .tessellate_polygon(poly)
                .map_err(vector_runtime_err)?,
        );
    }
    renderer
        .upload_polygons(&device, &queue, &packed)
        .map_err(vector_runtime_err)?;

    let (final_tex, final_view) =
        create_rgba_target(&device, "vf.Vector.PolygonFill.Final", width, height);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("vf.Vector.PolygonFill.Encoder"),
    });

    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("vf.Vector.PolygonFill.Render"),
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

        let fill = fill_rgba.unwrap_or((0.2, 0.4, 0.8, 1.0));
        let stroke = stroke_rgba.unwrap_or((0.0, 0.0, 0.0, 1.0));
        let total_indices: u32 = packed.iter().map(|poly| poly.indices.len() as u32).sum();

        renderer
            .render(
                &mut pass,
                &queue,
                &compute_polygon_view_proj(&polys, width, height),
                [fill.0, fill.1, fill.2, fill.3],
                [stroke.0, stroke.1, stroke.2, stroke.3],
                stroke_width.unwrap_or(1.0),
                total_indices,
            )
            .map_err(vector_runtime_err)?;
    }

    queue.submit(Some(encoder.finish()));
    device.poll(wgpu::Maintain::Wait);
    read_rgba_texture_to_py(
        py,
        &device,
        &queue,
        &final_tex,
        width,
        height,
        "vf.Vector.PolygonFill.Copy",
        "vf.Vector.PolygonFill.Read",
        "map_async cancelled",
    )
}
