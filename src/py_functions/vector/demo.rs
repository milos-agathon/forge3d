use super::*;
#[cfg(feature = "weighted-oit")]
use crate::vector::api::{PointDef, PolylineDef, VectorStyle};

#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(signature = (width, height, certificate=None, cache=None))]
pub(crate) fn vector_oit_and_pick_demo(
    py: Python<'_>,
    width: u32,
    height: u32,
    certificate: Option<Bound<'_, PyAny>>,
    cache: Option<Bound<'_, PyAny>>,
) -> PyResult<(Py<PyAny>, u32)> {
    let _ = cache;
    let _certificate_capture =
        crate::core::certificate::begin_render_capture("vector_oit_and_pick_demo");
    #[cfg(not(feature = "weighted-oit"))]
    {
        let _ = (py, width, height, certificate);
        Err(weighted_oit_not_enabled_err())
    }
    #[cfg(feature = "weighted-oit")]
    {
        let point_defs = vec![
            PointDef {
                position: glam::Vec2::new(-0.5, -0.5),
                style: VectorStyle {
                    fill_color: [1.0, 0.2, 0.2, 0.9],
                    stroke_color: [0.0, 0.0, 0.0, 1.0],
                    stroke_width: 1.0,
                    point_size: 24.0,
                },
            },
            PointDef {
                position: glam::Vec2::new(0.4, 0.2),
                style: VectorStyle {
                    fill_color: [0.2, 0.8, 1.0, 0.7],
                    stroke_color: [0.0, 0.0, 0.0, 1.0],
                    stroke_width: 1.0,
                    point_size: 32.0,
                },
            },
        ];
        let poly_defs = vec![PolylineDef {
            path: vec![
                glam::Vec2::new(-0.8, -0.8),
                glam::Vec2::new(0.8, 0.5),
                glam::Vec2::new(0.4, 0.8),
            ],
            style: VectorStyle {
                fill_color: [0.0, 0.0, 0.0, 0.0],
                stroke_color: [0.1, 0.9, 0.3, 0.6],
                stroke_width: 8.0,
                point_size: 4.0,
            },
        }];
        let mut scene = upload_vector_scene(&point_defs, &poly_defs)?;

        let oit = crate::vector::oit::WeightedOIT::new(
            &scene.device,
            width,
            height,
            wgpu::TextureFormat::Rgba8Unorm,
        )
        .map_err(vector_runtime_err)?;
        let (final_tex, final_view) =
            create_rgba_target(&scene.device, "vf.Vector.Demo.Final", width, height)?;
        let (pick_tex, pick_view) =
            create_pick_target(&scene.device, "vf.Vector.Demo.Pick", width, height)?;
        prepare_oit_scene(&mut scene, width, height);
        scene.flush_uploads();
        let mut timing = VectorPassTiming::new(scene.device.clone(), &scene.queue, 3)?;
        let accumulation_timing = timing.reserve("vector.demo.oit", 2);
        let compose_timing = timing.reserve("vector.demo.compose", 1);
        let pick_timing = timing.reserve("vector.demo.pick", 2);
        let mut accumulation_encoder =
            scene
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("vf.Vector.Demo.AccumulationEncoder"),
                });
        {
            let mut pass = oit.begin_accumulation_timed(
                &mut accumulation_encoder,
                timing.render_pass_writes(accumulation_timing),
            );
            render_oit_scene(&mut scene, &mut pass, width, height)?;
        }
        let accumulation_submission = scene.queue.submit(Some(accumulation_encoder.finish()));
        scene.device.poll(wgpu::Maintain::WaitForSubmissionIndex(
            accumulation_submission,
        ));

        let mut post_encoder =
            scene
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("vf.Vector.Demo.PostEncoder"),
                });
        {
            let mut pass = post_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("vf.Vector.Demo.Compose"),
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
                timestamp_writes: timing.render_pass_writes(compose_timing),
            });
            oit.compose(&mut pass);
        }
        {
            let mut pass = post_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("vf.Vector.Demo.PickPass"),
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
                timestamp_writes: timing.render_pass_writes(pick_timing),
            });
            render_pick_scene(&mut scene, &mut pass, width, height, 1)?;
        }
        timing.resolve(&mut post_encoder);

        scene.flush_uploads();
        scene.queue.submit(Some(post_encoder.finish()));
        scene.device.poll(wgpu::Maintain::Wait);

        let rgba = read_rgba_texture_to_py(
            py,
            &scene.device,
            &scene.queue,
            &final_tex,
            width,
            height,
            "vf.Vector.Demo.CopyFinal",
            "vf.Vector.Demo.FinalRead",
            "map_async cancelled",
        )?;
        let pick_id = read_single_u32_pixel(
            &scene.device,
            &scene.queue,
            &pick_tex,
            width / 2,
            height / 2,
            "vf.Vector.Demo.CopyPick",
            "vf.Vector.Demo.PickRead",
            "pick map cancelled",
        )?;
        if !timing.record_into_certificate() {
            crate::core::certificate::record_pass("vector.demo.oit", 0.0, 2);
            crate::core::certificate::record_pass("vector.demo.compose", 0.0, 1);
            crate::core::certificate::record_pass("vector.demo.pick", 0.0, 2);
        }
        _certificate_capture.finish();
        crate::core::certificate::emit_certificate_for_kwarg(py, certificate.as_ref())?;
        Ok((rgba, pick_id))
    }
}
