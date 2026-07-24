use super::math::rasterize_coverage_cpu;
use super::raster::{raster_shader_source, CoverageRasterizer};
use super::resolve::{resolve_coverage_cpu, CoverageResolver};
use super::types::FillRule;
use super::{CoverageBinner, CoverageGeometryBuilder};
use crate::vector::api::{PolygonDef, PolylineDef, VectorStyle};
use glam::Vec2;

#[test]
fn raster_shader_and_pinned_math_assemble_as_valid_wgsl() {
    let module = naga::front::wgsl::parse_str(&raster_shader_source())
        .expect("combined LIMES raster shader must parse");
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .expect("combined LIMES raster shader must validate");
}

#[test]
fn gpu_bin_and_raster_match_cpu_analytic_oracle() {
    let context = match crate::core::gpu::try_ctx() {
        Ok(context) => {
            eprintln!(
                "LIMES_GPU_ADAPTER={} backend={:?}",
                context.adapter.get_info().name,
                context.adapter.get_info().backend
            );
            if context.software_fallback {
                eprintln!(
                    "LIMES_GPU_SKIPPED=software fallback adapters are not an authoritative \
                     analytic-kernel dispatch lane"
                );
                return;
            }
            context
        }
        Err(error) => {
            // Non-GPU lanes still run the parser and exact CPU proofs. The
            // physical GPU lane is the authoritative dispatch gate.
            eprintln!("LIMES_GPU_SKIPPED={error}");
            return;
        }
    };
    let mut builder = CoverageGeometryBuilder::new(16, 16).unwrap();
    let fill_layer = builder
        .add_layer("triangle", FillRule::NonZero, [0.2, 0.4, 0.8, 1.0])
        .unwrap();
    builder
        .push_polygon(
            fill_layer,
            &PolygonDef {
                exterior: vec![
                    Vec2::new(1.25, 1.5),
                    Vec2::new(14.5, 2.25),
                    Vec2::new(4.75, 13.5),
                ],
                holes: Vec::new(),
                style: VectorStyle::default(),
            },
        )
        .unwrap();
    let stroke_layer = builder
        .add_layer("stroke", FillRule::NonZero, [0.8, 0.2, 0.1, 1.0])
        .unwrap();
    builder
        .push_round_polyline(
            stroke_layer,
            &PolylineDef {
                path: vec![Vec2::new(2.0, 12.0), Vec2::new(13.0, 5.0)],
                style: VectorStyle {
                    stroke_width: 1.5,
                    ..VectorStyle::default()
                },
            },
        )
        .unwrap();
    let geometry = builder.finish().unwrap();
    let cpu = rasterize_coverage_cpu(&geometry);
    let cpu_resolved = resolve_coverage_cpu(&geometry, &cpu);

    context
        .device
        .push_error_scope(wgpu::ErrorFilter::Validation);
    let binner = CoverageBinner::new(&context.device);
    let bins = binner.prepare(&context.device, &geometry).unwrap();
    let rasterizer = CoverageRasterizer::new(&context.device);
    let raster = rasterizer
        .prepare(&context.device, &geometry, &bins)
        .unwrap();
    let resolver = CoverageResolver::new(&context.device);
    let resolved = resolver
        .prepare(&context.device, &geometry, &bins, &raster)
        .unwrap();
    let readback = crate::core::resource_tracker::tracked_create_buffer(
        &context.device,
        &wgpu::BufferDescriptor {
            label: Some("vf.Vector.Coverage.TestReadback"),
            size: raster.coverage_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        },
    )
    .unwrap();
    let error_readback = crate::core::resource_tracker::tracked_create_buffer(
        &context.device,
        &wgpu::BufferDescriptor {
            label: Some("vf.Vector.Coverage.TestErrorReadback"),
            size: 16,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        },
    )
    .unwrap();
    let resolved_readback = crate::core::resource_tracker::tracked_create_buffer(
        &context.device,
        &wgpu::BufferDescriptor {
            label: Some("vf.Vector.Coverage.TestResolvedReadback"),
            size: resolved.output_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        },
    )
    .unwrap();
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("vf.Vector.Coverage.TestEncoder"),
        });
    binner.encode(
        &mut encoder,
        &bins,
        u32::try_from(geometry.primitives.len()).unwrap(),
    );
    rasterizer.encode(&mut encoder, &raster, &geometry);
    resolver.encode(&mut encoder, &resolved, &geometry);
    encoder.copy_buffer_to_buffer(&raster.coverage, 0, &readback, 0, raster.coverage_bytes);
    encoder.copy_buffer_to_buffer(&bins.overflow, 0, &error_readback, 0, 16);
    encoder.copy_buffer_to_buffer(
        &resolved.output,
        0,
        &resolved_readback,
        0,
        resolved.output_bytes,
    );
    context.queue.submit(Some(encoder.finish()));
    context.device.poll(wgpu::Maintain::Wait);
    let validation = pollster::block_on(context.device.pop_error_scope());
    assert!(validation.is_none(), "GPU validation error: {validation:?}");

    let errors = map_words(&context.device, &error_readback);
    assert_eq!(errors, vec![0_u32; 4], "structured GPU error flags");
    let gpu = map_floats(&context.device, &readback);
    assert_eq!(gpu.len(), cpu.len());
    let max_error = gpu
        .iter()
        .zip(&cpu)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_error < 5.0e-4,
        "GPU/CPU coverage max error {max_error:e}"
    );
    let gpu_resolved = map_floats(&context.device, &resolved_readback);
    let cpu_resolved_flat: Vec<f32> = cpu_resolved.into_iter().flatten().collect();
    assert_eq!(gpu_resolved.len(), cpu_resolved_flat.len());
    let resolve_max_error = gpu_resolved
        .iter()
        .zip(&cpu_resolved_flat)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        resolve_max_error < 1.0e-6,
        "GPU/CPU resolve max error {resolve_max_error:e}"
    );
}

fn map_words(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Vec<u32> {
    map_bytes(device, buffer)
        .chunks_exact(4)
        .map(|bytes| u32::from_ne_bytes(bytes.try_into().unwrap()))
        .collect()
}

fn map_floats(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Vec<f32> {
    map_bytes(device, buffer)
        .chunks_exact(4)
        .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
        .collect()
}

fn map_bytes(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Vec<u8> {
    let slice = buffer.slice(..);
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).ok();
    });
    device.poll(wgpu::Maintain::Wait);
    pollster::block_on(receiver.receive())
        .expect("map callback")
        .expect("map success");
    let data = slice.get_mapped_range();
    let result = data.to_vec();
    drop(data);
    buffer.unmap();
    result
}
