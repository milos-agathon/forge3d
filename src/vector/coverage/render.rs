use super::{
    CoverageBinner, CoverageGeometry, CoverageRasterizer, CoverageResolver, PrimitiveKind,
};
use crate::core::error::{RenderError, RenderResult};
use crate::core::resource_tracker::{tracked_create_buffer, TrackedBuffer};
use std::sync::Arc;
use std::time::Instant;

const COVERAGE_MEMORY_BUDGET: u64 = 512 * 1024 * 1024;

#[derive(Debug, Clone)]
pub struct CoverageRenderStats {
    pub tile_columns: u32,
    pub tile_rows: u32,
    pub tile_capacity: u32,
    pub measured_memberships: u64,
    pub allocation_bytes: u64,
    pub primitive_count: usize,
    pub line_count: usize,
    pub arc_count: usize,
    pub wall_ms: f64,
    pub output_sha256: String,
}

pub struct CoverageRenderOutput {
    pub coverage: Vec<f32>,
    pub linear_rgba: Vec<f32>,
    pub rgba8: Vec<u8>,
    pub errors: [u32; 4],
    pub stats: CoverageRenderStats,
}

/// Execute LIMES bin, analytic raster, and conflation-free resolve in order.
///
/// All three pass labels and shader hashes are recorded into the active render
/// certificate. Readbacks are included in the same 512 MiB fail-closed budget.
pub fn render_coverage(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    geometry: &CoverageGeometry,
) -> RenderResult<CoverageRenderOutput> {
    let started = Instant::now();
    let binner = CoverageBinner::new(device);
    let bins = binner.prepare(device, geometry)?;
    let rasterizer = CoverageRasterizer::new(device);
    let raster = rasterizer.prepare(device, geometry, &bins)?;
    let resolver = CoverageResolver::new(device);
    let resolved = resolver.prepare(device, geometry, &bins, &raster)?;

    let readback_bytes = raster
        .coverage_bytes
        .checked_add(resolved.output_bytes)
        .and_then(|bytes| bytes.checked_add(16))
        .ok_or_else(|| {
            RenderError::Budget("vector_coverage_readback_budget: byte-count overflow".into())
        })?;
    let allocation_bytes = resolved
        .allocation_bytes
        .checked_add(readback_bytes)
        .ok_or_else(|| {
            RenderError::Budget("vector_coverage_readback_budget: total-byte overflow".into())
        })?;
    if allocation_bytes > COVERAGE_MEMORY_BUDGET {
        return Err(RenderError::Budget(format!(
            "vector_coverage_readback_budget: required={allocation_bytes} \
             budget={COVERAGE_MEMORY_BUDGET}; refusing to truncate"
        )));
    }

    let coverage_readback =
        readback_buffer(device, "vf.Vector.Coverage.Readback", raster.coverage_bytes)?;
    let resolved_readback = readback_buffer(
        device,
        "vf.Vector.Coverage.ResolvedReadback",
        resolved.output_bytes,
    )?;
    let error_readback = readback_buffer(device, "vf.Vector.Coverage.ErrorReadback", 16)?;

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("vf.Vector.Coverage.RenderEncoder"),
    });
    let mut timing =
        crate::core::gpu_timing::OneShotTiming::for_device(device.clone(), queue.clone());
    let bin_scope = timing.begin(&mut encoder, "vector.coverage.bin");
    binner.encode(
        &mut encoder,
        &bins,
        u32::try_from(geometry.primitives.len()).map_err(|_| {
            RenderError::Budget("vector_coverage_render: primitive count exceeds u32".into())
        })?,
    );
    timing.end(&mut encoder, bin_scope, 1);

    let raster_scope = timing.begin(&mut encoder, "vector.coverage.raster");
    rasterizer.encode(&mut encoder, &raster, geometry);
    timing.end(&mut encoder, raster_scope, 1);

    let resolve_scope = timing.begin(&mut encoder, "vector.coverage.resolve");
    resolver.encode(&mut encoder, &resolved, geometry);
    timing.end(&mut encoder, resolve_scope, 1);
    timing.resolve(&mut encoder);

    encoder.copy_buffer_to_buffer(
        &raster.coverage,
        0,
        &coverage_readback,
        0,
        raster.coverage_bytes,
    );
    encoder.copy_buffer_to_buffer(
        &resolved.output,
        0,
        &resolved_readback,
        0,
        resolved.output_bytes,
    );
    encoder.copy_buffer_to_buffer(&bins.overflow, 0, &error_readback, 0, 16);
    queue.submit(Some(encoder.finish()));
    device.poll(wgpu::Maintain::Wait);

    if !timing.record_into_certificate() {
        crate::core::certificate::record_pass("vector.coverage.bin", 0.0, 1);
        crate::core::certificate::record_pass("vector.coverage.raster", 0.0, 1);
        crate::core::certificate::record_pass("vector.coverage.resolve", 0.0, 1);
    }

    let error_words = map_buffer::<u32>(device, &error_readback)?;
    let errors: [u32; 4] = error_words
        .as_slice()
        .try_into()
        .map_err(|_| RenderError::Readback("LIMES error block is not four words".into()))?;
    if errors != [0; 4] {
        return Err(RenderError::Render(format!(
            "{{\"status\":\"vector_coverage_overflow\",\"bin\":{},\
             \"active_list\":{},\"breakpoints\":{},\"reserved\":{}}}",
            errors[0], errors[1], errors[2], errors[3]
        )));
    }

    let coverage = map_buffer::<f32>(device, &coverage_readback)?;
    let linear_rgba = map_buffer::<f32>(device, &resolved_readback)?;
    let rgba8 = linear_to_straight_rgba8(&linear_rgba);
    let output_sha256 = crate::core::provenance::to_hex(&crate::core::provenance::sha256(&rgba8));
    let line_count = geometry
        .primitives
        .iter()
        .filter(|record| record.kind() == PrimitiveKind::Line)
        .count();
    let primitive_count = geometry.primitives.len();
    Ok(CoverageRenderOutput {
        coverage,
        linear_rgba,
        rgba8,
        errors,
        stats: CoverageRenderStats {
            tile_columns: bins.layout.tile_columns,
            tile_rows: bins.layout.tile_rows,
            tile_capacity: bins.layout.tile_capacity,
            measured_memberships: bins.layout.measured_memberships,
            allocation_bytes,
            primitive_count,
            line_count,
            arc_count: primitive_count - line_count,
            wall_ms: started.elapsed().as_secs_f64() * 1_000.0,
            output_sha256,
        },
    })
}

fn readback_buffer(
    device: &wgpu::Device,
    label: &'static str,
    size: u64,
) -> RenderResult<TrackedBuffer> {
    tracked_create_buffer(
        device,
        &wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        },
    )
}

fn map_buffer<T: bytemuck::Pod>(
    device: &wgpu::Device,
    buffer: &wgpu::Buffer,
) -> RenderResult<Vec<T>> {
    let slice = buffer.slice(..);
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).ok();
    });
    device.poll(wgpu::Maintain::Wait);
    pollster::block_on(receiver.receive())
        .ok_or_else(|| RenderError::Readback("LIMES map callback was cancelled".into()))?
        .map_err(|error| RenderError::Readback(format!("LIMES map failed: {error}")))?;
    let mapped = slice.get_mapped_range();
    let result = bytemuck::cast_slice(&mapped).to_vec();
    drop(mapped);
    buffer.unmap();
    Ok(result)
}

fn linear_to_straight_rgba8(linear: &[f32]) -> Vec<u8> {
    let mut output = Vec::with_capacity(linear.len());
    for pixel in linear.chunks_exact(4) {
        let alpha = pixel[3].clamp(0.0, 1.0);
        let inverse_alpha = if alpha > 0.0 { alpha.recip() } else { 0.0 };
        for &premultiplied in &pixel[..3] {
            let straight = (premultiplied * inverse_alpha).clamp(0.0, 1.0);
            output.push((straight * 255.0).round() as u8);
        }
        output.push((alpha * 255.0).round() as u8);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::linear_to_straight_rgba8;

    #[test]
    fn resolved_linear_premultiplied_values_convert_to_straight_rgba8() {
        assert_eq!(
            linear_to_straight_rgba8(&[0.25, 0.0, 0.5, 0.75]),
            [85, 0, 170, 191]
        );
        assert_eq!(linear_to_straight_rgba8(&[0.0; 4]), [0, 0, 0, 0]);
    }
}
