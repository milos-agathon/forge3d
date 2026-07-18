use anyhow::{ensure, Result};

use super::debug::{log_gpu_info_once, log_render_request, read_debug_dot_products};
use super::request::BrdfTileRenderRequest;
use super::resources::{
    encode_render_pass, record_runtime_contract_observation, MeshBuffers, RenderTargets,
    TimestampResources, UniformResources,
};

pub(super) fn render_brdf_tile(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    request: BrdfTileRenderRequest,
) -> Result<Vec<u8>> {
    log_gpu_info_once();

    let request = request.prepare()?;
    log_render_request(&request);

    let targets = RenderTargets::new(device, request.width, request.height)?;
    let mesh = MeshBuffers::new(device, request.sphere_sectors, request.sphere_stacks)?;
    let pipeline = crate::offscreen::pipeline::BrdfTilePipeline::new(device)?;
    let resources = UniformResources::new(device, &pipeline, &request)?;
    let timestamps = TimestampResources::new(device)?;

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("offscreen.brdf_tile.encoder"),
    });
    timestamps.write_begin(&mut encoder);
    encode_render_pass(
        &mut encoder,
        &pipeline,
        &resources,
        &targets,
        &mesh,
        &timestamps,
    );
    timestamps.resolve(&mut encoder);

    queue.submit(Some(encoder.finish()));
    device.poll(wgpu::Maintain::Wait);

    // CENSOR: certify what actually executed — the shader module bound by this
    // render and the pass's live GPU duration (0.0 only when TIMESTAMP_QUERY is
    // absent or the readback is implausible; never fabricated).
    crate::core::shader_registry::record_shader_use("brdf_tile.shader");
    let gpu_ms = timestamps.read_gpu_ms(device, queue).unwrap_or(0.0);
    crate::core::certificate::record_pass("brdf.tile", gpu_ms, 1);

    let buffer = crate::renderer::readback::read_texture_tight(
        device,
        queue,
        &targets.render_target,
        (request.width, request.height),
        wgpu::TextureFormat::Rgba8Unorm,
    )?;

    if request.debug_dot_products {
        read_debug_dot_products(device, queue, &resources.debug_buffer);
    }

    ensure!(
        buffer.len() == request.expected_buffer_size(),
        "readback size mismatch: got {} bytes, expected {} for {}x{} RGBA8",
        buffer.len(),
        request.expected_buffer_size(),
        request.width,
        request.height
    );
    record_runtime_contract_observation(&request, &buffer)?;

    Ok(buffer)
}
