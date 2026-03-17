use anyhow::{ensure, Result};

use super::debug::{log_gpu_info_once, log_render_request, read_debug_dot_products};
use super::request::BrdfTileRenderRequest;
use super::resources::{
    encode_render_pass, MeshBuffers, RenderTargets, TimestampResources, UniformResources,
};

pub(super) fn render_brdf_tile(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    request: BrdfTileRenderRequest,
) -> Result<Vec<u8>> {
    log_gpu_info_once();

    let request = request.prepare()?;
    log_render_request(&request);

    let targets = RenderTargets::new(device, request.width, request.height);
    let mesh = MeshBuffers::new(device, request.sphere_sectors, request.sphere_stacks);
    let pipeline = crate::offscreen::pipeline::BrdfTilePipeline::new(device)?;
    let resources = UniformResources::new(device, &pipeline, &request);
    let timestamps = TimestampResources::new(device);

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

    Ok(buffer)
}
