use std::sync::Once;

use super::request::PreparedBrdfTileRequest;

pub(super) fn log_gpu_info_once() {
    static LOG_GPU_INFO: Once = Once::new();
    LOG_GPU_INFO.call_once(|| {
        let gpu_ctx = crate::core::gpu::ctx();
        let adapter_info = gpu_ctx.adapter.get_info();
        log::info!(
            "[M0] GPU Adapter: {} ({})",
            adapter_info.name,
            adapter_info.backend.to_str()
        );
        log::info!("[M0] Device Type: {:?}", adapter_info.device_type);
    });
}

pub(super) fn log_render_request(request: &PreparedBrdfTileRequest) {
    log::info!(
        "[M0] BRDF Tile Render: model={} roughness={:.3} size={}x{} flags: ndf_only={} g_only={} dfg_only={} spec_only={} r_vis={} exposure={:.3} light={:.3} base=({:.2},{:.2},{:.2})",
        request.model_u32,
        request.roughness,
        request.width,
        request.height,
        request.ndf_only,
        request.g_only,
        request.dfg_only,
        request.spec_only,
        request.roughness_visualize,
        request.exposure,
        request.light_intensity,
        request.base_color[0],
        request.base_color[1],
        request.base_color[2]
    );
}

pub(super) fn read_debug_dot_products(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    debug_buffer: &wgpu::Buffer,
) {
    let debug_staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("offscreen.brdf_tile.debug_staging"),
        size: 16,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("offscreen.brdf_tile.debug_readback"),
    });
    encoder.copy_buffer_to_buffer(debug_buffer, 0, &debug_staging, 0, 16);
    queue.submit(Some(encoder.finish()));
    device.poll(wgpu::Maintain::Wait);

    let slice = debug_staging.slice(..);
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).ok();
    });
    device.poll(wgpu::Maintain::Wait);

    if let Some(Ok(())) = pollster::block_on(receiver.receive()) {
        let data = slice.get_mapped_range();
        let values: &[u32; 4] = bytemuck::from_bytes(&data[..16]);
        let denom = 4294967295.0_f32;
        let min_nl = (values[0] as f32) / denom;
        let max_nl = (values[1] as f32) / denom;
        let min_nv = (values[2] as f32) / denom;
        let max_nv = (values[3] as f32) / denom;

        log::info!("[M1] Debug Dot Products:");
        log::info!("  N·L range: [{:.4}, {:.4}]", min_nl, max_nl);
        log::info!("  N·V range: [{:.4}, {:.4}]", min_nv, max_nv);

        drop(data);
        debug_staging.unmap();
    }
}
