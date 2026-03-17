use super::*;

pub(super) fn dispatch(resources: &DispatchResources, width: u32, height: u32) {
    let g = ctx();
    let mut enc = g
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("pt-encoder"),
        });

    {
        let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("pt-cpass"),
            ..Default::default()
        });
        cpass.set_pipeline(&resources.pipeline);
        cpass.set_bind_group(0, &resources.bg0, &[]);
        cpass.set_bind_group(1, &resources.bg1, &[]);
        cpass.set_bind_group(2, &resources.bg2, &[]);
        cpass.set_bind_group(3, &resources.bg3, &[]);
        cpass.set_bind_group(4, &resources.bg4, &[]);
        cpass.dispatch_workgroups((width + 7) / 8, (height + 7) / 8, 1);
    }

    g.queue.submit([enc.finish()]);
    g.device.poll(wgpu::Maintain::Wait);
}
