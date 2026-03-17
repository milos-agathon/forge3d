use super::{MeshBuffers, RenderTargets, TimestampResources, UniformResources};

pub(crate) fn encode_render_pass(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &crate::offscreen::pipeline::BrdfTilePipeline,
    resources: &UniformResources,
    targets: &RenderTargets,
    mesh: &MeshBuffers,
    timestamps: &TimestampResources,
) {
    let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("offscreen.brdf_tile.render"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: &targets.render_view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: &targets.depth_view,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Clear(1.0),
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        timestamp_writes: timestamps.timestamp_writes(),
        occlusion_query_set: None,
    });

    render_pass.set_pipeline(pipeline.pipeline());
    render_pass.set_bind_group(0, &resources.bind_group, &[]);
    render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
    render_pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
    render_pass.draw_indexed(0..mesh.num_indices, 0, 0..1);
}
