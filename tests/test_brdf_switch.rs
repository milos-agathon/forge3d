// tests/test_brdf_switch.rs
// M2-05: Ensure runtime BRDF switching plumbs through the PBR pipeline without errors
// This is a sanity/integration test (not an image-based assertion).
// RELEVANT FILES: src/render/pbr_pass.rs, src/pipeline/pbr.rs, src/shaders/pbr.wgsl

#![cfg(all(feature = "enable-pbr", feature = "enable-tbn"))]

use forge3d::core::material::{PbrLighting, PbrMaterial};
use forge3d::pipeline::pbr::PbrSceneUniforms;
use forge3d::render::params::BrdfModel;
use forge3d::render::pbr_pass::PbrRenderPass;
use wgpu::{
    Color, CommandEncoderDescriptor, LoadOp, Operations, RenderPassColorAttachment,
    RenderPassDepthStencilAttachment, RenderPassDescriptor, TextureDescriptor, TextureDimension,
    TextureFormat, TextureUsages,
};

fn try_create_device_and_queue() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))?;

    let mut limits = adapter.limits();
    let baseline = wgpu::Limits::downlevel_defaults();
    limits = limits.using_resolution(baseline);
    let desired_storage_buffers = 8;
    limits.max_storage_buffers_per_shader_stage = limits
        .max_storage_buffers_per_shader_stage
        .max(desired_storage_buffers);

    let descriptor = wgpu::DeviceDescriptor {
        required_features: wgpu::Features::empty(),
        required_limits: limits,
        label: Some("pbr-brdf-switch-test-device"),
    };

    match pollster::block_on(adapter.request_device(&descriptor, None)) {
        Ok((device, queue)) => Some((device, queue)),
        Err(_) => None,
    }
}

#[test]
fn pbr_runtime_brdf_switch_executes() {
    let Some((device, queue)) = try_create_device_and_queue() else {
        eprintln!("Skipping BRDF switch test due to missing GPU adapter");
        return;
    };

    // Create pass and prepare once
    let material = PbrMaterial::default();
    let mut render_pass = PbrRenderPass::new(&device, &queue, material, /*enable_shadows=*/ true);

    let scene_uniforms = PbrSceneUniforms::default();
    let lighting = PbrLighting::default();

    render_pass.prepare(
        &device,
        &queue,
        TextureFormat::Rgba8Unorm,
        &scene_uniforms,
        &lighting,
    );

    // Render target setup
    let color_texture = device.create_texture(&TextureDescriptor {
        label: Some("pbr_pass_color"),
        size: wgpu::Extent3d { width: 4, height: 4, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let color_view = color_texture.create_view(&wgpu::TextureViewDescriptor::default());

    let depth_texture = device.create_texture(&TextureDescriptor {
        label: Some("pbr_pass_depth"),
        size: wgpu::Extent3d { width: 4, height: 4, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Depth32Float,
        usage: TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

    // Try several BRDFs sequentially; just ensure we can begin the pass without errors
    for model in [BrdfModel::Lambert, BrdfModel::CookTorranceGGX, BrdfModel::DisneyPrincipled] {
        render_pass.set_brdf_model(&queue, model);

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("pbr_brdf_switch_encoder"),
        });
        {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("pbr_brdf_switch_pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &color_view,
                    resolve_target: None,
                    ops: Operations { load: LoadOp::Clear(Color::BLACK), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(Operations { load: LoadOp::Clear(1.0), store: wgpu::StoreOp::Discard }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            render_pass.begin(&device, &mut pass);
            // No draw required for this sanity test; just binding/pipeline begin should succeed
        }
        queue.submit(Some(encoder.finish()));
    }
}
