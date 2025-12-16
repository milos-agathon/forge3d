// src/viewer/init/fallback_init.rs
// Fallback pipeline initialization for the Viewer

use std::sync::Arc;
use wgpu::{Device, RenderPipeline, TextureFormat};

/// Fallback shader source
pub const FALLBACK_SHADER: &str = r#"
@vertex
fn vs_fb(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    let x = f32((vid << 1u) & 2u);
    let y = f32(vid & 2u);
    return vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
}
@fragment
fn fs_fb() -> @location(0) vec4<f32> {
    return vec4<f32>(0.05, 0.0, 0.15, 1.0);
}
"#;

/// Create fallback render pipeline
pub fn create_fallback_pipeline(device: &Arc<Device>, surface_format: TextureFormat) -> RenderPipeline {
    let fb_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("viewer.fallback.shader"),
        source: wgpu::ShaderSource::Wgsl(FALLBACK_SHADER.into()),
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("viewer.fallback.pipeline"),
        layout: None,
        vertex: wgpu::VertexState {
            module: &fb_shader,
            entry_point: "vs_fb",
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &fb_shader,
            entry_point: "fs_fb",
            targets: &[Some(wgpu::ColorTargetState {
                format: surface_format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview: None,
    })
}
