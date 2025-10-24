// src/util/debug_pattern.rs
// Minimal GPU debug pattern renderer for validating readback correctness
// Exists to isolate row-padding bugs without touching terrain shaders
// RELEVANT FILES: src/renderer/readback.rs, src/util/image_write.rs, tests/readback_depad.rs

use anyhow::{ensure, Result};
use std::borrow::Cow;

const DEBUG_PATTERN_WGSL: &str = r#"
struct VertexOutput {
    @builtin(position) clip: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(-1.0, 3.0),
        vec2<f32>(3.0, -1.0)
    );
    var out: VertexOutput;
    let pos = positions[vid];
    out.clip = vec4<f32>(pos, 0.0, 1.0);
    return out;
}

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    let x = u32(frag_pos.x);
    let y = u32(frag_pos.y);
    let r = f32(x & 255u) / 255.0;
    let g = f32(y & 255u) / 255.0;
    let b = 64.0 / 255.0;
    return vec4<f32>(r, g, b, 1.0);
}
"#;

/// Render a deterministic pattern to a single-sample Rgba8UnormSrgb texture.
/// Pattern: RGB = (x % 256, y % 256, 64), A = 255.
pub fn render_debug_pattern(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    width: u32,
    height: u32,
) -> Result<wgpu::Texture> {
    ensure!(width > 0 && height > 0, "pattern size must be positive");

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("forge3d.debug-pattern.shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(DEBUG_PATTERN_WGSL)),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("forge3d.debug-pattern.pipeline-layout"),
        bind_group_layouts: &[],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("forge3d.debug-pattern.pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main",
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    });

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("forge3d.debug-pattern.texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });

    let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("forge3d.debug-pattern.encoder"),
    });

    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("forge3d.debug-pattern.pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &texture_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(&pipeline);
        pass.draw(0..3, 0..1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    device.poll(wgpu::Maintain::Wait);

    Ok(texture)
}
