// src/offscreen/forward.rs
// General-purpose offscreen forward-raster path: depth-tested multi-draw
// rendering into an HDR color target with tracked host-visible readback.
// This is the shared offscreen mesh raster harness: consumers supply their
// own pipelines/bind groups (shading is caller-defined) while target
// creation, pass encoding, submission, and HDR readback are owned here and
// routed through core::hdr_readback + the global memory tracker.
// RELEVANT FILES: src/offscreen/adjudication_raster.rs, src/core/hdr_readback.rs

use crate::core::error::RenderError;
use crate::core::resource_tracker::{tracked_create_texture, TrackedTexture};
use std::ops::Range;
use wgpu::{BindGroup, Buffer, CommandEncoder, Device, Queue, RenderPipeline, TextureView};

/// HDR color + depth attachment pair for one offscreen forward pass.
pub struct ForwardTargets {
    pub width: u32,
    pub height: u32,
    pub color_format: wgpu::TextureFormat,
    pub color: TrackedTexture,
    color_view: TextureView,
    depth_view: TextureView,
}

impl ForwardTargets {
    /// Create a color target (renderable + copy source) with a matching
    /// Depth32Float attachment.
    pub fn new(
        device: &Device,
        width: u32,
        height: u32,
        color_format: wgpu::TextureFormat,
    ) -> Result<Self, RenderError> {
        if width == 0 || height == 0 {
            return Err(RenderError::Render(
                "offscreen forward pass requires non-zero width/height".into(),
            ));
        }
        let extent = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let color = tracked_create_texture(
            device,
            &wgpu::TextureDescriptor {
                label: Some("offscreen-forward-color"),
                size: extent,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: color_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            },
        )?;
        let depth = tracked_create_texture(
            device,
            &wgpu::TextureDescriptor {
                label: Some("offscreen-forward-depth"),
                size: extent,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            },
        )?;
        let color_view = color.create_view(&wgpu::TextureViewDescriptor::default());
        let depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());
        Ok(Self {
            width,
            height,
            color_format,
            color,
            color_view,
            depth_view,
        })
    }
}

/// One draw in an offscreen forward pass, executed in slice order.
/// Indexed draws set `vertex_buffer` + `index_buffer` + `index_count`;
/// buffer-less draws (fullscreen triangles) leave the buffers `None` and use
/// the raw `vertices` range.
pub struct ForwardDraw<'a> {
    pub pipeline: &'a RenderPipeline,
    pub bind_group: Option<&'a BindGroup>,
    pub vertex_buffer: Option<&'a Buffer>,
    pub index_buffer: Option<&'a Buffer>,
    pub index_count: u32,
    pub vertices: Range<u32>,
}

/// Encode one depth-tested forward pass over `draws` (in order), clearing the
/// color target to `clear` and depth to 1.0.
pub fn encode_forward_pass(
    encoder: &mut CommandEncoder,
    targets: &ForwardTargets,
    clear: wgpu::Color,
    draws: &[ForwardDraw],
) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("offscreen-forward-pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: &targets.color_view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(clear),
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
        timestamp_writes: None,
        occlusion_query_set: None,
    });
    for draw in draws {
        pass.set_pipeline(draw.pipeline);
        if let Some(bind_group) = draw.bind_group {
            pass.set_bind_group(0, bind_group, &[]);
        }
        match (draw.vertex_buffer, draw.index_buffer) {
            (Some(vb), Some(ib)) => {
                pass.set_vertex_buffer(0, vb.slice(..));
                pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..draw.index_count, 0, 0..1);
            }
            (Some(vb), None) => {
                pass.set_vertex_buffer(0, vb.slice(..));
                pass.draw(draw.vertices.clone(), 0..1);
            }
            _ => {
                pass.draw(draw.vertices.clone(), 0..1);
            }
        }
    }
}

/// Render `draws` into `targets` and read the HDR color target back as
/// tightly-packed f32 RGBA (row-major, `width * height * 4` values) through
/// the established `core::hdr_readback::read_hdr_texture` path, whose wrapper
/// already accounts the transient host-visible staging allocation.
///
/// `timing` (CENSOR F-04): when supplied as `(one_shot, label)`, the forward
/// pass is bracketed in a certificate timing scope on the encoder that
/// executes it and the queries are resolved before this submit. The
/// `OneShotTiming` MUST live on the same wgpu device as `device`; pass `None`
/// when driving a standalone device (tests).
pub fn render_forward_hdr(
    device: &Device,
    queue: &Queue,
    targets: &ForwardTargets,
    clear: wgpu::Color,
    draws: &[ForwardDraw],
    mut timing: Option<(&mut crate::core::gpu_timing::OneShotTiming, &str)>,
) -> Result<Vec<f32>, RenderError> {
    match targets.color_format {
        wgpu::TextureFormat::Rgba32Float | wgpu::TextureFormat::Rgba16Float => {}
        other => {
            return Err(RenderError::Render(format!(
                "offscreen forward HDR readback: unsupported color format {other:?}"
            )))
        }
    }
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("offscreen-forward-encoder"),
    });
    let timing_scope = timing
        .as_mut()
        .and_then(|(t, label)| t.begin(&mut encoder, label));
    encode_forward_pass(&mut encoder, targets, clear, draws);
    if let Some((t, _)) = timing.as_mut() {
        t.end(&mut encoder, timing_scope, draws.len() as u32);
        t.resolve(&mut encoder);
    }
    queue.submit(std::iter::once(encoder.finish()));

    // read_hdr_texture's wrapper already accounts the staging-buffer footprint;
    // no manual mirror here (it would double-count host-visible bytes).
    crate::core::hdr::read_hdr_texture(
        device,
        queue,
        &targets.color,
        targets.width,
        targets.height,
        targets.color_format,
    )
    .map_err(RenderError::Readback)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Minimal consumer independent of the adjudication raster: proves the
    // harness is a general offscreen mesh raster path, not an
    // adjudication-private alias.
    const FLAT_TRIANGLE_WGSL: &str = r#"
@vertex
fn vs(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    let x = f32(i32(vid & 1u) * 4 - 1);
    let y = f32(i32(vid >> 1u) * 4 - 1);
    return vec4<f32>(x, y, 0.5, 1.0);
}
@fragment
fn fs() -> @location(0) vec4<f32> {
    return vec4<f32>(0.25, 0.5, 0.75, 1.0);
}
"#;

    #[test]
    fn forward_harness_renders_and_reads_back_independently() {
        let instance = wgpu::Instance::default();
        let Some(adapter) =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
        else {
            return; // no GPU: silent-skip convention shared with queues/types.rs
        };
        let Ok((device, queue)) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None))
        else {
            return;
        };

        let shader = crate::core::shader_registry::create_labeled_shader_module(
            &device,
            "forward-harness-test-shader",
            FLAT_TRIANGLE_WGSL,
        );
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("forward-harness-test-layout"),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });
        let pipeline = crate::core::shader_registry::create_render_pipeline_scoped(
            &device,
            &wgpu::RenderPipelineDescriptor {
                label: Some("forward-harness-test-pipeline"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: "vs",
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::LessEqual,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: "fs",
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba32Float,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview: None,
            },
        );

        let targets = ForwardTargets::new(&device, 8, 4, wgpu::TextureFormat::Rgba32Float)
            .expect("forward targets");
        let draws = [ForwardDraw {
            pipeline: &pipeline,
            bind_group: None,
            vertex_buffer: None,
            index_buffer: None,
            index_count: 0,
            vertices: 0..3,
        }];
        // timing = None: this test drives a standalone device, and a timing
        // manager from the global context would live on a different device.
        let pixels =
            render_forward_hdr(&device, &queue, &targets, wgpu::Color::BLACK, &draws, None)
                .expect("forward render + readback");

        assert_eq!(pixels.len(), 8 * 4 * 4);
        for px in pixels.chunks_exact(4) {
            assert!((px[0] - 0.25).abs() < 1e-6, "r = {}", px[0]);
            assert!((px[1] - 0.5).abs() < 1e-6, "g = {}", px[1]);
            assert!((px[2] - 0.75).abs() < 1e-6, "b = {}", px[2]);
        }
    }
}
