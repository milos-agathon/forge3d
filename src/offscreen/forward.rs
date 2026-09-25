// src/offscreen/forward.rs
// General-purpose offscreen forward-raster path: depth-tested multi-draw
// rendering into an HDR color target with tracked host-visible readback.
// This is the shared offscreen mesh raster harness: consumers supply their
// own pipelines/bind groups (shading is caller-defined) while target
// creation, pass encoding, submission, and HDR readback are owned here and
// routed through core::hdr_readback + the global memory tracker. The
// ANAMNESIS graph executor lives in src/core/anamnesis/hdr_graph.rs.
// RELEVANT FILES: src/viewer/pbr_scene/mod.rs, src/core/hdr_readback.rs,
// src/core/anamnesis/hdr_graph.rs

use crate::core::error::RenderError;
use crate::core::resource_tracker::{tracked_create_texture, TrackedTexture};
use std::ops::Range;
use wgpu::{BindGroup, Buffer, CommandEncoder, Device, Queue, RenderPipeline, TextureView};

pub use crate::core::anamnesis::ForwardCacheDeclaration;

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
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::COPY_DST,
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

pub trait ForwardRecorder {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn record<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>);
}

impl ForwardRecorder for [ForwardDraw<'_>] {
    fn len(&self) -> usize {
        <[ForwardDraw]>::len(self)
    }

    fn record<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        for draw in self {
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
                _ => pass.draw(draw.vertices.clone(), 0..1),
            }
        }
    }
}

/// Encode one depth-tested forward pass over `draws` (in order), clearing the
/// color target to `clear` and depth to 1.0.
pub fn encode_forward_pass(
    encoder: &mut CommandEncoder,
    targets: &ForwardTargets,
    clear: wgpu::Color,
    draws: &[ForwardDraw],
) {
    encode_forward_recorded(encoder, targets, clear, draws);
}

fn encode_forward_recorded<R: ForwardRecorder + ?Sized>(
    encoder: &mut CommandEncoder,
    targets: &ForwardTargets,
    clear: wgpu::Color,
    draws: &R,
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
    draws.record(&mut pass);
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
    timing: Option<(&mut crate::core::gpu_timing::OneShotTiming, &str)>,
) -> Result<Vec<f32>, RenderError> {
    render_forward_hdr_incremental(device, queue, targets, clear, draws, timing, None)
        .map(|(pixels, _)| pixels)
}

/// Execute the real offscreen graph with optional ANAMNESIS restoration.
///
/// A hit rehydrates the cached tightly packed HDR bytes into the graph's
/// actual color texture with `Queue::write_texture`; the declared readback
/// pass then consumes that texture under the same compiled barrier plan used
/// on a miss. This is the production resource-restoration path exercised by
/// the ANAMNESIS GPU acceptance test.
pub fn render_forward_hdr_incremental(
    device: &Device,
    queue: &Queue,
    targets: &ForwardTargets,
    clear: wgpu::Color,
    draws: &[ForwardDraw],
    timing: Option<(&mut crate::core::gpu_timing::OneShotTiming, &str)>,
    cache: Option<&ForwardCacheDeclaration>,
) -> Result<(Vec<f32>, crate::core::anamnesis::CacheReport), RenderError> {
    render_forward_hdr_recorded(device, queue, targets, clear, draws, timing, cache)
}

/// Thin caller over the shared ANAMNESIS HDR graph executor
/// (`core::anamnesis::render_hdr_graph`). The graph shape, pass labels,
/// cache behavior, timing bracketing, and readback semantics are unchanged
/// from the former inline implementation.
pub fn render_forward_hdr_recorded<R: ForwardRecorder + ?Sized>(
    device: &Device,
    queue: &Queue,
    targets: &ForwardTargets,
    clear: wgpu::Color,
    draws: &R,
    mut timing: Option<(&mut crate::core::gpu_timing::OneShotTiming, &str)>,
    cache: Option<&ForwardCacheDeclaration>,
) -> Result<(Vec<f32>, crate::core::anamnesis::CacheReport), RenderError> {
    let target = crate::core::anamnesis::HdrGraphTarget {
        texture: &targets.color,
        width: targets.width,
        height: targets.height,
        format: targets.color_format,
    };
    let labels = crate::core::anamnesis::HdrGraphLabels {
        graphics: "offscreen.forward",
        readback: "offscreen.readback",
    };
    let mut render = || -> Result<(), RenderError> {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("offscreen-forward-encoder"),
        });
        let timing_scope = timing
            .as_mut()
            .and_then(|(t, label)| t.begin(&mut encoder, label));
        encode_forward_recorded(&mut encoder, targets, clear, draws);
        if let Some((t, _)) = timing.as_mut() {
            t.end(&mut encoder, timing_scope, draws.len() as u32);
            t.resolve(&mut encoder);
        }
        queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    };
    crate::core::anamnesis::render_hdr_graph(device, queue, &target, &labels, cache, &mut render)
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

    #[test]
    fn forward_graph_restores_a_real_cached_texture_before_readback() {
        let instance = wgpu::Instance::default();
        let Some(adapter) =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
        else {
            return;
        };
        let Ok((device, queue)) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None))
        else {
            return;
        };
        let shader = crate::core::shader_registry::create_labeled_shader_module(
            &device,
            "forward-cache-test-shader",
            FLAT_TRIANGLE_WGSL,
        );
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("forward-cache-test-layout"),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });
        let pipeline = crate::core::shader_registry::create_render_pipeline_scoped(
            &device,
            &wgpu::RenderPipelineDescriptor {
                label: Some("forward-cache-test-pipeline"),
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
        let draws = [ForwardDraw {
            pipeline: &pipeline,
            bind_group: None,
            vertex_buffer: None,
            index_buffer: None,
            index_count: 0,
            vertices: 0..3,
        }];
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("forge3d-forward-cache-{nonce}"));
        let cache = ForwardCacheDeclaration {
            root: root.clone(),
            max_bytes: 128 * 1024,
            verify_reads: true,
            pipeline_descriptor_bytes: [
                FLAT_TRIANGLE_WGSL.as_bytes(),
                b"vs/fs;rgba32float;depth32float;less_equal;sample_count=1",
            ]
            .concat(),
            uniform_bytes: bytemuck::cast_slice(&[0.0f64; 5]).to_vec(),
            external_input_bytes: b"bufferless-fullscreen-triangle".to_vec(),
            capability_fingerprint_bytes: b"standalone-test-device".to_vec(),
            engine_fingerprint_bytes: crate::core::anamnesis::EngineFingerprint::current()
                .canonical_bytes(),
        };

        let cold_targets =
            ForwardTargets::new(&device, 8, 4, wgpu::TextureFormat::Rgba32Float).unwrap();
        let (cold_pixels, cold_report) = render_forward_hdr_incremental(
            &device,
            &queue,
            &cold_targets,
            wgpu::Color::BLACK,
            &draws,
            None,
            Some(&cache),
        )
        .unwrap();
        assert_eq!(
            cold_report.misses,
            ["offscreen.forward", "offscreen.readback"]
        );
        assert!(cold_report.hits.is_empty());

        let warm_targets =
            ForwardTargets::new(&device, 8, 4, wgpu::TextureFormat::Rgba32Float).unwrap();
        let (warm_pixels, warm_report) = render_forward_hdr_incremental(
            &device,
            &queue,
            &warm_targets,
            wgpu::Color::BLACK,
            &draws,
            None,
            Some(&cache),
        )
        .unwrap();
        assert_eq!(warm_report.hits, ["offscreen.forward"]);
        assert_eq!(warm_report.misses, ["offscreen.readback"]);
        assert!(warm_report.wall_ms_saved > 0.0);
        assert_eq!(
            bytemuck::cast_slice::<f32, u8>(&cold_pixels),
            bytemuck::cast_slice::<f32, u8>(&warm_pixels),
            "restored GPU texture must produce byte-identical readback"
        );
        std::fs::remove_dir_all(root).unwrap();
    }
}
