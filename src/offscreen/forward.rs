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
use std::collections::BTreeMap;
use std::ops::Range;
use std::path::PathBuf;
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

    fn restore_tight_bytes(&self, queue: &Queue, bytes: &[u8]) -> Result<(), RenderError> {
        let bytes_per_pixel = match self.color_format {
            wgpu::TextureFormat::Rgba32Float => 16usize,
            wgpu::TextureFormat::Rgba16Float => 8usize,
            other => {
                return Err(RenderError::Render(format!(
                    "offscreen forward cache cannot restore {other:?}"
                )))
            }
        };
        let expected = self.width as usize * self.height as usize * bytes_per_pixel;
        if bytes.len() != expected {
            return Err(RenderError::Render(format!(
                "offscreen forward cached byte length mismatch: got {}, expected {}",
                bytes.len(),
                expected
            )));
        }
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.color,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytes,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(self.width * bytes_per_pixel as u32),
                rows_per_image: Some(self.height),
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
        Ok(())
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

/// Complete key material for cacheable caller-owned forward draws.
///
/// `RenderPipeline` and `BindGroup` are intentionally opaque in wgpu, so the
/// shared harness cannot reconstruct their descriptors. A caller that wants
/// caching must supply the exact canonical pipeline descriptor, uploaded
/// uniform bytes, and every external draw input. Omitting this declaration
/// leaves the pass conservatively uncacheable.
pub struct ForwardCacheDeclaration {
    pub root: PathBuf,
    pub max_bytes: u64,
    pub verify_reads: bool,
    pub pipeline_descriptor_bytes: Vec<u8>,
    pub uniform_bytes: Vec<u8>,
    pub external_input_bytes: Vec<u8>,
    pub capability_fingerprint_bytes: Vec<u8>,
    pub engine_fingerprint_bytes: Vec<u8>,
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
    mut timing: Option<(&mut crate::core::gpu_timing::OneShotTiming, &str)>,
    cache: Option<&ForwardCacheDeclaration>,
) -> Result<(Vec<f32>, crate::core::anamnesis::CacheReport), RenderError> {
    use crate::core::framegraph_impl::{
        PassType, RendererGraphBuilder, ResourceDesc, ResourceType,
    };
    let extent = wgpu::Extent3d {
        width: targets.width,
        height: targets.height,
        depth_or_array_layers: 1,
    };
    let mut builder = RendererGraphBuilder::new();
    let external_input_resource = builder.add_resource(ResourceDesc {
        name: "offscreen.forward.external_inputs".into(),
        resource_type: ResourceType::StorageBuffer,
        format: None,
        extent: None,
        size: Some(
            cache
                .map(|declaration| declaration.external_input_bytes.len())
                .unwrap_or(1)
                .max(1) as u64,
        ),
        usage: None,
        can_alias: false,
        is_transient: false,
    });
    let color_resource = builder.add_resource(ResourceDesc {
        name: "offscreen.forward.color".into(),
        resource_type: ResourceType::ColorAttachment,
        format: Some(targets.color_format),
        extent: Some(extent),
        size: None,
        usage: Some(
            wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
        ),
        can_alias: false,
        is_transient: false,
    });
    let depth_resource = builder.add_resource(ResourceDesc {
        name: "offscreen.forward.depth".into(),
        resource_type: ResourceType::DepthStencilAttachment,
        format: Some(wgpu::TextureFormat::Depth32Float),
        extent: Some(extent),
        size: None,
        usage: Some(wgpu::TextureUsages::RENDER_ATTACHMENT),
        can_alias: false,
        is_transient: false,
    });
    let readback_resource = builder.add_resource(ResourceDesc {
        name: "offscreen.forward.readback".into(),
        resource_type: ResourceType::StorageBuffer,
        format: None,
        extent: None,
        size: Some(targets.width as u64 * targets.height as u64 * 16),
        usage: None,
        can_alias: false,
        is_transient: true,
    });
    let forward_descriptor = cache
        .map(|declaration| declaration.pipeline_descriptor_bytes.clone())
        .unwrap_or_else(|| {
            let mut descriptor = Vec::new();
            descriptor.extend_from_slice(b"forge3d.offscreen.forward/incomplete-v1\0");
            descriptor.extend_from_slice(format!("{:?}", targets.color_format).as_bytes());
            descriptor.extend_from_slice(&(draws.len() as u64).to_le_bytes());
            descriptor
        });
    let forward_uniforms = cache
        .map(|declaration| declaration.uniform_bytes.clone())
        .unwrap_or_else(|| {
            let mut uniforms = Vec::with_capacity(40);
            for value in [clear.r, clear.g, clear.b, clear.a, 1.0] {
                uniforms.extend_from_slice(&value.to_bits().to_le_bytes());
            }
            uniforms
        });
    builder.add_pass("offscreen.forward", PassType::Graphics, |pass| {
        pass.read(external_input_resource)
            .write(color_resource)
            .write(depth_resource)
            .pipeline_descriptor(forward_descriptor)
            .uniform_bytes(forward_uniforms);
        if cache.is_none() {
            pass.disable_cache(
                "wgpu render-pipeline descriptors are borrowed without reconstructible state",
            );
        }
        Ok(())
    })?;
    let mut readback_descriptor = Vec::new();
    readback_descriptor.extend_from_slice(b"forge3d.offscreen.readback/v1\0");
    readback_descriptor.extend_from_slice(format!("{:?}", targets.color_format).as_bytes());
    let mut readback_uniforms = Vec::new();
    readback_uniforms.extend_from_slice(&targets.width.to_le_bytes());
    readback_uniforms.extend_from_slice(&targets.height.to_le_bytes());
    builder.add_pass("offscreen.readback", PassType::Transfer, |pass| {
        pass.read(color_resource)
            .write(readback_resource)
            .pipeline_descriptor(readback_descriptor)
            .uniform_bytes(readback_uniforms)
            .disable_cache("GPU resource restore is owned by the ANAMNESIS scheduler");
        Ok(())
    })?;
    let mut graph = builder.compile()?;
    debug_assert_eq!(graph.labels, ["offscreen.forward", "offscreen.readback"]);
    match targets.color_format {
        wgpu::TextureFormat::Rgba32Float | wgpu::TextureFormat::Rgba16Float => {}
        other => {
            return Err(RenderError::Render(format!(
                "offscreen forward HDR readback: unsupported color format {other:?}"
            )))
        }
    }
    if let Some(declaration) = cache {
        use crate::core::anamnesis::{leaf_key, ContentStore, GraphScheduler};
        let store = ContentStore::new(
            &declaration.root,
            declaration.max_bytes.max(1),
            declaration.verify_reads,
        )
        .map_err(|error| RenderError::Render(error.to_string()))?;
        let leaf_keys = BTreeMap::from([(
            external_input_resource,
            leaf_key(&declaration.external_input_bytes),
        )]);
        let mut scheduler = GraphScheduler::new(
            store,
            declaration.capability_fingerprint_bytes.clone(),
            declaration.engine_fingerprint_bytes.clone(),
        );
        let mut result = None;
        scheduler
            .execute_graph(
                &graph,
                &leaf_keys,
                |pass, barriers| match pass.name.as_str() {
                    "offscreen.forward" => {
                        let mut encoder =
                            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
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
                        let pixels = crate::core::hdr::read_hdr_texture(
                            device,
                            queue,
                            &targets.color,
                            targets.width,
                            targets.height,
                            targets.color_format,
                        )
                        .map_err(std::io::Error::other)?;
                        Ok(bytemuck::cast_slice(&pixels).to_vec())
                    }
                    "offscreen.readback" => {
                        if barriers.is_empty() {
                            return Err(std::io::Error::other(
                                "offscreen readback lost its compiled color transition",
                            ));
                        }
                        result = Some(
                            crate::core::hdr::read_hdr_texture(
                                device,
                                queue,
                                &targets.color,
                                targets.width,
                                targets.height,
                                targets.color_format,
                            )
                            .map_err(std::io::Error::other)?,
                        );
                        Ok(Vec::new())
                    }
                    label => Err(std::io::Error::other(format!(
                        "unknown offscreen graph pass {label:?}"
                    ))),
                },
                |pass, bytes, _barriers| match pass.name.as_str() {
                    "offscreen.forward" => targets
                        .restore_tight_bytes(queue, bytes)
                        .map_err(|error| std::io::Error::other(error.to_string())),
                    label => Err(std::io::Error::other(format!(
                        "uncacheable pass {label:?} unexpectedly requested restoration"
                    ))),
                },
            )
            .map_err(|error| RenderError::Render(error.to_string()))?;
        let report = scheduler.into_report();
        let pixels = result.ok_or_else(|| {
            RenderError::Render("offscreen readback pass did not produce pixels".into())
        })?;
        return Ok((pixels, report));
    }

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("offscreen-forward-encoder"),
    });
    let timing_scope = timing
        .as_mut()
        .and_then(|(t, label)| t.begin(&mut encoder, label));
    graph.execute_with_barriers("offscreen.forward", |_barriers| {
        encode_forward_pass(&mut encoder, targets, clear, draws);
        Ok::<(), RenderError>(())
    })?;
    if let Some((t, _)) = timing.as_mut() {
        t.end(&mut encoder, timing_scope, draws.len() as u32);
        t.resolve(&mut encoder);
    }
    queue.submit(std::iter::once(encoder.finish()));
    let result = graph.execute_with_barriers("offscreen.readback", |barriers| {
        if barriers.is_empty() {
            return Err(RenderError::Render(
                "offscreen readback lost its compiled color transition".into(),
            ));
        }
        crate::core::hdr::read_hdr_texture(
            device,
            queue,
            &targets.color,
            targets.width,
            targets.height,
            targets.color_format,
        )
        .map_err(RenderError::Readback)
    })?;
    graph.finish()?;
    Ok((result, crate::core::anamnesis::CacheReport::default()))
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
