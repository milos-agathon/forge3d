// src/core/anamnesis/hdr_graph.rs
// Neutral HDR render graph executor shared by the offscreen forward harness
// and the viewer PBR-scene capture path.
//
// The executor owns the two-pass graph shape (graphics pass -> readback
// pass) and the ANAMNESIS scheduler integration: on a miss the caller's
// `render` closure fully renders and submits into `target`, then the HDR
// texture is read back through `core::hdr::read_hdr_texture` — once to
// produce the pass's cached bytes, once for the declared readback pass
// that yields the result. On a hit the cached tight bytes are rehydrated
// into the real texture with `Queue::write_texture` and the same readback
// pass consumes them.
// RELEVANT FILES: src/offscreen/forward.rs, src/viewer/pbr_scene/mod.rs,
// src/core/anamnesis/scheduler.rs

use std::collections::BTreeMap;
use std::path::PathBuf;

use wgpu::{Device, Queue, Texture, TextureFormat};

use crate::core::error::RenderError;

use super::CacheReport;

/// Complete key material for cacheable caller-owned render work.
///
/// `RenderPipeline` and `BindGroup` are intentionally opaque in wgpu, so the
/// shared executor cannot reconstruct their descriptors. A caller that wants
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

/// The HDR texture a graph renders into and reads back.
pub struct HdrGraphTarget<'a> {
    pub texture: &'a Texture,
    pub width: u32,
    pub height: u32,
    pub format: TextureFormat,
}

/// Pass labels for the compiled graph. The graphics pass renders into the
/// target; the readback pass consumes it. Callers keep their own stable
/// labels (`"offscreen.forward"`/`"offscreen.readback"`,
/// `"viewer.pbr_scene"`/`"viewer.readback"`).
#[derive(Clone, Copy)]
pub struct HdrGraphLabels {
    pub graphics: &'static str,
    pub readback: &'static str,
}

fn restore_tight_bytes(
    queue: &Queue,
    target: &HdrGraphTarget<'_>,
    bytes: &[u8],
) -> Result<(), RenderError> {
    let bytes_per_pixel = match target.format {
        TextureFormat::Rgba32Float => 16usize,
        TextureFormat::Rgba16Float => 8usize,
        other => {
            return Err(RenderError::Render(format!(
                "hdr graph cache cannot restore {other:?}"
            )))
        }
    };
    let expected = target.width as usize * target.height as usize * bytes_per_pixel;
    if bytes.len() != expected {
        return Err(RenderError::Render(format!(
            "hdr graph cached byte length mismatch: got {}, expected {}",
            bytes.len(),
            expected
        )));
    }
    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: target.texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytes,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(target.width * bytes_per_pixel as u32),
            rows_per_image: Some(target.height),
        },
        wgpu::Extent3d {
            width: target.width,
            height: target.height,
            depth_or_array_layers: 1,
        },
    );
    Ok(())
}

/// Execute the two-pass HDR render graph with optional ANAMNESIS
/// restoration. `render` fully renders and submits into `target` on a miss
/// (it is invoked only when the graphics pass actually executes).
pub fn render_hdr_graph(
    device: &Device,
    queue: &Queue,
    target: &HdrGraphTarget<'_>,
    labels: &HdrGraphLabels,
    cache: Option<&ForwardCacheDeclaration>,
    render: &mut dyn FnMut() -> Result<(), RenderError>,
) -> Result<(Vec<f32>, CacheReport), RenderError> {
    use crate::core::framegraph_impl::{
        PassType, RendererGraphBuilder, ResourceDesc, ResourceType,
    };
    let extent = wgpu::Extent3d {
        width: target.width,
        height: target.height,
        depth_or_array_layers: 1,
    };
    let mut builder = RendererGraphBuilder::new();
    let external_input_resource = builder.add_resource(ResourceDesc {
        name: format!("{}.external_inputs", labels.graphics),
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
        name: format!("{}.color", labels.graphics),
        resource_type: ResourceType::ColorAttachment,
        format: Some(target.format),
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
        name: format!("{}.depth", labels.graphics),
        resource_type: ResourceType::DepthStencilAttachment,
        format: Some(TextureFormat::Depth32Float),
        extent: Some(extent),
        size: None,
        usage: Some(wgpu::TextureUsages::RENDER_ATTACHMENT),
        can_alias: false,
        is_transient: false,
    });
    let readback_resource = builder.add_resource(ResourceDesc {
        name: format!("{}.readback", labels.readback),
        resource_type: ResourceType::StorageBuffer,
        format: None,
        extent: None,
        size: Some(target.width as u64 * target.height as u64 * 16),
        usage: None,
        can_alias: false,
        is_transient: true,
    });
    let graphics_descriptor = cache
        .map(|declaration| declaration.pipeline_descriptor_bytes.clone())
        .unwrap_or_else(|| {
            let mut descriptor = Vec::new();
            descriptor.extend_from_slice(format!("{}/incomplete-v1\0", labels.graphics).as_bytes());
            descriptor.extend_from_slice(format!("{:?}", target.format).as_bytes());
            descriptor
        });
    let graphics_uniforms = cache
        .map(|declaration| declaration.uniform_bytes.clone())
        .unwrap_or_default();
    builder.add_pass(labels.graphics, PassType::Graphics, |pass| {
        pass.read(external_input_resource)
            .write(color_resource)
            .write(depth_resource)
            .pipeline_descriptor(graphics_descriptor)
            .uniform_bytes(graphics_uniforms);
        if cache.is_none() {
            pass.disable_cache(
                "wgpu render-pipeline descriptors are borrowed without reconstructible state",
            );
        }
        Ok(())
    })?;
    let mut readback_descriptor = Vec::new();
    readback_descriptor.extend_from_slice(format!("{}/v1\0", labels.readback).as_bytes());
    readback_descriptor.extend_from_slice(format!("{:?}", target.format).as_bytes());
    let mut readback_uniforms = Vec::new();
    readback_uniforms.extend_from_slice(&target.width.to_le_bytes());
    readback_uniforms.extend_from_slice(&target.height.to_le_bytes());
    builder.add_pass(labels.readback, PassType::Transfer, |pass| {
        pass.read(color_resource)
            .write(readback_resource)
            .pipeline_descriptor(readback_descriptor)
            .uniform_bytes(readback_uniforms)
            .disable_cache("GPU resource restore is owned by the ANAMNESIS scheduler");
        Ok(())
    })?;
    let mut graph = builder.compile()?;
    debug_assert_eq!(graph.labels, [labels.graphics, labels.readback]);
    match target.format {
        TextureFormat::Rgba32Float | TextureFormat::Rgba16Float => {}
        other => {
            return Err(RenderError::Render(format!(
                "hdr graph readback: unsupported color format {other:?}"
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
                    name if name == labels.graphics => {
                        render().map_err(std::io::Error::other)?;
                        let pixels = crate::core::hdr::read_hdr_texture(
                            device,
                            queue,
                            target.texture,
                            target.width,
                            target.height,
                            target.format,
                        )
                        .map_err(std::io::Error::other)?;
                        Ok(bytemuck::cast_slice(&pixels).to_vec())
                    }
                    name if name == labels.readback => {
                        if barriers.is_empty() {
                            return Err(std::io::Error::other(
                                "hdr graph readback lost its compiled color transition",
                            ));
                        }
                        result = Some(
                            crate::core::hdr::read_hdr_texture(
                                device,
                                queue,
                                target.texture,
                                target.width,
                                target.height,
                                target.format,
                            )
                            .map_err(std::io::Error::other)?,
                        );
                        Ok(Vec::new())
                    }
                    label => Err(std::io::Error::other(format!(
                        "unknown hdr graph pass {label:?}"
                    ))),
                },
                |pass, bytes, _barriers| match pass.name.as_str() {
                    name if name == labels.graphics => restore_tight_bytes(queue, target, bytes)
                        .map_err(|error| std::io::Error::other(error.to_string())),
                    label => Err(std::io::Error::other(format!(
                        "uncacheable pass {label:?} unexpectedly requested restoration"
                    ))),
                },
            )
            .map_err(|error| RenderError::Render(error.to_string()))?;
        let report = scheduler.into_report();
        let pixels = result.ok_or_else(|| {
            RenderError::Render("hdr graph readback pass did not produce pixels".into())
        })?;
        return Ok((pixels, report));
    }

    graph.execute_with_barriers(labels.graphics, |_barriers| render())?;
    let result = graph.execute_with_barriers(labels.readback, |barriers| {
        if barriers.is_empty() {
            return Err(RenderError::Render(
                "hdr graph readback lost its compiled color transition".into(),
            ));
        }
        crate::core::hdr::read_hdr_texture(
            device,
            queue,
            target.texture,
            target.width,
            target.height,
            target.format,
        )
        .map_err(RenderError::Readback)
    })?;
    graph.finish()?;
    Ok((result, CacheReport::default()))
}
