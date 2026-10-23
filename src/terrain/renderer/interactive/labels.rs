use super::*;
use crate::core::text_overlay::{TextInstance, TextOverlayRenderer};
use numpy::PyUntypedArrayMethods;

pub(super) struct LabelBatch {
    renderer: TextOverlayRenderer,
    alpha: f64,
}

pub(super) fn prepare_labels(
    scene: &TerrainScene,
    payload: Option<&Bound<'_, PyAny>>,
    width: u32,
    height: u32,
) -> Result<Vec<LabelBatch>> {
    let mut batches = Vec::new();
    let Some(payload) = payload.filter(|value| !value.is_none()) else {
        return Ok(batches);
    };
    for item in payload.iter()? {
        let item = item?;
        let alpha = item.get_item("alpha")?.extract::<f64>()?;
        if !alpha.is_finite() || !(0.0..=1.0).contains(&alpha) {
            return Err(anyhow!("label run alpha must be finite and in [0,1]"));
        }
        let atlas_object = item.get_item("atlas")?;
        let atlas = atlas_object.extract::<numpy::PyReadonlyArray3<u8>>()?;
        let shape = atlas.shape();
        let (atlas_h, atlas_w, channels) = (shape[0], shape[1], shape[2]);
        if atlas_w == 0 || atlas_h == 0 || !matches!(channels, 1 | 3 | 4) {
            return Err(anyhow!(
                "label atlas must have nonzero dimensions and 1, 3 or 4 channels"
            ));
        }
        let max_size = scene.device.limits().max_texture_dimension_2d as usize;
        if atlas_w > max_size || atlas_h > max_size {
            return Err(anyhow!("label atlas exceeds device limits"));
        }
        let source = atlas.as_slice()?;
        let mut rgba = Vec::with_capacity(atlas_w * atlas_h * 4);
        for pixel in source.chunks_exact(channels) {
            match channels {
                1 => rgba.extend_from_slice(&[pixel[0], pixel[0], pixel[0], 255]),
                3 => rgba.extend_from_slice(&[pixel[0], pixel[1], pixel[2], 255]),
                _ => rgba.extend_from_slice(pixel),
            }
        }
        let texture = tracked_create_texture(
            &scene.device,
            &wgpu::TextureDescriptor {
                label: Some("concordia.labels.atlas"),
                size: wgpu::Extent3d {
                    width: atlas_w as u32,
                    height: atlas_h as u32,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
        )?;
        scene.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &rgba,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(atlas_w as u32 * 4),
                rows_per_image: Some(atlas_h as u32),
            },
            wgpu::Extent3d {
                width: atlas_w as u32,
                height: atlas_h as u32,
                depth_or_array_layers: 1,
            },
        );
        let view = texture.create_view(&Default::default());
        let mut renderer =
            TextOverlayRenderer::new(&scene.device, wgpu::TextureFormat::Rgba8Unorm)?;
        renderer.set_atlas(texture, view);
        renderer.recreate_bind_group(&scene.device, None)?;
        renderer.set_channels(item.get_item("channels")?.extract::<u32>()?);
        renderer.set_smoothing(1.0);
        renderer.set_resolution(width, height);
        renderer.set_enabled(true);
        renderer.set_alpha(1.0);
        renderer.upload_uniforms(&scene.queue);
        let rects = item.get_item("rects")?.extract::<Vec<Vec<f32>>>()?;
        let mut instances = Vec::with_capacity(rects.len());
        for rect in rects {
            if !matches!(rect.len(), 17 | 18) || !rect.iter().all(|value| value.is_finite()) {
                return Err(anyhow!(
                    "a native label rect must contain 17 or 18 finite values"
                ));
            }
            if rect[2] <= 0.0
                || rect[3] <= 0.0
                || rect[16] < 0.0
                || !(0.0 <= rect[4] && rect[4] < rect[6] && rect[6] <= 1.0)
                || !(0.0 <= rect[5] && rect[5] < rect[7] && rect[7] <= 1.0)
            {
                return Err(anyhow!("invalid native label rectangle or atlas UV"));
            }
            let rotation = rect.get(17).copied().unwrap_or(0.0);
            let (min, max) = if rotation == 0.0 {
                (
                    [rect[0].max(0.0), rect[1].max(0.0)],
                    [(rect[0] + rect[2]).max(0.0), (rect[1] + rect[3]).max(0.0)],
                )
            } else {
                ([rect[0], rect[1]], [rect[0] + rect[2], rect[1] + rect[3]])
            };
            let color = [rect[8], rect[9], rect[10], rect[11]].map(|value| value.clamp(0.0, 1.0));
            let halo = [rect[12], rect[13], rect[14], rect[15]].map(|value| value.clamp(0.0, 1.0));
            let mut instance =
                TextInstance::new(min, max, [rect[4], rect[5]], [rect[6], rect[7]], color)
                    .with_halo(halo, rect[16]);
            instance.rotation = rotation;
            instances.push(instance);
        }
        renderer.upload_instances(&scene.device, &scene.queue, &instances)?;
        batches.push(LabelBatch { renderer, alpha });
    }
    Ok(batches)
}

pub(super) fn render_labels(
    scene: &TerrainScene,
    encoder: &mut wgpu::CommandEncoder,
    mut output: Arc<TrackedTexture>,
    batches: &[LabelBatch],
) -> Result<Arc<TrackedTexture>> {
    for batch in batches {
        if batch.alpha == 0.0 {
            continue;
        }
        let (width, height) = (output.width(), output.height());
        let target = if batch.alpha == 1.0 {
            output.clone()
        } else {
            let opaque = ldr_texture(scene, width, height, "concordia.labels.opaque")?;
            encoder.copy_texture_to_texture(
                copy(&output),
                copy(&opaque),
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );
            opaque
        };
        let view = target.create_view(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("concordia.labels.draw"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            crate::core::shader_registry::record_shader_use("text_overlay_shader");
            batch.renderer.render(&mut pass);
        }
        if batch.alpha < 1.0 {
            output = fade(scene, encoder, &output, &target, batch.alpha)?;
        }
    }
    Ok(output)
}

fn copy(texture: &wgpu::Texture) -> wgpu::ImageCopyTexture<'_> {
    wgpu::ImageCopyTexture {
        texture,
        mip_level: 0,
        origin: wgpu::Origin3d::ZERO,
        aspect: wgpu::TextureAspect::All,
    }
}

pub(super) fn ldr_texture(
    scene: &TerrainScene,
    width: u32,
    height: u32,
    label: &str,
) -> Result<Arc<TrackedTexture>> {
    Ok(Arc::new(tracked_create_texture(
        &scene.device,
        &wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        },
    )?))
}

fn fade(
    scene: &TerrainScene,
    encoder: &mut wgpu::CommandEncoder,
    base: &TrackedTexture,
    opaque: &TrackedTexture,
    alpha: f64,
) -> Result<Arc<TrackedTexture>> {
    // The public CHRONOS operation is float64 interpolation followed by ties-
    // even byte rounding. An alpha-only lookup preserves every possible byte
    // pair exactly without reading either image back or requiring GPU f64.
    let mut lookup = Vec::with_capacity(256 * 256);
    for below in 0..256 {
        for above in 0..256 {
            lookup.push(
                (below as f64 + alpha * f64::from(above - below))
                    .round_ties_even()
                    .clamp(0.0, 255.0) as u8,
            );
        }
    }
    let table = tracked_create_texture(
        &scene.device,
        &wgpu::TextureDescriptor {
            label: Some("concordia.labels.fade.lookup"),
            size: wgpu::Extent3d {
                width: 256,
                height: 256,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Uint,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        },
    )?;
    scene.queue.write_texture(
        copy(&table),
        &lookup,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(256),
            rows_per_image: Some(256),
        },
        wgpu::Extent3d {
            width: 256,
            height: 256,
            depth_or_array_layers: 1,
        },
    );
    let shader = crate::core::shader_registry::create_labeled_shader_module(
        &scene.device,
        "concordia.label_fade.shader",
        include_str!("../../../shaders/concordia_label_fade.wgsl"),
    );
    let pipeline = crate::core::shader_registry::create_compute_pipeline_scoped(
        &scene.device,
        &wgpu::ComputePipelineDescriptor {
            label: Some("concordia.labels.fade"),
            layout: None,
            module: &shader,
            entry_point: "main",
        },
    );
    let result = ldr_texture(scene, base.width(), base.height(), "concordia.labels.faded")?;
    let views = [
        base.create_view(&Default::default()),
        opaque.create_view(&Default::default()),
        table.create_view(&Default::default()),
        result.create_view(&Default::default()),
    ];
    let entries: Vec<_> = views
        .iter()
        .enumerate()
        .map(|(binding, view)| wgpu::BindGroupEntry {
            binding: binding as u32,
            resource: wgpu::BindingResource::TextureView(view),
        })
        .collect();
    let group = scene.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("concordia.labels.fade.bind_group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &entries,
    });
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("concordia.labels.fade"),
        timestamp_writes: None,
    });
    pass.set_pipeline(&pipeline);
    pass.set_bind_group(0, &group, &[]);
    pass.dispatch_workgroups(base.width().div_ceil(8), base.height().div_ceil(8), 1);
    Ok(result)
}
