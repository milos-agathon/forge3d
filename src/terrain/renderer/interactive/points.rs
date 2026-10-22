//! Transparent 3-D layers share opaque visibility but own their representative
//! depth. Their OIT accumulation can never erase the terrain/building GBuffer.
use super::*;

#[cfg(feature = "weighted-oit")]
pub(super) struct PointLayer {
    renderer: crate::vector::PointRenderer,
    oit: crate::vector::oit::WeightedOIT,
    representative: wgpu::RenderPipeline,
    shared: wgpu::BindGroup,
    count: u32,
    _color: TrackedTexture,
    color_view: wgpu::TextureView,
    edl: Option<(
        wgpu::RenderPipeline,
        wgpu::BindGroup,
        TrackedTexture,
        wgpu::TextureView,
    )>,
    overlay: crate::core::overlays::OverlayRenderer,
}

#[cfg(not(feature = "weighted-oit"))]
pub(super) struct PointLayer;

pub(super) fn edl_layer_count(layers: &[PointLayer]) -> usize {
    #[cfg(feature = "weighted-oit")]
    {
        layers.iter().filter(|layer| layer.edl.is_some()).count()
    }
    #[cfg(not(feature = "weighted-oit"))]
    {
        let _ = layers;
        0
    }
}

pub(super) fn prepare(
    scene: &TerrainScene,
    payload: Option<&Bound<'_, PyAny>>,
    state: &ConcordiaFrame,
    params: &crate::terrain::render_params::TerrainRenderParams,
) -> Result<Vec<PointLayer>> {
    let Some(payload) = payload.filter(|value| !value.is_none()) else {
        return Ok(Vec::new());
    };
    if payload.len()? == 0 {
        return Ok(Vec::new());
    }
    #[cfg(not(feature = "weighted-oit"))]
    {
        let _ = (scene, state, params);
        Err(anyhow!(
            "CONCORDIA transparent points require the weighted-oit feature"
        ))
    }
    #[cfg(feature = "weighted-oit")]
    {
        use numpy::PyUntypedArrayMethods;
        let device = &scene.device;
        let (width, height) = params.size_px;
        let mut result = Vec::new();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("concordia.points.shared.layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        for item in payload.iter()? {
            let item = item?;
            let array_object = item.get_item("points")?;
            let array = array_object.extract::<numpy::PyReadonlyArray2<f32>>()?;
            if array.shape()[1] != 8 {
                return Err(anyhow!(
                    "transparent points require N x 8 [NDCxy,size,RGBA,depth] inputs"
                ));
            }
            let mut instances = Vec::with_capacity(array.shape()[0]);
            for row in array.as_array().rows() {
                if !row.iter().all(|v| v.is_finite())
                    || row[2] <= 0.0
                    || !(0.0..=1.0).contains(&row[7])
                    || row.iter().skip(3).take(4).any(|v| !(0.0..=1.0).contains(v))
                {
                    return Err(anyhow!("transparent points require finite positions, positive size and normalized color/depth"));
                }
                instances.push(crate::vector::data::PointInstance {
                    position: [row[0], row[1]],
                    size: row[2],
                    color: [row[3], row[4], row[5], row[6]],
                    rotation: 0.0,
                    uv_offset: [0.0, 0.0],
                    _pad: row[7],
                });
            }
            if instances.is_empty() {
                continue;
            }
            let count = u32::try_from(instances.len())
                .map_err(|_| anyhow!("point count exceeds GPU draw instance domain"))?;
            let mut renderer =
                crate::vector::PointRenderer::new(device, wgpu::TextureFormat::Rgba16Float)?;
            let representative = renderer.configure_shared_depth(device, &layout)?;
            renderer.upload_points(device, &scene.queue, &instances)?;
            renderer.prepare_oit(
                &scene.queue,
                &glam::Mat4::IDENTITY.to_cols_array_2d(),
                [width as f32, height as f32],
                1.0,
            );
            let shared = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("concordia.points.shared"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: state.frame_uniform.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(state.cloud.shadow_view()),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(state.cloud.shadow_sampler()),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(
                            state.cartographic_depth_view(),
                        ),
                    },
                ],
            });
            let oit = crate::vector::oit::WeightedOIT::new(
                device,
                width,
                height,
                wgpu::TextureFormat::Rgba16Float,
            )?;
            let color = hdr_texture(device, width, height, "concordia.points.color")?;
            let color_view = color.create_view(&Default::default());
            let edl = if item.get_item("edl")?.extract::<bool>()? {
                let strength = item.get_item("edl_strength")?.extract::<f32>()?;
                let radius = item.get_item("edl_radius_px")?.extract::<f32>()?;
                if !strength.is_finite() || strength < 0.0 || !radius.is_finite() || radius <= 0.0 {
                    return Err(anyhow!(
                        "EDL parameters require finite nonnegative strength and positive radius"
                    ));
                }
                let (pipeline, group) =
                    oit.create_edl_pipeline(device, &color_view, strength, radius)?;
                let texture = hdr_texture(device, width, height, "concordia.points.edl")?;
                let view = texture.create_view(&Default::default());
                Some((pipeline, group, texture, view))
            } else {
                None
            };
            let mut overlay = crate::core::overlays::OverlayRenderer::new(
                device,
                wgpu::TextureFormat::Rgba8Unorm,
                false,
            )?;
            let view = edl.as_ref().map(|value| &value.3).unwrap_or(&color_view);
            overlay.recreate_bind_group(device, Some(view), None, None, None)?;
            overlay.set_enabled(true);
            overlay.upload_uniforms(&scene.queue);
            result.push(PointLayer {
                renderer,
                oit,
                representative,
                shared,
                count,
                _color: color,
                color_view,
                edl,
                overlay,
            });
        }
        let upload = scene.queue.submit(std::iter::empty());
        scene
            .device
            .poll(wgpu::Maintain::WaitForSubmissionIndex(upload));
        Ok(result)
    }
}

pub(super) fn encode(
    encoder: &mut wgpu::CommandEncoder,
    output: &wgpu::TextureView,
    layers: &[PointLayer],
) {
    #[cfg(not(feature = "weighted-oit"))]
    let _ = (encoder, output, layers);
    #[cfg(feature = "weighted-oit")]
    for layer in layers {
        {
            let mut pass = layer.oit.begin_accumulation(encoder);
            pass.set_bind_group(1, &layer.shared, &[]);
            layer.renderer.render_oit_prepared(&mut pass, layer.count);
        }
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("concordia.points.representative_depth"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: layer.oit.depth_view(),
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_bind_group(1, &layer.shared, &[]);
            layer.renderer.render_representative_depth(
                &mut pass,
                &layer.representative,
                layer.count,
            );
        }
        {
            let mut pass = color_pass(encoder, &layer.color_view, "concordia.points.compose", true);
            layer.oit.compose(&mut pass);
        }
        if let Some((pipeline, group, _, view)) = &layer.edl {
            let mut pass = color_pass(encoder, view, "concordia.points.edl", true);
            crate::core::shader_registry::record_shader_use("vf.Vector.PointEDL");
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, group, &[]);
            pass.draw(0..3, 0..1);
        }
        let mut pass = color_pass(encoder, output, "concordia.points.blend", false);
        crate::core::shader_registry::record_shader_use("overlays_shader");
        pass.set_pipeline(&layer.overlay.pipeline);
        pass.set_bind_group(0, &layer.overlay.bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

#[cfg(feature = "weighted-oit")]
fn color_pass<'a>(
    encoder: &'a mut wgpu::CommandEncoder,
    view: &'a wgpu::TextureView,
    label: &'a str,
    clear: bool,
) -> wgpu::RenderPass<'a> {
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: if clear {
                    wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT)
                } else {
                    wgpu::LoadOp::Load
                },
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
    })
}

#[cfg(feature = "weighted-oit")]
impl PointLayer {
    pub(super) fn representative_texture(&self) -> &wgpu::Texture {
        self.oit.depth_texture()
    }
}
