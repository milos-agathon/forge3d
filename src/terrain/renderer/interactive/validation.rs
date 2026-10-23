//! Explicit after-beauty validation capture. This is excluded from the routine
//! single-readback frame and never supplies images back to GPU processing.
use super::*;
use numpy::IntoPyArray;

fn read(scene: &TerrainScene, texture: &wgpu::Texture) -> Result<Vec<u8>> {
    crate::core::texture_readback::read_texture_sync(
        &scene.device,
        &scene.queue,
        texture,
        texture.width(),
        texture.height(),
        texture.format(),
        "concordia.validation.readback",
    )
    .map_err(Into::into)
}

fn float_image<'py>(
    py: Python<'py>,
    scene: &TerrainScene,
    texture: &wgpu::Texture,
) -> Result<Bound<'py, numpy::PyArray2<f32>>> {
    let bytes = read(scene, texture)?;
    let values: Vec<f32> = match texture.format() {
        wgpu::TextureFormat::R32Float | wgpu::TextureFormat::Depth32Float => bytes
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes([v[0], v[1], v[2], v[3]]))
            .collect(),
        wgpu::TextureFormat::R8Unorm => bytes.into_iter().map(|v| f32::from(v) / 255.0).collect(),
        other => return Err(anyhow!("validation scalar format {other:?} is unavailable")),
    };
    Ok(ndarray::Array2::from_shape_vec(
        (texture.height() as usize, texture.width() as usize),
        values,
    )?
    .into_pyarray_bound(py))
}

/// Replay only the exact submitted cartography into transparent targets.
/// These extra draws and transfers are explicitly requested after beauty; they
/// are never inputs to the routine interactive frame or its certificate.
fn cartographic_coverage(
    scene: &TerrainScene,
    state: &ConcordiaFrame,
) -> Result<(Arc<TrackedTexture>, Arc<TrackedTexture>)> {
    let config = state.gbuffer().config();
    let labels = super::labels::ldr_texture(
        scene,
        config.width,
        config.height,
        "concordia.validation.label_coverage",
    )?;
    let vectors = super::labels::ldr_texture(
        scene,
        config.width,
        config.height,
        "concordia.validation.vector_coverage",
    )?;
    let label_view = labels.create_view(&Default::default());
    let vector_view = vectors.create_view(&Default::default());
    let mut encoder = scene
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("concordia.validation.cartography"),
        });
    {
        let attachments = [&label_view, &vector_view].map(|view| {
            Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })
        });
        let _clear = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("concordia.validation.coverage.clear"),
            color_attachments: &attachments,
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
    }
    super::overlays::draw(&mut encoder, &vector_view, &state.vector_batches);
    let labels = super::labels::render_labels(scene, &mut encoder, labels, &state.label_batches)?;
    scene.queue.submit(Some(encoder.finish()));
    Ok((labels, vectors))
}

fn analytic_ao_resolve(scene: &TerrainScene, state: &ConcordiaFrame) -> Result<Vec<f32>> {
    fn input(
        scene: &TerrainScene,
        format: wgpu::TextureFormat,
        bytes: &[u8],
    ) -> Result<TrackedTexture> {
        let texture = tracked_create_texture(
            &scene.device,
            &wgpu::TextureDescriptor {
                label: Some("concordia.validation.analytic_input"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
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
            bytes,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: None,
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        Ok(texture)
    }
    let white_bits = [half::f16::ONE.to_bits(); 4];
    let white = input(
        scene,
        wgpu::TextureFormat::Rgba16Float,
        bytemuck::cast_slice(&white_bits),
    )?;
    let white_view = white.create_view(&Default::default());
    let depth = input(scene, wgpu::TextureFormat::R32Float, &1.0f32.to_le_bytes())?;
    let depth_view = depth.create_view(&Default::default());
    let uniform = tracked_create_buffer_init(
        &scene.device,
        &wgpu::util::BufferInitDescriptor {
            label: Some("concordia.validation.analytic_uniform"),
            contents: bytemuck::bytes_of(&ComposeUniforms {
                inv_view: glam::Mat4::IDENTITY.to_cols_array_2d(),
                inv_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
                terrain_scale_bias: [0.0; 4],
                weights: [0.0; 4],
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        },
    )?;
    let mut encoder = scene
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("concordia.validation.analytic_resolve"),
        });
    let mut outputs = Vec::new();
    for (height, gtao, kind) in [
        (0.75f32, 0.75f32, 1u8),
        (1.0, 1.0, 1),
        (0.5, 1.0, 1),
        (1.0, 0.5, 1),
        (0.75, 0.75, 2),
    ] {
        let height = input(scene, wgpu::TextureFormat::R32Float, &height.to_le_bytes())?;
        let gtao = input(scene, wgpu::TextureFormat::R32Float, &gtao.to_le_bytes())?;
        let kind = input(scene, wgpu::TextureFormat::R8Uint, &[kind])?;
        let h = height.create_view(&Default::default());
        let g = gtao.create_view(&Default::default());
        let k = kind.create_view(&Default::default());
        let output = hdr_texture(&scene.device, 1, 1, "concordia.validation.analytic_output")?;
        let view = output.create_view(&Default::default());
        let views = [&white_view, &g, &h, &white_view, &white_view, &depth_view];
        let mut entries: Vec<_> = views
            .iter()
            .enumerate()
            .map(|(binding, view)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: wgpu::BindingResource::TextureView(view),
            })
            .collect();
        entries.extend([
            wgpu::BindGroupEntry {
                binding: 6,
                resource: uniform.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: wgpu::BindingResource::TextureView(&k),
            },
        ]);
        let group = scene.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("concordia.validation.analytic_group"),
            layout: &state.compose_layout,
            entries: &entries,
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("concordia.validation.analytic_ao"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&state.ao_pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        outputs.push(output);
    }
    scene.queue.submit(Some(encoder.finish()));
    outputs
        .iter()
        .map(|texture| {
            let bytes = read(scene, texture)?;
            Ok(half::f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32())
        })
        .collect()
}

#[pymethods]
impl TerrainRenderer {
    /// Read measurement data after the final Frame has been read. This explicit
    /// validation operation is outside render_interactive's beauty pipeline.
    fn _interactive_validation_data<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let capture = || -> Result<Bound<'py, pyo3::types::PyDict>> {
            let scene = &self.scene;
            if !scene
                .concordia_readbacks
                .as_ref()
                .is_some_and(|counter| counter.load(std::sync::atomic::Ordering::Relaxed) > 0)
            {
                return Err(anyhow!(
                    "read the final interactive beauty Frame before requesting validation data"
                ));
            }
            let state = scene
                .concordia
                .as_ref()
                .ok_or_else(|| anyhow!("no completed interactive frame"))?;
            let data = pyo3::types::PyDict::new_bound(py);
            data.set_item("analytic_ao_resolve", analytic_ao_resolve(scene, state)?)?;
            if let Some(depth) = state.cartography_depth.as_ref() {
                data.set_item("cartography_depth", float_image(py, scene, &depth.linear)?)?;
            }
            let gbuffer = state.gbuffer();
            data.set_item("depth", float_image(py, scene, &gbuffer.depth_texture)?)?;
            data.set_item(
                "surface_kind",
                ndarray::Array2::from_shape_vec(
                    (
                        gbuffer.config().height as usize,
                        gbuffer.config().width as usize,
                    ),
                    read(scene, &state._surface_kind)?,
                )?
                .into_pyarray_bound(py),
            )?;
            #[cfg(feature = "weighted-oit")]
            {
                let point_depths = state
                    .point_layers
                    .iter()
                    .map(|layer| float_image(py, scene, layer.representative_texture()))
                    .collect::<Result<Vec<_>>>()?;
                data.set_item("point_depths", point_depths)?;
                data.set_item("point_depth_convention", "normalized_device_depth")?;
            }
            if let Some(texture) = state.effects.ao_raw_texture() {
                data.set_item("gtao_raw", float_image(py, scene, texture)?)?;
            }
            let material = read(scene, &gbuffer.material_texture)?;
            data.set_item(
                "material",
                ndarray::Array3::from_shape_vec(
                    (
                        gbuffer.config().height as usize,
                        gbuffer.config().width as usize,
                        4,
                    ),
                    material,
                )?
                .into_pyarray_bound(py),
            )?;
            if let Some(texture) = state.effects.ao_resolved_texture() {
                data.set_item("gtao_visibility", float_image(py, scene, texture)?)?;
            }
            if state.height_ao_active {
                if let Some(texture) = scene
                    .height_ao_texture
                    .lock()
                    .map_err(|_| anyhow!("height AO mutex poisoned"))?
                    .as_ref()
                {
                    data.set_item("height_visibility", float_image(py, scene, texture)?)?;
                }
            }
            let cloud = state.cloud.shadow_texture();
            let bytes = read(scene, cloud)?;
            data.set_item(
                "cloud_field",
                ndarray::Array3::from_shape_vec(
                    (cloud.height() as usize, cloud.width() as usize, 4),
                    bytes,
                )?
                .into_pyarray_bound(py),
            )?;
            let velocity: Vec<f32> = read(scene, &gbuffer.velocity_texture)?
                .chunks_exact(2)
                .map(|v| half::f16::from_bits(u16::from_le_bytes([v[0], v[1]])).to_f32())
                .collect();
            data.set_item(
                "velocity",
                ndarray::Array3::from_shape_vec(
                    (
                        gbuffer.config().height as usize,
                        gbuffer.config().width as usize,
                        2,
                    ),
                    velocity,
                )?
                .into_pyarray_bound(py),
            )?;
            let normal = read(scene, &gbuffer.normal_texture)?;
            let normal: Vec<f32> = normal
                .chunks_exact(2)
                .map(|v| half::f16::from_bits(u16::from_le_bytes([v[0], v[1]])).to_f32())
                .collect();
            data.set_item(
                "normal_roughness",
                ndarray::Array3::from_shape_vec(
                    (
                        gbuffer.config().height as usize,
                        gbuffer.config().width as usize,
                        4,
                    ),
                    normal,
                )?
                .into_pyarray_bound(py),
            )?;
            if let Some(texture) = state.geometry_hdr.as_ref() {
                let values = crate::core::hdr::read_hdr_texture(
                    &scene.device,
                    &scene.queue,
                    texture,
                    texture.width(),
                    texture.height(),
                    texture.format(),
                )
                .map_err(|error| anyhow!(error))?;
                data.set_item(
                    "geometry_hdr",
                    ndarray::Array3::from_shape_vec(
                        (texture.height() as usize, texture.width() as usize, 4),
                        values,
                    )?
                    .into_pyarray_bound(py),
                )?;
            }
            if state.composition_encoded {
                let texture = &state.scratch[state.current_scratch];
                let values = crate::core::hdr::read_hdr_texture(
                    &scene.device,
                    &scene.queue,
                    texture,
                    texture.width(),
                    texture.height(),
                    texture.format(),
                )
                .map_err(|error| anyhow!(error))?;
                data.set_item(
                    "last_composition_hdr",
                    ndarray::Array3::from_shape_vec(
                        (texture.height() as usize, texture.width() as usize, 4),
                        values,
                    )?
                    .into_pyarray_bound(py),
                )?;
            }
            let (labels, vectors) = cartographic_coverage(scene, state)?;
            data.set_item(
                "label_coverage",
                ndarray::Array3::from_shape_vec(
                    (labels.height() as usize, labels.width() as usize, 4),
                    read(scene, &labels)?,
                )?
                .into_pyarray_bound(py),
            )?;
            // Preserve the actual output-space RGBA8 coverage bytes.
            data.set_item(
                "vector_coverage",
                ndarray::Array3::from_shape_vec(
                    (vectors.height() as usize, vectors.width() as usize, 4),
                    read(scene, &vectors)?,
                )?
                .into_pyarray_bound(py),
            )?;
            let u = &state.current_uniforms;
            data.set_item("view", u.view)?;
            data.set_item("proj", u.proj)?;
            data.set_item("inv_view", u.inv_view)?;
            data.set_item("inv_proj", u.inv_proj)?;
            data.set_item("cloud_scale_bias", u.cloud_scale_bias)?;
            Ok(data)
        };
        capture().map_err(|error| {
            PyRuntimeError::new_err(format!("CONCORDIA validation capture failed: {error:#}"))
        })
    }
}
