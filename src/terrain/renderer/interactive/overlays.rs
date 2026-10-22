use super::*;
use crate::core::overlays::OverlayRenderer;
use numpy::PyUntypedArrayMethods;

pub(super) fn prepare(
    scene: &TerrainScene,
    payload: Option<&Bound<'_, PyAny>>,
    params: &crate::terrain::render_params::TerrainRenderParams,
    depth: &wgpu::TextureView,
    surface_kind: Option<&Arc<wgpu::TextureView>>,
    format: wgpu::TextureFormat,
) -> Result<Vec<OverlayRenderer>> {
    let Some(payload) = payload.filter(|value| !value.is_none()) else {
        return Ok(Vec::new());
    };
    let mut result = Vec::new();
    let (_, _, projection) = super::camera_matrices(params);
    for item in payload.iter()? {
        let item = item?;
        let rgba_object = item.get_item("rgba")?;
        let rgba = rgba_object.extract::<numpy::PyReadonlyArray3<u8>>()?;
        let shape = rgba.shape();
        let (height, width) = (shape[0] as u32, shape[1] as u32);
        if shape[2] != 4
            || width == 0
            || height == 0
            || width > scene.device.limits().max_texture_dimension_2d
            || height > scene.device.limits().max_texture_dimension_2d
        {
            return Err(anyhow!(
                "cartographic input must be a device-sized RGBA image"
            ));
        }
        let depth_test = item.get_item("depth_test")?.extract::<bool>()?;
        let mut renderer = OverlayRenderer::new_shared_depth(&scene.device, format)?;
        if let Some(view) = surface_kind {
            renderer.shared_surface_kind = Some(Arc::clone(view));
        }
        let texture = upload(
            scene,
            width,
            height,
            wgpu::TextureFormat::Rgba8Unorm,
            rgba.as_slice()?,
            4,
        )?;
        let view = texture.create_view(&Default::default());
        renderer.overlay_tex = Some(texture);
        renderer.overlay_view = Some(view);
        let authored_depth = if depth_test {
            let depth_object = item.get_item("depth")?;
            let depth = depth_object.extract::<numpy::PyReadonlyArray2<f32>>()?;
            if depth.shape() != &shape[..2]
                || !depth
                    .as_array()
                    .iter()
                    .all(|v| v.is_finite() && (0.0..=1.0).contains(v))
            {
                return Err(anyhow!(
                    "cartographic depth must match coverage and contain normalized finite depth"
                ));
            }
            Some(upload(
                scene,
                width,
                height,
                wgpu::TextureFormat::R32Float,
                bytemuck::cast_slice(depth.as_slice()?),
                4,
            )?)
        } else {
            None
        };
        let authored_view = authored_depth
            .as_ref()
            .map(|texture| texture.create_view(&Default::default()));
        renderer.recreate_bind_group(
            &scene.device,
            None,
            authored_view.as_ref(),
            None,
            Some(depth),
        )?;
        renderer.set_enabled(true);
        renderer.set_depth_test_enabled(depth_test);
        let draped = item
            .get_item("terrain_draped")
            .ok()
            .map(|value| value.extract::<bool>())
            .transpose()?
            .unwrap_or(false);
        if draped && surface_kind.is_none() {
            return Err(anyhow!(
                "terrain drapes require shared surface classification"
            ));
        }
        renderer.uniforms.depth_params[3] = f32::from(u8::from(draped));
        renderer.uniforms.view_proj = projection.to_cols_array_2d();
        renderer.upload_uniforms(&scene.queue);
        result.push(renderer);
    }
    Ok(result)
}

fn upload(
    scene: &TerrainScene,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    data: &[u8],
    bytes: u32,
) -> Result<TrackedTexture> {
    let texture = tracked_create_texture(
        &scene.device,
        &wgpu::TextureDescriptor {
            label: Some("concordia.cartography.input"),
            size: wgpu::Extent3d {
                width,
                height,
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
        data,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(width * bytes),
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    Ok(texture)
}

pub(super) fn draw(
    encoder: &mut wgpu::CommandEncoder,
    output: &wgpu::TextureView,
    overlays: &[OverlayRenderer],
) {
    if overlays.is_empty() {
        return;
    }
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("concordia.cartography.draw"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: output,
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
    for renderer in overlays {
        crate::core::shader_registry::record_shader_use("concordia.overlays.shader");
        pass.set_pipeline(&renderer.pipeline);
        pass.set_bind_group(0, &renderer.bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}
