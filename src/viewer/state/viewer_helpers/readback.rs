use std::path::Path;

use crate::gis::raster_write::{
    write_raster, CrsSpec, RasterArray, RasterData, WriteRasterOptions,
};
use crate::gis::types::{AffineTransform, PixelRegistration, HEIGHT_SYSTEM_UNSPECIFIED};
use crate::renderer::readback::read_texture_tight;
use crate::util::image_write;
use crate::viewer::viewer_render_helpers::render_view_to_rgba8_ex;
use crate::viewer::viewer_types::{ActiveCameraKind, FrameCamera};
use crate::viewer::Viewer;
use anyhow::{bail, ensure, Context};
use glam::{DVec2, DVec3, Vec3};

impl Viewer {
    /// Render the outer terrain samples at the viewport edges, then write
    /// the Area grid of the framebuffer in the loaded raster's CRS.
    pub(crate) fn snapshot_terrain_geotiff(
        &mut self,
        path: &str,
        width: u32,
        height: u32,
        overwrite: bool,
    ) -> anyhow::Result<()> {
        let limit = self.device.limits().max_texture_dimension_2d;
        ensure!(
            (1..=limit).contains(&width) && (1..=limit).contains(&height),
            "GeoTIFF dimensions must be positive and fit the device texture limit ({limit})"
        );
        let anchor = self.current_frame_camera().anchor;
        let scene = self.terrain_viewer.as_mut().context("no terrain loaded")?;
        let terrain = scene.terrain.as_ref().context("no terrain loaded")?;
        let info = &terrain.raster_info;
        ensure!(
            info.is_georeferenced,
            "GeoTIFF snapshot requires georeferenced terrain"
        );
        let crs = CrsSpec::from_raster_info(info).context("terrain CRS is unavailable")?;
        let (dx, _, west, _, dy, north) =
            info.transform.context("terrain affine is unavailable")?;
        let lens = &scene.pbr_config.lens_effects;
        ensure!(
            !lens.enabled || (lens.distortion.abs() <= 0.001 && lens.chromatic_aberration <= 0.001),
            "GeoTIFF snapshot requires an undistorted orthographic projection"
        );
        ensure!(
            !scene.pbr_config.motion_blur.enabled,
            "GeoTIFF snapshot requires a single camera pose"
        );

        let sample_intervals = DVec2::new(
            f64::from(info.width.saturating_sub(1).max(1)),
            f64::from(info.height.saturating_sub(1).max(1)),
        );
        let offsets = DVec2::new(
            if info.width == 1 { 0.0 } else { 0.5 },
            if info.height == 1 { 0.0 } else { 0.5 },
        );
        let transform = AffineTransform::new([
            dx * sample_intervals.x / f64::from(width),
            0.0,
            west + dx * offsets.x,
            0.0,
            dy * sample_intervals.y / f64::from(height),
            north + dy * offsets.y,
        ])?;

        let span = terrain.world_span_xz * sample_intervals
            / DVec2::new(f64::from(info.width), f64::from(info.height));
        let mut centre = terrain.default_camera_target();
        centre.y = 0.0;
        let elevation_span = f64::from(terrain.height_range() * terrain.z_scale).abs();
        let margin = span.max_element().max(elevation_span);
        let far = (2.0 * margin + elevation_span) as f32;
        let half_extents = [(span.x * 0.5) as f32, (span.y * 0.5) as f32];
        ensure!(
            far.is_finite() && half_extents.iter().all(|&v| v.is_finite() && v > 0.0),
            "terrain extent cannot be represented by the snapshot camera"
        );
        let frame = FrameCamera {
            kind: ActiveCameraKind::Terrain,
            anchor,
            eye_world: centre + DVec3::Y * (margin + elevation_span),
            target_world: centre,
            up: Vec3::NEG_Z,
            fov_deg: terrain.cam_fov_deg,
            near: (margin * 0.5) as f32,
            far,
            orthographic_half_extents: Some(half_extents),
        };
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("viewer.geotiff_snapshot"),
            });
        let texture = scene
            .render_to_texture(
                &mut encoder,
                self.config.format,
                width,
                height,
                self.selected_feature_id,
                frame,
            )
            .context("failed to render orthographic terrain snapshot")?;
        ensure!(
            !scene.pbr_config.enabled || scene.pbr_pipeline.is_some(),
            "failed to initialize terrain PBR for GeoTIFF snapshot"
        );

        let labels_enabled = self.label_manager.is_enabled();
        if labels_enabled {
            self.resize_labels(width, height);
            self.update_labels_for_viewport(frame, width, height);
            if self.label_manager.visible_count() > 0 {
                self.label_manager
                    .upload_to_renderer(&self.device, &self.queue, &mut self.hud);
                self.hud.set_enabled(true);
                self.hud.set_resolution(width, height);
                self.hud.upload_uniforms(&self.queue);
                let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("viewer.geotiff_snapshot.labels"),
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
                self.hud.render(&mut pass);
            }
            self.resize_labels(self.config.width, self.config.height);
        }
        self.queue.submit(Some(encoder.finish()));
        if labels_enabled {
            self.hud.set_resolution(self.config.width, self.config.height);
            self.hud.upload_uniforms(&self.queue);
        }

        let rgba = self.read_snapshot_rgba8(&texture)?;
        let bands = (0..4)
            .flat_map(|band| rgba.chunks_exact(4).map(move |pixel| pixel[band]))
            .collect();
        write_raster(
            path,
            RasterArray::new(RasterData::U8(bands), &[4, height as usize, width as usize])?,
            WriteRasterOptions {
                crs: Some(crs),
                transform: Some(transform),
                nodata: vec![None; 4],
                driver: "GTiff".into(),
                overwrite,
                creation_options: Default::default(),
                creation_options_explicit: false,
                like_info: None,
                height_system: HEIGHT_SYSTEM_UNSPECIFIED.into(),
                registration: PixelRegistration::Area,
            },
        )?;
        Ok(())
    }

    pub(crate) fn snapshot_swapchain_to_png(
        &mut self,
        tex: &wgpu::Texture,
        path: &str,
    ) -> anyhow::Result<()> {
        let size = tex.size();
        let data = self.read_snapshot_rgba8(tex)?;
        image_write::write_png_rgba8(Path::new(path), &data, size.width, size.height)
            .context("failed to write PNG")
    }

    fn read_snapshot_rgba8(&self, tex: &wgpu::Texture) -> anyhow::Result<Vec<u8>> {
        let size = tex.size();
        let format = tex.format();
        let bgra = match format {
            wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb => false,
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb => true,
            other => bail!(
                "snapshot only supports RGBA8/BGRA8 surfaces (got {:?})",
                other
            ),
        };
        let mut data = read_texture_tight(
            &self.device,
            &self.queue,
            tex,
            (size.width, size.height),
            format,
        )
        .context("readback failed")?;
        for pixel in data.chunks_exact_mut(4) {
            if bgra {
                pixel.swap(0, 2);
            }
            pixel[3] = 255;
        }
        Ok(data)
    }

    pub(crate) fn capture_material_rgba8(&self) -> anyhow::Result<Vec<u8>> {
        let gi = self.gi.as_ref().context("GI manager not available")?;
        let far = self.viz_depth_max_override.unwrap_or(self.view_config.zfar);
        self.with_comp_pipeline(|comp_pl, comp_bgl| {
            let fog_view = if self.fog_enabled {
                &self.fog_output_view
            } else {
                &self.fog_zero_view
            };
            render_view_to_rgba8_ex(crate::viewer::viewer_render_helpers::RenderViewArgs {
                device: &self.device,
                queue: &self.queue,
                comp_pl,
                comp_bgl,
                sky_view: &self.sky_output_view,
                depth_view: &gi.gbuffer().depth_view,
                fog_view,
                surface_format: self.config.format,
                width: self.config.width,
                height: self.config.height,
                far,
                src_view: gi
                    .material_with_ssr_view()
                    .or_else(|| gi.material_with_ssgi_view())
                    .or_else(|| gi.material_with_ao_view())
                    .unwrap_or(&gi.gbuffer().material_view),
                mode: 0,
            })
        })
    }

    pub(crate) fn read_ssgi_filtered_bytes(&self) -> anyhow::Result<(Vec<u8>, (u32, u32))> {
        let gi = self.gi.as_ref().context("GI manager not available")?;
        let dims = gi
            .ssgi_dimensions()
            .context("SSGI dimensions unavailable")?;
        let tex = gi
            .ssgi_filtered_texture()
            .context("SSGI filtered texture unavailable")?;
        let bytes = read_texture_tight(
            &self.device,
            &self.queue,
            tex,
            dims,
            wgpu::TextureFormat::Rgba16Float,
        )
        .context("read SSGI filtered texture")?;
        Ok((bytes, dims))
    }

    pub(crate) fn read_ssgi_hit_bytes(&self) -> anyhow::Result<(Vec<u8>, (u32, u32))> {
        let gi = self.gi.as_ref().context("GI manager not available")?;
        let dims = gi
            .ssgi_dimensions()
            .context("SSGI dimensions unavailable")?;
        let tex = gi
            .ssgi_hit_texture()
            .context("SSGI hit texture unavailable")?;
        let bytes = read_texture_tight(
            &self.device,
            &self.queue,
            tex,
            dims,
            wgpu::TextureFormat::Rgba16Float,
        )
        .context("read SSGI hit texture")?;
        Ok((bytes, dims))
    }
}
