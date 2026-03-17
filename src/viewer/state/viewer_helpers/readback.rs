use std::path::Path;

use crate::renderer::readback::read_texture_tight;
use crate::util::image_write;
use crate::viewer::viewer_render_helpers::render_view_to_rgba8_ex;
use crate::viewer::Viewer;
use anyhow::{bail, Context};

impl Viewer {
    pub(crate) fn snapshot_swapchain_to_png(
        &mut self,
        tex: &wgpu::Texture,
        path: &str,
    ) -> anyhow::Result<()> {
        let size = tex.size();
        let w = size.width;
        let h = size.height;
        let fmt = tex.format();

        match fmt {
            wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb => {
                let mut data = read_texture_tight(&self.device, &self.queue, tex, (w, h), fmt)
                    .context("readback failed")?;
                for px in data.chunks_exact_mut(4) {
                    px[3] = 255;
                }
                image_write::write_png_rgba8(Path::new(path), &data, w, h)
                    .context("failed to write PNG")?;
                Ok(())
            }
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb => {
                let mut data = read_texture_tight(&self.device, &self.queue, tex, (w, h), fmt)
                    .context("readback failed")?;
                for px in data.chunks_exact_mut(4) {
                    px.swap(0, 2);
                    px[3] = 255;
                }
                image_write::write_png_rgba8(Path::new(path), &data, w, h)
                    .context("failed to write PNG")?;
                Ok(())
            }
            other => bail!(
                "snapshot only supports RGBA8/BGRA8 surfaces (got {:?})",
                other
            ),
        }
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
