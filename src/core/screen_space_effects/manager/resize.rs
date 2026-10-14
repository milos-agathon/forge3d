use super::*;
use crate::core::screen_space_effects::{
    hzb::PreparedHzbTargets, ssao::PreparedSsaoTargets, ssgi::PreparedSsgiTargets,
    ssr::PreparedSsrTargets,
};

pub(crate) struct PreparedScreenSpaceTargets {
    gbuffer: GBuffer,
    hzb: Option<PreparedHzbTargets>,
    ssao: Option<PreparedSsaoTargets>,
    ssgi: Option<PreparedSsgiTargets>,
    ssr: Option<PreparedSsrTargets>,
}

impl ScreenSpaceEffectsManager {
    #[cfg(all(test, windows))]
    pub(crate) fn viewport_textures(&self) -> Vec<&crate::core::resource_tracker::TrackedTexture> {
        let mut textures = vec![
            &self.gbuffer.depth_texture,
            &self.gbuffer.normal_texture,
            &self.gbuffer.material_texture,
            &self.gbuffer.velocity_texture,
        ];
        if let Some(hzb) = self.hzb.as_ref() {
            textures.push(&hzb.tex);
        }
        if let Some(ssao) = self.ssao_renderer.as_ref() {
            textures.extend(ssao.viewport_textures());
        }
        if let Some(ssgi) = self.ssgi_renderer.as_ref() {
            textures.extend(ssgi.viewport_textures());
        }
        if let Some(ssr) = self.ssr_renderer.as_ref() {
            textures.extend(ssr.viewport_textures());
        }
        textures
    }

    #[cfg(all(test, windows))]
    pub(crate) fn assert_viewport_dimensions(&self, width: u32, height: u32) {
        let hzb = self.hzb.as_ref().unwrap();
        assert_eq!((hzb.tex.width(), hzb.tex.height()), (width, height));
        assert_eq!(hzb.mip_count, u32::BITS - width.max(height).leading_zeros());
        assert_eq!(hzb.tex.mip_level_count(), hzb.mip_count);
        if let Some(ssao) = self.ssao_renderer.as_ref() {
            for texture in ssao.viewport_textures() {
                assert_eq!((texture.width(), texture.height()), (width, height));
            }
        }
        if let Some(ssgi) = self.ssgi_renderer.as_ref() {
            let trace = if ssgi.is_half_res() {
                ((width / 2).max(1), (height / 2).max(1))
            } else {
                (width, height)
            };
            for (index, texture) in ssgi.viewport_textures().iter().enumerate() {
                let expected = if index < 4 { trace } else { (width, height) };
                assert_eq!((texture.width(), texture.height()), expected);
            }
        }
        if let Some(ssr) = self.ssr_renderer.as_ref() {
            for texture in ssr.viewport_textures() {
                assert_eq!((texture.width(), texture.height()), (width, height));
            }
        }
    }

    pub(crate) fn prepare_resize(
        &self,
        device: &Device,
        width: u32,
        height: u32,
    ) -> RenderResult<PreparedScreenSpaceTargets> {
        let mut config = self.gbuffer.config().clone();
        config.width = width;
        config.height = height;
        Ok(PreparedScreenSpaceTargets {
            gbuffer: GBuffer::new(device, config)?,
            hzb: self
                .hzb
                .as_ref()
                .map(|r| r.prepare_resize(device, width, height))
                .transpose()?,
            ssao: self
                .ssao_renderer
                .as_ref()
                .map(|r| r.prepare_resize(device, width, height))
                .transpose()?,
            ssgi: self
                .ssgi_renderer
                .as_ref()
                .map(|r| {
                    r.prepare_resize(device, width, height, self.gbuffer.config().material_format)
                })
                .transpose()?,
            ssr: self
                .ssr_renderer
                .as_ref()
                .map(|r| r.prepare_resize(device, width, height))
                .transpose()?,
        })
    }

    pub(crate) fn install_resize(&mut self, prepared: PreparedScreenSpaceTargets, queue: &Queue) {
        self.gbuffer = prepared.gbuffer;
        if let (Some(renderer), Some(targets)) = (self.hzb.as_mut(), prepared.hzb) {
            renderer.install_resize(targets);
        }
        if let (Some(renderer), Some(targets)) = (self.ssao_renderer.as_mut(), prepared.ssao) {
            renderer.install_resize(targets, queue);
        }
        if let (Some(renderer), Some(targets)) = (self.ssgi_renderer.as_mut(), prepared.ssgi) {
            renderer.install_resize(targets, queue);
        }
        if let (Some(renderer), Some(targets)) = (self.ssr_renderer.as_mut(), prepared.ssr) {
            renderer.install_resize(targets, queue);
        }
        self.last_hzb_ms = 0.0;
    }
}
