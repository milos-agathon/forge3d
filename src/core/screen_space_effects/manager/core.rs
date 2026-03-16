use super::*;

impl ScreenSpaceEffectsManager {
    pub fn new(device: &Device, width: u32, height: u32) -> RenderResult<Self> {
        let gbuffer_config = GBufferConfig {
            width,
            height,
            ..Default::default()
        };
        let gbuffer = GBuffer::new(device, gbuffer_config)?;
        let hzb = HzbPyramid::new(device, width, height).ok();

        Ok(Self {
            gbuffer,
            ssao_renderer: None,
            ssgi_renderer: None,
            ssr_renderer: None,
            enabled_effects: Vec::new(),
            hzb,
            ssr_params: SsrParams::default(),
            last_hzb_ms: 0.0,
        })
    }

    pub fn set_environment_texture(&mut self, device: &Device, env_texture: &Texture) {
        if let Some(ref mut ssgi) = self.ssgi_renderer {
            ssgi.set_environment_texture(device, env_texture);
        }
        if let Some(ref mut ssr) = self.ssr_renderer {
            ssr.set_environment_texture(device, env_texture);
        }
    }

    pub fn enable_effect(
        &mut self,
        device: &Device,
        effect: ScreenSpaceEffect,
    ) -> RenderResult<()> {
        if !self.enabled_effects.contains(&effect) {
            self.enabled_effects.push(effect);
        }

        match effect {
            ScreenSpaceEffect::SSAO => {
                if self.ssao_renderer.is_none() {
                    let (width, height) = self.gbuffer.dimensions();
                    let mat_fmt = self.gbuffer.config().material_format;
                    self.ssao_renderer = Some(SsaoRenderer::new(device, width, height, mat_fmt)?);
                }
            }
            ScreenSpaceEffect::SSGI => {
                if self.ssgi_renderer.is_none() {
                    let (width, height) = self.gbuffer.dimensions();
                    let mat_fmt = self.gbuffer.config().material_format;
                    self.ssgi_renderer = Some(SsgiRenderer::new(device, width, height, mat_fmt)?);
                }
            }
            ScreenSpaceEffect::SSR => {
                if self.ssr_renderer.is_none() {
                    let (width, height) = self.gbuffer.dimensions();
                    self.ssr_renderer = Some(SsrRenderer::new(device, width, height)?);
                }
            }
        }

        Ok(())
    }

    pub fn disable_effect(&mut self, effect: ScreenSpaceEffect) {
        self.enabled_effects.retain(|&e| e != effect);
    }

    pub fn gbuffer(&self) -> &GBuffer {
        &self.gbuffer
    }

    pub fn gbuffer_mut(&mut self) -> &mut GBuffer {
        &mut self.gbuffer
    }

    pub fn build_hzb(
        &mut self,
        device: &Device,
        encoder: &mut CommandEncoder,
        src_depth: &TextureView,
        reversed_z: bool,
    ) {
        if let Some(ref hzb) = self.hzb {
            let t0 = Instant::now();
            hzb.build(device, encoder, src_depth, reversed_z);
            self.last_hzb_ms = t0.elapsed().as_secs_f32() * 1000.0;
        }
    }

    pub fn hzb_texture_and_mips(&self) -> Option<(&Texture, u32)> {
        self.hzb.as_ref().map(|h| (&h.tex, h.mip_count))
    }

    pub fn set_ssr_params(&mut self, queue: &Queue, params: &SsrParams) {
        self.ssr_params = *params;
        self.refresh_ssr_settings(queue);
    }

    fn refresh_ssr_settings(&mut self, queue: &Queue) {
        let settings = self.build_ssr_settings();
        if let Some(ref mut renderer) = self.ssr_renderer {
            renderer.update_settings(queue, settings);
        }
    }

    fn build_ssr_settings(&self) -> SsrSettings {
        let (mut w, mut h) = self.gbuffer.dimensions();
        if w == 0 {
            w = 1;
        }
        if h == 0 {
            h = 1;
        }
        SsrSettings {
            max_steps: self.ssr_params.ssr_max_steps,
            thickness: self.ssr_params.ssr_thickness,
            inv_resolution: [1.0 / w as f32, 1.0 / h as f32],
            ..SsrSettings::default()
        }
    }
}
