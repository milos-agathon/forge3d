use super::settings::SsaoTemporalParamsUniform;
use super::*;

impl ScreenSpaceEffectsManager {
    pub fn gi_debug_view(&self) -> Option<&TextureView> {
        if let Some(ref ssr) = self.ssr_renderer {
            #[allow(dead_code)]
            {
                return Some(ssr.get_output());
            }
        }
        if let Some(ref ssgi) = self.ssgi_renderer {
            #[allow(dead_code)]
            {
                return Some(ssgi.get_output_for_display());
            }
        }
        None
    }

    pub fn material_with_ao_view(&self) -> Option<&TextureView> {
        if self.enabled_effects.contains(&ScreenSpaceEffect::SSAO) {
            if let Some(ref ssao) = self.ssao_renderer {
                return Some(ssao.get_composited());
            }
        }
        None
    }

    pub fn material_with_ssr_view(&self) -> Option<&TextureView> {
        if self.enabled_effects.contains(&ScreenSpaceEffect::SSR) {
            if let Some(ref ssr) = self.ssr_renderer {
                return Some(ssr.composite_view());
            }
        }
        None
    }

    pub fn ssr_spec_view(&self) -> Option<&TextureView> {
        self.ssr_renderer.as_ref().map(|ssr| ssr.spec_view())
    }

    pub fn ssr_final_view(&self) -> Option<&TextureView> {
        self.ssr_renderer.as_ref().map(|ssr| ssr.final_view())
    }

    pub fn set_ssr_scene_color_view(&mut self, view: TextureView) {
        if let Some(ref mut ssr) = self.ssr_renderer {
            ssr.set_scene_color_view(view);
        }
    }

    pub fn collect_ssr_stats(
        &mut self,
        device: &Device,
        queue: &Queue,
        stats: &mut SsrStats,
    ) -> RenderResult<()> {
        if let Some(ref mut ssr) = self.ssr_renderer {
            ssr.collect_stats_into(device, queue, stats)
        } else {
            stats.clear();
            Ok(())
        }
    }

    pub fn material_with_ssgi_view(&self) -> Option<&TextureView> {
        if self.enabled_effects.contains(&ScreenSpaceEffect::SSGI) {
            if let Some(ref ssgi) = self.ssgi_renderer {
                return Some(ssgi.get_composited());
            }
        }
        None
    }

    pub fn set_ssgi_composite_intensity(&mut self, queue: &Queue, intensity: f32) {
        if let Some(ref mut ssgi) = self.ssgi_renderer {
            ssgi.set_composite_intensity(queue, intensity);
        }
    }

    pub fn ao_raw_view(&self) -> Option<&TextureView> {
        self.ssao_renderer.as_ref().map(|r| r.get_raw_ao_view())
    }

    pub fn ao_tmp_view(&self) -> Option<&TextureView> {
        self.ssao_renderer.as_ref().map(|r| r.get_tmp_ao_view())
    }

    pub fn ao_blur_view(&self) -> Option<&TextureView> {
        self.ssao_renderer.as_ref().map(|r| r.get_output())
    }

    pub fn ao_resolved_view(&self) -> Option<&TextureView> {
        self.ssao_renderer
            .as_ref()
            .map(|r| r.get_resolved_ao_view())
    }

    pub fn ao_raw_texture(&self) -> Option<&Texture> {
        self.ssao_renderer.as_ref().map(|r| r.raw_ao_texture())
    }

    pub fn ao_blur_texture(&self) -> Option<&Texture> {
        self.ssao_renderer.as_ref().map(|r| r.blurred_ao_texture())
    }

    pub fn ao_resolved_texture(&self) -> Option<&Texture> {
        self.ssao_renderer.as_ref().map(|r| r.resolved_ao_texture())
    }

    pub fn ao_composited_texture(&self) -> Option<&Texture> {
        self.ssao_renderer.as_ref().map(|r| r.composited_texture())
    }

    pub fn ssr_hit_view(&self) -> Option<&TextureView> {
        self.ssr_renderer.as_ref().map(|r| r.hit_data_view())
    }

    pub fn ssr_output_texture(&self) -> Option<&Texture> {
        self.ssr_renderer.as_ref().map(|r| r.output_texture())
    }

    pub fn ssr_hit_texture(&self) -> Option<&Texture> {
        self.ssr_renderer.as_ref().map(|r| r.hit_data_texture())
    }

    pub fn ssr_timings_ms(&self) -> Option<(f32, f32, f32)> {
        self.ssr_renderer.as_ref().map(|r| r.timings_ms())
    }

    pub fn set_ssao_temporal(&mut self, on: bool) {
        if let Some(ref mut r) = self.ssao_renderer {
            r.set_temporal_enabled(on);
        }
    }

    pub fn set_ssao_temporal_alpha(&mut self, queue: &Queue, alpha: f32) {
        if let Some(ref mut r) = self.ssao_renderer {
            r.temporal_alpha = alpha.clamp(0.0, 1.0);
            let params = SsaoTemporalParamsUniform {
                temporal_alpha: r.temporal_alpha,
                _pad: [0.0; 7],
            };
            queue.write_buffer(&r.temporal_params_buffer, 0, bytemuck::bytes_of(&params));
        }
    }

    pub fn set_ssao_composite_multiplier(&mut self, queue: &Queue, mul: f32) {
        if let Some(ref mut ssao) = self.ssao_renderer {
            ssao.set_composite_multiplier(queue, mul);
        }
    }

    pub fn ssao_timings_ms(&self) -> Option<(f32, f32, f32)> {
        self.ssao_renderer.as_ref().map(|r| r.timings_ms())
    }

    pub fn is_enabled(&self, effect: ScreenSpaceEffect) -> bool {
        self.enabled_effects.contains(&effect)
    }

    pub fn ssao_settings(&self) -> SsaoSettings {
        self.ssao_renderer
            .as_ref()
            .map(|r| r.settings)
            .unwrap_or_default()
    }

    pub fn ssao_temporal_alpha(&self) -> f32 {
        self.ssao_renderer
            .as_ref()
            .map(|r| r.temporal_alpha)
            .unwrap_or(0.0)
    }

    pub fn ssao_temporal_enabled(&self) -> bool {
        self.ssao_renderer
            .as_ref()
            .map(|r| r.temporal_enabled)
            .unwrap_or(false)
    }

    pub fn update_ssao_settings(&mut self, queue: &Queue, mut f: impl FnMut(&mut SsaoSettings)) {
        if let Some(ref mut r) = self.ssao_renderer {
            let mut s = r.get_settings();
            f(&mut s);
            r.update_settings(queue, s);
        }
    }

    pub fn set_ssao_blur(&mut self, on: bool) {
        if let Some(ref mut r) = self.ssao_renderer {
            r.set_blur_enabled(on);
        }
    }

    pub fn set_ssao_bias(&mut self, queue: &Queue, bias: f32) {
        if let Some(ref mut r) = self.ssao_renderer {
            let mut s = r.get_settings();
            s.bias = bias.max(0.0);
            r.update_settings(queue, s);
        }
    }

    pub fn update_ssgi_settings(&mut self, queue: &Queue, mut f: impl FnMut(&mut SsgiSettings)) {
        if let Some(ref mut r) = self.ssgi_renderer {
            let mut s = r.get_settings();
            f(&mut s);
            r.update_settings(queue, s);
        }
    }

    pub fn update_ssr_settings(&mut self, queue: &Queue, mut f: impl FnMut(&mut SsrSettings)) {
        if let Some(ref mut r) = self.ssr_renderer {
            let mut s = r.get_settings();
            f(&mut s);
            r.update_settings(queue, s);
        }
    }

    pub fn set_env_for_all(&mut self, device: &Device, env_texture: &Texture) {
        if let Some(ref mut ssgi) = self.ssgi_renderer {
            ssgi.set_environment_texture(device, env_texture);
        }
        if let Some(ref mut ssr) = self.ssr_renderer {
            ssr.set_environment_texture(device, env_texture);
        }
    }

    pub fn set_ssgi_env(&mut self, device: &Device, env_texture: &Texture) {
        if let Some(ref mut r) = self.ssgi_renderer {
            r.set_environment_texture(device, env_texture);
        }
    }

    pub fn set_ssr_env(&mut self, device: &Device, env_texture: &Texture) {
        if let Some(ref mut r) = self.ssr_renderer {
            r.set_environment_texture(device, env_texture);
        }
    }

    pub fn set_ssgi_half_res(&mut self, device: &Device, on: bool) {
        if let Some(ref mut _ssgi) = self.ssgi_renderer {
            let _ = (device, on);
        }
    }

    pub fn set_ssgi_half_res_with_queue(&mut self, device: &Device, queue: &Queue, on: bool) {
        if let Some(ref mut ssgi) = self.ssgi_renderer {
            let _ = ssgi.set_half_res(device, queue, on);
        }
    }

    pub fn ssgi_settings(&self) -> Option<SsgiSettings> {
        self.ssgi_renderer.as_ref().map(|r| r.get_settings())
    }

    pub fn ssgi_timings_ms(&self) -> Option<(f32, f32, f32, f32)> {
        self.ssgi_renderer.as_ref().map(|r| r.timings_ms())
    }

    pub fn ssgi_dimensions(&self) -> Option<(u32, u32)> {
        self.ssgi_renderer.as_ref().map(|r| r.dimensions())
    }

    pub fn ssgi_half_res(&self) -> Option<bool> {
        self.ssgi_renderer.as_ref().map(|r| r.is_half_res())
    }

    pub fn ssgi_hit_texture(&self) -> Option<&Texture> {
        self.ssgi_renderer.as_ref().map(|r| r.hit_texture())
    }

    pub fn ssgi_filtered_texture(&self) -> Option<&Texture> {
        self.ssgi_renderer.as_ref().map(|r| r.filtered_texture())
    }

    pub fn ssgi_history_texture(&self) -> Option<&Texture> {
        self.ssgi_renderer.as_ref().map(|r| r.history_texture())
    }

    pub fn ssgi_upscaled_texture(&self) -> Option<&Texture> {
        self.ssgi_renderer.as_ref().map(|r| r.upscaled_texture())
    }

    pub fn ssgi_output_for_display_view(&self) -> Option<&TextureView> {
        self.ssgi_renderer
            .as_ref()
            .map(|r| r.get_output_for_display())
    }

    pub fn ssgi_reset_history(&mut self, device: &Device, queue: &Queue) -> RenderResult<()> {
        if let Some(ref mut ssgi) = self.ssgi_renderer {
            ssgi.reset_history(device, queue)?;
        }
        Ok(())
    }
}
