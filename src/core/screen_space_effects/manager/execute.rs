use super::*;

impl ScreenSpaceEffectsManager {
    pub fn execute(
        &mut self,
        device: &Device,
        encoder: &mut CommandEncoder,
        ssr_stats: Option<&mut SsrStats>,
        mut timing: Option<&mut GpuTimingManager>,
    ) -> RenderResult<()> {
        let mut ssr_stats_opt = ssr_stats;
        for effect in &self.enabled_effects {
            match effect {
                ScreenSpaceEffect::SSAO => {
                    if let (Some(ssao), Some(hzb)) =
                        (self.ssao_renderer.as_mut(), self.hzb.as_ref())
                    {
                        let hzb_view = hzb.texture_view();
                        if let Some(timer) = timing.as_deref_mut() {
                            let scope_id = timer.begin_scope(encoder, "p5.ssao");
                            ssao.execute(device, encoder, &self.gbuffer, &hzb_view)?;
                            timer.end_scope(encoder, scope_id);
                        } else {
                            ssao.execute(device, encoder, &self.gbuffer, &hzb_view)?;
                        }
                    }
                }
                ScreenSpaceEffect::SSGI => {
                    if let Some(ref mut ssgi) = self.ssgi_renderer {
                        let timing_scope = timing
                            .as_deref_mut()
                            .map(|timer| timer.begin_scope(encoder, "p5.ssgi"));
                        if let Some(ref hzb) = self.hzb {
                            let hzb_view = hzb.texture_view();
                            ssgi.execute(device, encoder, &self.gbuffer, &hzb_view)?;
                        } else {
                            ssgi.execute(device, encoder, &self.gbuffer, &self.gbuffer.depth_view)?;
                        }
                        if let Some(scope_id) = timing_scope {
                            if let Some(timer) = timing.as_deref_mut() {
                                timer.end_scope(encoder, scope_id);
                            }
                        }
                    }
                }
                ScreenSpaceEffect::SSR => {
                    if self.ssr_params.ssr_enable {
                        if let Some(ref mut ssr) = self.ssr_renderer {
                            if let Some(timer) = timing.as_deref_mut() {
                                let scope_id = timer.begin_scope(encoder, "p5.ssr");
                                ssr.execute(
                                    device,
                                    encoder,
                                    &self.gbuffer,
                                    ssr_stats_opt.as_deref_mut(),
                                )?;
                                timer.end_scope(encoder, scope_id);
                            } else {
                                ssr.execute(
                                    device,
                                    encoder,
                                    &self.gbuffer,
                                    ssr_stats_opt.as_deref_mut(),
                                )?;
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    pub fn update_camera(&mut self, queue: &Queue, camera: &CameraParams) {
        if let Some(ref mut ssao) = self.ssao_renderer {
            ssao.update_camera(queue, camera);
        }
        if let Some(ref mut ssgi) = self.ssgi_renderer {
            ssgi.update_camera(queue, camera);
        }
        if let Some(ref mut ssr) = self.ssr_renderer {
            ssr.update_camera(queue, camera);
        }
    }

    pub fn advance_frame(&mut self, queue: &Queue) {
        if let Some(ref mut ssgi) = self.ssgi_renderer {
            ssgi.advance_frame(queue);
        }
    }

    pub fn hzb_ms(&self) -> f32 {
        self.last_hzb_ms
    }

    pub fn set_gi_seed(&mut self, device: &Device, queue: &Queue, seed: u32) -> RenderResult<()> {
        if let Some(ref mut ssao) = self.ssao_renderer {
            ssao.set_seed(queue, seed);
        }
        if let Some(ref mut ssgi) = self.ssgi_renderer {
            ssgi.set_seed(queue, seed);
        }
        self.ssgi_reset_history(device, queue)?;
        Ok(())
    }
}
