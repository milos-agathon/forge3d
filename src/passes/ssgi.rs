// src/passes/ssgi.rs
// P5.2: SSGI pass wrapper that delegates to core::screen_space_effects::SsgiRenderer

use crate::core::error::RenderResult;
use crate::core::gbuffer::GBuffer;
use crate::core::screen_space_effects::{CameraParams, SsgiRenderer, SsgiSettings};
use wgpu::{CommandEncoder, Device, Queue, TextureFormat, TextureView};

pub struct SsgiPass {
    renderer: SsgiRenderer,
}

impl SsgiPass {
    pub fn new(device: &Device, width: u32, height: u32) -> RenderResult<Self> {
        let renderer = SsgiRenderer::new(device, width, height, TextureFormat::Rgba8Unorm)?;
        Ok(Self { renderer })
    }

    pub fn update_settings(&mut self, queue: &Queue, f: impl FnOnce(&mut SsgiSettings)) {
        let mut s = self.renderer.get_settings();
        (f)(&mut s);
        self.renderer.update_settings(queue, s);
    }

    pub fn update_camera(&mut self, queue: &Queue, cam: &CameraParams) {
        self.renderer.update_camera(queue, cam);
    }

    pub fn set_half_res(&mut self, device: &Device, queue: &Queue, on: bool) -> RenderResult<()> {
        self.renderer.set_half_res(device, queue, on)
    }

    pub fn execute(
        &mut self,
        device: &Device,
        encoder: &mut CommandEncoder,
        gbuffer: &GBuffer,
        hzb_view: &TextureView,
    ) -> RenderResult<()> {
        self.renderer.execute(device, encoder, gbuffer, hzb_view)
    }

    pub fn output_for_display(&self) -> &TextureView {
        self.renderer.get_output_for_display()
    }
}
