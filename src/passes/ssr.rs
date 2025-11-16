//! src/passes/ssr.rs
//! Thin wrapper for SSR renderer pass used by P5 harness tooling.

use crate::core::gbuffer::GBuffer;
use crate::core::screen_space_effects::{CameraParams, SsrRenderer, SsrSettings};
use crate::error::RenderResult;
use wgpu::{CommandEncoder, Device, Queue, TextureView};

pub use crate::render::params::SsrParams;

pub struct SsrPass {
    renderer: SsrRenderer,
}

impl SsrPass {
    pub fn new(device: &Device, width: u32, height: u32) -> RenderResult<Self> {
        Ok(Self {
            renderer: SsrRenderer::new(device, width, height)?,
        })
    }

    pub fn update_settings(&mut self, queue: &Queue, f: impl FnOnce(&mut SsrSettings)) {
        let mut settings = self.renderer.get_settings();
        (f)(&mut settings);
        self.renderer.update_settings(queue, settings);
    }

    pub fn update_camera(&mut self, queue: &Queue, camera: &CameraParams) {
        self.renderer.update_camera(queue, camera);
    }

    pub fn execute(
        &mut self,
        device: &Device,
        encoder: &mut CommandEncoder,
        gbuffer: &GBuffer,
    ) -> RenderResult<()> {
        self.renderer.execute(device, encoder, gbuffer)
    }

    pub fn composite_view(&self) -> &TextureView {
        self.renderer.composite_view()
    }
}
