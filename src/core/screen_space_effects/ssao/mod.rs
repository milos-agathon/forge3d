use super::settings::SsaoTemporalParamsUniform;
use super::*;

mod accessors;
mod constructor;
mod passes;
mod pipelines;
mod resources;
mod runtime;
mod temporal;

pub use constructor::SsaoRenderer;

pub(crate) struct PreparedSsaoTargets {
    width: u32,
    height: u32,
    textures: resources::SsaoViewportTextures,
}

impl SsaoRenderer {
    #[cfg(all(test, windows))]
    pub(crate) fn viewport_textures(&self) -> Vec<&crate::core::resource_tracker::TrackedTexture> {
        vec![
            &self.ssao_texture,
            &self.ssao_blurred,
            &self.ssao_history,
            &self.ssao_resolved,
            &self._ssao_tmp,
            &self.ssao_composited,
        ]
    }

    pub(crate) fn prepare_resize(
        &self,
        device: &Device,
        width: u32,
        height: u32,
    ) -> RenderResult<PreparedSsaoTargets> {
        Ok(PreparedSsaoTargets {
            width,
            height,
            textures: resources::create_viewport_textures(device, width, height)?,
        })
    }

    pub(crate) fn install_resize(&mut self, prepared: PreparedSsaoTargets, queue: &Queue) {
        let textures = prepared.textures;
        self.ssao_texture = textures.ssao_texture;
        self.ssao_view = textures.ssao_view;
        self.ssao_blurred = textures.ssao_blurred;
        self.ssao_blurred_view = textures.ssao_blurred_view;
        self.ssao_history = textures.ssao_history;
        self.ssao_history_view = textures.ssao_history_view;
        self.ssao_resolved = textures.ssao_resolved;
        self.ssao_resolved_view = textures.ssao_resolved_view;
        self.ssao_tmp_view = textures.ssao_tmp_view;
        self.ssao_composited = textures.ssao_composited;
        self.ssao_composited_view = textures.ssao_composited_view;
        self._ssao_tmp = textures.ssao_tmp;
        self.settings.proj_scale *= prepared.height as f32 / self.height as f32;
        self.width = prepared.width;
        self.height = prepared.height;
        self.settings.inv_resolution = [1.0 / self.width as f32, 1.0 / self.height as f32];
        self.update_settings(queue, self.settings);
        self.invalidate_history();
    }
}
