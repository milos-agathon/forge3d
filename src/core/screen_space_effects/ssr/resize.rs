use super::*;

pub(crate) struct PreparedSsrTargets {
    width: u32,
    height: u32,
    textures: constructor::ViewportTextures,
}

impl SsrRenderer {
    #[cfg(all(test, windows))]
    pub(crate) fn viewport_textures(&self) -> Vec<&crate::core::resource_tracker::TrackedTexture> {
        vec![
            &self._ssr_spec_texture,
            &self._ssr_final_texture,
            &self.ssr_history_texture,
            &self.ssr_filtered_texture,
            &self.ssr_hit_texture,
            &self._ssr_composited_texture,
        ]
    }

    pub(crate) fn prepare_resize(
        &self,
        device: &Device,
        width: u32,
        height: u32,
    ) -> RenderResult<PreparedSsrTargets> {
        Ok(PreparedSsrTargets {
            width,
            height,
            textures: constructor::resources::create_viewport_textures(device, width, height)?,
        })
    }

    pub(crate) fn install_resize(&mut self, prepared: PreparedSsrTargets, queue: &Queue) {
        let textures = prepared.textures;
        self.ssr_spec_view = textures.ssr_spec_view;
        self.ssr_final_view = textures.ssr_final_view;
        self.ssr_history_texture = textures.ssr_history_texture;
        self.ssr_history_view = textures.ssr_history_view;
        self.ssr_filtered_texture = textures.ssr_filtered_texture;
        self.ssr_filtered_view = textures.ssr_filtered_view;
        self.ssr_hit_texture = textures.ssr_hit_texture;
        self.ssr_hit_view = textures.ssr_hit_view;
        self.ssr_composited_view = textures.ssr_composited_view;
        self._ssr_spec_texture = textures.ssr_spec_texture;
        self._ssr_final_texture = textures.ssr_final_texture;
        self._ssr_composited_texture = textures.ssr_composited_texture;
        self.width = prepared.width;
        self.height = prepared.height;
        self.settings.inv_resolution = [1.0 / self.width as f32, 1.0 / self.height as f32];
        self.update_settings(queue, self.settings);
        self.scene_color_override = None;
        self.invalidate_history();
    }
}
