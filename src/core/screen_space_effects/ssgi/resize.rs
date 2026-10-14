use super::*;

pub(crate) struct PreparedSsgiTargets {
    width: u32,
    height: u32,
    textures: constructor::ViewportTextures,
}

impl SsgiRenderer {
    #[cfg(all(test, windows))]
    pub(crate) fn viewport_textures(&self) -> Vec<&crate::core::resource_tracker::TrackedTexture> {
        vec![
            &self.ssgi_hit,
            &self.ssgi_texture,
            &self.ssgi_history,
            &self.ssgi_filtered,
            &self.ssgi_upscaled,
            &self._ssgi_composited,
            &self.scene_history[0],
            &self.scene_history[1],
        ]
    }

    pub(crate) fn prepare_resize(
        &self,
        device: &Device,
        width: u32,
        height: u32,
        material_format: TextureFormat,
    ) -> RenderResult<PreparedSsgiTargets> {
        Ok(PreparedSsgiTargets {
            width,
            height,
            textures: constructor::resources::create_viewport_textures(
                device,
                width,
                height,
                material_format,
                self.half_res,
            )?,
        })
    }

    pub(crate) fn install_resize(&mut self, prepared: PreparedSsgiTargets, queue: &Queue) {
        let textures = prepared.textures;
        self.ssgi_hit = textures.ssgi_hit;
        self.ssgi_hit_view = textures.ssgi_hit_view;
        self.ssgi_texture = textures.ssgi_texture;
        self.ssgi_view = textures.ssgi_view;
        self.ssgi_history = textures.ssgi_history;
        self.ssgi_history_view = textures.ssgi_history_view;
        self.ssgi_filtered = textures.ssgi_filtered;
        self.ssgi_filtered_view = textures.ssgi_filtered_view;
        self.ssgi_upscaled = textures.ssgi_upscaled;
        self.ssgi_upscaled_view = textures.ssgi_upscaled_view;
        self.ssgi_composited_view = textures.ssgi_composited_view;
        self.scene_history = textures.scene_history;
        self.scene_history_views = textures.scene_history_views;
        self._ssgi_composited = textures.ssgi_composited;
        self.width = prepared.width;
        self.height = prepared.height;
        let (width, height) = if self.half_res {
            ((self.width / 2).max(1), (self.height / 2).max(1))
        } else {
            (self.width, self.height)
        };
        self.settings.inv_resolution = [1.0 / width as f32, 1.0 / height as f32];
        self.update_settings(queue, self.settings);
        self.reset_history();
    }
}
