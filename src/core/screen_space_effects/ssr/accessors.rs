use super::*;

impl SsrRenderer {
    pub fn get_output(&self) -> &TextureView {
        &self.ssr_filtered_view
    }

    pub fn output_texture(&self) -> &Texture {
        &self.ssr_filtered_texture
    }

    pub fn set_scene_color_view(&mut self, view: TextureView) {
        self.scene_color_override = Some(view);
    }

    pub(super) fn clear_scene_color_override(&mut self) {
        self.scene_color_override = None;
    }

    pub fn update_settings(&mut self, queue: &Queue, settings: SsrSettings) {
        self.settings = settings;
        queue.write_buffer(&self.settings_buffer, 0, bytemuck::bytes_of(&self.settings));
    }

    pub fn update_camera(&mut self, queue: &Queue, camera: &CameraParams) {
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(camera));
    }

    pub fn spec_view(&self) -> &TextureView {
        &self.ssr_spec_view
    }

    pub fn final_view(&self) -> &TextureView {
        &self.ssr_final_view
    }

    pub fn hit_data_view(&self) -> &TextureView {
        &self.ssr_hit_view
    }

    pub fn hit_data_texture(&self) -> &Texture {
        &self.ssr_hit_texture
    }

    pub fn composite_view(&self) -> &TextureView {
        &self.ssr_composited_view
    }

    pub fn timings_ms(&self) -> (f32, f32, f32) {
        (
            self.last_trace_ms,
            self.last_shade_ms,
            self.last_fallback_ms,
        )
    }

    pub fn get_settings(&self) -> SsrSettings {
        self.settings
    }

    pub fn set_environment(&mut self, _env_view: &TextureView, _env_sampler: &Sampler) {
        // Deprecated in favor of set_environment_texture; kept for API-compat.
    }

    pub fn set_environment_texture(&mut self, device: &Device, env_texture: &Texture) {
        println!("[SSR] Updating environment texture");
        let view = env_texture.create_view(&TextureViewDescriptor {
            label: Some("p5.ssr.env.view"),
            format: None,
            dimension: Some(TextureViewDimension::Cube),
            aspect: TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
        });
        self.env_view = view;
        self.env_sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("gi.env.cube.sampler"),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Linear,
            ..Default::default()
        });
    }
}
